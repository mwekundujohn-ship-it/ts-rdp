#!/usr/bin/env python
r"""pull_store.py - resumable, sharded SFTP puller for the QuantStore.

Runners use this because Windows has no sshpass and the bundled curl.exe has
SFTP compiled out. paramiko does password auth in-process, so nothing else is
needed beyond `pip install paramiko`.

Examples
--------
  # one whole dataset (all 68 monthly files, gz preferred)
  python pull_store.py --host 100.125.90.54 --user quantstore \
      --password PW --remote /ticks/Jump_10_Index --local C:\w\ticks\Jump_10_Index

  # node 2 of 3 (deterministic split, safe to re-run)
  python pull_store.py --host 100.125.90.54 --password PW \
      --remote /ticks/Jump_10_Index --local C:\w\ticks\Jump_10_Index --shard 1/3

  # only the gz copies, skip raw csv
  python pull_store.py ... --only '.csv.gz'

  # upload results back to the writable /data tree
  python pull_store.py ... --push C:\w\out --to /data/alpha/master/w1/stage2

Exit codes: 0 = everything present, 1 = some file failed, 2 = fatal.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import socket
import sys
import time

try:
    import paramiko
except ImportError:  # pragma: no cover
    sys.exit("paramiko missing: python -m pip install --quiet paramiko")


def connect(args):
    """Open a tuned SFTP session. Big window + prefetch is what makes this fast."""
    sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
    transport = paramiko.Transport(sock)
    transport.default_window_size = 32 * 1024 * 1024
    transport.packetizer.REKEY_BYTES = 1 << 30
    transport.packetizer.REKEY_PACKETS = 1 << 30
    transport.banner_timeout = args.timeout
    transport.auth_timeout = args.timeout
    transport.connect(username=args.user, password=args.password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get_channel().settimeout(args.timeout)
    return transport, sftp


def want(name: str, args) -> bool:
    if args.only and not fnmatch.fnmatch(name.lower(), args.only.lower()):
        return False
    if args.skip and fnmatch.fnmatch(name.lower(), args.skip.lower()):
        return False
    return True


def shard_of(names, args):
    """Deterministic split so 3 workers never fetch the same file."""
    if not args.shard:
        return names
    idx, total = args.shard
    if not 0 <= idx < total:
        sys.exit(f"--shard index must be 0..{total - 1}")
    if args.shard_by == "size":
        ordered = sorted(names)
    else:
        ordered = sorted(names)
    return [n for i, n in enumerate(ordered) if i % total == idx]


def pull_one(sftp, remote_path: str, local_path: str, args):
    size = sftp.stat(remote_path).st_size
    have = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    if have == size and size > 0:
        return size, 0.0, "cached"
    if have > size:  # corrupt/partial from another epoch: restart
        os.remove(local_path)
        have = 0
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    part = local_path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else have

    t0 = time.perf_counter()
    with sftp.open(remote_path, "rb") as src:
        if have:
            src.seek(have)
        mode = "ab" if have else "wb"
        with open(part, mode) as dst:
            if args.prefetch:
                src.prefetch(size)
            src.set_pipelined(True)
            while True:
                buf = src.read(1 << 20)
                if not buf:
                    break
                dst.write(buf)
    got = os.path.getsize(part)
    if size and got != size:
        raise IOError(f"short read: {got}/{size}")
    os.replace(part, local_path)
    dt = max(time.perf_counter() - t0, 1e-6)
    return got, dt, "resumed" if have else "new"


def do_pull(sftp, args):
    entries = [a for a in sftp.listdir_attr(args.remote) if not a.st_mode & 0o040000]
    names = [e.filename for e in entries if want(e.filename, args)]
    # gz preference: if both foo.csv and foo.csv.gz exist and no --only given,
    # keep the gz and drop the raw, so we do not pull 4.66x the bytes.
    if not args.only and not args.keep_raw:
        gz = {n[:-3] for n in names if n.lower().endswith(".gz")}
        names = [n for n in names if n.lower().endswith(".gz") or n not in gz]
    names = shard_of(sorted(names), args)
    total = sum(sftp.stat(os.path.join(args.remote, n).replace("\\", "/")).st_size for n in names)
    print(f"pull {len(names)} files / {total / 1048576:.1f} MiB from {args.remote}", flush=True)

    done = failed = 0
    moved = 0
    t_start = time.perf_counter()
    for i, name in enumerate(names, 1):
        remote_path = args.remote.rstrip("/") + "/" + name
        local_path = os.path.join(args.local, name)
        try:
            got, dt, how = pull_one(sftp, remote_path, local_path, args)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(names)}] FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        done += 1
        if how != "cached":
            moved += got
        rate = got / 1048576 / dt if dt else 0
        print(f"[{i}/{len(names)}] {how:7} {name} {got / 1048576:7.1f} MiB {dt:6.1f}s {rate:5.2f} MB/s", flush=True)

    wall = max(time.perf_counter() - t_start, 1e-6)
    print(f"DONE ok={done} fail={failed} moved={moved / 1073741824:.2f} GiB in {wall:.0f}s "
          f"({moved / 1048576 / wall:.2f} MB/s avg)")
    return 0 if failed == 0 else 1


def do_push(sftp, args):
    files = []
    for root, _dirs, names in os.walk(args.push):
        for n in names:
            files.append(os.path.join(root, n))
    files.sort()
    total = sum(os.path.getsize(f) for f in files)
    print(f"push {len(files)} files / {total / 1048576:.1f} MiB -> {args.to}", flush=True)

    def mkdirs(path):
        cur = ""
        for part in path.strip("/").split("/"):
            if not part:
                continue
            cur += "/" + part
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass

    mkdirs(args.to)
    failed = 0
    t_start = time.perf_counter()
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, args.push).replace("\\", "/")
        dest = args.to.rstrip("/") + "/" + rel
        mkdirs(os.path.dirname(dest))
        size = os.path.getsize(f)
        try:
            have = sftp.stat(dest).st_size
        except IOError:
            have = -1
        if have == size:
            print(f"[{i}/{len(files)}] cached {rel} ({size / 1048576:.1f} MiB)", flush=True)
            continue
        try:
            tmp = dest + ".part"
            t0 = time.perf_counter()
            sftp.put(f, tmp, confirm=True)
            try:
                sftp.remove(dest)
            except IOError:
                pass
            sftp.rename(tmp, dest)
            dt = max(time.perf_counter() - t0, 1e-6)
            print(f"[{i}/{len(files)}] put {rel} {size / 1048576:.1f} MiB {dt:.1f}s "
                  f"{size / 1048576 / dt:.2f} MB/s", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(files)}] FAIL {rel}: {type(exc).__name__}: {exc}", flush=True)

    wall = max(time.perf_counter() - t_start, 1e-6)
    print(f"PUSH DONE ok={len(files) - failed} fail={failed} in {wall:.0f}s")
    return 0 if failed == 0 else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="resumable sharded SFTP sync for QuantStore")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--user", default="quantstore")
    p.add_argument("--password", default=os.environ.get("QUANTSTORE_PW", ""))
    p.add_argument("--remote", help="remote dir to pull from (store-absolute, e.g. /ticks/X)")
    p.add_argument("--local", help="local dir to pull into")
    p.add_argument("--shard", help="INDEX/TOTAL, e.g. 1/3")
    p.add_argument("--shard-by", choices=["name", "size"], default="name")
    p.add_argument("--only", help="glob filter, e.g. '*.csv.gz'")
    p.add_argument("--skip", help="glob exclusion")
    p.add_argument("--keep-raw", action="store_true", help="keep .csv when a .csv.gz exists")
    p.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    p.add_argument("--push", help="local dir to upload")
    p.add_argument("--to", help="remote dir for --push, e.g. /data/alpha/master/w1/s2")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args(argv)

    if not args.password:
        sys.exit("no password: pass --password or set QUANTSTORE_PW")
    if args.shard:
        try:
            a, b = args.shard.split("/")
            args.shard = (int(a), int(b))
        except Exception:
            sys.exit("--shard must look like 1/3")
    if args.push and not args.to:
        sys.exit("--push needs --to")
    if not args.push and not (args.remote and args.local):
        sys.exit("need --remote and --local (or --push/--to)")

    transport = sftp = None
    try:
        transport, sftp = connect(args)
        if args.push:
            return do_push(sftp, args)
        return do_pull(sftp, args)
    except paramiko.AuthenticationException:
        return print("AUTH FAILED - check the store password") or 2
    except Exception as exc:
        return print(f"FATAL {type(exc).__name__}: {exc}") or 2
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()


if __name__ == "__main__":
    sys.exit(main())

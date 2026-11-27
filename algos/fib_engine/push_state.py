"""Push a stage directory to the QuantStore and GitHub in one shot.

    python push_state.py --dir out/encode-15m-0of3 --stage encode-15m-0of3 \
        --fleet alpha --wave w1 --node runnervm-abc \
        --repo . --branch quant/w1

Store : everything in the stage dir, over SFTP, into /data/<fleet>/<node>/<wave>/<stage>
GitHub: code, logs, JSON, chat, catalogs and any file under --git-max-bytes

The store leg is delegated to pull_store.py (same SFTP client, same host), so the
password comes from the QUANTSTORE_PW environment variable and never appears in argv.
The stage is only marked pushed when both legs succeed; the exit code is the number of
failed legs, so a runner wrapper can retry the whole push safely.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

STORE_HOST = os.environ.get("QUANTSTORE_HOST", "100.125.90.54")
STORE_ROOT = {"alpha": "/data/alpha", "beta": "/data/beta"}
HERE = Path(__file__).resolve().parent


def find_puller() -> Path | None:
    """The prompt writes tools\\pull_store.py; older layouts put it beside us."""
    for cand in (
        HERE / "pull_store.py",
        HERE.parent.parent / "tools" / "pull_store.py",
        Path.cwd() / "tools" / "pull_store.py",
        Path.cwd() / "pull_store.py",
    ):
        if cand.is_file():
            return cand
    return None


def run(cmd: list[str], cwd: str | None = None, quiet: bool = False) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if not quiet:
        print(f"$ {' '.join(cmd[:4])}... -> {p.returncode}")
    if p.returncode != 0:
        print(out[-2000:])
    return p.returncode, out


def push_store(stage_dir: Path, fleet: str, wave: str, node: str) -> int:
    if not os.environ.get("QUANTSTORE_PW"):
        print("store: QUANTSTORE_PW is not set")
        return 1
    puller = find_puller()
    if not puller:
        print("store: pull_store.py not found (expected tools\\pull_store.py)")
        return 1
    dest = f"{STORE_ROOT[fleet]}/{node}/{wave}/{stage_dir.name}"
    rc, _ = run([
        sys.executable, str(puller),
        "--host", STORE_HOST, "--user", "quantstore",
        "--push", str(stage_dir), "--to", dest, "--timeout", "120",
    ])
    return rc


def push_git(stage_dir: Path, repo: Path, branch: str, wave: str, node: str,
             max_bytes: int) -> int:
    if not (repo / ".git").exists():
        print(f"git: {repo} is not a repository")
        return 1
    dest = repo / "runs" / wave / node / stage_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in stage_dir.rglob("*"):
        if not p.is_file() or p.stat().st_size > max_bytes:
            continue
        rel = p.relative_to(stage_dir)
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, tgt)
        copied += 1
    print(f"git: staged {copied} files under {dest}")

    cmds = [
        ["git", "checkout", "-B", branch],
        ["git", "add", str(dest.relative_to(repo))],
        ["git", "-c", "user.name=quant-runner", "-c", "user.email=runner@local",
         "commit", "-m", f"{stage_dir.name}: stage packet from {node}"],
    ]
    for c in cmds:
        rc, out = run(c, cwd=str(repo))
        if rc != 0:
            if "nothing to commit" in out:
                print("git: nothing to commit")
                continue
            return rc
    rc, _ = run(["git", "push", "-u", "origin", branch, "--force-with-lease"], cwd=str(repo))
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--fleet", required=True, choices=sorted(STORE_ROOT))
    ap.add_argument("--wave", required=True)
    ap.add_argument("--node", default=os.environ.get("HOSTNAME", "node"))
    ap.add_argument("--repo", default=".")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--git-max-bytes", type=int, default=25 * 1024 * 1024)
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--no-store", "--no-mega", dest="no_store", action="store_true")
    ap.add_argument("--store-only", action="store_true")
    args = ap.parse_args(argv)

    stage_dir = Path(args.dir)
    if not stage_dir.is_dir():
        print(f"missing stage dir: {stage_dir}")
        return 2

    failed = 0
    if not args.no_store:
        failed += push_store(stage_dir, args.fleet, args.wave, args.node) != 0
    if not args.no_git and not args.store_only:
        failed += push_git(stage_dir, Path(args.repo), args.branch or f"quant/{args.wave}",
                           args.wave, args.node, args.git_max_bytes) != 0

    print(f"done: {failed} failed leg(s)")
    return failed


if __name__ == "__main__":
    sys.exit(main())

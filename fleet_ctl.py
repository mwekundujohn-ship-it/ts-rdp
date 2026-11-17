import argparse
import base64
import datetime as dt
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
CFG_PATH = BASE / "config.json"
ROLES = ("master", "worker1", "worker2")


def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def fleet_cfg(cfg, name):
    merged = {k: cfg.get(k) for k in ("repo", "workflow", "ref", "password", "authkey", "durationMinutes", "nvidiaKey", "mt5Login", "mt5Password", "mt5Server", "megaEmail", "megaPassword", "fleet")}
    merged.update(cfg.get("fleets", {}).get(name, {}))
    return merged


def gh(cfg, path, method="GET", payload=None):
    url = f"https://api.github.com/repos/{cfg['owner']}/{cfg['repo']}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"token {cfg['token']}",
        "User-Agent": "fleet-ctl",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            raw = res.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"github {method} {path}: {exc.code} {detail}")


def get_json_file(cfg, path):
    obj = gh(cfg, f"/contents/{path}")
    return json.loads(base64.b64decode(obj["content"].replace("\n", "")).decode("utf-8"))


def put_json_file(cfg, path, obj, message):
    endpoint = f"/contents/{path}"
    sha = None
    try:
        sha = gh(cfg, endpoint)["sha"]
    except Exception:
        pass
    payload = {"message": message, "content": base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii"), "branch": cfg["ref"]}
    if sha:
        payload["sha"] = sha
    gh(cfg, endpoint, "PUT", payload)


def run_list(cfg):
    obj = gh(cfg, f"/actions/workflows/{cfg['workflow']}/runs?per_page=20")
    return obj.get("workflow_runs", [])


def dispatch_inputs(cfg, role):
    inputs = {
        "fleet": str(cfg.get("fleet") or "alpha"),
        "node_role": role,
        "duration_minutes": str(cfg["durationMinutes"]),
        "rdp_password": str(cfg["password"]),
        "ts_authkey": str(cfg["authkey"]),
        "nvidia_api_key": str(cfg["nvidiaKey"]),
    }
    mapping = {
        "mt5_login": "mt5Login",
        "mt5_password": "mt5Password",
        "mt5_server": "mt5Server",
        "mega_email": "megaEmail",
        "mega_password": "megaPassword",
    }
    for key, name in mapping.items():
        value = str(cfg.get(name) or "")
        if value and not value.startswith("<"):
            inputs[key] = value
    return inputs


def dispatch_role(cfg, role):
    before = max([int(r["id"]) for r in run_list(cfg)] or [0])
    gh(cfg, f"/actions/workflows/{cfg['workflow']}/dispatches", "POST", {"ref": cfg["ref"], "inputs": dispatch_inputs(cfg, role)})
    deadline = time.time() + 180
    while time.time() < deadline:
        ids = [int(r["id"]) for r in run_list(cfg) if int(r["id"]) > before]
        if ids:
            return max(ids)
        time.sleep(3)
    raise TimeoutError(f"no run created for {role}")


def wait_port(ip, port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=3):
                return True
        except OSError:
            time.sleep(3)
    return False


def wait_ready(cfg, run_id, role, fleet, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            st = get_json_file(cfg, f"node-{fleet}-{role}.json")
            if str(st.get("run_id")) == str(run_id) and st.get("status") == "ready":
                return st
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"timed out waiting for {role} run {run_id}")


def send_worker(ip, action, prompt):
    payload = json.dumps({"action": action, "cmd": prompt}).encode("utf-8")
    with socket.create_connection((ip, 5555), timeout=15) as sock:
        sock.settimeout(300)
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            part = sock.recv(65536)
            if not part:
                break
            chunks.append(part)
    return json.loads(b"".join(chunks).decode("utf-8"))


def spawn(name, replace):
    cfg = fleet_cfg(load_cfg(), name)
    old = [str(r["id"]) for r in run_list(cfg) if r.get("status") in ("queued", "in_progress")]
    nodes = []
    expires = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=int(cfg["durationMinutes"]) + 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for role in ROLES:
        run_id = dispatch_role(cfg, role)
        print(f"{name}/{role}: run {run_id} queued", flush=True)
        st = wait_ready(cfg, run_id, role, name)
        node = {
            "role": role,
            "run_id": str(run_id),
            "ip": st.get("ip"),
            "host": st.get("host") or st.get("self"),
            "self": st.get("self"),
            "user": st.get("user") or "rdp",
            "port": 3389,
            "worker_port": 5555,
            "expires": expires,
            "ready": True,
        }
        if not wait_port(node["ip"], 5555, 90):
            raise TimeoutError(f"{name}/{role}: port 5555 never opened on {node['ip']}")
        if role == "master" and not wait_port(node["ip"], 3389, 90):
            raise TimeoutError(f"{name}/{role}: port 3389 never opened on {node['ip']}")
        nodes.append(node)
        print(f"{name}/{role}: ready ip={node['ip']} host={node['host']}", flush=True)
    if replace:
        for run_id in old:
            try:
                gh(cfg, f"/actions/runs/{run_id}/cancel", "POST")
            except Exception:
                pass
    for node in nodes:
        response = send_worker(node["ip"], "fast_cmd", "echo connected")
        if response.get("status") != "completed":
            raise RuntimeError(f"{name}/{node['role']}: connect check failed: {response}")
    print(f"{name}: all 3 nodes connected", flush=True)
    result = {"status": "ready", "fleet": name, "nodes": nodes}
    local = BASE / f"cloud_nodes-{name}.json"
    local.write_text(json.dumps(result, indent=2), encoding="utf-8")
    registry = cfg.get("registry") or f"registry/fleet-{name}.json"
    put_json_file(cfg, registry, result, f"fleet {name} ready")
    print(json.dumps(result, indent=2))


def spawn_all(replace):
    for name in ("alpha", "beta"):
        spawn(name, replace)


def cmd(name, node_role, action, prompt):
    path = BASE / f"cloud_nodes-{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    node = next((n for n in data["nodes"] if n["role"] == node_role), None)
    if not node:
        raise SystemExit(f"no {node_role} in {path}")
    response = send_worker(node["ip"], action, prompt)
    print(response.get("stdout", ""))
    if response.get("stderr"):
        print(response["stderr"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    spawn_parser = sub.add_parser("spawn")
    spawn_parser.add_argument("--fleet", choices=("alpha", "beta", "all"), default="all")
    spawn_parser.add_argument("--replace", action="store_true")
    cmd_parser = sub.add_parser("cmd")
    cmd_parser.add_argument("--fleet", choices=("alpha", "beta"), default="alpha")
    cmd_parser.add_argument("--node", choices=ROLES, default="master")
    cmd_parser.add_argument("--action", choices=("fast_cmd", "opencode"), default="opencode")
    cmd_parser.add_argument("prompt", nargs="+")
    args = parser.parse_args()
    if args.command == "spawn":
        if args.fleet == "all":
            spawn_all(args.replace)
        else:
            spawn(args.fleet, args.replace)
    else:
        cmd(args.fleet, args.node, args.action, " ".join(args.prompt))


if __name__ == "__main__":
    main()

import argparse
import json
import socket
import subprocess
from pathlib import Path

REGISTRY = Path(r"C:\cluster\workers.json")


def local_ips():
    try:
        output = subprocess.run(
            [r"C:\Program Files\Tailscale\tailscale.exe", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return {line.strip() for line in output.splitlines() if line.strip()}
    except Exception:
        return set()


def local_run(prompt):
    try:
        p = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", prompt],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {"status": "completed", "stdout": p.stdout, "stderr": p.stderr, "code": p.returncode}
    except Exception as e:
        return {"status": "error", "stderr": str(e)}

STATUS_PS = r'''
$py = @(Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { [ordered]@{ id = $_.Id; cpu = [math]::Round($_.CPU, 1); start = $_.StartTime.ToString('HH:mm:ss') } })
$mt = @(Get-Process terminal64 -ErrorAction SilentlyContinue | ForEach-Object { [ordered]@{ id = $_.Id; cpu = [math]::Round($_.CPU, 1); start = $_.StartTime.ToString('HH:mm:ss') } })
$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$log = @(Get-ChildItem 'C:\DerivMT5\logs' -Filter '20*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)[0]
$last = ''
if ($log) { $last = [string](Get-Content $log.FullName -Tail 1) }
$o = [ordered]@{
  host = $env:COMPUTERNAME
  time = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  uptime = ((Get-Date) - $boot).ToString()
  python = $py
  mt5 = $mt
  mt5_last_log = $last
}
$o | ConvertTo-Json -Compress -Depth 5
'''


def read_request(conn):
    data = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
        try:
            return json.loads(data.decode("utf-8", errors="replace"))
        except Exception:
            continue
    return None


def send(ip, action, prompt):
    payload = json.dumps({"action": action, "cmd": prompt}).encode("utf-8")
    with socket.create_connection((ip, 5555), timeout=15) as sock:
        sock.settimeout(300)
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode("utf-8", errors="replace"))


def load_registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    sub.add_parser("status")
    sub.add_parser("connect")
    command = sub.add_parser("cmd")
    command.add_argument("--node", required=True)
    command.add_argument("--action", default="fast_cmd", choices=("fast_cmd", "opencode"))
    command.add_argument("prompt", nargs="+")
    args = parser.parse_args()
    registry = load_registry()
    nodes = registry.get("nodes", [])
    if args.command == "list":
        print(json.dumps(registry, indent=2))
        return
    if args.command == "connect":
        for node in nodes:
            print(f"mstsc /v:{node['ip']} /admin")
        return
    if args.command == "check":
        ips = local_ips()
        for node in nodes:
            response = local_run("echo alive") if node["ip"] in ips else send(node["ip"], "fast_cmd", "echo alive")
            print(f"{node['role']:<8} {node['ip']:<15} {response.get('status')} {response.get('stdout', '').strip()}")
        return
    if args.command == "status":
        ips = local_ips()
        for node in nodes:
            response = local_run(STATUS_PS) if node["ip"] in ips else send(node["ip"], "fast_cmd", STATUS_PS)
            raw = response.get("stdout", "").strip()
            try:
                detail = json.loads(raw)
            except Exception:
                detail = {"raw": raw}
            summary = {"role": node["role"], "ip": node["ip"], "status": response.get("status")}
            summary.update(detail)
            print(json.dumps(summary, separators=(",", ":")))
        return
    node = next((n for n in nodes if n["role"] == args.node), None)
    if not node:
        raise SystemExit(f"no worker {args.node}")
    response = send(node["ip"], args.action, " ".join(args.prompt))
    print(response.get("stdout", ""))
    if response.get("stderr"):
        print(response["stderr"])


if __name__ == "__main__":
    main()

import socket
import subprocess
import json
import os

WORKDIR = r"C:\mt5-opencode"
OPENCODE_BIN = r"C:\ProgramData\opencode\opencode.cmd"
os.environ["OPENCODE_CONFIG"] = r"C:\ProgramData\opencode\opencode.json"
os.environ["MT5_OPENCODE_HOME"] = r"C:\mt5-opencode"


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


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 5555))
s.listen(5)
while True:
    conn, addr = s.accept()
    try:
        req = read_request(conn)
        if not req:
            raise ValueError("empty request")
        action = req.get("action", "fast_cmd")
        prompt = req.get("cmd", "")
        if action == "set_self":
            with open(r"C:\cluster\self.json", "w", encoding="utf-8") as fh:
                json.dump(prompt, fh)
            res = {"status": "completed", "stdout": "self.json updated"}
        elif action == "set_workers":
            with open(r"C:\cluster\workers.json", "w", encoding="utf-8") as fh:
                json.dump(prompt, fh)
            res = {"status": "completed", "stdout": "workers.json updated"}
        elif action == "opencode":
            p = subprocess.run(["cmd.exe", "/c", OPENCODE_BIN, "run", "--continue", "--auto", prompt], cwd=WORKDIR, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
            res = {"status": "completed", "stdout": p.stdout, "stderr": p.stderr, "code": p.returncode}
        else:
            p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", prompt], cwd=WORKDIR, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
            res = {"status": "completed", "stdout": p.stdout, "stderr": p.stderr, "code": p.returncode}
    except Exception as e:
        res = {"status": "error", "stderr": str(e)}
    try:
        conn.sendall(json.dumps(res).encode("utf-8"))
    except Exception:
        pass
    conn.close()

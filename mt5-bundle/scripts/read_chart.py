import json
import sys
from pathlib import Path

def load_chart_state(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception as e:
        print(f"ERR: cannot read {path}: {e}", file=sys.stderr)
        return None

def main():
    path = Path(r"C:\Users\MICROAPLE\AppData\Roaming\MetaQuotes\Terminal\FB9A56D617EDDDFE29EE54EBEFFE96C1\MQL5\Files\opencode\chart_state.json")
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--path", "-p") and len(sys.argv)>2:
            path = Path(sys.argv[2])
    if not path.exists():
        print("ERR: chart_state.json not found. Is ChartReaderEA attached and running?", file=sys.stderr)
        sys.exit(1)
    data = load_chart_state(path)
    if not data: sys.exit(2)
    print(f"symbol={data.get('symbol')} tf={data.get('timeframe','')} bid={data.get('bid')} ask={data.get('ask')} spread={data.get('spread')}")
    for w in data.get('windows', []):
        print(f"window {w.get('index')} sub={w.get('subwindow')} indicators={len(w.get('indicators',[]))}")
        for i in w.get('indicators',[]):
            bufs = i.get('buffers') or []
            print(f"  {i.get('name')} (sw={i.get('subwindow')}, h={i.get('handle')}): {len(bufs)} buffers")
    if any(a in sys.argv for a in ("--json","-j")):
        print(json.dumps(data, indent=2 if "--pretty" in sys.argv else None))
    return 0

if __name__ == "__main__":
    sys.exit(main())

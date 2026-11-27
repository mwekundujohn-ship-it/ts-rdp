#!/usr/bin/env python3
r"""mt5.py - opencode <-> MetaTrader 5 automation channel.

Actions:
  eas                                    list compiled EAs in MQL5\Experts
  compile <source>                       headless MetaEditor compile + log parse
  backtest --ea --symbol --from --to ... portable-clone Strategy Tester run + report parse
  chart                                  read chart_state.json (ChartReaderEA bridge)

Config: C:\mt5-opencode\config\mt5.json (or %MT5_OPENCODE_HOME%\config\mt5.json)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ.get("MT5_OPENCODE_HOME", r"C:\mt5-opencode"))
CONFIG_PATH = HOME / "config" / "mt5.json"
MODELS = {
    "every_tick": 0, "ohlc_1min": 1, "open_prices": 2,
    "m1_ohlc": 3, "real_ticks": 4, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
}


def load_cfg():
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_common(clone: Path) -> list[str]:
    p = clone / "config" / "startup.ini"
    if not p.exists():
        return []
    out = ["[Common]"]
    for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        s = line.strip()
        if s.startswith(("Login=", "Password=", "Server=")):
            out.append(s)
    return out if len(out) > 1 else []


# ---------------------------------------------------------------- eas
def cmd_eas(args):
    cfg = load_cfg()
    exps = Path(cfg["mql5_dir"]) / "Experts"
    out = []
    for p in sorted(exps.rglob("*.ex5")):
        out.append(str(p.relative_to(exps)).replace("\\", "/"))
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for r in out:
            print(r)
    return 0


# ------------------------------------------------------------- compile
def cmd_compile(args):
    cfg = load_cfg()
    src = Path(args.source)
    if not src.is_absolute():
        cands = [
            src,
            Path(cfg["mql5_dir"]) / args.source,
            Path(cfg["mql5_dir"]).parent / args.source,
        ]
        for c in cands:
            if c.exists():
                src = c
                break
    if not src.exists():
        print(f"ERR: source not found: {args.source}", file=sys.stderr)
        return 1

    log = Path(cfg["bridge_dir"]) / "compile.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [cfg["metaeditor_exe"], f"/compile:{src}", f"/log:{log}"],
        capture_output=True,
    )
    if not log.exists():
        print("ERR: no compile log produced", file=sys.stderr)
        return 3
    txt = log.read_text(encoding="utf-16", errors="replace")
    m = re.findall(r"Result:\s*(\d+)\s*errors?,\s*(\d+)\s*warnings?", txt)
    if m:
        errs, warns = int(m[-1][0]), int(m[-1][1])
        print(f"compile: {errs} errors, {warns} warnings  ({src})")
        for line in txt.splitlines():
            if "error" in line.lower() or "Result:" in line:
                print("  " + line.strip())
        if args.json:
            print(json.dumps({"errors": errs, "warnings": warns}))
        return 0 if errs == 0 else 2
    print(txt[-4000:])
    return 3


# ------------------------------------------------------------ backtest
def cmd_backtest(args):
    cfg = load_cfg()
    clone = Path(cfg["clone_dir"])
    if not (clone / "terminal64.exe").exists():
        print(
            "ERR: portable clone missing at %s\n"
            "Build it once:\n"
            '  robocopy "C:\\Program Files\\MetaTrader 5 Terminal" "%s" /E /NFL /NDL /NJH /NJS' % (clone, clone),
            file=sys.stderr,
        )
        return 1

    exp_rel = args.ea
    if exp_rel.lower().endswith(".ex5"):
        exp_rel = exp_rel[:-4]
    exp_rel = exp_rel.replace("/", "\\")
    if not exp_rel.lower().endswith(".ex5"):
        exp_rel = exp_rel + ".ex5"

    # mirror newest .ex5 from live data dir into the clone (same paths exist)
    live_ex5 = Path(cfg["mql5_dir"]) / "Experts" / exp_rel
    if live_ex5.exists():
        dst = clone / "MQL5" / "Experts" / exp_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(live_ex5.read_bytes())

    reports = clone / "MQL5" / "Files" / "opencode" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    name = args.name or ("bt_%d" % (abs(hash(args.ea + args.symbol + args.to)) % 1000000))
    for ext in (".htm", ".log", ".tst"):
        (reports / (name + ext)).unlink(missing_ok=True)

    ini = reports / "bt_run.ini"
    model = MODELS.get(str(args.model).lower(), MODELS.get(str(args.model), 2))
    lines = read_common(clone)
    lines += [
        "[Tester]",
        "Expert=" + args.ea.replace("/", "\\").replace(".ex5", ""),
        "Symbol=" + args.symbol,
        "Period=" + args.period,
        "Model=" + str(model),
        "FromDate=" + args.from_date,
        "ToDate=" + args.to,
        "Deposit=%s" % (args.deposit or 10000),
        "Currency=" + (args.currency or "USD"),
        "Leverage=%s" % (args.leverage or 1000),
        "Optimization=0",
        "Visual=0",
        "ShutdownTerminal=1",
        "ReplaceReport=1",
        "Report=MQL5\\Files\\opencode\\reports\\" + name,
    ]
    if args.inputs:
        lines.append("[TesterInputs]")
        lines.extend(args.inputs)
    ini.write_text("\n".join(lines), encoding="utf-8")

    cmd = [clone / "terminal64.exe", "/portable", f"/config:{ini}"]
    try:
        proc = subprocess.run(
            [str(c) for c in cmd], timeout=args.timeout or 600
        )
    except subprocess.TimeoutExpired:
        print("ERR: backtest timed out", file=sys.stderr)
        return 4
    except FileNotFoundError as e:
        print("ERR: %s" % e, file=sys.stderr)
        return 1

    htm = reports / (name + ".htm")
    if not htm.exists():
        print(
            "ERR: report not produced (exit=%s). See %s" % (proc.returncode, clone / "MQL5\\Files\\opencode\\reports"),
            file=sys.stderr,
        )
        return 5
    metrics = parse_report(htm)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        for k, v in metrics.items():
            print("%-42s %s" % (k, v))
    return 0


def parse_report(htm: Path) -> dict:
    raw_bytes = htm.read_bytes()
    if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif raw_bytes[:3] == b"\xef\xbb\xbf":
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    raw = raw_bytes.decode(enc, errors="replace")
    txt = re.sub(r"<[^>]+>", "|", raw)
    txt = txt.replace("&nbsp;", " ").replace("&mdash;", "-")
    txt = re.sub(r"\|+", " | ", txt)
    txt = txt.replace("\r", "").replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt)
    low = txt.lower()

    def grab(labels):
        for lab in labels:
            i = low.find(lab.lower())
            if i != -1:
                seg = txt[i + len(lab):]
                seg = seg.lstrip(" :|")
                seg = seg.split("|")[0] if "|" in seg else seg[:60]
                seg = seg.strip()
                if seg:
                    return seg
        return None

    order = [
        ("Start Date", ["Start date"]),
        ("Start Time", ["Start time"]),
        ("End Date", ["End date"]),
        ("End Time", ["End time"]),
        ("Model", ["Model"]),
        ("Parameters", ["Parameters"]),
        ("Initial Deposit", ["Initial deposit"]),
        ("Total Net Profit", ["Total net profit"]),
        ("Gross Profit", ["Gross profit"]),
        ("Gross Loss", ["Gross loss"]),
        ("Profit Factor", ["Profit factor"]),
        ("Total Trades", ["Total trades"]),
        ("Short Trades", ["Short trades", "Short trades (won %)"]),
        ("Long Trades", ["Long trades", "Long trades (won %)"]),
        ("Balance Drawdown Maximal", ["Balance drawdown maximal", "Balance drawdown, maximal"]),
        ("Equity Drawdown Maximal", ["Equity drawdown maximal"]),
        ("Sharpe Ratio", ["Sharpe ratio"]),
        ("Recovery Factor", ["Recovery factor"]),
        ("History Quality", ["History quality", "Quality of history"]),
    ]
    metrics = {}
    for key, labels in order:
        v = grab(labels)
        if v is not None:
            metrics[key] = v
    return metrics


# --------------------------------------------------------------- chart
def cmd_chart(args):
    cfg = load_cfg()
    p = Path(cfg["chart_state_file"])
    if not p.exists():
        print("ERR: chart_state.json not found - is ChartReaderEA attached?", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    if args.json:
        print(json.dumps(data, indent=2 if args.pretty else None))
        return 0
    print("symbol=%s tf=%s bid=%s ask=%s spread=%s bars=%s" % (
        data.get("symbol"), data.get("timeframe"), data.get("bid"),
        data.get("ask"), data.get("spread"), data.get("bars")))
    for w in data.get("windows", []):
        print("window %s sub=%s indicators=%s" % (
            w.get("index"), w.get("subwindow"), len(w.get("indicators", []))))
        for i in w.get("indicators", []):
            bufs = i.get("buffers") or []
            print("   %s (sw=%s): %d buffers" % (i.get("name"), i.get("subwindow"), len(bufs)))
    return 0


def main():
    p = argparse.ArgumentParser(prog="mt5")
    sp = p.add_subparsers(dest="cmd")

    c = sp.add_parser("eas"); c.add_argument("--json", action="store_true")
    c = sp.add_parser("compile"); c.add_argument("source"); c.add_argument("--json", action="store_true")
    c = sp.add_parser("backtest")
    c.add_argument("--ea", required=True)
    c.add_argument("--symbol", default="BTCUSD")
    c.add_argument("--period", default="H1")
    c.add_argument("--from", dest="from_date", default="2025.01.01")
    c.add_argument("--to", default="2026.10.14")
    c.add_argument("--model", default="ohlc_1min")
    c.add_argument("--deposit", type=float, default=10000)
    c.add_argument("--currency", default="USD")
    c.add_argument("--leverage", default="1000")
    c.add_argument("--input", dest="inputs", action="append")
    c.add_argument("--name")
    c.add_argument("--timeout", type=int, default=600)
    c.add_argument("--json", action="store_true")
    c = sp.add_parser("chart")
    c.add_argument("--json", action="store_true")
    c.add_argument("--pretty", action="store_true")

    args = p.parse_args()
    if args.cmd == "eas":
        return cmd_eas(args)
    if args.cmd == "compile":
        return cmd_compile(args)
    if args.cmd == "backtest":
        return cmd_backtest(args)
    if args.cmd == "chart":
        return cmd_chart(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

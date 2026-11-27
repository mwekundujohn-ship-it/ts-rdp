"""Stage 5/6: Macro cascade + extremum-hold execution backtest.

This stage:
1. Loads anchor vault and cascade results
2. Runs extremum-hold execution backtest (L6)
3. Reports: trades, win rate, PF, expectancy in sigma units, max DD,
   exit-node distribution, per-side metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit
import cascade
import compare
import dalton
import epicycle
import lattice


# ========================================================================
# Load tick data for backtest
# ========================================================================

def load_ticks_for_backtest(tick_dir: str, start_ms: int, end_ms: int):
    """Load tick data for a specific time range."""
    import glob
    import pandas as pd

    files = sorted(glob.glob(os.path.join(tick_dir, "*.csv.gz")))
    if not files:
        files = sorted(glob.glob(os.path.join(tick_dir, "*.csv")))

    ms_parts, px_parts = [], []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["epoch_ms", "bid", "ask"])
            ms = df["epoch_ms"].to_numpy(dtype=np.int64)
            px = 0.5 * (df["bid"].to_numpy(dtype=float) + df["ask"].to_numpy(dtype=float))
        except Exception:
            continue
        if ms.size == 0:
            continue
        ok = np.isfinite(ms) & np.isfinite(px)
        ms, px = ms[ok], px[ok]
        mask = (ms >= start_ms) & (ms < end_ms)
        if np.any(mask):
            ms_parts.append(ms[mask])
            px_parts.append(px[mask])

    if not ms_parts:
        return np.empty(0, dtype=np.int64), np.empty(0)

    ms = np.concatenate(ms_parts)
    px = np.concatenate(px_parts)
    order = np.argsort(ms, kind="stable")
    ms, px = ms[order], px[order]
    keep = np.concatenate(([True], np.diff(ms) > 0))
    return ms[keep].astype(np.int64), px[keep]


# ========================================================================
# Build forward path for a matched anchor family
# ========================================================================

def build_forward_path(anchor_matches: list, catalog: list, curves: dict, n_nodes: int = 80):
    """Build the empirical forward path from matched anchors.

    anchor_matches: list of match dicts for this anchor
    catalog: catalog rows
    curves: curve data for the timeframe
    """
    # Get forward curves (nodes 80-159) for each matched window
    forward_curves = []
    curve_stack = curves["curves"]
    price_idx = list(curves["order"]).index("price")

    for m in anchor_matches:
        window_b = m["window_b"]
        if window_b < len(curve_stack):
            price_curve = curve_stack[window_b, price_idx, 80:]  # forward leg
            if len(price_curve) > 0:
                forward_curves.append(price_curve)

    if not forward_curves:
        return None

    # Pad to same length
    max_len = max(len(c) for c in forward_curves)
    padded = []
    for c in forward_curves:
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)), mode="edge")
        padded.append(c)

    mean_forward = np.mean(padded, axis=0)
    return mean_forward


# ========================================================================
# Extremum-hold backtest
# ========================================================================

def run_backtest(
    vault: list,
    catalogs: dict,
    curves: dict,
    tick_dir: str,
    spread_tick: float = 0.5,
    max_trades: int = 0,
) -> dict:
    """Run extremum-hold execution backtest on vault anchors."""

    all_trades = []
    trade_id = 0

    for entry in vault:
        tf = entry["tf"]
        window_id = entry["window_id"]
        direction = 1 if entry["mean_forward_corr"] > 0 else -1

        # Get window metadata
        cat = catalogs[tf]
        if window_id >= len(cat):
            continue
        row = cat[window_id]

        # Window time bounds
        start_ms = int(row["start_ms"])
        stop_ms = int(row["stop_ms"])

        # Load ticks for this window + forward period
        # Forward period = window duration
        window_dur = stop_ms - start_ms
        forward_end = stop_ms + window_dur

        ms, px = load_ticks_for_backtest(tick_dir, start_ms, forward_end)
        if len(ms) < 100:
            continue

        # Find the anchor extremum (node 40 = tau=-2)
        # Resample window to lattice
        window_mask = (ms >= start_ms) & (ms < stop_ms)
        if np.sum(window_mask) < 30:
            continue

        w_ms = ms[window_mask]
        w_px = px[window_mask]

        # Resample to 160 nodes
        ys = lattice.resample(w_ms, w_px)
        sigma_val = float(np.std(ys))
        if sigma_val < 1e-12:
            continue

        # Anchor price at node 40
        anchor_idx = min(40, len(ys) - 1)
        anchor_price = ys[anchor_idx]

        # Entry at node 80 (tau=0) = last price of window
        entry_price = ys[-1] if len(ys) > 80 else ys[-1]

        # Stop at anchor extremum +/- 0.15 sigma
        stop_price = anchor_price - direction * 0.15 * sigma_val

        # Build forward path from matches
        # (In production, use the actual matched anchor family)
        # For now, use the actual forward ticks
        forward_mask = (ms >= stop_ms) & (ms < forward_end)
        if np.sum(forward_mask) < 10:
            continue

        fwd_ms = ms[forward_mask]
        fwd_px = px[forward_mask]

        # Resample forward to 80 nodes
        fwd_ys = lattice.resample(fwd_ms, fwd_px, n=80)

        # Target = empirical extremum of forward path
        signed_fwd = direction * fwd_ys
        peak_idx = int(np.argmax(signed_fwd))
        target_price = fwd_ys[peak_idx]

        # Dynamic exit check: correlation with anchor family forward path
        # For now, use simple correlation with mean forward
        # (In production, compare with matched family's mean forward)
        exit_node = peak_idx
        exit_reason = "target"

        # Check dynamic exit
        if len(fwd_ys) > 5:
            # Rolling correlation
            for i in range(10, len(fwd_ys)):
                if i < 20:
                    continue
                ref = fwd_ys[:i]
                live = fwd_ys[max(0, i-20):i]
                if len(ref) > 5 and len(live) > 5:
                    corr = np.corrcoef(ref, live)[0, 1]
                    if corr < 0.50:
                        exit_node = i
                        exit_reason = "lock_lost"
                        target_price = fwd_ys[i]
                        break

        # Check stop hit
        d = direction
        min_price = np.min(fwd_ys[:exit_node+1]) if d > 0 else np.max(fwd_ys[:exit_node+1])
        if d > 0 and min_price <= stop_price:
            exit_reason = "stop"
            target_price = stop_price
        elif d < 0 and min_price >= stop_price:
            exit_reason = "stop"
            target_price = stop_price

        # PnL in sigma units
        pnl = d * (target_price - entry_price)
        pnl_sigma = pnl / sigma_val

        # Risk in sigma units
        risk = d * (entry_price - stop_price)
        risk_sigma = risk / sigma_val

        # R-multiple
        r_multiple = pnl_sigma / risk_sigma if risk_sigma != 0 else 0

        all_trades.append({
            "trade_id": trade_id,
            "tf": tf,
            "window_id": window_id,
            "direction": int(direction),
            "entry_price": float(entry_price),
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "exit_price": float(target_price),
            "exit_node": int(exit_node),
            "exit_reason": exit_reason,
            "pnl_sigma": float(pnl_sigma),
            "risk_sigma": float(risk_sigma),
            "r_multiple": float(r_multiple),
            "sigma": float(sigma_val),
            "anchor_node": int(row.get("anchor_node", 40)),
            "dpoc": float(row.get("dpoc", 0)),
            "bimodality": float(row.get("bimodality", 0)),
            "verified": entry.get("verified_profitable", False),
        })
        trade_id += 1

        if max_trades and trade_id >= max_trades:
            break

    return all_trades


# ========================================================================
# Metrics computation
# ========================================================================

def compute_metrics(trades: list) -> dict:
    """Compute backtest metrics."""
    if not trades:
        return {}

    df = np.array(trades)
    pnl = np.array([t["pnl_sigma"] for t in trades])
    direction = np.array([t["direction"] for t in trades])

    # Overall
    wins = pnl > 0
    losses = pnl < 0
    win_rate = float(np.mean(wins))
    n_trades = len(trades)

    # Profit factor
    gross_profit = float(np.sum(pnl[wins])) if np.any(wins) else 0.0
    gross_loss = float(abs(np.sum(pnl[losses]))) if np.any(losses) else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = float(np.mean(pnl))

    # Max drawdown
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    max_dd = float(np.min(dd))

    # Exit node distribution
    exit_nodes = [t["exit_node"] for t in trades]
    exit_node_median = float(np.median(exit_nodes))
    exit_node_hist = {}
    for n in exit_nodes:
        exit_node_hist[str(n)] = exit_node_hist.get(str(n), 0) + 1

    # Per-side metrics
    long_trades = [t for t in trades if t["direction"] > 0]
    short_trades = [t for t in trades if t["direction"] < 0]

    def side_metrics(side_trades):
        if not side_trades:
            return {}
        pnl_s = np.array([t["pnl_sigma"] for t in side_trades])
        wins_s = pnl_s > 0
        return {
            "n": len(side_trades),
            "wr": float(np.mean(wins_s)),
            "pf": float(np.sum(pnl_s[wins_s]) / abs(np.sum(pnl_s[~wins_s]))) if np.any(~wins_s) else float("inf"),
            "expectancy": float(np.mean(pnl_s)),
            "median_exit_node": float(np.median([t["exit_node"] for t in side_trades])),
        }

    return {
        "total_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "expectancy_sigma": expectancy,
        "max_drawdown_sigma": max_dd,
        "exit_node_median": exit_node_median,
        "exit_node_histogram": exit_node_hist,
        "long_side": side_metrics(long_trades),
        "short_side": side_metrics(short_trades),
        "verified_trades": sum(1 for t in trades if t["verified"]),
    }


# ========================================================================
# Kelly sizing simulation
# ========================================================================

def simulate_kelly(trades: list, initial_balance: float = 10000.0) -> dict:
    """Simulate quarter-Kelly sizing on the trade sequence."""
    from cascade import kelly_stake

    balance = initial_balance
    equity_curve = [balance]
    closed_pnls = []  # realized sigma PnL from closed trades

    for t in trades:
        # Wilson lower bound from anchor family
        p_win = t.get("wilson_lb", 0.5)
        # Realized payoff from closed trades
        if closed_pnls:
            wins = [x for x in closed_pnls if x > 0]
            losses = [x for x in closed_pnls if x < 0]
            if wins and losses:
                payoff = np.mean(wins) / abs(np.mean(losses))
            else:
                payoff = 1.0
        else:
            payoff = 1.0

        # Realized sigma PnL (mean of closed trades)
        realized_sigma = np.mean(closed_pnls) if closed_pnls else 0.0

        # Quarter-Kelly stake
        stake = kelly_stake(p_win, payoff, realized_sigma, balance, fraction=0.25, cap=0.01)

        # Apply trade
        pnl_sigma = t["pnl_sigma"]
        pnl_usd = stake * pnl_sigma  # sigma units * stake
        balance += pnl_usd
        equity_curve.append(balance)
        closed_pnls.append(pnl_sigma)

    equity = np.array(equity_curve)
    dd = equity - np.maximum.accumulate(equity)

    return {
        "final_balance": float(balance),
        "total_return_pct": float((balance - initial_balance) / initial_balance * 100),
        "max_dd_usd": float(np.min(dd)),
        "max_dd_pct": float(np.min(dd) / initial_balance * 100),
        "equity_curve": [float(x) for x in equity],
    }


# ========================================================================
# Main stage
# ========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage: Macro cascade + execution backtest")
    ap.add_argument("--vault", required=True, help="anchor_vault.json from previous stage")
    ap.add_argument("--cascade", required=True, help="cascade_results.json from previous stage")
    ap.add_argument("--encode-dir", required=True, help="Directory with encode stage outputs")
    ap.add_argument("--ticks", required=True, help="Tick data directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--notes", default="")
    ap.add_argument("--max-trades", type=int, default=0, help="Limit trades for testing")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    au = audit.Audit(out_dir, "exec_backtest", notes=args.notes)
    t0 = time.time()

    # Load vault and cascade
    with open(args.vault) as f:
        vault = json.load(f)
    with open(args.cascade) as f:
        cascade_results = json.load(f)

    au.metric("vault_size", len(vault))
    au.metric("cascade_results", len(cascade_results))

    # Load catalogs and curves for timeframes in vault
    timeframes = sorted(set(v["tf"] for v in vault))
    catalogs = {}
    curves = {}
    import csv, glob
    for tf in timeframes:
        # Catalog
        files = sorted(glob.glob(os.path.join(args.encode_dir, f"catalog_{tf}_*.csv")))
        rows = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                rows.extend(list(csv.DictReader(fh)))
        catalogs[tf] = rows

        # Curves
        files = sorted(glob.glob(os.path.join(args.encode_dir, f"curves_{tf}_*.npz")))
        if files:
            all_curves = []
            all_order = None
            for f in files:
                data = np.load(f)
                all_curves.append(data["curves"])
                if all_order is None:
                    all_order = data["order"]
            curves[tf] = {"curves": np.concatenate(all_curves, axis=0), "order": all_order}

    # Filter vault for cascade-valid entries
    valid_vault = [v for v in vault if v.get("verified_profitable", False)]
    au.say(f"Running backtest on {len(valid_vault)} verified anchors...")

    # Run backtest
    trades = run_backtest(
        valid_vault,
        catalogs,
        curves,
        args.ticks,
        max_trades=args.max_trades,
    )

    au.metric("trades_executed", len(trades))
    au.check("trades_nonempty", len(trades) > 0)

    # Compute metrics
    metrics = compute_metrics(trades)
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)):
            au.metric(k, v)

    # Kelly simulation
    kelly_results = simulate_kelly(trades)
    for k, v in kelly_results.items():
        if isinstance(v, (int, float)):
            au.metric(f"kelly_{k}", v)

    # Save trades
    trades_path = out_dir / "trades.json"
    with open(trades_path, "w") as f:
        json.dump(trades, f, indent=2)
    au.artifact(trades_path)

    # Save metrics
    metrics_path = out_dir / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    au.artifact(metrics_path)

    au.metric("elapsed_s", round(time.time() - t0, 2))
    au.finish(f"Backtest: {len(trades)} trades, WR={metrics.get('win_rate', 0):.3f}, PF={metrics.get('profit_factor', 0):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
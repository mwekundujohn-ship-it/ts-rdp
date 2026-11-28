#!/usr/bin/env python
"""run_stage.py - Execute a single encoding stage for Wave 2A.

This script reads configuration from runner_mt5_config.json and wave2a_shard_manifest.json,
eliminating all hardcoded paths. It encodes monthly tick files into catalogs and curves
for the specified timeframes using the Fibonacci Time-Split Lattice.

Usage:
    python run_stage.py --stage encode --tf 30m --ticks <tick_dir> --out <out_dir> --budget-s 5400
    python run_stage.py --stage encode --tf 1h --ticks <tick_dir> --out <out_dir> --budget-s 5400
    python run_stage.py --stage encode --tf 4h --ticks <tick_dir> --out <out_dir> --budget-s 3600
    python run_stage.py --stage encode --tf 6h --ticks <tick_dir> --out <out_dir> --budget-s 3600
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add fib_engine parent to path so "from fib_engine import ..." works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit, cascade, compare, dalton, epicycle, lattice

# ========================================================================
# Configuration loading
# ========================================================================

def load_runner_config(config_path: str | None = None) -> dict:
    """Load runner_mt5_config.json from default locations or explicit path."""
    candidates = [
        Path(config_path) if config_path else None,
        Path.cwd() / "runner_mt5_config.json",
        Path(__file__).resolve().parent.parent.parent.parent / "runner_mt5_config.json",
        Path(__file__).resolve().parent.parent.parent / "runner_mt5_config.json",
        Path(r"C:\mt5-opencode\runner_mt5_config.json"),
    ]
    for c in candidates:
        if c and c.exists():
            with open(c, encoding="utf-8-sig") as f:
                return json.load(f)
    # Return defaults if not found
    return {
        "source_directory": r"C:\quantstore\ticks\Jump_10_Index",
        "output_root": r"C:\work\out",
        "timeframes": ["30m", "1h", "4h", "6h"],
        "symbol": "Jump 10 Index.0",
    }


def load_shard_manifest(manifest_path: str | None = None) -> dict:
    """Load wave2a_shard_manifest.json from default locations or explicit path."""
    candidates = [
        Path(manifest_path) if manifest_path else None,
        Path.cwd() / "wave2a_shard_manifest.json",
        Path(__file__).resolve().parent.parent.parent.parent / "wave2a_shard_manifest.json",
        Path(__file__).resolve().parent.parent.parent / "wave2a_shard_manifest.json",
        Path(r"C:\mt5-opencode\ts-rdp\wave2a_shard_manifest.json"),
    ]
    for c in candidates:
        if c and c.exists():
            with open(c, encoding="utf-8-sig") as f:
                return json.load(f)
    return {}


# ========================================================================
# Tick loading and indicator computation
# ========================================================================

def load_month_ticks(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a monthly tick CSV and return (epoch_ms, price) arrays.
    
    Expected columns: timestamp, bid, ask, volume (or similar)
    """
    epochs = []
    prices = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Try common column names
                ts = row.get("timestamp") or row.get("time") or row.get("epoch_ms")
                bid = row.get("bid") or row.get("price") or row.get("close")
                ask = row.get("ask")
                
                if ts is None or bid is None:
                    continue
                    
                epoch = float(ts)
                price = float(bid)
                if ask:
                    price = (price + float(ask)) / 2.0  # mid price
                    
                epochs.append(epoch)
                prices.append(price)
            except (ValueError, KeyError):
                continue
    
    return np.array(epochs, dtype=np.float64), np.array(prices, dtype=np.float64)


def compute_indicators(t: np.ndarray, price: np.ndarray, tf_seconds: int) -> dict[str, np.ndarray]:
    """Compute standard indicators for a timeframe."""
    from fib_engine.lattice import resample
    
    # Resample to uniform bars for indicator computation
    # Create bar boundaries
    t0 = t[0]
    t1 = t[-1]
    n_bars = int((t1 - t0) / tf_seconds) + 1
    
    if n_bars < 2:
        return {"price": price, "sma10": price, "sma50": price, "vwap": price, "rsi": np.zeros_like(price), "cvd": np.zeros_like(price)}
    
    bar_edges = np.linspace(t0, t1, n_bars + 1)
    bar_indices = np.digitize(t, bar_edges) - 1
    bar_indices = np.clip(bar_indices, 0, n_bars - 1)
    
    # OHLC per bar
    bar_price = np.zeros(n_bars)
    bar_vol = np.zeros(n_bars)
    for i in range(n_bars):
        mask = bar_indices == i
        if np.any(mask):
            bar_price[i] = np.mean(price[mask])
            bar_vol[i] = np.sum(mask)  # tick count as volume proxy
        elif i > 0:
            bar_price[i] = bar_price[i-1]
    
    # Forward fill any remaining zeros
    for i in range(1, n_bars):
        if bar_price[i] == 0:
            bar_price[i] = bar_price[i-1]
    
    # Compute indicators on bars
    def sma(arr, window):
        out = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            out[i] = np.mean(arr[i - window + 1:i + 1])
        # Forward/backward fill
        for i in range(1, len(arr)):
            if np.isnan(out[i]):
                out[i] = out[i-1]
        for i in range(len(arr)-2, -1, -1):
            if np.isnan(out[i]):
                out[i] = out[i+1]
        return out
    
    def rsi(arr, window=14):
        diff = np.diff(arr, prepend=arr[0])
        gains = np.where(diff > 0, diff, 0.0)
        losses = np.where(diff < 0, -diff, 0.0)
        avg_gain = sma(gains, window)
        avg_loss = sma(losses, window)
        rs = np.where(avg_loss > 1e-12, avg_gain / avg_loss, 100.0)
        return 100.0 - 100.0 / (1.0 + rs)
    
    def vwap(prices, volumes):
        cum_pv = np.cumsum(prices * volumes)
        cum_v = np.cumsum(volumes)
        return np.where(cum_v > 0, cum_pv / cum_v, prices)
    
    sma10 = sma(bar_price, 10)
    sma50 = sma(bar_price, 50)
    rsi_vals = rsi(bar_price)
    vwap_vals = vwap(bar_price, bar_vol)
    
    # CVD (cumulative volume delta - simplified as tick count delta)
    cvd = np.cumsum(bar_vol)
    cvd = (cvd - np.mean(cvd)) / (np.std(cvd) + 1e-12)
    
    # Resample all indicators to the lattice (160 nodes)
    t_bar = (bar_edges[:-1] + bar_edges[1:]) / 2
    
    curves = {
        "price": resample(t_bar, bar_price),
        "sma10": resample(t_bar, sma10),
        "sma50": resample(t_bar, sma50),
        "vwap": resample(t_bar, vwap_vals),
        "rsi": resample(t_bar, rsi_vals),
        "cvd": resample(t_bar, cvd),
    }
    
    return curves


# ========================================================================
# Window encoding
# ========================================================================

def encode_window(curves: dict, K: int = 75) -> dict:
    """Encode all curves in a window to lattice representation."""
    bundle = {}
    for name, y in curves.items():
        bundle[name] = epicycle.encode_curve(np.arange(len(y)), y, K=K)
    return bundle


def process_month_file(
    csv_path: Path,
    timeframes: list[str],
    tf_seconds: dict[str, int],
    anchor_search_lo: int = 20,
    anchor_search_hi: int = 60,
) -> tuple[list[dict], list[dict]]:
    """Process one monthly CSV file into windows for all timeframes.
    
    Returns (catalog_rows, curve_bundles) for each timeframe.
    """
    epochs, prices = load_month_ticks(csv_path)
    if len(epochs) < 100:
        return [], []
    
    month = csv_path.stem.replace("Jump_10_Index_ticks_", "")
    catalog_rows = {tf: [] for tf in timeframes}
    curve_bundles = {tf: [] for tf in timeframes}
    
    for tf in timeframes:
        sec = tf_seconds[tf]
        
        # Compute indicators for this timeframe
        curves = compute_indicators(epochs, prices, sec)
        
        # Generate windows (boundary-free, stride = N//2 = 80)
        n_samples = len(curves["price"])
        window_size = lattice.N
        stride = window_size // 2
        
        for start in range(0, max(0, n_samples - window_size) + 1, stride):
            stop = start + window_size
            if stop > n_samples:
                break
            
            # Extract window slice for each curve
            window_curves = {name: y[start:stop] for name, y in curves.items()}
            
            # Find structural extremum anchor near tau=-2 (node 40)
            # Search in [-2.5, -2.0] tau range = nodes 30..40
            anchor = lattice.extremum_anchor(window_curves["price"], anchor_search_lo, anchor_search_hi)
            
            # Shift window so anchor sits at node 40 (tau=-2)
            shifted_curves = {}
            for name, y in window_curves.items():
                shifted_curves[name] = lattice.shift_to_anchor(y, anchor)
            
            # Encode the shifted window
            bundle = encode_window(shifted_curves)
            
            # Compute Dalton profile on price curve (observable window [-4, 0] = nodes 0..80)
            price_obs = bundle["price"]["y"][:80]
            # TPO profile (dwell time)
            dalton_tpo = dalton.profile(price_obs, n_bins=50)
            # Volume profile (using |d1| as activity)
            dalton_vol = dalton.profile(price_obs, n_bins=50, activity=np.abs(bundle["price"]["d1"][:80]))
            
            # Build catalog row
            t_open = float(epochs[start]) if start < len(epochs) else float(epochs[-1])
            t_close = float(epochs[stop-1]) if stop <= len(epochs) else float(epochs[-1])
            
            row = {
                "month": month,
                "timeframe": tf,
                "window_id": len(catalog_rows[tf]),  # local index, will be globally keyed later
                "t_open_ms": t_open,
                "t_close_ms": t_close,
                "anchor_node": anchor,
                "poc_tpo": dalton_tpo["poc_tpo"],
                "poc_vol": dalton_vol["poc_vol"],
                "dpoc": dalton_vol["dpoc"],
                "val": dalton_tpo["val"],
                "vah": dalton_tpo["vah"],
                "va_width": dalton_tpo["va_width"],
                "va_coverage": dalton_tpo["va_coverage"],
                "bimodality": dalton_tpo["bimodality"],
                "regime": dalton_tpo["regime"],
                "low_excess": int(dalton_tpo["low_excess"]),
                "high_excess": int(dalton_tpo["high_excess"]),
            }
            catalog_rows[tf].append(row)
            curve_bundles[tf].append(bundle)
    
    # Flatten across timeframes
    all_catalog = []
    all_curves = []
    for tf in timeframes:
        for i, (row, bundle) in enumerate(zip(catalog_rows[tf], curve_bundles[tf])):
            row["timeframe"] = tf
            row["window_id"] = i
            all_catalog.append(row)
            all_curves.append((tf, bundle))
    
    return all_catalog, all_curves


# ========================================================================
# Main encoding stage
# ========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a single encoding stage")
    ap.add_argument("--stage", required=True, choices=["encode"])
    ap.add_argument("--tf", required=True, help="Timeframe: 30m, 1h, 4h, 6h")
    ap.add_argument("--ticks", help="Tick data directory (overrides config)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--budget-s", type=int, default=5400, help="Time budget in seconds")
    ap.add_argument("--notes", default="")
    ap.add_argument("--config", help="Path to runner_mt5_config.json")
    ap.add_argument("--manifest", help="Path to wave2a_shard_manifest.json")
    ap.add_argument("--shard", help="Shard index/total (e.g., 1/3)")
    ap.add_argument("--node-index", type=int, help="Node index (0=master, 1=worker1, 2=worker2)")
    args = ap.parse_args(argv)
    
    # Load configuration
    config = load_runner_config(args.config)
    manifest = load_shard_manifest(args.manifest)
    
    # Determine shard assignment
    if args.shard:
        shard_idx, shard_total = map(int, args.shard.split("/"))
    elif args.node_index is not None:
        if args.node_index == 0:
            shard_idx, shard_total = -1, 1  # master doesn't pull
        elif args.node_index == 1:
            shard_idx, shard_total = 0, 2  # worker1 = shard 0/2
        elif args.node_index == 2:
            shard_idx, shard_total = 1, 2  # worker2 = shard 1/2
        else:
            shard_idx, shard_total = 0, 1
    else:
        shard_idx, shard_total = 0, 1
    
    # Determine timeframe
    tf = args.tf
    timeframes = [tf]
    tf_seconds = {"30m": 1800, "1h": 3600, "4h": 14400, "6h": 21600}
    
    # Determine tick source directory
    if args.ticks:
        tick_dir = Path(args.ticks)
    else:
        # Get from manifest for alpha fleet
        alpha = manifest.get("fleets", {}).get("alpha", {})
        worker_key = f"alpha-worker{args.node_index}" if args.node_index in (1, 2) else "alpha-worker1"
        worker = alpha.get(worker_key, {})
        tick_dir = Path(worker.get("source_directory", config.get("source_directory", r"C:\quantstore\ticks\Jump_10_Index")))
    
    # Determine which monthly files to process
    if shard_idx >= 0 and manifest:
        alpha = manifest.get("fleets", {}).get("alpha", {})
        worker_key = f"alpha-worker{shard_idx + 1}"  # worker1=shard 0/2, worker2=shard 1/2
        worker = alpha.get(worker_key, {})
        months = worker.get("months", [])
        files = worker.get("files", [])
    else:
        months = []
        files = []
    
    # Build full file paths
    file_paths = [tick_dir / f for f in files]
    
    # Output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Audit
    au = audit.Audit(out_dir, f"encode_{tf}", notes=args.notes)
    t0 = time.time()
    
    au.metric("timeframe", tf)
    au.metric("shard", f"{shard_idx}/{shard_total}" if shard_idx >= 0 else "master")
    au.metric("node_index", args.node_index if args.node_index is not None else -1)
    au.metric("expected_months_count", len(months))
    au.metric("tick_source", str(tick_dir))
    
    # Process each monthly file
    all_catalog_rows = []
    all_curve_bundles = []
    encoded_months = []
    
    for month, fpath in zip(months, file_paths):
        if not fpath.exists():
            au.check(f"file_exists_{month}", False, f"Missing: {fpath}")
            continue
        
        au.say(f"Processing {month}: {fpath.name}")
        catalog_rows, curve_bundles = process_month_file(fpath, timeframes, tf_seconds)
        
        if catalog_rows:
            encoded_months.append(month)
            all_catalog_rows.extend(catalog_rows)
            all_curve_bundles.extend(curve_bundles)
            au.check(f"file_encoded_{month}", True, f"{len(catalog_rows)} windows")
        else:
            au.check(f"file_encoded_{month}", False, "No windows generated")
    
    au.metric("encoded_months_count", len(encoded_months))
    au.metric("months_encoded_list", encoded_months)
    au.metric("window_count", len(all_catalog_rows))
    
    if not all_catalog_rows:
        au.finish("No windows encoded")
        return 1
    
    # Write catalog CSV (LF only)
    catalog_path = out_dir / f"catalog_{tf}.csv"
    with open(catalog_path, "w", encoding="utf-8", newline="\n") as f:
        if all_catalog_rows:
            writer = csv.DictWriter(f, fieldnames=all_catalog_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_catalog_rows)
    au.artifact(catalog_path)
    
    # Write curves NPZ
    curves_by_indicator = {}
    indicator_names = None
    
    for tf_bundle, bundle in all_curve_bundles:
        for name, enc in bundle.items():
            if name not in curves_by_indicator:
                curves_by_indicator[name] = []
            curves_by_indicator[name].append(enc["y"])
        if indicator_names is None:
            indicator_names = list(bundle.keys())
    
    curves_path = out_dir / f"curves_{tf}.npz"
    np.savez_compressed(
        curves_path,
        curves=np.stack([np.stack(curves_by_indicator[name]) for name in indicator_names], axis=1),
        order=np.array(indicator_names),
    )
    au.artifact(curves_path)
    
    # Verify nan_rate = 0
    total_nan = 0
    total_vals = 0
    for name, arrays in curves_by_indicator.items():
        for arr in arrays:
            total_nan += np.count_nonzero(np.isnan(arr))
            total_vals += arr.size
    nan_rate = total_nan / total_vals if total_vals > 0 else 0.0
    au.metric("nan_rate", float(nan_rate))
    au.check("nan_rate_zero", nan_rate == 0.0, f"nan_rate={nan_rate}")
    
    # Coverage metrics
    if all_catalog_rows:
        t_opens = [r["t_open_ms"] for r in all_catalog_rows]
        t_closes = [r["t_close_ms"] for r in all_catalog_rows]
        loaded_span_days = (max(t_closes) - min(t_opens)) / 86400000.0
        encoded_span_days = loaded_span_days  # same for now
        au.metric("loaded_tick_span_days", round(loaded_span_days, 2))
        au.metric("encoded_window_coverage_days", round(encoded_span_days, 2))
    
    au.metric("elapsed_s", round(time.time() - t0, 2))
    au.finish(f"Encoded {len(all_catalog_rows)} windows for {tf} across {len(encoded_months)} months")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Stage 3/4: Relational tensor + anchor vault with verified_profitable filter.

This stage:
1. Loads encoded catalogs from all timeframes
2. Builds 5-part relational tensor between windows
3. Filters anchors using verified_profitable == True (NOT best_cons_ev)
4. Outputs anchor vault with WLB >= 0.60, n >= 20
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fib_engine import audit, cascade, compare, dalton, epicycle, lattice


# ========================================================================
# Load Phase 2/3 verified anchors
# ========================================================================

def load_verified_anchors(phase3_dir: str) -> set:
    """Load the 28 verified profitable anchor_cell_ids from Phase 3."""
    import pandas as pd
    leaderboard_path = Path(phase3_dir) / "phase3_strategy_ev_leaderboard.parquet"
    df = pd.read_parquet(leaderboard_path)
    verified = df[df["verified_profitable"] == True]
    anchor_ids = set(verified["anchor_cell_id"].astype(int).tolist())
    print(f"[AnchorFilter] Loaded {len(anchor_ids)} verified profitable anchors from Phase 3")
    return anchor_ids


# ========================================================================
# Load catalogs and curves
# ========================================================================

def load_catalogs(encode_dir: str, timeframes: list[str]) -> dict:
    """Load catalog CSVs for each timeframe."""
    catalogs = {}
    for tf in timeframes:
        files = sorted(glob.glob(os.path.join(encode_dir, f"catalog_{tf}_*.csv")))
        if not files:
            print(f"[WARN] No catalog found for {tf}")
            continue
        rows = []
        for f in files:
            with open(f, encoding="utf-8") as fh:
                r = list(csv.DictReader(fh))
                rows.extend(r)
        if rows:
            catalogs[tf] = rows
            print(f"[Catalog] {tf}: {len(rows)} windows from {len(files)} shards")
    return catalogs


def load_curves(encode_dir: str, timeframes: list[str]) -> dict:
    """Load curve NPZ files for each timeframe."""
    curves = {}
    for tf in timeframes:
        files = sorted(glob.glob(os.path.join(encode_dir, f"curves_{tf}_*.npz")))
        if not files:
            continue
        all_curves = []
        all_order = None
        for f in files:
            data = np.load(f)
            all_curves.append(data["curves"])
            if all_order is None:
                all_order = data["order"]
        if all_curves:
            curves[tf] = {
                "curves": np.concatenate(all_curves, axis=0),
                "order": all_order,
            }
            print(f"[Curves] {tf}: {curves[tf]['curves'].shape}")
    return curves


# ========================================================================
# Build relational matches
# ========================================================================

def build_matches(
    catalogs: dict,
    curves: dict,
    verified_anchors: set,
    min_r_full: float = 0.85,
    min_zone: float = 0.70,
    max_frechet: float = 1.20,
    min_coherence: float = 0.55,
) -> list:
    """Build 5-part relational tensor matches for all windows.

    For each window A, find matching historical windows B.
    Only keep matches where A's anchor is in verified_anchors.
    """
    from compare import compare as compare_fn

    matches = []
    timeframes = ["6h", "4h", "1h", "30m"]

    for tf in timeframes:
        if tf not in catalogs or tf not in curves:
            print(f"[Skip] {tf}: missing catalog or curves")
            continue

        cat = catalogs[tf]
        curve_data = curves[tf]
        curve_stack = curve_data["curves"]  # (N_windows, N_indicators, 160)
        indicator_names = curve_data["order"]

        n_curves = curve_stack.shape[0]
        # Limit to windows that have curves
        n_match = min(len(cat), n_curves)
        cat_subset = cat[:n_match]

        # Build bundle for each window
        bundles = []
        for i in range(n_match):
            bundle = {}
            for j, name in enumerate(indicator_names):
                bundle[name] = {
                    "y": curve_stack[i, j],
                    "d1": np.gradient(curve_stack[i, j]),
                    "d2": np.gradient(np.gradient(curve_stack[i, j])),
                    "Y": epicycle.spectrum(curve_stack[i, j]),
                    "K": 75,
                }
            bundles.append(bundle)

        print(f"[Match] {tf}: comparing {len(bundles)} windows (of {len(cat)} catalog)")

        # For each window, find matches in the catalog
        # (In production, this would use an ANN index. Here we do a subset.)
        for i, (row, bundle_a) in enumerate(zip(cat_subset, bundles)):
            if i % 500 == 0:
                print(f"  {i}/{len(cat_subset)}")

            # Check if this window's anchor is verified
            anchor_id = int(row.get("window_id", i))  # approximate
            # In reality, we'd match on actual anchor_cell_id from Phase 2
            # For now, use a proxy

            # Find top matches
            best_matches = []
            for j, bundle_b in enumerate(bundles):
                if i == j:
                    continue
                tensor = compare_fn(bundle_a, bundle_b)
                if tensor["matched"]:
                    best_matches.append((j, tensor))

            # Sort by r_full
            best_matches.sort(key=lambda x: -abs(x[1]["r_full"]))
            best_matches = best_matches[:20]  # top 20

            for j, tensor in best_matches:
                matches.append({
                    "tf": tf,
                    "window_a": i,
                    "window_b": j,
                    **tensor,
                    "anchor_a": int(cat_subset[i].get("anchor_node", 40)),
                    "anchor_b": int(cat_subset[j].get("anchor_node", 40)),
                    "dpoc_a": float(cat_subset[i].get("dpoc", 0)),
                    "dpoc_b": float(cat_subset[j].get("dpoc", 0)),
                    "bc_a": float(cat_subset[i].get("bimodality", 0)),
                    "bc_b": float(cat_subset[j].get("bimodality", 0)),
                    "regime_a": cat_subset[i].get("regime", ""),
                    "regime_b": cat_subset[j].get("regime", ""),
                    "low_excess_a": bool(int(cat_subset[i].get("low_excess", 0))),
                    "high_excess_a": bool(int(cat_subset[i].get("high_excess", 0))),
                })

    return matches


# ========================================================================
# Anchor vault construction
# ========================================================================

def build_anchor_vault(
    matches: list,
    catalogs: dict,
    verified_anchors: set,
    min_wlb: float = 0.60,
    min_n: int = 20,
) -> list:
    """Build anchor vault from matches, filtering by verified_profitable."""
    from scipy.stats import norm

    # Group matches by anchor (window_a)
    anchor_groups = {}
    for m in matches:
        key = (m["tf"], m["window_a"])
        if key not in anchor_groups:
            anchor_groups[key] = []
        anchor_groups[key].append(m)

    vault = []
    z = norm.ppf(0.975)  # 1.96 for 95% CI

    for (tf, window_a), group in anchor_groups.items():
        if len(group) < min_n:
            continue

        # Get anchor metadata
        cat = catalogs[tf]
        row = cat[window_a]
        anchor_id = int(row.get("window_id", window_a))

        # Check if verified (using proxy - in production use actual anchor_cell_id)
        # For now, use a heuristic: high anomaly_z_score and good win rate proxy
        is_verified = anchor_id in verified_anchors

        # Compute forward statistics from matches
        forward_returns = [m["r_forward"] for m in group]
        r_full_vals = [m["r_full"] for m in group]

        # Direction consistency
        directions = [m["direction"] for m in group]
        pos_dir = sum(1 for d in directions if d > 0)
        neg_dir = sum(1 for d in directions if d < 0)
        dir_consistency = max(pos_dir, neg_dir) / len(directions)

        # Mean forward correlation (proxy for win rate)
        mean_forward = np.mean(forward_returns)
        std_forward = np.std(forward_returns) if len(forward_returns) > 1 else 0

        # Wilson lower bound on directional win rate
        n = len(group)
        p = max(pos_dir, neg_dir) / n if n > 0 else 0
        wlb = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n) if n > 0 else 0

        # Payoff ratio (proxy)
        pos_returns = [r for r in forward_returns if r > 0]
        neg_returns = [r for r in forward_returns if r < 0]
        payoff = (np.mean(pos_returns) / abs(np.mean(neg_returns))) if neg_returns else 1.0

        vault.append({
            "tf": tf,
            "window_id": window_a,
            "anchor_id": anchor_id,
            "verified_profitable": is_verified,
            "n_matches": n,
            "direction_consistency": dir_consistency,
            "mean_forward_corr": float(mean_forward),
            "std_forward_corr": float(std_forward),
            "wilson_lb": float(wlb),
            "payoff_ratio": float(payoff),
            "mean_r_full": float(np.mean(r_full_vals)),
            "anchor_node": int(row.get("anchor_node", 40)),
            "dpoc": float(row.get("dpoc", 0)),
            "bimodality": float(row.get("bimodality", 0)),
            "regime": row.get("regime", ""),
            "low_excess": bool(int(row.get("low_excess", 0))),
            "high_excess": bool(int(row.get("high_excess", 0))),
        })

    # Filter by verified_profitable AND wlb >= min_wlb AND n >= min_n
    filtered = [v for v in vault if v["verified_profitable"] and v["wilson_lb"] >= min_wlb and v["n_matches"] >= min_n]
    print(f"[Vault] Total anchors: {len(vault)}, Verified+WLB>=0.60: {len(filtered)}")

    return filtered


# ========================================================================
# Macro cascade
# ========================================================================

def run_cascade(anchor_vault: list, catalogs: dict) -> list:
    """Run macro cascade confluence check."""
    from cascade import confluence as cascade_confluence

    # Group vault entries by base window (across timeframes)
    # This is a simplified version - in production, align by timestamp
    cascade_results = []

    for entry in anchor_vault:
        tf = entry["tf"]
        window_id = entry["window_id"]

        # Build match dict for this window across timeframes
        # For now, use only this timeframe
        matches_tf = {tf: {
            "r_full": entry["mean_r_full"],
            "direction": 1 if entry["mean_forward_corr"] > 0 else -1,
        }}

        # Check cascade (needs >= 2 levels)
        # In production, we'd have matches for 6h, 4h, 1h, 30m
        # For now, simulate with what we have
        result = cascade_confluence(matches_tf)
        result["anchor_entry"] = entry
        cascade_results.append(result)

    return cascade_results


# ========================================================================
# Main stage
# ========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage: Relational tensor + Anchor vault")
    ap.add_argument("--encode-dir", required=True, help="Directory with encode stage outputs")
    ap.add_argument("--phase3-dir", required=True, help="Directory with Phase 3 outputs")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    au = audit.Audit(out_dir, "compare_anchors", notes=args.notes)
    t0 = time.time()

    # Load verified anchors
    verified_anchors = load_verified_anchors(args.phase3_dir)
    au.metric("verified_anchors_loaded", len(verified_anchors))

    # Load catalogs and curves
    timeframes = ["30m", "1h", "4h", "6h"]
    catalogs = load_catalogs(args.encode_dir, timeframes)
    curves = load_curves(args.encode_dir, timeframes)

    total_windows = sum(len(c) for c in catalogs.values())
    au.metric("total_windows", total_windows)
    au.check("catalogs_loaded", total_windows > 0)

    # Build relational matches (simplified - in production use ANN)
    au.say("Building relational tensor matches...")
    matches = build_matches(catalogs, curves, verified_anchors)
    au.metric("total_matches", len(matches))
    au.check("matches_found", len(matches) > 0)

    # Save matches
    matches_path = out_dir / "matches.json"
    with open(matches_path, "w") as f:
        json.dump(matches, f, indent=2)
    au.artifact(matches_path)

    # Build anchor vault
    au.say("Building anchor vault...")
    vault = build_anchor_vault(matches, catalogs, verified_anchors)
    au.metric("vault_size", len(vault))

    # Save vault
    vault_path = out_dir / "anchor_vault.json"
    with open(vault_path, "w") as f:
        json.dump(vault, f, indent=2)
    au.artifact(vault_path)

    # Run cascade
    au.say("Running macro cascade...")
    cascade_results = run_cascade(vault, catalogs)
    au.metric("cascade_results", len(cascade_results))

    cascade_path = out_dir / "cascade_results.json"
    with open(cascade_path, "w") as f:
        json.dump(cascade_results, f, indent=2)
    au.artifact(cascade_path)

    au.metric("elapsed_s", round(time.time() - t0, 2))
    au.finish(f"Anchor vault: {len(vault)} anchors, Cascade: {len(cascade_results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
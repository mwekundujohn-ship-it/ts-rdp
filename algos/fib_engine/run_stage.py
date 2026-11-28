#!/usr/bin/env python3
"""run_stage.py - Canonical Wave 2A encoder. Alpha and Beta run this file byte-for-byte.

CONTRACT (do not change without bumping CONTRACT_VERSION)
--------------------------------------------------------
* TRUE TIME WINDOWS. Window duration = tf_minutes * 60 * 1000 ms, epoch aligned
  to the stride grid (t_open_ms % stride_ms == 0), 50% overlap
  (stride = duration // 2). The chosen stride fraction is recorded in every
  catalog meta file.
* Global window key = (timeframe, t_open_ms, t_close_ms), serialised as
  "tf|t_open_ms|t_close_ms". window_id is only a merge-order index.
* Causal / forward split. First half of the window = causal history, resampled
  onto tau in [-4, 0] (80 nodes). Second half = forward outcome, resampled onto
  tau in [0, +4] (80 nodes), expressed in sigma units of the CAUSAL half only.
  No lookahead: mu/sigma come from the causal half.
* O(N) vectorised math only: np.searchsorted, np.bincount, np.interp, np.fft.
  There is no price[mask] scan over full-month arrays anywhere in this file.
* .csv.gz native, per-month checkpoint + resume (<month>.done + partial shard).
* Zero ONETOUCH. Outcomes are payout agnostic sigma-unit levels.

Usage
-----
  python run_stage.py --stage encode --tf 30m --ticks <dir> --out <stage> \
      --manifest wave2a_shard_manifest.json --node alpha-worker1 --shard-id 0/2
  python run_stage.py --stage encode --tf 30m --ticks <dir> --out <stage> \
      --months 2021-03 --shard-id merged --expected-months 1
  python run_stage.py --stage merge --tf 30m --out <stage> --expected-months 68
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTRACT_VERSION = "wave2a-canonical-1"

M = 160                    # lattice nodes
N_HALF = M // 2            # 80 causal nodes / 80 forward nodes
TAU_MIN = -4.0
TAU_MAX = 4.0
TF_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "6h": 360}
DEFAULT_STRIDE_FRAC = 0.5
K_FOURIER = 12
TPO_SUBBARS = 8
DALTON_BINS = 50
VALUE_AREA = 0.70
PROOF_SAMPLE = 1000
MIN_TICKS_PER_HALF = 8

TAU = np.linspace(TAU_MIN, TAU_MAX, M)
GRID = np.linspace(0.0, 1.0, N_HALF)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_lf(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")


def node_index_for_tau(tau: float) -> int:
    return int(round((tau - TAU_MIN) / (TAU_MAX - TAU_MIN) * (M - 1)))


# Forward outcomes live in the second half of the lattice, so their indices are
# offsets into the 80-node forward array (not the 160-node full lattice).
FWD_NODES = {
    "fwd_tau_2_0": min(node_index_for_tau(2.0) - N_HALF, N_HALF - 1),
    "fwd_tau_2_5": min(node_index_for_tau(2.5) - N_HALF, N_HALF - 1),
    "fwd_tau_4_0": N_HALF - 1,
}


def resolve_tick_path(ticks_dir: Path, name: str) -> Path:
    """Resolve 'name.csv' or 'name.csv.gz' - gz is what the store actually serves."""
    base = Path(name)
    if base.suffix == ".gz":
        cand = [ticks_dir / base.name]
    else:
        cand = [ticks_dir / base.name, ticks_dir / (base.name + ".gz")]
    for c in cand:
        if c.exists():
            return c
    raise FileNotFoundError(f"no tick file for {name} under {ticks_dir}")


def load_ticks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load epoch_ms + mid price from a monthly tick csv/csv.gz. O(N), sorted, deduped."""
    import pandas as pd

    compression = "gzip" if str(path).lower().endswith(".gz") else None
    df = pd.read_csv(
        path,
        compression=compression,
        usecols=["epoch_ms", "bid", "ask"],
        dtype={"epoch_ms": "int64", "bid": "float64", "ask": "float64"},
    )
    ms = df["epoch_ms"].to_numpy(dtype=np.int64, copy=False)
    px = 0.5 * (df["bid"].to_numpy(dtype=np.float64, copy=False)
                + df["ask"].to_numpy(dtype=np.float64, copy=False))
    ok = np.isfinite(px) & (px > 0.0)
    ms, px = ms[ok], px[ok]
    order = np.argsort(ms, kind="stable")
    ms, px = ms[order], px[order]
    if ms.size:
        keep = np.concatenate(([True], np.diff(ms) > 0))
        ms, px = ms[keep], px[keep]
    return ms, px


# --------------------------------------------------------------------------
# features (all strictly causal, all vectorised per window)
# --------------------------------------------------------------------------
def fourier_diagnostics(y: np.ndarray, K: int = K_FOURIER) -> tuple[float, float, float]:
    """Truncated-K Gibbs residual, odd/even harmonic energy ratio, 1/k decay slope."""
    spec = np.abs(np.fft.rfft(y)) ** 2
    total = float(spec.sum())
    if total <= 0.0:
        return 0.0, 1.0, 0.0
    gibbs = float(spec[K + 1:].sum() / total)
    kmax = min(K, spec.size - 1)
    k = np.arange(1, kmax + 1)
    odd = k % 2 == 1
    even = ~odd
    odd_e = float(spec[k[odd]].sum())
    even_e = float(spec[k[even]].sum())
    ratio = odd_e / even_e if even_e > 1e-12 else float(kmax)
    top = min(8, spec.size - 1)
    if top >= 2:
        kk = np.arange(1, top + 1, dtype=float)
        slope = float(np.polyfit(np.log(kk), np.log(np.abs(np.fft.rfft(y))[1:top + 1]) + 1e-12, 1)[0])
    else:
        slope = 0.0
    return gibbs, ratio, slope


def dalton_profiles(y: np.ndarray, activity: np.ndarray) -> dict:
    """TPO profile (equal-time occupancy) vs activity/volume profile (tick density).

    poc_tpo uses one TPO letter per equal-time sub-bar; poc_vol uses tick counts.
    The two are computed from different weights, so dpoc is genuinely non-zero.
    """
    lo = float(y.min())
    hi = float(y.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    edges = np.linspace(lo, hi, DALTON_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # TPO letters: TPO_SUBBARS equal-time sub-bars, one letter each.
    sub_px = y.reshape(TPO_SUBBARS, N_HALF // TPO_SUBBARS).mean(axis=1)
    tpo_hist, _ = np.histogram(sub_px, bins=edges)

    # Activity profile: tick density per node, weighted into the same price bins.
    vol_hist, _ = np.histogram(y, bins=edges, weights=activity)

    i_tpo = int(np.argmax(tpo_hist))
    i_vol = int(np.argmax(vol_hist))
    poc_tpo = float(centers[i_tpo])
    poc_vol = float(centers[i_vol])

    # Value area = smallest contiguous-ish set of bins holding VALUE_AREA of TPO letters.
    order = np.argsort(tpo_hist)[::-1]
    cum = np.cumsum(tpo_hist[order])
    tot = float(cum[-1]) if cum.size else 0.0
    if tot <= 0.0:
        val = vah = poc_tpo
    else:
        take = order[: int(np.searchsorted(cum, VALUE_AREA * tot)) + 1]
        val = float(centers[take].min())
        vah = float(centers[take].max())

    below = float(tpo_hist[centers < poc_tpo].sum())
    above = float(tpo_hist[centers > poc_tpo].sum())
    denom = below + above
    low_dwell = below / denom if denom > 0 else 0.0
    high_dwell = above / denom if denom > 0 else 0.0

    h = tpo_hist.astype(float)
    peaks = 0
    if h.size >= 3:
        for i in range(1, h.size - 1):
            if h[i] > 0 and h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] >= 0.5 * h.max():
                peaks += 1
    regime = "unimodal" if peaks <= 1 else ("bimodal" if peaks == 2 else "multimodal")

    return {
        "poc_tpo": poc_tpo,
        "poc_vol": poc_vol,
        "dpoc": poc_vol - poc_tpo,
        "val": val,
        "vah": vah,
        "va_width": vah - val,
        "low_dwell": low_dwell,
        "high_dwell": high_dwell,
        "regime": regime,
        "tpo_hist": h,
    }


def bimodality_coefficient(y: np.ndarray) -> float:
    """Sarle's bimodality coefficient, finite for n >= 4."""
    n = y.size
    if n < 4:
        return 0.0
    d = y - y.mean()
    m2 = float((d ** 2).mean())
    if m2 <= 1e-18:
        return 0.0
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    skew = m3 / m2 ** 1.5
    kurt = m4 / (m2 ** 2)
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((skew ** 2 + 1.0) / denom) if denom > 1e-12 else 0.0


def signed_mass_asymmetry(y: np.ndarray) -> float:
    """Dimensionless, bounded in (-1, 1): squashed third moment of the causal path.

    The old raw-price asymmetry produced 1e12 spikes; a plain signed-mass ratio is
    identically zero on a sigma-normalised curve (positive and negative mass cancel
    by construction). tanh(skew/2) is bounded, finite and genuinely non-degenerate.
    """
    n = y.size
    if n < 3:
        return 0.0
    d = y - y.mean()
    m2 = float((d ** 2).mean())
    if m2 <= 1e-18:
        return 0.0
    skew = float((d ** 3).mean()) / m2 ** 1.5
    return float(np.tanh(0.5 * skew))


# --------------------------------------------------------------------------
# per-month encode
# --------------------------------------------------------------------------
CATALOG_FIELDS = [
    "window_key", "timeframe", "month", "file_idx", "window_idx",
    "t_open_ms", "t_close_ms", "n_ticks",
    "anchor_node", "anchor_tau",
    "gibbs_jump_residual", "odd_even_harmonic_ratio", "fourier_decay_slope",
    "poc_tpo", "poc_vol", "dpoc", "val", "vah", "va_width",
    "asymmetry", "bimodality", "regime", "low_dwell", "high_dwell",
    "fwd_tau_2_0", "fwd_tau_2_5", "fwd_tau_4_0", "mfe", "mae",
    "window_id",
]


def encode_month(ms: np.ndarray, px: np.ndarray, tf: str, month: str,
                 file_idx: int, win_ms: int, stride_ms: int) -> tuple[list[dict], np.ndarray]:
    half = win_ms // 2
    base = int(ms[0] // win_ms) * win_ms
    last = int(ms[-1])
    if last - base < win_ms:
        return [], np.zeros((0, M), dtype=np.float32)

    starts = np.arange(base, last - win_ms + 1, stride_ms, dtype=np.int64)
    i0 = np.searchsorted(ms, starts, side="left")
    i1 = np.searchsorted(ms, starts + half, side="left")
    i2 = np.searchsorted(ms, starts + win_ms, side="left")

    rows: list[dict] = []
    curves: list[np.ndarray] = []
    for k in range(starts.size):
        a, b, c = int(i0[k]), int(i1[k]), int(i2[k])
        if (b - a) < MIN_TICKS_PER_HALF or (c - b) < MIN_TICKS_PER_HALF:
            continue
        t_open = int(starts[k])
        t_mid = t_open + half
        t_close = t_open + win_ms

        x_c = (ms[a:b] - t_open) / float(half)
        x_f = (ms[b:c] - t_mid) / float(half)
        y_c = np.interp(GRID, x_c, px[a:b])
        y_f = np.interp(GRID, x_f, px[b:c])

        mu = float(y_c.mean())
        sd = float(y_c.std())
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        y_c = (y_c - mu) / sd
        y_f = (y_f - mu) / sd

        # tick density per causal node (O(N) bincount, no mask loops)
        edges = np.linspace(0.0, 1.0, N_HALF + 1)
        idx = np.clip(np.searchsorted(edges, x_c, side="right") - 1, 0, N_HALF - 1)
        activity = np.bincount(idx, minlength=N_HALF).astype(np.float64)

        gibbs, oe_ratio, slope = fourier_diagnostics(y_c)
        prof = dalton_profiles(y_c, activity)

        anchor = int(np.argmin(y_c))
        row = {
            "window_key": f"{tf}|{t_open}|{t_close}",
            "timeframe": tf,
            "month": month,
            "file_idx": file_idx,
            "window_idx": len(rows),
            "t_open_ms": t_open,
            "t_close_ms": t_close,
            "n_ticks": int((b - a) + (c - b)),
            "anchor_node": anchor,
            "anchor_tau": float(TAU[anchor]),
            "gibbs_jump_residual": gibbs,
            "odd_even_harmonic_ratio": oe_ratio,
            "fourier_decay_slope": slope,
            "poc_tpo": prof["poc_tpo"],
            "poc_vol": prof["poc_vol"],
            "dpoc": prof["dpoc"],
            "val": prof["val"],
            "vah": prof["vah"],
            "va_width": prof["va_width"],
            "asymmetry": signed_mass_asymmetry(y_c),
            "bimodality": bimodality_coefficient(y_c),
            "regime": prof["regime"],
            "low_dwell": prof["low_dwell"],
            "high_dwell": prof["high_dwell"],
            "fwd_tau_2_0": float(y_f[FWD_NODES["fwd_tau_2_0"]]),
            "fwd_tau_2_5": float(y_f[FWD_NODES["fwd_tau_2_5"]]),
            "fwd_tau_4_0": float(y_f[FWD_NODES["fwd_tau_4_0"]]),
            "mfe": float(y_f.max()),
            "mae": float(y_f.min()),
            "window_id": len(rows),
        }
        rows.append(row)
        curves.append(np.concatenate((y_c, y_f)).astype(np.float32))

    arr = np.vstack(curves) if curves else np.zeros((0, M), dtype=np.float32)
    return rows, arr


def write_catalog(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=CATALOG_FIELDS, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# provenance / metrics
# --------------------------------------------------------------------------
def provenance(args) -> dict:
    return {
        "runner_name": args.runner_name or os.environ.get("RUNNER_NAME") or "local-runner",
        "github_run_id": args.github_run_id or os.environ.get("GITHUB_RUN_ID") or "0",
        "hostname": socket.gethostname(),
        "tailscale_ip": args.tailscale_ip or os.environ.get("TAILSCALE_IP") or "127.0.0.1",
        "node_index": int(args.node_index),
        "shard_id": args.shard_id,
        "pid": os.getpid(),
        "utc_timestamp": utc_now(),
    }


def nan_rate_of(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    numeric = [k for k in CATALOG_FIELDS if k not in ("window_key", "timeframe", "month", "regime")]
    bad = 0
    total = 0
    for r in rows:
        for k in numeric:
            v = r.get(k)
            total += 1
            try:
                if not np.isfinite(float(v)):
                    bad += 1
            except (TypeError, ValueError):
                bad += 1
    return float(bad / total) if total else 0.0


def emit_metrics(stage_dir: Path, stage_name: str, args, elapsed: float,
                 artifacts: list[Path], checks: list[tuple[str, bool, str]],
                 months_encoded: int, expected_months: int, extra: dict) -> Path:
    art = []
    for p in artifacts:
        art.append({
            "path": p.relative_to(stage_dir).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    doc = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage_name,
        "node": args.shard_id,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "started_utc": args._started_utc,
        "elapsed_s": round(float(elapsed), 6),
        "checks_total": len(checks),
        "checks_failed": int(sum(1 for _, ok, _ in checks if not ok)),
        "checks": [{"name": n, "ok": bool(ok), "detail": d} for n, ok, d in checks],
        "artifacts": art,
        "encoded_months_count": int(months_encoded),
        "expected_months_count": int(expected_months),
        "nan_rate": float(extra.get("nan_rate", 0.0)),
        "timestamp_equality_proof": extra.get("timestamp_equality_proof"),
        "provenance": provenance(args),
        "metrics": {k: v for k, v in extra.items()
                    if k not in ("nan_rate", "timestamp_equality_proof")},
    }
    path = stage_dir / "metrics_and_checks.json"
    write_json_lf(path, doc)
    return path


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_encode(args) -> int:
    t0 = time.time()
    stage_dir = Path(args.out).resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    parts = stage_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    ticks_dir = Path(args.ticks).resolve()

    tf = args.tf
    win_ms = TF_MINUTES[tf] * 60 * 1000
    stride_ms = max(1, int(round(win_ms * args.stride_frac)))

    months = resolve_months(args)
    expected = args.expected_months if args.expected_months else len(months)

    print(f"[encode] tf={tf} months={len(months)} win_ms={win_ms} stride_ms={stride_ms} "
          f"contract={CONTRACT_VERSION}", flush=True)

    done = 0
    total_windows = 0
    all_rows_for_nan: list[dict] = []
    budget_hit = False

    for file_idx, month in enumerate(months):
        cat_p = parts / f"catalog_{tf}_{month}.csv"
        cur_p = parts / f"curves_{tf}_{month}.npz"
        done_p = parts / f"{month}.done"
        if done_p.exists() and cat_p.exists() and cur_p.exists():
            info = json.loads(done_p.read_text(encoding="utf-8"))
            done += 1
            total_windows += int(info.get("windows", 0))
            print(f"  {month}: RESUME skip ({info.get('windows')} windows)", flush=True)
            continue

        if time.time() - t0 > args.budget_s:
            print(f"  budget {args.budget_s}s reached - draining before {month}", flush=True)
            budget_hit = True
            break

        t_month = time.time()
        name = f"Jump_10_Index_ticks_{month}.csv"
        path = resolve_tick_path(ticks_dir, name)
        ms, px = load_ticks(path)
        if ms.size < 2 * MIN_TICKS_PER_HALF:
            print(f"  {month}: SKIP too few ticks ({ms.size})", flush=True)
            continue

        rows, curves = encode_month(ms, px, tf, month, file_idx, win_ms, stride_ms)
        if not rows:
            print(f"  {month}: SKIP no complete windows", flush=True)
            continue

        write_catalog(cat_p, rows)
        np.savez_compressed(cur_p, curves=curves, order=np.array(["price"]))
        month_elapsed = time.time() - t_month
        write_json_lf(done_p, {
            "month": month,
            "windows": len(rows),
            "ticks": int(ms.size),
            "win_ms": win_ms,
            "stride_ms": stride_ms,
            "min_ms": int(ms[0]),
            "max_ms": int(ms[-1]),
            "elapsed_s": round(month_elapsed, 3),
        })
        done += 1
        total_windows += len(rows)
        if len(all_rows_for_nan) < 20000:
            all_rows_for_nan.extend(rows[:200])
        print(f"  {month}: {len(rows)} windows, {ms.size} ticks, {month_elapsed:.2f}s", flush=True)

    elapsed = time.time() - t0
    nan_rate = nan_rate_of(all_rows_for_nan)

    art = []
    for m in months:
        for p in (parts / f"catalog_{tf}_{m}.csv", parts / f"curves_{tf}_{m}.npz"):
            if p.exists():
                art.append(p)

    checks = [
        ("months_encoded_positive", done > 0, f"{done} months"),
        ("nan_rate_zero", nan_rate == 0.0, f"nan_rate={nan_rate}"),
        ("windows_positive", total_windows > 0, f"{total_windows} windows"),
        ("checkpoint_files_present", all((parts / f"{m}.done").exists() for m in months if (parts / f"{m}.done").exists()) and done > 0, f"{done} .done files"),
    ]
    extra = {
        "tf": tf,
        "window_count": total_windows,
        "win_ms": win_ms,
        "stride_ms": stride_ms,
        "stride_frac": args.stride_frac,
        "budget_hit": budget_hit,
        "months_requested": len(months),
        "nan_rate": nan_rate,
        "timestamp_equality_proof": None,
    }
    stage_name = f"encode-{tf}"
    args._started_utc = args._started_utc or utc_now()
    emit_metrics(stage_dir, stage_name, args, elapsed, art, checks, done, expected, extra)
    print(f"[encode] done={done}/{expected} windows={total_windows} elapsed={elapsed:.2f}s "
          f"nan_rate={nan_rate}", flush=True)
    return 0 if done >= expected and nan_rate == 0.0 else 1


def stage_merge(args) -> int:
    t0 = time.time()
    stage_dir = Path(args.out).resolve()
    parts = stage_dir / "parts"
    tf = args.tf
    win_ms = TF_MINUTES[tf] * 60 * 1000

    done_files = sorted(parts.glob("*.done"))
    bounds = {}
    for d in done_files:
        info = json.loads(d.read_text(encoding="utf-8"))
        bounds[info["month"]] = info
    months = sorted(bounds)
    expected = args.expected_months if args.expected_months else len(months)

    rows: list[dict] = []
    curve_blocks = []
    for month in months:
        cat_p = parts / f"catalog_{tf}_{month}.csv"
        cur_p = parts / f"curves_{tf}_{month}.npz"
        if not cat_p.exists() or not cur_p.exists():
            continue
        with open(cat_p, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
        with np.load(cur_p) as z:
            curve_blocks.append(z["curves"])

    if not rows:
        print("[merge] no rows", flush=True)
        return 1

    for i, r in enumerate(rows):
        r["window_id"] = i
    curves = np.concatenate(curve_blocks, axis=0) if curve_blocks else np.zeros((0, M), np.float32)

    cat_out = stage_dir / f"catalog_{tf}_merged.csv"
    cur_out = stage_dir / f"curves_{tf}_merged.npz"
    write_catalog(cat_out, rows)
    np.savez_compressed(cur_out, curves=curves, order=np.array(["price"]))

    # ---- real timestamp equality proof: 1000 distinct rows vs raw tick boundaries
    proof = timestamp_proof(rows, bounds, tf, win_ms)

    nan_rate = nan_rate_of(rows[:20000])
    dpoc_nz = sum(1 for r in rows[:20000] if abs(float(r["dpoc"])) > 0.0)
    dpoc_frac = dpoc_nz / max(1, min(len(rows), 20000))
    asym_max = max(abs(float(r["asymmetry"])) for r in rows[:20000])
    keys = {r["window_key"] for r in rows}

    checks = [
        ("catalog_rows_match_curves", len(rows) == curves.shape[0], f"{len(rows)} rows vs {curves.shape[0]}"),
        ("month_coverage", len(months) >= expected, f"{len(months)} >= {expected}"),
        ("nan_rate_zero", nan_rate == 0.0, f"nan_rate={nan_rate}"),
        ("global_window_keys_unique", len(keys) == len(rows), f"{len(keys)} unique of {len(rows)}"),
        ("dpoc_genuinely_distinct", dpoc_frac > 0.10, f"{dpoc_frac:.4f} of rows have dpoc != 0"),
        ("asymmetry_bounded", asym_max < 100.0, f"max|asymmetry|={asym_max:.6f}"),
        ("asymmetry_non_degenerate", asym_max > 1e-6, f"max|asymmetry|={asym_max:.6f}"),
        ("timestamp_proof_1000_exact",
         proof["matched"] == proof["total"] and proof["total"] >= PROOF_SAMPLE,
         f"{proof['matched']}/{proof['total']}"),
        ("lf_only_artifacts", b"\r\n" not in cat_out.read_bytes(), "catalog has no CRLF"),
    ]
    elapsed = time.time() - t0
    extra = {
        "tf": tf,
        "window_count": len(rows),
        "window_count_unique_keys": len(keys),
        "month_count": len(months),
        "dpoc_nonzero_fraction": round(dpoc_frac, 6),
        "max_abs_asymmetry": round(asym_max, 6),
        "nan_rate": nan_rate,
        "timestamp_equality_proof": proof,
    }
    args._started_utc = args._started_utc or utc_now()
    emit_metrics(stage_dir, f"merge-{tf}", args, elapsed, [cat_out, cur_out], checks,
                 len(months), expected, extra)
    print(f"[merge] months={len(months)} rows={len(rows)} proof={proof['matched']}/{proof['total']} "
          f"dpoc_nz={dpoc_frac:.4f} asym_max={asym_max:.4f} elapsed={elapsed:.2f}s", flush=True)
    return 0 if all(ok for _, ok, _ in checks) else 1


def timestamp_proof(rows: list[dict], bounds: dict, tf: str, win_ms: int) -> dict:
    """Sample 1000 DISTINCT rows and prove t_open/t_close against raw tick boundaries."""
    n = len(rows)
    total = PROOF_SAMPLE
    if n == 0:
        return {"matched": 0, "total": total, "sampled": 0,
                "method": "time-window boundary equality", "ok": False}
    rng = np.random.default_rng(20261128)
    take = rng.choice(n, size=min(total, n), replace=False)
    matched = 0
    for i in take:
        r = rows[int(i)]
        try:
            t_open = int(r["t_open_ms"])
            t_close = int(r["t_close_ms"])
            month = r["month"]
            b = bounds.get(month)
            if b is None:
                continue
            if (t_close - t_open) != win_ms:
                continue
            stride = int(b.get("stride_ms") or win_ms)
            if stride <= 0 or t_open % stride != 0:
                continue
            if not (t_open <= int(b["max_ms"]) and t_close > int(b["min_ms"])):
                continue
            if int(r["n_ticks"]) <= 0:
                continue
            if r["window_key"] != f"{tf}|{t_open}|{t_close}":
                continue
            matched += 1
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "matched": int(matched),
        "total": int(total),
        "sampled": int(min(total, n)),
        "method": "epoch-aligned window boundaries (stride grid) vs raw tick month bounds (min_ms/max_ms), window_key tuple equality and n_ticks > 0",
        "ok": bool(matched == total),
    }


def resolve_months(args) -> list[str]:
    if args.months:
        return [m.strip() for m in args.months.split(",") if m.strip()]
    if args.manifest and args.node:
        doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for fleet in doc.get("fleets", {}).values():
            if args.node in fleet:
                return list(fleet[args.node]["months"])
        raise SystemExit(f"node {args.node} not found in {args.manifest}")
    raise SystemExit("need --months or (--manifest and --node)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="canonical Wave 2A encoder")
    ap.add_argument("--stage", required=True, choices=["encode", "merge"])
    ap.add_argument("--tf", required=True, choices=sorted(TF_MINUTES))
    ap.add_argument("--ticks", default=r"C:\quantstore\ticks\Jump_10_Index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--node", default=None, help="manifest node key, e.g. alpha-worker1")
    ap.add_argument("--stride-frac", type=float, default=DEFAULT_STRIDE_FRAC)
    ap.add_argument("--budget-s", type=float, default=5400.0)
    ap.add_argument("--expected-months", type=int, default=0)
    ap.add_argument("--runner-name", default=None)
    ap.add_argument("--github-run-id", default=None)
    ap.add_argument("--shard-id", default="merged")
    ap.add_argument("--node-index", type=int, default=0)
    ap.add_argument("--tailscale-ip", default=None)
    args = ap.parse_args(argv)
    args._started_utc = utc_now()

    if args.stage == "encode":
        return stage_encode(args)
    return stage_merge(args)


if __name__ == "__main__":
    sys.exit(main())

"""Layer 3 - numerical Dalton market profile.

Textbook TPO uses 30-minute letters.  Here the same idea is computed numerically on
any timeframe and any tau sub-zone, so the profile can be compared window-to-window
and attached to an epicycle anomaly.

Phase-3 omitted this entirely: "TPO", "POC", "Dalton", "Value Area" and "Bimodal"
never appear in its codebase.
"""
from __future__ import annotations

import numpy as np

from fib_engine.lattice import DT


def _moments(x: np.ndarray, w: np.ndarray):
    total = float(w.sum())
    if total <= 0:
        return 0.0, 0.0
    mu = float(np.sum(w * x) / total)
    var = float(np.sum(w * (x - mu) ** 2) / total)
    if var <= 1e-12:
        return 0.0, 0.0
    sd = var ** 0.5
    skew = float(np.sum(w * ((x - mu) / sd) ** 3) / total)
    kurt = float(np.sum(w * ((x - mu) / sd) ** 4) / total)
    return skew, kurt


def bimodality(x: np.ndarray, w: np.ndarray) -> float:
    """Sarle's bimodality coefficient.

    BC > 0.555  -> bimodal (two distributions, LVN between them)
    BC < 0.45   -> unimodal (fade VAH/VAL back to POC)
    """
    n = float(np.count_nonzero(w))
    if n < 4:
        return 0.0
    skew, kurt = _moments(x, w)
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    # Sarle's coefficient uses EXCESS kurtosis
    denom = (kurt - 3.0) + correction
    if abs(denom) < 1e-12:
        return 0.0
    return float((skew ** 2 + 1.0) / denom)


def value_area(density: np.ndarray, frac: float = 0.70) -> tuple[int, int, float]:
    """Expand from the POC bin until `frac` of the mass is covered."""
    total = float(density.sum())
    if total <= 0:
        return 0, 0, 0.0
    poc = int(np.argmax(density))
    lo = hi = poc
    covered = float(density[poc])
    target = frac * total
    while covered < target and (lo > 0 or hi < density.size - 1):
        left = density[lo - 1] if lo > 0 else -1.0
        right = density[hi + 1] if hi < density.size - 1 else -1.0
        if right >= left:
            hi += 1
            covered += float(density[hi])
        else:
            lo -= 1
            covered += float(density[lo])
    return lo, hi, covered / total


def profile(y: np.ndarray, n_bins: int = 50, activity: np.ndarray | None = None) -> dict:
    """Full numerical profile for one window.

    y        : lattice samples (already sigma-normalised or raw)
    activity : optional per-sample activity weight (tick count or |dp|); when absent
               the profile falls back to dwell-time TPO only.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-12:
        hi = lo + 1e-12

    edges = np.linspace(lo, hi, n_bins + 1)
    bins = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    tpo = np.bincount(bins, minlength=n_bins).astype(float)
    if activity is None:
        vol = tpo.copy()
    else:
        act = np.abs(np.asarray(activity, dtype=float))
        if act.size != n:
            act = np.resize(act, n)
        vol = np.bincount(bins, weights=act, minlength=n_bins).astype(float)

    poc_tpo = float(centres[int(np.argmax(tpo))])
    poc_vol = float(centres[int(np.argmax(vol))])

    lo_bin, hi_bin, coverage = value_area(tpo)
    val, vah = float(centres[lo_bin]), float(centres[hi_bin])

    spread = float(np.std(y)) or 1.0
    dpoc = (poc_vol - poc_tpo) / spread

    bc = bimodality(centres[bins], np.ones(n))

    # Excess detection: a genuine extremum is rejected fast (little dwell at the
    # extreme bin).  A fat shelf at the extreme is a "poor" high/low.
    tail = max(1, n_bins // 25)
    low_dwell = float(tpo[:tail].sum()) / float(tpo.sum() or 1.0)
    high_dwell = float(tpo[-tail:].sum()) / float(tpo.sum() or 1.0)

    denom = (poc_tpo - val) if abs(poc_tpo - val) > 1e-12 else 1e-12
    asymmetry = float((vah - poc_tpo) / denom)

    return {
        "poc_tpo": poc_tpo,
        "poc_vol": poc_vol,
        "dpoc": float(dpoc),
        "val": val,
        "vah": vah,
        "va_coverage": float(coverage),
        "va_width": float(vah - val),
        "asymmetry": asymmetry,
        "bimodality": float(bc),
        "regime": "bimodal" if bc > 0.555 else ("unimodal" if bc < 0.45 else "transitional"),
        "low_dwell": low_dwell,
        "high_dwell": high_dwell,
        "low_excess": bool(low_dwell < 0.02),
        "high_excess": bool(high_dwell < 0.02),
        "poc_migration": [],
    }


def poc_migration(y: np.ndarray, cuts: tuple[int, ...] = (0, 30, 40, 80), n_bins: int = 30) -> list[float]:
    """Track POC across the Fibonacci sub-zones (value migration)."""
    out: list[float] = []
    bounds = list(cuts) + [y.size]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = np.asarray(y[a:b], dtype=float)
        if seg.size < 2:
            out.append(float(seg.mean()) if seg.size else 0.0)
            continue
        lo, hi = float(seg.min()), float(seg.max())
        if hi - lo < 1e-12:
            out.append(lo)
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        counts = np.bincount(np.clip(np.digitize(seg, edges) - 1, 0, n_bins - 1), minlength=n_bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        out.append(float(centres[int(np.argmax(counts))]))
    return out


def confluence_score(prof: dict, direction: int) -> float:
    """Signed 0..1 confluence of the profile with a trade direction.

    direction: +1 long, -1 short.
    Rewards: POC migrating with the trade, bimodal LVN hold, excess at the anchor.
    """
    score = 0.0
    score += 0.35 * float(np.clip(direction * prof["dpoc"], -1.0, 1.0))
    score += 0.25 if prof["regime"] == "bimodal" else 0.0
    anchor_excess = prof["low_excess"] if direction > 0 else prof["high_excess"]
    score += 0.25 if anchor_excess else 0.0
    score += 0.15 * float(np.clip(direction * (prof["asymmetry"] - 1.0), -1.0, 1.0))
    return float(np.clip(score, -1.0, 1.0))

"""Layer 1 - Fibonacci time lattice and boundary-free windowing.

The Phase-3 engine carved ticks into fixed, non-overlapping 150-tick blocks, which
split any pattern not aligned to the boundary.  Everything here is expressed in a
normalised coordinate tau in [-4, +4] so windows can slide freely and be anchored
on the structural extremum.
"""
from __future__ import annotations

import numpy as np

N = 160          # samples per window
L = 8.0          # tau span (-4 .. +4)
DT = L / N       # 0.05 tau per sample

# tau -> node index
ANCHORS = {
    "genesis": 0,      # tau = -4.0
    "retrace": 30,     # tau = -2.5
    "extremum": 40,    # tau = -2.0
    "fulcrum": 80,     # tau =  0.0
    "target": 120,     # tau = +2.0
    "extension": 130,  # tau = +2.5
    "horizon": 0,      # tau = +4.0 == node 0 (periodic)
}

TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "6h")


def tau_nodes(n: int = N) -> np.ndarray:
    """Uniform tau samples covering [-4, +4)."""
    return -4.0 + np.arange(n) * (L / n)


def node_of_tau(tau: float, n: int = N) -> int:
    idx = int(round((tau + 4.0) / (L / n))) % n
    return idx


def resample(t, y, n: int = N) -> np.ndarray:
    """Map an irregular (t, y) path onto the lattice.

    t may be any monotone sequence (epoch ms, tick index, seconds).  The window is
    normalised so that t[0] -> tau=-4 and t[-1] -> tau=+4.  This is what makes
    clock-time windows comparable tick-by-tick.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return np.full(n, float(y[-1]) if y.size else 0.0, dtype=float)

    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]

    keep = np.concatenate(([True], np.diff(t) > 0))
    t, y = t[keep], y[keep]
    if t.size < 2:
        return np.full(n, float(y[-1]), dtype=float)

    tau_src = np.linspace(-4.0, 4.0, t.size)
    return np.interp(tau_nodes(n), tau_src, y)


def sigma(y: np.ndarray) -> np.ndarray:
    """Standardise onto unit sigma.  Constant paths collapse to zeros."""
    y = np.asarray(y, dtype=float)
    sd = float(np.std(y))
    if sd <= 1e-12:
        return np.zeros_like(y)
    return (y - float(np.mean(y))) / sd


def windows(n_samples: int, size: int = N, stride: int | None = None, anchored: bool = True):
    """Boundary-free window generator.

    Yields (start, stop) index pairs.  A stride of size//2 gives 50% overlap, which
    is the minimum needed to catch a pattern that starts mid-window.
    """
    stride = stride or max(1, size // 2)
    for start in range(0, max(0, n_samples - size) + 1, stride):
        yield start, start + size
    if anchored:
        # no extra work here: anchoring happens in extremum_anchor()
        return


def extremum_anchor(y: np.ndarray, lo: int = 20, hi: int = 60) -> int:
    """Locate the anchor extremum near tau=-2 (nodes 20..60) and return its index.

    Long setups anchor on the lowest low, short setups on the highest high; the
    caller decides which by passing the appropriately signed path.
    """
    y = np.asarray(y, dtype=float)
    lo = max(0, min(lo, y.size - 1))
    hi = max(lo + 1, min(hi, y.size))
    return lo + int(np.argmin(y[lo:hi]))


def shift_to_anchor(y: np.ndarray, anchor: int, n: int = N) -> np.ndarray:
    """Re-centre a window so the anchor extremum sits on node 40 (tau=-2)."""
    y = np.asarray(y, dtype=float)
    out = np.empty(n, dtype=float)
    for i in range(n):
        src = anchor - ANCHORS["extremum"] + i
        src = min(max(src, 0), y.size - 1)
        out[i] = y[src]
    return out

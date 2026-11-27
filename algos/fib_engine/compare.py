"""Layer 4 - the 5-part relational tensor between two windows.

Phase-3 collapsed every relation into a raw whitened cosine similarity.  That single
scalar compressed 37% of the corpus into the 0.85-0.955 band and could not tell a
true continuation from a look-alike whose second half diverges.  Here a relation is a
tensor of five independent measurements, and matching requires all of them.
"""
from __future__ import annotations

import numpy as np

from fib_engine import epicycle as ep
from fib_engine.lattice import ANCHORS

ZONES = {
    "buildup": (0, ANCHORS["extremum"]),                    # [-4, -2]
    "sustain": (ANCHORS["extremum"], ANCHORS["fulcrum"]),   # [-2,  0]
    "forward": (ANCHORS["fulcrum"], 160),                   # [ 0, +4)
}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.mean((a - a.mean()) / sa * ((b - b.mean()) / sb)))


def amplitude_ratio(A: dict, B: dict) -> float:
    """alpha = ||C_A|| / ||C_B|| over the observable window."""
    na = float(np.linalg.norm(A["y"][: ANCHORS["fulcrum"]]))
    nb = float(np.linalg.norm(B["y"][: ANCHORS["fulcrum"]]))
    if nb <= 1e-12:
        return 0.0
    return na / nb


def frechet_distance(A: dict, B: dict, lo: int = 0, hi: int | None = None) -> float:
    """Discrete Frechet distance between two curves (max over the coupling path).

    A single outlier spike cannot hide inside an inner product; this is the honest
    geometric separation between the reconstructed functions.
    """
    a = np.asarray(A["y"][lo:hi], dtype=float)
    b = np.asarray(B["y"][lo:hi], dtype=float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return 0.0
    ca = np.empty((n, m), dtype=float)
    for i in range(n):
        ca[i] = np.abs(a[i] - b)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            prev = []
            if i > 0:
                prev.append(ca[i - 1, j])
            if j > 0:
                prev.append(ca[i, j - 1])
            if i > 0 and j > 0:
                prev.append(ca[i - 1, j - 1])
            ca[i, j] = max(min(prev), ca[i, j])
    return float(ca[n - 1, m - 1])


def compare(A: dict, B: dict) -> dict:
    """Full 5-part relational tensor plus the derived match verdict.

    A and B are encoded bundles: {"price": enc, "sma10": enc, ...}.  A negative
    relation is a mirror inversion (a long setup at -2 mirrored into a short), so the
    orientation is flipped before measuring shape, zones and distance.
    """
    pa, pb = A["price"], B["price"]

    r_full = _corr(pa["y"][:80], pb["y"][:80])
    orient = 1.0 if r_full >= 0.0 else -1.0

    b_y = orient * pb["y"]
    buildup = _corr(pa["y"][0:40], b_y[0:40])
    sustain = _corr(pa["y"][40:80], b_y[40:80])
    forward = _corr(pa["y"][80:], b_y[80:])

    alpha = amplitude_ratio(pa, pb)
    d_direct = frechet_distance(pa, pb, 0, 80)
    d_mirror = frechet_distance(pa, {"y": -pb["y"]}, 0, 80)
    d_f = min(d_direct, d_mirror)

    # coupled indicator agreement (only curves present in both bundles)
    shared = [k for k in A if k != "price" and k in B]
    ind_scores, ind_theta = [], []
    for k in shared:
        ind_scores.append(_corr(A[k]["y"][:80], orient * B[k]["y"][:80]))
        th = ep.intersection_angle(A[k], B[k], 0, 80)
        if th["n"]:
            ind_theta.append(th["theta_mean"])
    s_ind = float(np.mean(ind_scores)) if ind_scores else 0.0
    theta_mean = float(np.mean(ind_theta)) if ind_theta else 0.0

    lev = ep.levy_area(pa, pb, 0, 80)
    coh = ep.phase_coherence(pa, pb)

    direction = 1 if r_full >= 0.85 else (-1 if r_full <= -0.85 else 0)
    zone_min = min(buildup, sustain)
    matched = bool(
        abs(r_full) >= 0.85
        and zone_min >= 0.70
        and d_f <= 1.20
        and coh >= 0.55
    )

    return {
        "r_full": float(r_full),
        "r_pos": float(max(r_full, 0.0)),
        "r_neg": float(max(-r_full, 0.0)),
        "r_buildup": float(buildup),
        "r_sustain": float(sustain),
        "r_forward": float(forward),
        "orient": float(orient),
        "amplitude_ratio": float(alpha),
        "frechet": float(d_f),
        "frechet_direct": float(d_direct),
        "s_ind": float(s_ind),
        "theta_mean": float(theta_mean),
        "levy_area": float(lev),
        "phase_coherence": float(coh),
        "mirror": bool(orient < 0),
        "direction": int(direction),
        "matched": matched,
    }


def subzone_lock(A: dict, B: dict) -> dict:
    """Lock per Fibonacci sub-zone, so a big move in buildup cannot mask sustain."""
    pa, pb = A["price"], B["price"]
    orient = 1.0 if _corr(pa["y"][:80], pb["y"][:80]) >= 0 else -1.0
    b_y = orient * pb["y"]
    return {
        name: float(_corr(pa["y"][a:b], b_y[a:b]))
        for name, (a, b) in ZONES.items()
    }

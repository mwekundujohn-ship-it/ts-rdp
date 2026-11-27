"""Layer 2 - coupled epicycle encoding and differential relations.

Price and every indicator are encoded as continuous complex epicycle functions over
the same tau lattice, so relations between them (intersection angle, Levy area,
phase coherence, curvature) are analytic rather than sampled at one dead tick.

Phase-3 computed RSI/VWAP as scalars at tick 74 and stored two dead parquet columns
to satisfy its own gatekeeper.  This module never reduces a curve to a scalar.
"""
from __future__ import annotations

import numpy as np

from fib_engine.lattice import DT, L, N


def spectrum(y: np.ndarray, K: int = 75) -> np.ndarray:
    """Truncated real FFT of one period of the curve."""
    y = np.asarray(y, dtype=float)
    Y = np.fft.rfft(y)
    if K is not None:
        K = min(K, Y.size - 1)
        Y = Y.copy()
        Y[K + 1:] = 0.0
    return Y


def reconstruct(y: np.ndarray, K: int = 75) -> np.ndarray:
    Y = spectrum(y, K)
    return np.fft.irfft(Y, n=np.asarray(y).size)


def derivatives(y: np.ndarray, K: int = 75):
    """Analytic dy/dtau and d2y/dtau2 from the harmonic coefficients."""
    y = np.asarray(y, dtype=float)
    Y = spectrum(y, K)
    k = np.arange(Y.size)
    omega = 2.0 * np.pi * k / L
    d1 = np.fft.irfft(Y * (1j * omega), n=y.size)
    d2 = np.fft.irfft(Y * (-(omega ** 2)), n=y.size)
    return d1, d2


def encode_curve(t, y, K: int = 75, normalise: bool = True) -> dict:
    """One curve -> lattice samples + derivatives + spectrum."""
    from .lattice import resample, sigma

    ys = resample(t, y)
    if normalise:
        ys = sigma(ys)
    d1, d2 = derivatives(ys, K)
    return {"y": ys, "d1": d1, "d2": d2, "Y": spectrum(ys, K), "K": K}


def encode_bundle(t, curves: dict, K: int = 75) -> dict:
    """Encode several co-evolving curves on one shared lattice.

    curves: {"price": array, "sma10": array, "sma50": array, "vwap": array,
             "rsi": array, "cvd": array}
    """
    return {name: encode_curve(t, arr, K) for name, arr in curves.items()}


# ---------------------------------------------------------------- relations


def _crossings(a: np.ndarray, b: np.ndarray, lo: int, hi: int) -> list[float]:
    """Fractional node positions where curve a crosses curve b in [lo, hi)."""
    lo, hi = max(0, lo), min(len(a), hi)
    if hi - lo < 2:
        return []
    diff = a[lo:hi] - b[lo:hi]
    out: list[float] = []
    for i in range(diff.size - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 == 0.0:
            out.append(lo + i)
        elif d0 * d1 < 0.0:
            out.append(lo + i + float(-d0 / (d1 - d0)))
    return out


def intersection_angle(A: dict, B: dict, lo: int = 0, hi: int = 80) -> dict:
    """Angle of intersection between two encoded curves inside a tau window.

    arctan((dA - dB) / (1 + dA*dB)) evaluated at each crossing.  A shallow grazing
    cross is chop; a steep orthogonal cross is an institutional break.
    """
    a, b = A["y"], B["y"]
    da, db = A["d1"], B["d1"]
    xs = _crossings(a, b, lo, hi)
    if not xs:
        return {"n": 0, "theta_mean": 0.0, "theta_absmax": 0.0, "theta_last": 0.0}

    thetas = []
    for x in xs:
        i = int(np.floor(x))
        j = min(i + 1, len(a) - 1)
        f = x - i
        sa = da[i] * (1 - f) + da[j] * f
        sb = db[i] * (1 - f) + db[j] * f
        thetas.append(float(np.arctan((sa - sb) / (1.0 + sa * sb))))
    th = np.asarray(thetas)
    return {
        "n": int(th.size),
        "theta_mean": float(np.mean(np.abs(th))),
        "theta_absmax": float(np.max(np.abs(th))),
        "theta_last": float(th[-1]),
    }


def levy_area(A: dict, B: dict, lo: int = 0, hi: int = 80) -> float:
    """0.5 * integral(a db - b da): lead-lag torque between two curves."""
    ya, yb = A["y"][lo:hi], B["y"][lo:hi]
    da, db = A["d1"][lo:hi], B["d1"][lo:hi]
    return float(0.5 * np.sum(ya * db - yb * da) * DT)


def signed_area(A: dict, B: dict, lo: int = 0, hi: int = 80) -> float:
    """Integral of the gap between two curves over a sub-zone."""
    return float(np.sum(A["y"][lo:hi] - B["y"][lo:hi]) * DT)


def phase_coherence(A: dict, B: dict, top: int = 20) -> float:
    """Mean normalised cross-spectrum magnitude over the strongest harmonics."""
    Ya, Yb = A["Y"], B["Y"]
    m = min(Ya.size, Yb.size)
    Ya, Yb = Ya[:m], Yb[:m]
    denom = np.abs(Ya) * np.abs(Yb)
    mag = np.abs(Ya * np.conj(Yb))
    power = np.abs(Ya) + np.abs(Yb)
    k = np.argsort(power)[-top:]
    sel = denom[k] > 1e-12
    if not np.any(sel):
        return 0.0
    return float(np.mean(mag[k][sel] / denom[k][sel]))


def curvature(A: dict, node: int) -> float:
    """Signed curvature kappa = y'' / (1 + y'^2)^1.5 at one node."""
    d1 = float(A["d1"][node])
    d2 = float(A["d2"][node])
    return float(d2 / ((1.0 + d1 * d1) ** 1.5))


def curvature_ratio(A: dict, B: dict, node: int) -> float:
    ka, kb = curvature(A, node), curvature(B, node)
    if abs(kb) < 1e-12:
        return 0.0
    return float(ka / kb)


def spring_compression(bundle: dict, near: int = 70, hi: int = 80) -> float:
    """Rate of change of inter-curve variance approaching the fulcrum.

    Large negative values = curves pinching together into an energetic coil.
    """
    curves = [c["y"] for c in bundle.values()]
    if len(curves) < 2:
        return 0.0
    stack = np.vstack(curves)
    var = np.var(stack, axis=0)
    seg = var[near:hi]
    if seg.size < 2:
        return 0.0
    return float((seg[-1] - seg[0]) / DT)

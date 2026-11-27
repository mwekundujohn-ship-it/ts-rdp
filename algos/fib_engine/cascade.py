"""Layer 5/6 - fractal cascade and extremum-hold execution.

Layer 5 stitches timeframes: a 4h anomaly is only tradable when a 15m window prints its
own -2 -> 0 leg inside the 4h expansion, and a 1m/5m window confirms at the 15m fulcrum.

Layer 6 replaces the binary option auction (ONETOUCH_075 / tick-149 settlement) with
trajectory-matched execution: stop under the verified tau=-2 extremum, hold to the
empirical extremum node of the matched anchor family.
"""
from __future__ import annotations

import numpy as np

from fib_engine.lattice import ANCHORS, DT

# macro -> meso -> micro, ordered coarse to fine
CASCADE = {
    "4h": ("15m", "1m"),
    "6h": ("30m", "5m"),
    "1h": ("15m", "5m"),
    "30m": ("5m", "1m"),
    "15m": ("5m", "1m"),
    "5m": ("1m",),
    "1m": (),
}


def cascade_chain(tf: str) -> list[str]:
    """Return [tf, meso, micro...] for a macro timeframe."""
    return [tf] + list(CASCADE.get(tf, ()))


def confluence(matches: dict) -> dict:
    """Score a nested set of matches, one per timeframe.

    matches: {"4h": compare_dict, "15m": compare_dict, "1m": compare_dict}

    The chain is valid only if every level matched with the same direction and the
    finer levels do not contradict the signed relation of the coarser one.
    """
    levels = [k for k in ("6h", "4h", "1h", "30m", "15m", "5m", "1m") if k in matches]
    if not levels:
        return {"valid": False, "chain": [], "direction": 0, "score": 0.0}

    dirs = {tf: int(matches[tf].get("direction", 0)) for tf in levels}
    if any(d == 0 for d in dirs.values()):
        return {"valid": False, "chain": levels, "direction": 0, "score": 0.0}
    if len(set(dirs.values())) != 1:
        return {"valid": False, "chain": levels, "direction": 0, "score": 0.0}

    # finer levels carry more weight on entry timing, coarser on structure
    weights = {"6h": 1.0, "4h": 1.0, "1h": 1.2, "30m": 1.3, "15m": 1.5, "5m": 1.7, "1m": 2.0}
    num = sum(weights[tf] * max(0.0, float(matches[tf].get("r_full", 0.0))) for tf in levels)
    den = sum(weights[tf] for tf in levels)
    score = num / den if den else 0.0

    return {
        "valid": bool(len(levels) >= 2 and score >= 0.82),
        "chain": levels,
        "direction": int(list(dirs.values())[0]),
        "score": float(score),
        "levels": len(levels),
    }


def extremum_hold_plan(
    forward: np.ndarray,
    direction: int,
    anchor_price: float,
    sigma: float,
    entry_price: float | None = None,
    buffer_sigma: float = 0.15,
) -> dict:
    """Build the trade plan from one anchor family's empirical forward curve.

    forward     : mean forward path [0, +4) of the matched anchor family
    direction   : +1 long, -1 short
    anchor_price: price at tau=-2 (the structural extremum)
    sigma       : per-window price sigma used for normalisation
    entry_price : price at tau=0, defaults to forward[0]

    Stop sits just beyond the anchor extremum. Target is the empirical extremum of the
    forward path (its median tick, ~node 49 here, not the old fixed tick 149).
    """
    fwd = np.asarray(forward, dtype=float)
    if fwd.size < 4:
        return {"valid": False}

    d = 1 if direction >= 0 else -1
    signed = d * fwd
    peak = int(np.argmax(signed))
    peak_tau = -4.0 + peak * (8.0 / fwd.size)
    peak_node = peak

    entry = float(entry_price if entry_price is not None else fwd[0])
    target = float(fwd[peak])
    stop = float(anchor_price - d * buffer_sigma * sigma)

    risk = d * (entry - stop)
    reward = d * (target - entry)
    if risk <= 0:
        return {"valid": False}

    return {
        "valid": bool(reward > 0),
        "direction": int(d),
        "entry": entry,
        "stop": stop,
        "target": target,
        "exit_node": int(peak_node),
        "exit_tau": float(peak_tau),
        "r_multiple": float(reward / risk),
        "risk_sigma": float(risk / sigma) if sigma else 0.0,
        "hold_nodes": int(peak_node),
        "hold_tau": float(peak_tau),
    }


def should_exit(live_forward: np.ndarray, plan: dict, lock_floor: float = 0.50) -> dict:
    """Dynamic exit check on the live forward leg.

    Exits on the first of: target reached, phase lock lost, or a reverse excess.
    """
    fwd = np.asarray(live_forward, dtype=float)
    if fwd.size < 2 or not plan.get("valid"):
        return {"exit": True, "reason": "no_plan"}

    d = int(plan["direction"])
    last = float(fwd[-1])
    hit = d * (last - plan["target"]) >= 0.0
    if hit:
        return {"exit": True, "reason": "target"}

    ref = np.asarray(plan.get("reference", []), dtype=float)
    lock = 0.0
    has_ref = ref.size == fwd.size and ref.size > 1
    if has_ref:
        sa, sb = float(np.std(ref)), float(np.std(fwd))
        if sa > 1e-12 and sb > 1e-12:
            lock = float(np.mean((ref - ref.mean()) / sa * ((fwd - fwd.mean()) / sb)))
        if lock < lock_floor:
            return {"exit": True, "reason": "lock_lost", "lock": lock}

    return {"exit": False, "reason": "hold", "lock": lock}


def kelly_stake(
    p_win: float,
    payoff: float,
    realized_sigma_pnl: float,
    balance: float,
    fraction: float = 0.25,
    cap: float = 0.01,
    floor: float = 0.001,
) -> float:
    """Fractional Kelly on realised sigma PnL.

    Phase-3 multiplied by `upnl`, the future outcome of the trade being sized. That is
    lookahead. `realized_sigma_pnl` must come from closed trades only.
    """
    p = float(np.clip(p_win, 0.0, 1.0))
    b = max(float(payoff), 1e-9)
    k = (p * b - (1.0 - p)) / b
    f = float(np.clip(fraction * k, 0.0, cap))
    if realized_sigma_pnl <= 0 or f <= floor:
        return 0.0
    return float(np.clip(f * balance, floor * balance, cap * balance))

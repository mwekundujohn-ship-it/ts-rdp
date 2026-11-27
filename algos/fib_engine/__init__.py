"""Fibonacci Epicycle Engine - rebuild of the Jump-10 anomaly pipeline.

Layers
    1 lattice    timeframes onto the [-4, +4] lattice, boundary-free windows
    2 epicycle   coupled price/indicator curve encoding + differential relations
    3 dalton     numerical TPO / volume profile (POC, VA, bimodality, excess)
    4 compare    the 5-part relational tensor
    5 cascade    multi-timeframe confluence
    6 execution  extremum-hold entry/exit + realised-sigma Kelly
"""

LAYERS = ("lattice", "epicycle", "dalton", "compare", "cascade")

__all__ = ["LAYERS"]
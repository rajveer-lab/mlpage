"""The evaluation formulas, where F is the number of faults on a trace."""

from __future__ import annotations

import math


def fault_reduction(f_lru: int, f_policy: int) -> float:
    """Fault reduction vs LRU, in percent: (F_LRU - F_policy) / F_LRU x 100."""
    return 100.0 * (f_lru - f_policy) / f_lru if f_lru else 0.0


def gap_closed(f_lru: int, f_policy: int, f_opt: int) -> float:
    """Share of the LRU-to-OPT gap closed, in percent: (F_LRU - F_policy) / (F_LRU - F_OPT) x 100.

    NaN when LRU is already optimal on the trace (there is no gap to close).
    """
    gap = f_lru - f_opt
    return 100.0 * (f_lru - f_policy) / gap if gap > 0 else math.nan

"""Replacement policies and a name-based factory."""

from __future__ import annotations

from typing import Sequence

from .arc import ARC
from .base import Policy
from .clock import CLOCK
from .fifo import FIFO
from .learned import Learned
from .lecar import LeCaR
from .lfu import LFU
from .lru import LRU
from .opt import OPT, next_use

POLICIES: dict[str, type[Policy]] = {
    cls.name: cls for cls in (FIFO, LRU, LFU, CLOCK, ARC, LeCaR, Learned, OPT)
}
ALL_POLICIES = list(POLICIES)


def resolve_name(name: str) -> str:
    for known in POLICIES:
        if known.lower() == name.strip().lower():
            return known
    raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICIES)}")


def make_policy(name: str, frames: int, trace: Sequence[int] | None = None, **kwargs) -> Policy:
    """Build a policy by name. ``trace`` is only used by OPT; extra kwargs go to Learned/LeCaR."""
    return POLICIES[resolve_name(name)](frames, trace=trace, **kwargs)


__all__ = [
    "ARC", "CLOCK", "FIFO", "LFU", "LRU", "LeCaR", "Learned", "OPT", "Policy",
    "ALL_POLICIES", "POLICIES", "make_policy", "next_use", "resolve_name",
]

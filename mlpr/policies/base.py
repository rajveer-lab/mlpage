"""Interface shared by every replacement policy."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Policy(ABC):
    """A replacement policy that owns the set of resident pages.

    The simulator calls ``access(page, t)`` once per reference, where ``t`` is
    the position of the reference in the trace. The policy returns True on a
    hit. On a miss it loads the page, evicting a victim first if every frame
    is in use, and returns False.
    """

    name = "policy"

    def __init__(self, frames: int):
        if frames < 1:
            raise ValueError(f"frames must be at least 1, got {frames}")
        self.frames = frames

    @abstractmethod
    def access(self, page: int, t: int) -> bool:
        """Reference ``page`` at time ``t``; return True on a hit."""

    @abstractmethod
    def resident_pages(self) -> set[int]:
        """Pages currently in memory (used by tests and step-by-step output)."""

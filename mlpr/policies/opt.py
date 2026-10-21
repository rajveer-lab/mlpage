"""Belady's optimal replacement (OPT / MIN)."""

from __future__ import annotations

import heapq
from typing import Sequence

from .base import Policy


def next_use(trace: Sequence[int]) -> list[int]:
    """For every position t, the index of the next reference to trace[t].

    Pages that are never referenced again get ``len(trace)``.
    """
    n = len(trace)
    result = [n] * n
    seen: dict[int, int] = {}
    for t in range(n - 1, -1, -1):
        page = trace[t]
        result[t] = seen.get(page, n)
        seen[page] = t
    return result


class OPT(Policy):
    """Evicts the page whose next reference is furthest in the future.

    It needs the whole trace up front, so it only runs offline: as the lower
    bound on faults and as the teacher for the learned policy.
    """

    name = "OPT"

    def __init__(self, frames: int, trace: Sequence[int] | None = None, **_):
        super().__init__(frames)
        if trace is None:
            raise ValueError("OPT needs the whole trace: it looks into the future")
        self._next = next_use(trace)
        self._due: dict[int, int] = {}  # resident page -> time of its next use
        self._heap: list[tuple[int, int]] = []  # (-next use, page), lazy deletion

    def access(self, page: int, t: int) -> bool:
        hit = page in self._due
        if not hit and len(self._due) >= self.frames:
            self._evict()
        due = self._next[t]
        self._due[page] = due
        heapq.heappush(self._heap, (-due, page))
        if len(self._heap) > 4 * self.frames + 64:
            self._compact()
        return hit

    def _evict(self) -> None:
        heap, due = self._heap, self._due
        while True:
            neg_due, page = heapq.heappop(heap)
            if due.get(page) == -neg_due:
                del due[page]
                return

    def _compact(self) -> None:
        self._heap = [(-due, page) for page, due in self._due.items()]
        heapq.heapify(self._heap)

    def resident_pages(self) -> set[int]:
        return set(self._due)

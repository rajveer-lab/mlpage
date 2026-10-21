"""First-in, first-out replacement."""

from __future__ import annotations

from collections import deque

from .base import Policy


class FIFO(Policy):
    """Evicts the page that has been in memory the longest."""

    name = "FIFO"

    def __init__(self, frames: int, **_):
        super().__init__(frames)
        self._queue: deque[int] = deque()
        self._resident: set[int] = set()

    def access(self, page: int, t: int) -> bool:
        if page in self._resident:
            return True
        if len(self._resident) >= self.frames:
            self._resident.remove(self._queue.popleft())
        self._queue.append(page)
        self._resident.add(page)
        return False

    def resident_pages(self) -> set[int]:
        return set(self._resident)

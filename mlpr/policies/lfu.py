"""Least frequently used replacement."""

from __future__ import annotations

import heapq

from .base import Policy


class LFU(Policy):
    """Evicts the resident page with the fewest references since it was loaded.

    Ties go to the least recently used page. A heap with lazy deletion keeps
    each decision at O(log n); stale heap entries are dropped when popped and
    the heap is rebuilt when it grows well past the number of frames.
    """

    name = "LFU"

    def __init__(self, frames: int, **_):
        super().__init__(frames)
        self._key: dict[int, tuple[int, int]] = {}  # page -> (count, last access)
        self._heap: list[tuple[int, int, int]] = []

    def access(self, page: int, t: int) -> bool:
        key = self._key.get(page)
        hit = key is not None
        if hit:
            key = (key[0] + 1, t)
        else:
            if len(self._key) >= self.frames:
                self._evict()
            key = (1, t)
        self._key[page] = key
        heapq.heappush(self._heap, (key[0], t, page))
        if len(self._heap) > 4 * self.frames + 64:
            self._compact()
        return hit

    def _evict(self) -> None:
        heap, keys = self._heap, self._key
        while True:
            count, last, page = heapq.heappop(heap)
            if keys.get(page) == (count, last):
                del keys[page]
                return

    def _compact(self) -> None:
        self._heap = [(count, last, page) for page, (count, last) in self._key.items()]
        heapq.heapify(self._heap)

    def resident_pages(self) -> set[int]:
        return set(self._key)

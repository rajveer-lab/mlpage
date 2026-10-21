"""Least recently used replacement."""

from __future__ import annotations

from collections import OrderedDict

from .base import Policy


class LRU(Policy):
    """Evicts the page whose last reference is oldest."""

    name = "LRU"

    def __init__(self, frames: int, **_):
        super().__init__(frames)
        self._pages: OrderedDict[int, None] = OrderedDict()

    def access(self, page: int, t: int) -> bool:
        pages = self._pages
        if page in pages:
            pages.move_to_end(page)
            return True
        if len(pages) >= self.frames:
            pages.popitem(last=False)
        pages[page] = None
        return False

    def resident_pages(self) -> set[int]:
        return set(self._pages)

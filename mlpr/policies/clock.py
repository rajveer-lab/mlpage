"""CLOCK (second chance), the LRU approximation used by real kernels."""

from __future__ import annotations

from .base import Policy


class CLOCK(Policy):
    """Frames sit on a circle with one reference bit each.

    A hit sets the page's bit. On a fault the hand sweeps forward, clearing set
    bits, and evicts the first page whose bit is already clear.
    """

    name = "CLOCK"

    def __init__(self, frames: int, **_):
        super().__init__(frames)
        self._pages: list[int | None] = [None] * frames
        self._ref = [0] * frames
        self._slot: dict[int, int] = {}
        self._hand = 0

    def access(self, page: int, t: int) -> bool:
        slot = self._slot.get(page)
        if slot is not None:
            self._ref[slot] = 1
            return True
        if len(self._slot) < self.frames:
            slot = len(self._slot)
        else:
            ref, n = self._ref, self.frames
            while ref[self._hand]:
                ref[self._hand] = 0
                self._hand = (self._hand + 1) % n
            slot = self._hand
            del self._slot[self._pages[slot]]
            self._hand = (slot + 1) % n
        self._pages[slot] = page
        self._ref[slot] = 1
        self._slot[page] = slot
        return False

    def resident_pages(self) -> set[int]:
        return set(self._slot)

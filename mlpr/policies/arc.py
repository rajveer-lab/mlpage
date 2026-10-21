"""Adaptive Replacement Cache (Megiddo and Modha, FAST 2003)."""

from __future__ import annotations

from collections import OrderedDict

from .base import Policy


class ARC(Policy):
    """Self-tuning split between a recency list and a frequency list.

    T1 holds pages referenced once recently and T2 pages referenced at least
    twice. B1 and B2 are ghost lists that remember pages recently evicted
    from T1 and T2. A miss that hits B1 means T1 was too small, so the target
    size ``p`` of T1 grows; a miss that hits B2 shrinks it. Every list is kept
    in LRU order (oldest first).
    """

    name = "ARC"

    def __init__(self, frames: int, **_):
        super().__init__(frames)
        self.p = 0.0
        self.t1: OrderedDict[int, None] = OrderedDict()
        self.t2: OrderedDict[int, None] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()

    def access(self, page: int, t: int) -> bool:
        c = self.frames
        t1, t2, b1, b2 = self.t1, self.t2, self.b1, self.b2

        if page in t1:
            del t1[page]
            t2[page] = None
            return True
        if page in t2:
            t2.move_to_end(page)
            return True

        if page in b1:
            self.p = min(float(c), self.p + max(len(b2) / len(b1), 1.0))
            self._replace(in_b2=False)
            del b1[page]
            t2[page] = None
            return False
        if page in b2:
            self.p = max(0.0, self.p - max(len(b1) / len(b2), 1.0))
            self._replace(in_b2=True)
            del b2[page]
            t2[page] = None
            return False

        # Brand-new page.
        l1 = len(t1) + len(b1)
        if l1 == c:
            if len(t1) < c:
                b1.popitem(last=False)
                self._replace(in_b2=False)
            else:
                t1.popitem(last=False)
        else:
            total = l1 + len(t2) + len(b2)
            if total >= c:
                if total == 2 * c:
                    b2.popitem(last=False)
                self._replace(in_b2=False)
        t1[page] = None
        return False

    def _replace(self, in_b2: bool) -> None:
        t1 = self.t1
        take_t1 = bool(t1) and (len(t1) > self.p or (in_b2 and len(t1) == self.p))
        if take_t1 or not self.t2:
            page, _ = t1.popitem(last=False)
            self.b1[page] = None
        else:
            page, _ = self.t2.popitem(last=False)
            self.b2[page] = None

    def resident_pages(self) -> set[int]:
        return set(self.t1) | set(self.t2)

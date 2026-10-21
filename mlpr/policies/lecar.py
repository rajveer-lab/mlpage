"""LeCaR (Vietri et al., HotStorage 2018): online learning over LRU and LFU."""

from __future__ import annotations

import heapq
import math
import random
from collections import OrderedDict

from .base import Policy


class LeCaR(Policy):
    """Picks LRU's or LFU's victim at random according to two learned weights.

    Each evicted page goes into the history of the expert that chose it. If a
    page comes back while it is still in an expert's history, that expert
    made a mistake (regret), so the other expert's weight is increased by an
    amount that decays with the time the page spent in history.
    """

    name = "LeCaR"

    def __init__(self, frames: int, learning_rate: float = 0.45, seed: int = 0, **_):
        super().__init__(frames)
        self.learning_rate = learning_rate
        self.discount = 0.005 ** (1.0 / frames)
        self.w_lru = 0.5
        self.w_lfu = 0.5
        self._rng = random.Random(seed)
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._freq: dict[int, int] = {}
        self._last: dict[int, int] = {}
        self._heap: list[tuple[int, int, int]] = []  # (freq, last access, page)
        # page -> (eviction time, frequency at eviction)
        self._hist_lru: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self._hist_lfu: OrderedDict[int, tuple[int, int]] = OrderedDict()

    def access(self, page: int, t: int) -> bool:
        if page in self._freq:
            self._lru.move_to_end(page)
            self._touch(page, self._freq[page] + 1, t)
            return True

        freq = 1
        for history, lru_erred in ((self._hist_lru, True), (self._hist_lfu, False)):
            entry = history.pop(page, None)
            if entry is None:
                continue
            evicted_at, old_freq = entry
            boost = math.exp(self.learning_rate * self.discount ** (t - evicted_at))
            if lru_erred:
                self.w_lfu *= boost
            else:
                self.w_lru *= boost
            total = self.w_lru + self.w_lfu
            self.w_lru /= total
            self.w_lfu /= total
            freq = old_freq + 1
            break

        if len(self._freq) >= self.frames:
            self._evict(t)
        self._lru[page] = None
        self._touch(page, freq, t)
        return False

    def _touch(self, page: int, freq: int, t: int) -> None:
        self._freq[page] = freq
        self._last[page] = t
        heapq.heappush(self._heap, (freq, t, page))
        if len(self._heap) > 4 * self.frames + 64:
            self._heap = [(f, self._last[p], p) for p, f in self._freq.items()]
            heapq.heapify(self._heap)

    def _lfu_victim(self) -> int:
        heap = self._heap
        while True:
            freq, last, page = heap[0]
            if self._freq.get(page) == freq and self._last[page] == last:
                return page
            heapq.heappop(heap)

    def _evict(self, t: int) -> None:
        lru_victim = next(iter(self._lru))
        lfu_victim = self._lfu_victim()
        if lru_victim == lfu_victim:
            victim, history = lru_victim, None
        elif self._rng.random() < self.w_lru:
            victim, history = lru_victim, self._hist_lru
        else:
            victim, history = lfu_victim, self._hist_lfu
        if history is not None:
            history[victim] = (t, self._freq[victim])
            if len(history) > self.frames:
                history.popitem(last=False)
        del self._lru[victim]
        del self._freq[victim]
        del self._last[victim]

    def resident_pages(self) -> set[int]:
        return set(self._freq)

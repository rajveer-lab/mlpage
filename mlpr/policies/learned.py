"""The learned replacement policy: a tiny model trained offline to imitate OPT."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from ..features import MetadataTable
from ..models import load_model
from .base import Policy


class SlotSampler:
    """Draws k frame slots uniformly at random, refilling from a pre-drawn batch.

    Slots are drawn with replacement, which keeps each draw O(k); a repeated
    slot just means fewer distinct candidates that time. With k >= frames
    every slot is a candidate.
    """

    def __init__(self, frames: int, k: int, seed: int = 0, batch: int = 2048):
        self.frames = frames
        self.k = k
        self.batch = batch
        self.every_slot = list(range(frames)) if k >= frames else None
        self._rng = np.random.default_rng(seed)
        self._rows: list[list[int]] = []
        self._pos = 0

    def draw(self) -> list[int]:
        if self.every_slot is not None:
            return self.every_slot
        if self._pos >= len(self._rows):
            self._rows = self._rng.integers(0, self.frames, size=(self.batch, self.k)).tolist()
            self._pos = 0
        row = self._rows[self._pos]
        self._pos += 1
        return row


def choose_victim(table: MetadataTable, t: int, slots: Sequence[int], model, threshold: float) -> tuple[int, bool]:
    """Pick the slot to evict among ``slots``; also return whether LRU fallback was used."""
    scores = model.predict_proba(table.features(t, slots))
    best = int(np.argmax(scores))
    if scores[best] >= threshold:
        return slots[best], False
    return min(slots, key=table.last), True


class RegretGuard:
    """Switches the learned policy to plain LRU while it is losing to LRU.

    A shadow LRU directory (page numbers only, no data) counts the faults LRU
    would have taken on the same references. At the end of every window the
    learned policy's faults are compared with the shadow's. If they exceed
    them by more than ``margin``, eviction switches to exact LRU for
    ``backoff`` windows, then the model is tried again; ``backoff`` doubles
    after every losing window and halves after every winning one. This puts
    the robustness lesson of Lykouris and Vassilvitskii [20] into practice:
    a predictor that is usually right can still be trusted too much.
    """

    def __init__(self, frames: int, window: int | None = None, margin: float = 0.02, max_backoff: int = 64):
        self.frames = frames
        self.window = window if window is not None else max(1000, 8 * frames)
        self.margin = margin
        self.max_backoff = max_backoff
        self.shadow: OrderedDict[int, None] = OrderedDict()
        self.use_lru = False
        self.backoff = 1
        self._lru_windows_left = 0
        self._seen = self._faults = self._shadow_faults = 0
        self.windows = 0
        self.lru_windows = 0

    def observe(self, page: int, faulted: bool) -> None:
        shadow = self.shadow
        if page in shadow:
            shadow.move_to_end(page)
        else:
            self._shadow_faults += 1
            shadow[page] = None
            if len(shadow) > self.frames:
                shadow.popitem(last=False)
        self._faults += faulted
        self._seen += 1
        if self._seen >= self.window:
            self._end_window()

    def _end_window(self) -> None:
        self.windows += 1
        if self.use_lru:
            self.lru_windows += 1
            self._lru_windows_left -= 1
            if self._lru_windows_left <= 0:
                self.use_lru = False
        elif self._faults > self._shadow_faults * (1.0 + self.margin):
            self.use_lru = True
            self._lru_windows_left = self.backoff
            self.backoff = min(2 * self.backoff, self.max_backoff)
        else:
            self.backoff = max(1, self.backoff // 2)
        self._seen = self._faults = self._shadow_faults = 0


class Learned(Policy):
    """Sampled eviction scored by a small model, with two LRU safety nets.

    On every fault with memory full, k resident pages are sampled and the
    model estimates for each the probability that it will not be needed
    soon. The page with the highest score is evicted, so the cost per fault
    is constant in the number of frames.

    Safety nets: if even the best score is below ``threshold`` the model is
    treated as unsure and the least recently used sampled page is evicted
    instead; and with ``guard`` on, a ``RegretGuard`` hands eviction to exact
    LRU while the model is doing worse than LRU would.
    """

    name = "Learned"

    def __init__(
        self,
        frames: int,
        model=None,
        k: int | None = None,
        threshold: float | None = None,
        ghost: bool | None = None,
        guard: bool | None = None,
        seed: int = 0,
        **_,
    ):
        super().__init__(frames)
        self.model = model if model is not None else load_model()
        meta = self.model.meta
        self.k = int(k if k is not None else meta.get("k", 16))
        self.threshold = float(threshold if threshold is not None else meta.get("threshold", 0.5))
        use_ghost = bool(ghost if ghost is not None else meta.get("ghost", True))
        use_guard = bool(guard if guard is not None else meta.get("guard", True))
        self.table = MetadataTable(frames, ghost_size=frames if use_ghost else 0)
        self.sampler = SlotSampler(frames, self.k, seed)
        self.guard = RegretGuard(frames) if use_guard else None
        # Exact recency order of resident pages, only needed when the guard hands over to LRU.
        self._recency: OrderedDict[int, None] | None = OrderedDict() if use_guard else None
        self.decisions = 0
        self.fallbacks = 0
        self.guarded = 0

    def access(self, page: int, t: int) -> bool:
        table = self.table
        recency = self._recency
        hit = page in table.slot_of
        if hit:
            table.touch(page, t)
            if recency is not None:
                recency.move_to_end(page)
        else:
            slot = table.free_slot()
            if slot is None:
                slot = self._choose_slot(t)
                victim = table.evict(slot)
                if recency is not None:
                    del recency[victim]
            table.load(page, t, slot)
            if recency is not None:
                recency[page] = None
        if self.guard is not None:
            self.guard.observe(page, not hit)
        return hit

    def _choose_slot(self, t: int) -> int:
        self.decisions += 1
        if self.guard is not None and self.guard.use_lru:
            self.guarded += 1
            return self.table.slot_of[next(iter(self._recency))]
        slot, fell_back = choose_victim(self.table, t, self.sampler.draw(), self.model, self.threshold)
        self.fallbacks += fell_back
        return slot

    @property
    def fallback_rate(self) -> float:
        """Share of evictions where the model was unsure and LRU among the sample decided."""
        return self.fallbacks / self.decisions if self.decisions else 0.0

    @property
    def guarded_rate(self) -> float:
        """Share of evictions made by exact LRU because the guard had taken over."""
        return self.guarded / self.decisions if self.decisions else 0.0

    def resident_pages(self) -> set[int]:
        return set(self.table.slot_of)

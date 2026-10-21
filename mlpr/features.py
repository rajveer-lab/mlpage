"""Per-page metadata and the feature vector the learned policy scores.

Every feature is known at fault time without any future knowledge:

    age                 time since the page was last referenced
    count               number of references seen so far
    gap1, gap2, gap3    the last three gaps between consecutive references
    mean_gap            average gap between references
    age_over_mean_gap   how "overdue" the page is relative to its usual gap

Time quantities are divided by the number of frames so a single model works
across memory sizes. Every feature is capped at ``CAP`` and log-scaled.

A gap that has not been observed yet is censored: the page has gone at least
``age`` references without reuse. So the first missing gap (and the mean gap
of a page seen once) is filled in with the age, and older missing gaps with
the newest known one. A page loaded a moment ago then looks hot, and a page
that has sat unused for a long time looks cold, as it should.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

FEATURE_NAMES = ("age", "count", "gap1", "gap2", "gap3", "mean_gap", "age_over_mean_gap")
NUM_FEATURES = len(FEATURE_NAMES)

# Feature groups used by the ablation study.
FEATURE_GROUPS = {
    "recency": ("age",),
    "frequency": ("count",),
    "gap history": ("gap1", "gap2", "gap3", "mean_gap", "age_over_mean_gap"),
}

# Raw metadata kept per page, from which the features are computed.
RAW_FIELDS = ("last", "count", "gap1", "gap2", "gap3", "gap_sum")
LAST, COUNT, GAP1, GAP2, GAP3, GAP_SUM = range(len(RAW_FIELDS))

UNKNOWN = -1
CAP = 1024.0

# Size of the metadata if laid out as a C struct of six uint32 fields, plus
# the page number for a ghost entry of a recently evicted page.
METADATA_BYTES_PER_PAGE = 4 * len(RAW_FIELDS)
GHOST_BYTES_PER_ENTRY = 8 + METADATA_BYTES_PER_PAGE


def compute_features(t: int, raw: np.ndarray, frames: int) -> np.ndarray:
    """Feature matrix with one row per page.

    ``raw`` has one row per page with the columns of ``RAW_FIELDS``.
    """
    count = raw[:, COUNT]
    gaps_seen = count - 1
    x = np.empty((len(raw), NUM_FEATURES))
    age = x[:, 0]
    age[:] = t - raw[:, LAST]
    x[:, 1] = count
    x[:, 2] = np.where(raw[:, GAP1] >= 0, raw[:, GAP1], age)
    x[:, 3] = np.where(raw[:, GAP2] >= 0, raw[:, GAP2], x[:, 2])
    x[:, 4] = np.where(raw[:, GAP3] >= 0, raw[:, GAP3], x[:, 3])
    x[:, 5] = np.where(gaps_seen > 0, raw[:, GAP_SUM] / np.maximum(gaps_seen, 1), age)
    x[:, 6] = age / np.maximum(x[:, 5], 1.0)
    x[:, 0] /= frames
    x[:, 2:6] /= frames
    np.minimum(x, CAP, out=x)
    x += 1.0
    return np.log2(x, out=x)


def new_record(t: int) -> list[int]:
    """Metadata of a page referenced for the first time at ``t``."""
    return [t, 1, UNKNOWN, UNKNOWN, UNKNOWN, 0]


def update_record(record: list[int], t: int) -> None:
    """Account for a new reference at ``t``."""
    gap = t - record[LAST]
    record[GAP3] = record[GAP2]
    record[GAP2] = record[GAP1]
    record[GAP1] = gap
    record[GAP_SUM] += gap
    record[COUNT] += 1
    record[LAST] = t


class MetadataTable:
    """Metadata for resident pages, stored per frame slot.

    Slots fill in order while memory has free frames; afterwards a newly
    loaded page always takes the slot of the page it replaced. Metadata of
    recently evicted pages is kept in a bounded ghost table (like the shadow
    entries Linux keeps to measure refault distance), so a page that comes
    back keeps its reference count and gap history.
    """

    def __init__(self, frames: int, ghost_size: int | None = None):
        self.frames = frames
        self.ghost_size = frames if ghost_size is None else ghost_size
        self.pages: list[int | None] = [None] * frames
        self.slot_of: dict[int, int] = {}
        self.records: list[list[int]] = [new_record(0) for _ in range(frames)]
        self.ghost: OrderedDict[int, list[int]] = OrderedDict()

    def free_slot(self) -> int | None:
        used = len(self.slot_of)
        return used if used < self.frames else None

    def last(self, slot: int) -> int:
        return self.records[slot][LAST]

    def touch(self, page: int, t: int) -> None:
        """Record a hit on a resident page."""
        update_record(self.records[self.slot_of[page]], t)

    def load(self, page: int, t: int, slot: int) -> None:
        """Place a faulting page in ``slot``, restoring its history if it is a ghost."""
        record = self.ghost.pop(page, None)
        if record is None:
            record = new_record(t)
        else:
            update_record(record, t)
        self.pages[slot] = page
        self.slot_of[page] = slot
        self.records[slot] = record

    def evict(self, slot: int) -> int:
        """Remove the page in ``slot`` and remember its metadata in the ghost table."""
        page = self.pages[slot]
        del self.slot_of[page]
        if self.ghost_size > 0:
            self.ghost[page] = self.records[slot]
            if len(self.ghost) > self.ghost_size:
                self.ghost.popitem(last=False)
        return page

    def features(self, t: int, slots: Sequence[int]) -> np.ndarray:
        """Feature matrix for the pages in ``slots``."""
        records = self.records
        raw = np.array([records[s] for s in slots], dtype=np.float64)
        return compute_features(t, raw, self.frames)

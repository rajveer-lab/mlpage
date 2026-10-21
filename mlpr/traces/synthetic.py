"""Synthetic page reference patterns, each stressing one known weakness of LRU."""

from __future__ import annotations

import numpy as np

# The classic reference string from OS textbooks. With 3 frames:
# FIFO gives 15 faults, LRU 12 and OPT 9.
TEXTBOOK = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    p = np.arange(1, n + 1, dtype=np.float64) ** -alpha
    return p / p.sum()


def loop_trace(n_pages: int = 2000, length: int = 100_000, noise: float = 0.0, seed: int = 0) -> list[int]:
    """A loop over more pages than fit in memory: LRU's worst case.

    LRU always evicts exactly the page that the loop needs next. ``noise`` is
    the fraction of references replaced by a random page from the loop.
    """
    rng = np.random.default_rng(seed)
    trace = np.arange(length) % n_pages
    if noise > 0:
        hit = rng.random(length) < noise
        trace[hit] = rng.integers(0, n_pages, int(hit.sum()))
    return trace.tolist()


def zipf_trace(n_pages: int = 5000, length: int = 100_000, alpha: float = 1.0, seed: int = 0) -> list[int]:
    """Independent references with skewed (Zipf) popularity: frequency matters more than recency."""
    rng = np.random.default_rng(seed)
    pages = rng.permutation(n_pages)
    return pages[rng.choice(n_pages, size=length, p=_zipf_probs(n_pages, alpha))].tolist()


def phase_trace(
    n_pages: int = 4000,
    length: int = 100_000,
    n_phases: int = 6,
    working_set: int = 500,
    alpha: float = 0.7,
    seed: int = 0,
) -> list[int]:
    """The working set moves: each phase draws (Zipf) from its own random subset of pages."""
    rng = np.random.default_rng(seed)
    probs = _zipf_probs(working_set, alpha)
    out = []
    per_phase = length // n_phases
    for phase in range(n_phases):
        n = per_phase if phase < n_phases - 1 else length - per_phase * (n_phases - 1)
        ws = rng.choice(n_pages, size=working_set, replace=False)
        out.append(ws[rng.choice(working_set, size=n, p=probs)])
    return np.concatenate(out).tolist()


def scan_hot_trace(
    hot_pages: int = 500,
    scan_pages: int = 6000,
    length: int = 100_000,
    hot_burst: int = 2000,
    scan_len: int = 1000,
    alpha: float = 0.8,
    seed: int = 0,
) -> list[int]:
    """A hot set with long sequential scans mixed in; each scan flushes the hot set under LRU.

    Pages ``0..hot_pages-1`` are the hot set; scans read ``scan_len``
    consecutive pages of a large region (like a full table scan) starting at
    a random offset.
    """
    rng = np.random.default_rng(seed)
    probs = _zipf_probs(hot_pages, alpha)
    out: list[np.ndarray] = []
    size = 0
    while size < length:
        hot = rng.choice(hot_pages, size=hot_burst, p=probs)
        start = int(rng.integers(0, scan_pages))
        scan = hot_pages + (start + np.arange(scan_len)) % scan_pages
        out += [hot, scan]
        size += hot_burst + scan_len
    return np.concatenate(out)[:length].tolist()


def mixed_trace(
    loop_pages: int = 1200,
    zipf_pages: int = 3000,
    length: int = 100_000,
    alpha: float = 0.9,
    loop_share: float = 0.5,
    seed: int = 0,
) -> list[int]:
    """Two programs sharing memory: one loops over an array, the other has Zipf popularity."""
    rng = np.random.default_rng(seed)
    from_loop = rng.random(length) < loop_share
    n_loop = int(from_loop.sum())
    trace = np.empty(length, dtype=np.int64)
    trace[from_loop] = np.arange(n_loop) % loop_pages
    zipf_ids = rng.permutation(zipf_pages) + loop_pages
    trace[~from_loop] = zipf_ids[rng.choice(zipf_pages, size=length - n_loop, p=_zipf_probs(zipf_pages, alpha))]
    return trace.tolist()

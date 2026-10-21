"""The workload set, split into training, validation and unseen test traces.

Every family appears in all three splits with different seeds and slightly
different parameters, so test results measure generalisation rather than
memorisation. Validation traces are used to pick the model and tune its
settings; test traces are only used for the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .traces import (
    TEXTBOOK,
    bfs_trace,
    load_msr,
    loop_trace,
    matmul_trace,
    mergesort_trace,
    mixed_trace,
    phase_trace,
    scan_hot_trace,
    zipf_trace,
)

MEMORY_FRACTIONS = (0.01, 0.02, 0.05, 0.10)
MIN_FRAMES = 8
MSR_DIR = Path(__file__).resolve().parent.parent / "data" / "msr"
MSR_MAX_ACCESSES = 200_000


@dataclass(frozen=True)
class Workload:
    name: str
    kind: str  # "synthetic", "program" or "real"
    tests: str
    build: Callable[[], list[int]]


def _params(split: str) -> dict:
    if split == "train":
        return dict(seed=100, length=60_000, loop=(1500, 0.0), zipf=(4000, 0.9), phase=(3000, 5, 400),
                    scan=(400, 5000, 900), mixed=(1000, 2500), matmul=32, sort=4096, bfs=4000)
    if split == "validation":
        return dict(seed=200, length=60_000, loop=(1800, 0.01), zipf=(4500, 0.95), phase=(3500, 5, 450),
                    scan=(450, 5500, 950), mixed=(1100, 2800), matmul=36, sort=6000, bfs=5000)
    if split == "test":
        return dict(seed=300, length=100_000, loop=(2000, 0.02), zipf=(5000, 1.0), phase=(4000, 6, 500),
                    scan=(500, 6000, 1000), mixed=(1200, 3000), matmul=40, sort=8192, bfs=6000)
    raise ValueError(f"unknown split {split!r}")


def workloads(split: str) -> list[Workload]:
    p = _params(split)
    s, n = p["seed"], p["length"]
    return [
        Workload("loop", "synthetic", "loop larger than memory (LRU's worst case)",
                 lambda: loop_trace(p["loop"][0], n, noise=p["loop"][1], seed=s)),
        Workload("zipf", "synthetic", "skewed popularity (frequency beats recency)",
                 lambda: zipf_trace(p["zipf"][0], n, alpha=p["zipf"][1], seed=s + 1)),
        Workload("phase", "synthetic", "working set shifts over time",
                 lambda: phase_trace(p["phase"][0], n, n_phases=p["phase"][1], working_set=p["phase"][2], seed=s + 2)),
        Workload("scan_hot", "synthetic", "one-time scans flushing a hot set",
                 lambda: scan_hot_trace(p["scan"][0], p["scan"][1], n, scan_len=p["scan"][2], seed=s + 3)),
        Workload("mixed", "synthetic", "a loop and a Zipf program sharing memory",
                 lambda: mixed_trace(p["mixed"][0], p["mixed"][1], n, seed=s + 4)),
        Workload("matmul", "program", "naive matrix multiply",
                 lambda: matmul_trace(p["matmul"])),
        Workload("mergesort", "program", "bottom-up merge sort",
                 lambda: mergesort_trace(p["sort"], seed=s + 5)),
        Workload("bfs", "program", "breadth-first search on a graph",
                 lambda: bfs_trace(p["bfs"], seed=s + 6)),
    ]


def msr_workloads(directory: Path = MSR_DIR, max_accesses: int = MSR_MAX_ACCESSES) -> list[Workload]:
    """One workload per MSR Cambridge CSV found in ``data/msr`` (none are bundled)."""
    files = sorted(directory.glob("*.csv")) + sorted(directory.glob("*.csv.gz"))
    return [
        Workload(f"msr_{f.name.split('.')[0]}", "real", "MSR Cambridge enterprise block I/O",
                 lambda f=f: load_msr(f, max_accesses=max_accesses))
        for f in files
    ]


@lru_cache(maxsize=None)
def _build_split(split: str, with_msr: bool) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    chosen = workloads(split) + (msr_workloads() if with_msr else [])
    return tuple((w.name, w.kind, tuple(w.build())) for w in chosen)


def load_split(split: str, with_msr: bool | None = None) -> dict[str, list[int]]:
    """Traces of a split by name. Real MSR traces join the test split when present."""
    if with_msr is None:
        with_msr = split == "test"
    return {name: list(trace) for name, _, trace in _build_split(split, with_msr)}


def workload_kinds(split: str = "test") -> dict[str, str]:
    return {name: kind for name, kind, _ in _build_split(split, split == "test")}


def frames_for(trace: list[int], fraction: float, minimum: int = MIN_FRAMES) -> int:
    """Memory size as a fraction of the trace's distinct pages, with a floor."""
    return max(minimum, int(round(fraction * len(set(trace)))))


def memory_sizes(trace: list[int], fractions=MEMORY_FRACTIONS) -> list[tuple[float, int]]:
    """(fraction, frames) pairs, dropping fractions that collapse onto the same frame count."""
    seen, sizes = set(), []
    for fraction in fractions:
        frames = frames_for(trace, fraction)
        if frames not in seen:
            seen.add(frames)
            sizes.append((fraction, frames))
    return sizes


__all__ = [
    "MEMORY_FRACTIONS", "TEXTBOOK", "Workload", "frames_for", "load_split",
    "memory_sizes", "msr_workloads", "workload_kinds", "workloads",
]

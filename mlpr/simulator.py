"""Trace-driven simulation: replay a page reference string through a policy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .policies import Policy


@dataclass
class RunResult:
    policy: str
    frames: int
    accesses: int
    faults: int
    fault_ns: int = 0  # wall time spent inside faulting accesses
    hit_ns: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def hits(self) -> int:
        return self.accesses - self.faults

    @property
    def fault_rate(self) -> float:
        return self.faults / self.accesses if self.accesses else 0.0

    @property
    def hit_ratio(self) -> float:
        return 1.0 - self.fault_rate

    @property
    def us_per_fault(self) -> float:
        return self.fault_ns / self.faults / 1000.0 if self.faults else 0.0


def simulate(trace: Sequence[int], policy: Policy, timed: bool = True) -> RunResult:
    """Run ``policy`` over ``trace`` and count faults.

    With ``timed`` set, each access is timed and the time is attributed to
    hits or faults, which gives the decision cost per fault.
    """
    access = policy.access
    faults = 0
    fault_ns = hit_ns = 0
    if timed:
        clock = time.perf_counter_ns
        for t, page in enumerate(trace):
            start = clock()
            hit = access(page, t)
            elapsed = clock() - start
            if hit:
                hit_ns += elapsed
            else:
                faults += 1
                fault_ns += elapsed
    else:
        for t, page in enumerate(trace):
            if not access(page, t):
                faults += 1
    result = RunResult(policy.name, policy.frames, len(trace), faults, fault_ns, hit_ns)
    for stat in ("fallback_rate", "guarded_rate"):
        if hasattr(policy, stat):
            result.extra[stat] = getattr(policy, stat)
    return result


def trace_steps(trace: Sequence[int], policy: Policy) -> list[tuple[int, bool, list[int]]]:
    """Per-reference (page, hit, resident pages) for printing small examples."""
    steps = []
    for t, page in enumerate(trace):
        hit = policy.access(page, t)
        steps.append((page, hit, sorted(policy.resident_pages())))
    return steps

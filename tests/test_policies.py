import random
from collections import OrderedDict

import numpy as np
import pytest

from mlpr.models import LogisticModel
from mlpr.policies import ALL_POLICIES, LRU, OPT, Learned, make_policy, next_use
from mlpr.simulator import simulate
from mlpr.traces import TEXTBOOK, loop_trace, zipf_trace

# A model that scores every page alike: always above any threshold < 0.5.
NEUTRAL = LogisticModel(weights=np.zeros(7), bias=0.0)


def build(name, frames, trace, **kwargs):
    if name == "Learned":
        kwargs.setdefault("model", NEUTRAL)
    return make_policy(name, frames, trace=trace, **kwargs)


def random_trace(seed, n=3000, pages=60):
    rng = random.Random(seed)
    return [rng.randrange(pages) if rng.random() < 0.7 else rng.randrange(pages // 6) for _ in range(n)]


@pytest.mark.parametrize("name,faults", [("FIFO", 15), ("LRU", 12), ("OPT", 9)])
def test_textbook_reference_string(name, faults):
    assert simulate(TEXTBOOK, make_policy(name, 3, trace=TEXTBOOK)).faults == faults


def test_next_use():
    assert next_use([1, 2, 1, 3, 2]) == [2, 4, 5, 5, 5]


@pytest.mark.parametrize("name", ALL_POLICIES)
def test_never_exceeds_frames_and_reports_hits_correctly(name):
    trace = random_trace(1, n=1500)
    frames = 7
    policy = build(name, frames, trace)
    for t, page in enumerate(trace):
        was_resident = page in policy.resident_pages()
        hit = policy.access(page, t)
        assert hit == was_resident
        resident = policy.resident_pages()
        assert page in resident
        assert len(resident) <= frames


@pytest.mark.parametrize("seed", range(5))
def test_opt_is_a_lower_bound(seed):
    trace = random_trace(seed)
    for frames in (4, 10, 25):
        opt = simulate(trace, OPT(frames, trace=trace)).faults
        assert opt >= len(set(trace))  # compulsory misses
        for name in ALL_POLICIES:
            assert simulate(trace, build(name, frames, trace)).faults >= opt, name


def test_lru_matches_a_naive_implementation():
    trace = random_trace(7)
    frames = 9
    stack: OrderedDict[int, None] = OrderedDict()
    expected = 0
    for page in trace:
        if page in stack:
            stack.move_to_end(page)
        else:
            expected += 1
            stack[page] = None
            if len(stack) > frames:
                stack.popitem(last=False)
    assert simulate(trace, LRU(frames)).faults == expected


def test_learned_with_full_sample_and_forced_fallback_is_exactly_lru():
    trace = random_trace(3)
    frames = 12
    learned = Learned(frames, model=NEUTRAL, k=frames, threshold=1.1, guard=False)
    assert simulate(trace, learned).faults == simulate(trace, LRU(frames)).faults
    assert learned.fallback_rate == 1.0


def test_guard_limits_damage_from_a_bad_model():
    # Weights that prefer evicting the most recently used page: terrible on Zipf.
    bad = LogisticModel(weights=[-5.0, 0, 0, 0, 0, 0, 0], bias=5.0)
    trace = zipf_trace(800, 60_000, alpha=1.0, seed=1)
    frames = 40
    lru = simulate(trace, LRU(frames)).faults
    unguarded = simulate(trace, Learned(frames, model=bad, k=16, threshold=0.0, guard=False)).faults
    guarded_policy = Learned(frames, model=bad, k=16, threshold=0.0, guard=True)
    guarded = simulate(trace, guarded_policy).faults
    assert unguarded > 1.2 * lru
    assert guarded < 1.1 * lru
    assert guarded_policy.guarded_rate > 0.5


def test_arc_resists_a_scan():
    hot = [i % 20 for i in range(2000)]
    scan = list(range(1000, 1400))
    trace = hot + scan + hot
    frames = 30
    assert simulate(trace, make_policy("ARC", frames)).faults < simulate(trace, LRU(frames)).faults


def test_loop_is_lru_worst_case():
    trace = loop_trace(50, 1000)
    assert simulate(trace, LRU(40)).faults == 1000
    assert simulate(trace, OPT(40, trace=trace)).faults < 400

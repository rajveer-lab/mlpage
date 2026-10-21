"""Offline training: turn OPT's future knowledge into labelled examples and fit a tiny model.

A training trace is replayed with a "driver" policy (OPT by default). At an
eviction at time t, k resident pages are sampled exactly as the online
policy would sample them, and for each one we record

    x = features(page, t)                     (known at fault time)
    y = 1 if next_use(page) - t > H else 0    (1 = not needed soon; needs the future)

where the horizon H is a multiple of the number of frames. The labels only
depend on the trace, so the same code can also collect examples while the
learned policy itself is driving (a DAgger round, as in Parrot), which shows
the model the states its own decisions lead to.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np

from .features import FEATURE_NAMES, NUM_FEATURES, MetadataTable
from .models import LogisticModel, TreeModel
from .policies import LRU, Learned, next_use
from .policies.learned import SlotSampler, choose_victim
from .simulator import simulate
from .workloads import MEMORY_FRACTIONS, memory_sizes


@dataclass
class TrainConfig:
    horizon: float = 2.0  # label horizon H, in multiples of the number of frames
    k: int = 16  # pages sampled per eviction
    threshold: float = 0.5  # below this best score, fall back to LRU
    ghost: bool = True  # keep metadata of recently evicted pages
    guard: bool = True  # hand over to LRU while the model loses to LRU
    fractions: Sequence[float] = MEMORY_FRACTIONS
    evictions_per_run: int = 4000  # recorded evictions per (trace, memory size), roughly
    dagger_rounds: int = 2  # extra rounds collecting examples while the model drives
    tree_depth: int = 4
    seed: int = 0
    features: list[int] = field(default_factory=lambda: list(range(NUM_FEATURES)))


def collect_samples(
    trace: Sequence[int],
    frames: int,
    cfg: TrainConfig,
    driver: str = "opt",
    model=None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay ``trace`` and record (features, label) for sampled pages at evictions.

    ``driver`` decides which page is actually evicted: "opt" (furthest next
    use), "lru", or "model" (the learned policy with ``model``).
    """
    nxt = next_use(trace)
    horizon = cfg.horizon * frames
    table = MetadataTable(frames, ghost_size=frames if cfg.ghost else 0)
    sampler = SlotSampler(frames, cfg.k, seed)
    coin = random.Random(seed)
    record_prob = min(1.0, cfg.evictions_per_run / max(1, len(trace)))
    slot_next = [0] * frames  # next use of the page held in each slot
    xs, ys = [], []

    for t, page in enumerate(trace):
        slot = table.slot_of.get(page)
        if slot is not None:
            table.touch(page, t)
            slot_next[slot] = nxt[t]
            continue
        slot = table.free_slot()
        if slot is None:
            slots = sampler.draw()
            if coin.random() < record_prob:
                xs.append(table.features(t, slots))
                ys.append([slot_next[s] - t > horizon for s in slots])
            if driver == "opt":
                slot = slot_next.index(max(slot_next))
            elif driver == "lru":
                slot = min(range(frames), key=table.last)
            elif driver == "model":
                slot, _ = choose_victim(table, t, slots, model, cfg.threshold)
            else:
                raise ValueError(f"unknown driver {driver!r}")
            table.evict(slot)
        table.load(page, t, slot)
        slot_next[slot] = nxt[t]

    if not xs:
        return np.empty((0, NUM_FEATURES)), np.empty(0)
    return np.vstack(xs), np.concatenate(ys).astype(np.float64)


def build_dataset(
    traces: dict[str, list[int]],
    cfg: TrainConfig,
    driver: str = "opt",
    model=None,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i, trace in enumerate(traces.values()):
        for j, (_, frames) in enumerate(memory_sizes(trace, cfg.fractions)):
            X, y = collect_samples(trace, frames, cfg, driver, model, seed=cfg.seed + 97 * i + j)
            xs.append(X)
            ys.append(y)
    return np.vstack(xs), np.concatenate(ys)


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, cfg: TrainConfig):
    if kind == "logistic":
        return LogisticModel(features=cfg.features).fit(X, y)
    if kind == "tree":
        return TreeModel(features=cfg.features).fit(X, y, max_depth=cfg.tree_depth)
    raise ValueError(f"unknown model kind {kind!r}")


def relative_faults(
    model,
    traces: dict[str, list[int]],
    cfg: TrainConfig,
    lru_faults: dict | None = None,
    **policy_kwargs,
) -> float:
    """Mean of (learned faults / LRU faults) over traces and memory sizes; lower is better."""
    ratios = []
    for name, trace in traces.items():
        for _, frames in memory_sizes(trace, cfg.fractions):
            key = (name, frames)
            if lru_faults is not None and key in lru_faults:
                base = lru_faults[key]
            else:
                base = simulate(trace, LRU(frames), timed=False).faults
                if lru_faults is not None:
                    lru_faults[key] = base
            kwargs = dict(k=cfg.k, threshold=cfg.threshold, ghost=cfg.ghost, guard=cfg.guard)
            kwargs.update(policy_kwargs)
            learned = simulate(trace, Learned(frames, model=model, **kwargs), timed=False).faults
            ratios.append(learned / base if base else 1.0)
    return float(np.mean(ratios))


def train(
    train_traces: dict[str, list[int]],
    val_traces: dict[str, list[int]],
    cfg: TrainConfig,
    kinds: Sequence[str] = ("logistic", "tree"),
    log: Callable[[str], None] = print,
):
    """Collect OPT-labelled data, fit each model kind, keep the one with the fewest validation faults.

    Selection uses simulated faults (relative to LRU) on the validation
    traces, not classification accuracy.
    """
    start = time.perf_counter()
    X, y = build_dataset(train_traces, cfg, driver="opt")
    collect_seconds = time.perf_counter() - start
    log(f"collected {len(y):,} examples from OPT replays ({y.mean():.1%} labelled 'not needed soon') "
        f"in {collect_seconds:.1f}s")

    lru_cache: dict = {}
    best = None
    for kind in kinds:
        fit_start = time.perf_counter()
        model = fit_model(kind, X, y, cfg)
        data_X, data_y = X, y
        for round_ in range(cfg.dagger_rounds):
            more_X, more_y = build_dataset(train_traces, cfg, driver="model", model=model)
            data_X, data_y = np.vstack((data_X, more_X)), np.concatenate((data_y, more_y))
            model = fit_model(kind, data_X, data_y, cfg)
            log(f"  {kind}: DAgger round {round_ + 1} added {len(more_y):,} examples")
        fit_seconds = time.perf_counter() - fit_start
        accuracy = float(np.mean((model.predict_proba(data_X) >= 0.5) == data_y))
        score = relative_faults(model, val_traces, cfg, lru_cache)
        log(f"  {kind:8s} params={model.n_params:3d} train acc={accuracy:.3f} "
            f"validation faults vs LRU={score:.4f} fit={fit_seconds:.2f}s")
        model.meta.update(
            {
                "k": cfg.k,
                "threshold": cfg.threshold,
                "ghost": cfg.ghost,
                "guard": cfg.guard,
                "horizon": cfg.horizon,
                "feature_names": [FEATURE_NAMES[i] for i in cfg.features],
                "train_examples": int(len(data_y)),
                "train_accuracy": round(accuracy, 4),
                "train_seconds": round(collect_seconds + fit_seconds, 2),
                "validation_faults_vs_lru": round(score, 4),
                "config": {k: v for k, v in asdict(cfg).items() if k != "features"},
            }
        )
        if best is None or score < best[0]:
            best = (score, model)
    return best[1]

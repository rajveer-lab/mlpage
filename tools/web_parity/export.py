"""Export the cases the JavaScript port (docs/engine.js) is checked against.

Run from the repository root:

    python3 tools/web_parity/export.py

It writes tools/web_parity/cases.json with Python's answers for:

  - fault counts of every policy on the textbook string (1 to 7 frames),
    on 12 random traces (1, 2, 3, 8 and 37 frames) and on the first 15000
    references of every test workload (2 memory sizes); the learned policy
    runs with k = frames (exact, guard on and off) and with its default
    k = 16 (sampled, so only comparable statistically);
  - compute_features and predict_proba (logistic and tree) on random raw
    metadata rows;
  - matmul_trace, next_use and CPython's random.Random sequences;
  - every random workload generator on small parameters, together with the
    NumPy draws it made, so the JavaScript generator can be fed the same
    draws and must then produce the same trace.

``node tools/web_parity/check.js`` recomputes all of it and fails on any
mismatch.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mlpr.features import FEATURE_NAMES, compute_features  # noqa: E402
from mlpr.models import DEFAULT_MODEL_PATH, TreeModel, load_model  # noqa: E402
from mlpr.policies import make_policy, next_use  # noqa: E402
from mlpr.simulator import simulate  # noqa: E402
from mlpr.traces import (  # noqa: E402
    TEXTBOOK,
    bfs_trace,
    loop_trace,
    matmul_trace,
    mergesort_trace,
    mixed_trace,
    phase_trace,
    scan_hot_trace,
    zipf_trace,
)
from mlpr.metrics import fault_reduction, gap_closed  # noqa: E402
from mlpr.workloads import frames_for, load_split, memory_sizes, workloads  # noqa: E402

OUT = Path(__file__).resolve().parent / "cases.json"

PREFIX = 15_000
WORKLOAD_FRACTIONS = (0.02, 0.10)
RANDOM_FRAMES = (1, 2, 3, 8, 37)
SPREAD_SEEDS = (1, 2, 3, 4)

# (id, policy, keyword arguments); k="frames" means k equal to the frame count.
CONFIGS = [
    ("FIFO", "FIFO", {}),
    ("LRU", "LRU", {}),
    ("LFU", "LFU", {}),
    ("CLOCK", "CLOCK", {}),
    ("ARC", "ARC", {}),
    ("LeCaR", "LeCaR", {"seed": 0}),
    ("OPT", "OPT", {}),
    ("Learned k=frames", "Learned", {"k": "frames"}),
    ("Learned k=frames no guard", "Learned", {"k": "frames", "guard": False}),
    ("Learned k=16", "Learned", {}),
]
APPROX = "Learned k=16"


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ------------------------------------------------------------------ policies


def build(policy: str, frames: int, trace: list[int], kwargs: dict, model):
    kwargs = dict(kwargs)
    if kwargs.get("k") == "frames":
        kwargs["k"] = frames
    if policy == "Learned":
        kwargs["model"] = model
    return make_policy(policy, frames, trace=trace, **kwargs)


def run_case(trace: list[int], frames: int, model) -> dict:
    run: dict = {"frames": frames, "faults": {}, "learned_stats": {}}
    for cid, policy_name, kwargs in CONFIGS:
        policy = build(policy_name, frames, trace, kwargs, model)
        run["faults"][cid] = simulate(trace, policy, timed=False).faults
        if policy_name == "Learned":
            run["learned_stats"][cid] = [policy.decisions, policy.fallbacks, policy.guarded]
        if policy_name == "LeCaR":
            run["lecar_w_lru"] = policy.w_lru
    # Python's own spread over sampler seeds, to put the JS difference for k=16 in context.
    run["learned16_other_seeds"] = [
        simulate(trace, build("Learned", frames, trace, {"seed": s}, model), timed=False).faults for s in SPREAD_SEEDS
    ]
    return run


def random_traces() -> list[tuple[str, list[int]]]:
    """12 random traces, 2000 to 6000 references over 5 to 300 pages: uniform, skewed, looping and phased."""
    out = []
    page_ranges = [5, 12, 300, 40, 9, 150, 60, 25, 220, 80, 17, 110]
    for i, pages in enumerate(page_ranges):
        rng = random.Random(1000 + i)
        n = rng.randint(2000, 6000)
        style = ("uniform", "skewed", "looping", "phased")[i % 4]
        if style == "uniform":
            trace = [rng.randrange(pages) for _ in range(n)]
        elif style == "skewed":
            hot = max(1, pages // 8)
            trace = [rng.randrange(hot) if rng.random() < 0.8 else rng.randrange(pages) for _ in range(n)]
        elif style == "looping":
            span = max(2, (3 * pages) // 4)
            trace = [(t % span) if rng.random() > 0.05 else rng.randrange(pages) for t in range(n)]
        else:
            trace = []
            while len(trace) < n:
                ws = rng.sample(range(pages), max(1, min(pages, rng.randint(3, 25))))
                weights = [1.0 / (r + 1) for r in range(len(ws))]
                trace += rng.choices(ws, weights, k=rng.randint(100, 700))
            trace = trace[:n]
        out.append((f"random{i + 1}-{style}-{pages}p", trace))
    return out


# ------------------------------------------------------------------ features


def feature_cases(model) -> tuple[list[dict], dict]:
    """200 random raw metadata rows in 20 groups of 10, each group with its own t and frames."""
    rng = random.Random(7)
    groups = []
    all_x = []
    for g in range(20):
        frames = rng.choice([1, 2, 3, 8, 37, 100, 1000, 5000])
        t = rng.choice([0, 1, 10, 1000, 123_457, 10_000_000])
        rows = []
        for _ in range(10):
            count = rng.choice([1, 1, 2, 3, 4, 5, rng.randint(6, 50), rng.randint(51, 5000)])
            age = rng.choice([0, 1, rng.randint(2, 100), rng.randint(100, 50_000), t])
            age = min(age, t)
            known = [rng.choice([1, rng.randint(1, 50), rng.randint(50, 2_000_000)]) for _ in range(3)]
            gaps = [known[i] if count >= i + 2 else -1 for i in range(3)]
            gap_sum = sum(x for x in gaps if x > 0)
            if count > 4:
                gap_sum += rng.randint(count - 4, 1000 * (count - 4))
            rows.append([t - age, count, gaps[0], gaps[1], gaps[2], gap_sum])
        x = compute_features(t, np.array(rows, dtype=np.float64), frames)
        all_x.append(x)
        groups.append(
            {
                "t": t,
                "frames": frames,
                "raw": rows,
                "x": x.tolist(),
                "p_logistic": model.predict_proba(x).tolist(),
            }
        )
    X = np.vstack(all_x)
    y = (X[:, 0] + 0.3 * X[:, 6] > np.median(X[:, 0] + 0.3 * X[:, 6])).astype(np.float64)
    tree = TreeModel().fit(X, y, max_depth=3, min_leaf=10)
    for group, x in zip(groups, all_x):
        group["p_tree"] = tree.predict_proba(x).tolist()
    return groups, tree.to_dict()


# ---------------------------------------------------------------- generators


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class RecordingGenerator:
    """Wraps numpy.random.Generator and logs every call the trace generators make."""

    def __init__(self, gen: np.random.Generator):
        self.gen = gen
        self.log: list = []

    def random(self, size=None):
        out = self.gen.random(size)
        self.log.append(["random", {"size": _size(size)}, _plain(out)])
        return out

    def integers(self, low, high=None, size=None):
        out = self.gen.integers(low, high, size)
        self.log.append(["integers", {"low": int(low), "high": int(high), "size": _size(size)}, _plain(out)])
        return out

    def permutation(self, x):
        out = self.gen.permutation(x)
        self.log.append(["permutation", {"n": int(x)}, _plain(out)])
        return out

    def choice(self, a, size=None, replace=True, p=None):
        out = self.gen.choice(a, size=size, replace=replace, p=p)
        args = {"n": int(a), "size": _size(size)}
        if p is not None:
            args["p"] = np.asarray(p).tolist()
        self.log.append(["choice" if replace else "choiceNoReplace", args, _plain(out)])
        return out


def _size(size):
    if size is None:
        return None
    return int(np.prod(size))


def generator_cases() -> list[dict]:
    """Small instances of every trace generator, with the NumPy draws they consumed."""
    specs = [
        ("loop", loop_trace, dict(n_pages=200, length=3000, noise=0.05, seed=1),
         dict(nPages=200, length=3000, noise=0.05)),
        ("zipf", zipf_trace, dict(n_pages=300, length=3000, alpha=1.0, seed=2),
         dict(nPages=300, length=3000, alpha=1.0)),
        ("phase", phase_trace, dict(n_pages=400, length=3001, n_phases=6, working_set=50, alpha=0.7, seed=3),
         dict(nPages=400, length=3001, nPhases=6, workingSet=50, alpha=0.7)),
        ("scanHot", scan_hot_trace,
         dict(hot_pages=50, scan_pages=600, length=2950, hot_burst=200, scan_len=100, alpha=0.8, seed=4),
         dict(hotPages=50, scanPages=600, length=2950, hotBurst=200, scanLen=100, alpha=0.8)),
        ("mixed", mixed_trace, dict(loop_pages=120, zipf_pages=300, length=3000, alpha=0.9, loop_share=0.5, seed=5),
         dict(loopPages=120, zipfPages=300, length=3000, alpha=0.9, loopShare=0.5)),
        ("mergesort", mergesort_trace, dict(n=500, page_bytes=64, seed=6), dict(n=500, pageBytes=64)),
        ("bfs", bfs_trace, dict(n_nodes=300, degree=5, runs=2, local_share=0.5, page_bytes=64, seed=7),
         dict(nNodes=300, degree=5, runs=2, localShare=0.5, pageBytes=64)),
    ]
    original = np.random.default_rng
    cases = []
    try:
        for name, fn, py_args, js_args in specs:
            recorders: list[RecordingGenerator] = []

            def recording(seed=None, _recorders=recorders):
                rec = RecordingGenerator(original(seed))
                _recorders.append(rec)
                return rec

            np.random.default_rng = recording
            trace = fn(**py_args)
            assert len(recorders) == 1, name
            cases.append({"name": name, "options": js_args, "draws": recorders[0].log, "trace": list(trace)})
    finally:
        np.random.default_rng = original
    return cases


# -------------------------------------------------------------------- export


def main() -> None:
    start = time.time()
    model = load_model()
    model_dict = json.loads(Path(DEFAULT_MODEL_PATH).read_text())
    cases = []

    cases.append(
        {"group": "textbook", "name": "textbook", "trace": TEXTBOOK,
         "runs": [run_case(TEXTBOOK, f, model) for f in range(1, 8)]}
    )
    for name, trace in random_traces():
        cases.append({"group": "random", "name": name, "trace": trace,
                      "runs": [run_case(trace, f, model) for f in RANDOM_FRAMES]})
        print(f"  {name:28s} {len(trace)} refs", flush=True)

    split = load_split("test", with_msr=False)
    for name, full in split.items():
        trace = full[:PREFIX]
        sizes = memory_sizes(trace, WORKLOAD_FRACTIONS)
        cases.append({"group": "workload", "name": f"{name}[:{PREFIX}]", "trace": trace,
                      "runs": [run_case(trace, f, model) for _, f in sizes]})
        print(f"  {name:28s} frames {[f for _, f in sizes]}", flush=True)

    features, tree = feature_cases(model)
    small = [1, 2, 1, 3, 2, 7, 7, 1, 9, 2, 3, 3, 0, 1]
    doc = {
        "about": "Python answers for docs/engine.js; written by tools/web_parity/export.py, checked by check.js",
        "source": git_hash(),
        "model": model_dict,
        "configs": [{"id": cid, "policy": p, "opts": kw, "approx": cid == APPROX} for cid, p, kw in CONFIGS],
        "feature_names": list(FEATURE_NAMES),
        "workloads": [
            {"name": w.name, "kind": w.kind, "tests": w.tests,
             "accesses": len(split[w.name]), "distinct_pages": len(set(split[w.name]))}
            for w in workloads("test")
        ],
        "next_use": {"trace": small, "next": next_use(small)},
        "frames_for": [
            {"distinct": d, "fraction": f, "frames": frames_for(list(range(d)), f)}
            for d in (30, 250, 990, 1000, 2010, 4871, 6500) for f in (0.01, 0.02, 0.05, 0.10, 0.25)
        ],
        "metrics": [
            {"f_lru": a, "f": b, "f_opt": c, "reduction": fault_reduction(a, b),
             "gap_closed": None if gap_closed(a, b, c) != gap_closed(a, b, c) else gap_closed(a, b, c)}
            for a, b, c in ((100, 80, 50), (65796, 54700, 45500), (10, 12, 10), (0, 0, 0), (7, 9, 3))
        ],
        "mt19937": [
            {"seed": seed, "random": [r.random() for r in [random.Random(seed)] for _ in range(6)]}
            for seed in (0, 1, 42, 123456789, 2**32 - 1, 2**32 + 5, 2**40 + 3, -7)
        ],
        "matmul": [
            {"n": n, "page_bytes": pb, "trace": matmul_trace(n, pb)} for n, pb in ((40, 64), (12, 64), (10, 128))
        ],
        "tree_model": tree,
        "features": features,
        "generators": generator_cases(),
        "cases": cases,
    }
    write(doc, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - start:.0f} s")


def write(doc: dict, path: Path) -> None:
    """Compact JSON with one list entry per line for the long lists, so diffs stay readable."""
    dumps = lambda v: json.dumps(v, separators=(",", ":"))  # noqa: E731
    parts = []
    for key, value in doc.items():
        if key in ("cases", "features", "generators", "matmul", "mt19937", "workloads", "configs", "metrics"):
            parts.append(f"{json.dumps(key)}:[\n" + ",\n".join(dumps(v) for v in value) + "\n]")
        else:
            parts.append(f"{json.dumps(key)}:{dumps(value)}")
    path.write_text("{\n" + ",\n".join(parts) + "\n}\n")


if __name__ == "__main__":
    main()

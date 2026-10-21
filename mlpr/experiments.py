"""Experiments: the main evaluation on unseen traces, the ablation study and the parameter sweeps.

Everything writes plain CSV plus a Markdown summary into ``results/``, and
the figures into ``results/plots/``.
"""

from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable

from .features import FEATURE_GROUPS, FEATURE_NAMES, GHOST_BYTES_PER_ENTRY, METADATA_BYTES_PER_PAGE
from .metrics import fault_reduction, gap_closed
from .models import load_model, model_size_bytes
from .policies import ALL_POLICIES, make_policy
from .simulator import simulate
from .training import TrainConfig, relative_faults, train
from .workloads import load_split, memory_sizes, workload_kinds

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Exact LRU order of resident pages and the guard's shadow LRU directory,
# each as a C list node: page number plus two links.
GUARD_BYTES_PER_FRAME = 2 * (8 + 16)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


# ---------------------------------------------------------------- evaluation


def run_grid(
    traces: dict[str, list[int]],
    policies: Iterable[str],
    model=None,
    fractions=None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run every policy on every trace at every memory size."""
    kinds = workload_kinds("test")
    rows = []
    for name, trace in traces.items():
        distinct = len(set(trace))
        sizes = memory_sizes(trace) if fractions is None else memory_sizes(trace, fractions)
        for fraction, frames in sizes:
            line = []
            for policy_name in policies:
                policy = make_policy(policy_name, frames, trace=trace, model=model)
                r = simulate(trace, policy)
                rows.append(
                    {
                        "workload": name,
                        "kind": kinds.get(name, "custom"),
                        "accesses": len(trace),
                        "distinct_pages": distinct,
                        "memory_fraction": fraction,
                        "frames": frames,
                        "policy": r.policy,
                        "faults": r.faults,
                        "fault_rate": round(r.fault_rate, 6),
                        "hit_ratio": round(r.hit_ratio, 6),
                        "us_per_fault": round(r.us_per_fault, 3),
                        "fallback_rate": round(r.extra.get("fallback_rate", 0.0), 4),
                        "guarded_rate": round(r.extra.get("guarded_rate", 0.0), 4),
                    }
                )
                line.append(f"{r.policy}={r.fault_rate:.3f}")
            log(f"  {name:10s} frames={frames:5d}  " + "  ".join(line))
    return rows


def compare(rows: list[dict], policy: str) -> list[dict]:
    """Per (workload, memory size): faults of ``policy`` against LRU and OPT."""
    by_key: dict[tuple, dict[str, int]] = {}
    for r in rows:
        by_key.setdefault((r["workload"], int(r["frames"]), float(r["memory_fraction"])), {})[r["policy"]] = int(r["faults"])
    out = []
    for (workload, frames, fraction), faults in by_key.items():
        if policy not in faults or "LRU" not in faults or "OPT" not in faults:
            continue
        f_lru, f_pol, f_opt = faults["LRU"], faults[policy], faults["OPT"]
        out.append(
            {
                "workload": workload,
                "frames": frames,
                "memory_fraction": fraction,
                "reduction": fault_reduction(f_lru, f_pol),
                "gap_closed": gap_closed(f_lru, f_pol, f_opt),
                "worse_than_lru": 100.0 * (f_pol - f_lru) / f_lru if f_lru else 0.0,
            }
        )
    return out


def pooled_gap_closed(rows: list[dict], policy: str, workload: str | None = None) -> float:
    """Gap closed on faults summed over memory sizes (and workloads, if ``workload`` is None)."""
    totals = {"LRU": 0, "OPT": 0, policy: 0}
    for r in rows:
        if r["policy"] in totals and (workload is None or r["workload"] == workload):
            totals[r["policy"]] += int(r["faults"])
    return gap_closed(totals["LRU"], totals[policy], totals["OPT"])


def write_summary(rows: list[dict], model, path: Path) -> str:
    policies = [p for p in ALL_POLICIES if any(r["policy"] == p for r in rows)]
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    lines = ["# Results on unseen test traces", ""]
    lines += [
        "Generated by `python -m mlpr evaluate`. Every policy runs on the same traces at the same memory sizes;",
        "memory is a fraction of each trace's distinct pages (at least 8 frames).",
        "",
        "## Fault rate (%)",
        "",
        "| Workload | Memory | Frames | " + " | ".join(policies) + " |",
        "|---|---|---|" + "---|" * len(policies),
    ]
    cells: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        cells.setdefault((r["workload"], int(r["frames"]), float(r["memory_fraction"])), {})[r["policy"]] = r
    for (workload, frames, fraction), by_policy in cells.items():
        rates = {p: float(by_policy[p]["fault_rate"]) for p in policies if p in by_policy}
        online = {p: v for p, v in rates.items() if p != "OPT"}
        best = min(online.values()) if online else None
        row = []
        for p in policies:
            if p not in rates:
                row.append("")
                continue
            text = f"{100 * rates[p]:.1f}"
            row.append(f"**{text}**" if p != "OPT" and rates[p] == best else text)
        lines.append(f"| {workload} | {100 * fraction:g}% | {frames} | " + " | ".join(row) + " |")
    lines += ["", "Bold: best online policy (OPT needs the future and is only the lower bound).", ""]

    lines += [
        "## Learned policy against LRU and OPT",
        "",
        "Fault reduction vs LRU = (F_LRU - F_policy) / F_LRU x 100, averaged over the memory sizes.",
        "Gap closed = (F_LRU - F_policy) / (F_LRU - F_OPT) x 100, on faults summed over the memory sizes",
        "(averaging the ratio itself would let a size with a near-zero gap dominate).",
        "",
        "| Workload | ARC reduction | LeCaR reduction | Learned reduction "
        "| ARC gap closed | LeCaR gap closed | Learned gap closed | Learned worst vs LRU |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rivals = ("ARC", "LeCaR", "Learned")
    per_size = {p: compare(rows, p) for p in rivals}
    for workload in workloads + [None]:
        def pick(p):
            return [c for c in per_size[p] if workload is None or c["workload"] == workload]

        if not pick("Learned"):
            continue
        cols = [_fmt(mean(c["reduction"] for c in pick(p))) for p in rivals]
        cols += [_fmt(pooled_gap_closed(rows, p, workload)) for p in rivals]
        cols.append(_fmt(max(c["worse_than_lru"] for c in pick("Learned"))))
        if workload is None:
            lines.append("| **all** | " + " | ".join(f"**{x}**" for x in cols) + " |")
        else:
            lines.append(f"| {workload} | " + " | ".join(cols) + " |")
    lines += ["", "All values in %. Worst vs LRU = most extra faults at any memory size; negative means the",
              "learned policy beat LRU at every memory size of that workload.", ""]

    lines += [
        "## Cost",
        "",
        "Decision time is measured in this Python simulator, so absolute numbers are far above what",
        "a C implementation would take; compare policies with each other, not with a kernel.",
        "",
        "| Policy | Median decision time per fault (us) |",
        "|---|---|",
    ]
    for p in policies:
        times = [float(r["us_per_fault"]) for r in rows if r["policy"] == p and int(r["faults"]) > 0]
        lines.append(f"| {p} | {median(times):.2f} |")
    learned_rows = [r for r in rows if r["policy"] == "Learned"]
    if model is not None and learned_rows:
        meta = model.meta
        fallback = mean(float(r["fallback_rate"]) for r in learned_rows)
        guarded = mean(float(r["guarded_rate"]) for r in learned_rows)
        lines += [
            "",
            "| Learned policy | Value |",
            "|---|---|",
            f"| Model | {model.kind}, {model.n_params} parameters ({model_size_bytes(model)} bytes as float32) |",
            f"| Sampled pages per fault (k) | {meta.get('k')} |",
            f"| Metadata per resident page | {METADATA_BYTES_PER_PAGE} bytes |",
            f"| Ghost entry per evicted page (up to one per frame) | {GHOST_BYTES_PER_ENTRY} bytes |",
            f"| Guard state per frame (LRU order + shadow LRU) | {GUARD_BYTES_PER_FRAME} bytes |",
            f"| Training examples | {meta.get('train_examples', 0):,} |",
            f"| Training time | {meta.get('train_seconds')} s |",
            f"| Evictions decided by the LRU fallback (model unsure) | {100 * fallback:.1f}% |",
            f"| Evictions made while the guard had handed over to LRU | {100 * guarded:.1f}% |",
        ]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def evaluate(
    model_path: str | Path | None = None,
    out_dir: Path = RESULTS_DIR,
    policies: Iterable[str] = ALL_POLICIES,
    with_msr: bool = True,
    log: Callable[[str], None] = print,
) -> list[dict]:
    model = load_model(model_path)
    traces = load_split("test", with_msr=with_msr)
    log(f"evaluating {len(list(policies))} policies on {len(traces)} unseen test traces")
    start = time.perf_counter()
    rows = run_grid(traces, policies, model=model, log=log)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / "results.csv")
    write_summary(rows, model, out_dir / "summary.md")
    log(f"wrote {out_dir / 'results.csv'} and {out_dir / 'summary.md'} ({time.perf_counter() - start:.0f}s)")
    return rows


# ------------------------------------------------------- ablation and sweeps


def _validation_score(job: tuple[str, TrainConfig, tuple, dict | None]) -> dict:
    """Train with ``cfg`` (or reuse a saved model) and score it on the validation traces."""
    label, cfg, kinds, policy_kwargs = job
    va = load_split("validation")
    if kinds:
        model = train(load_split("train"), va, cfg, kinds=kinds, log=lambda _: None)
        score = model.meta["validation_faults_vs_lru"]
    else:
        score = relative_faults(load_model(), va, cfg, **(policy_kwargs or {}))
    return {"variant": label, "validation_faults_vs_lru": round(score, 4)}


def _run_jobs(jobs: list, n_jobs: int, log: Callable[[str], None]) -> list[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for result in pool.map(_validation_score, jobs):
            log(f"  {result['variant']:38s} {result['validation_faults_vs_lru']:.4f}")
            results.append(result)
    return results


def ablation(base: TrainConfig, out_dir: Path = RESULTS_DIR, n_jobs: int = 4, log=print) -> list[dict]:
    """Which features and design choices matter, scored on the validation traces.

    Feature variants are trained without the guard so that the model's own
    quality shows; the guard would hide part of the difference.
    """
    raw = replace(base, guard=False)
    every = list(range(len(FEATURE_NAMES)))
    jobs = [("all features", raw, ("logistic",), None)]
    for i, name in enumerate(FEATURE_NAMES):
        jobs.append((f"without {name}", replace(raw, features=[j for j in every if j != i]), ("logistic",), None))
    for group, names in FEATURE_GROUPS.items():
        keep = [FEATURE_NAMES.index(n) for n in names]
        jobs.append((f"only {group}", replace(raw, features=keep), ("logistic",), None))
    jobs += [
        ("design: full (guard on)", base, ("logistic",), None),
        ("design: no DAgger rounds", replace(base, dagger_rounds=0), ("logistic",), None),
        ("design: no ghost table", replace(base, ghost=False), ("logistic",), None),
        ("design: tree instead of logistic", base, ("tree",), None),
    ]
    log(f"ablation: {len(jobs)} variants on the validation traces")
    rows = _run_jobs(jobs, n_jobs, log)
    _write_csv(rows, out_dir / "ablation.csv")
    return rows


def sweep(base: TrainConfig, out_dir: Path = RESULTS_DIR, n_jobs: int = 4, log=print) -> list[dict]:
    """Sample size k and confidence threshold (saved model) and label horizon H (retrained)."""
    jobs = []
    for k in (4, 8, 16, 32, 64):
        jobs.append((f"k={k}", replace(base, k=k), (), {"k": k}))
    for threshold in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        jobs.append((f"threshold={threshold}", replace(base, threshold=threshold), (), {"threshold": threshold}))
    for horizon in (0.5, 1.0, 2.0, 4.0, 8.0):
        jobs.append((f"horizon={horizon}", replace(base, horizon=horizon), ("logistic",), None))
    log(f"sweep: {len(jobs)} settings on the validation traces")
    rows = _run_jobs(jobs, n_jobs, log)
    for row in rows:
        name, value = row["variant"].split("=")
        row["parameter"], row["value"] = name, float(value)
    _write_csv(rows, out_dir / "sweep.csv")
    return rows


def write_study_summary(out_dir: Path = RESULTS_DIR) -> None:
    """Markdown tables for the ablation and the sweep, if they have been run."""
    lines = ["# Ablation and parameter sweep (validation traces)", "",
             "Score: mean over traces and memory sizes of F_Learned / F_LRU (lower is better, 1.0 = LRU).", ""]
    if (out_dir / "ablation.csv").exists():
        lines += ["## Ablation", "", "| Variant | Faults vs LRU |", "|---|---|"]
        lines += [f"| {r['variant']} | {float(r['validation_faults_vs_lru']):.4f} |" for r in _read_csv(out_dir / "ablation.csv")]
        lines += ["", "Feature variants run without the guard, so the model's own quality shows.", ""]
    if (out_dir / "sweep.csv").exists():
        lines += ["## Sweep", "", "| Parameter | Value | Faults vs LRU |", "|---|---|---|"]
        lines += [f"| {r['parameter']} | {float(r['value']):g} | {float(r['validation_faults_vs_lru']):.4f} |"
                  for r in _read_csv(out_dir / "sweep.csv")]
        lines.append("")
    (out_dir / "ablation_and_sweep.md").write_text("\n".join(lines) + "\n")

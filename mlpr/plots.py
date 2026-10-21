"""Figures for the results: fault rate vs memory, gap closed, cost, ablation and sweeps.

Colours follow the policy, never its rank, so a policy has the same colour
in every figure. OPT is drawn as a neutral dashed reference line because it
is a bound, not a competing policy.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .experiments import pooled_gap_closed  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#7a7974"

COLORS = {
    "Learned": "#2a78d6",
    "LRU": "#eb6834",
    "ARC": "#1baf7a",
    "LeCaR": "#eda100",
    "CLOCK": "#e87ba4",
    "FIFO": "#008300",
    "LFU": "#6250d6",
}
MARKERS = {"Learned": "o", "LRU": "s", "ARC": "^", "LeCaR": "D", "CLOCK": "v", "FIFO": "P", "LFU": "X"}
LINE_POLICIES = ("LRU", "ARC", "LeCaR", "Learned")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9,
        }
    )


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fault_rate_figure(rows: list[dict], path: Path) -> None:
    """Small multiples: fault rate against memory size, one panel per workload."""
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    cols = 4
    nrows = math.ceil(len(workloads) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 2.6 * nrows + 0.6), squeeze=False)
    for ax, workload in zip(axes.flat, workloads):
        mine = [r for r in rows if r["workload"] == workload]
        fractions = sorted({float(r["memory_fraction"]) for r in mine})
        xs = range(len(fractions))

        def series(policy: str) -> list[float]:
            by_fraction = {float(r["memory_fraction"]): 100 * float(r["fault_rate"]) for r in mine if r["policy"] == policy}
            return [by_fraction.get(f, math.nan) for f in fractions]

        ax.plot(xs, series("OPT"), color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.5, label="OPT (lower bound)")
        for policy in LINE_POLICIES:
            ax.plot(xs, series(policy), color=COLORS[policy], marker=MARKERS[policy], markersize=6,
                    linewidth=2.6 if policy == "Learned" else 1.8, label=policy, zorder=3 if policy == "Learned" else 2)
        ax.set_title(workload, loc="left")
        ax.set_xticks(list(xs), [f"{100 * f:g}%" for f in fractions])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("memory (share of distinct pages)")
        ax.set_ylabel("fault rate (%)")
    for ax in list(axes.flat)[len(workloads):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gap_closed_figure(rows: list[dict], path: Path, policies=("ARC", "LeCaR", "Learned")) -> None:
    """Share of the LRU-to-OPT gap each adaptive policy closes, on faults summed over memory sizes."""
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    values = {p: [pooled_gap_closed(rows, p, w) for w in workloads] for p in policies}
    values = {p: [0.0 if math.isnan(v) else v for v in vs] for p, vs in values.items()}

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(workloads) + 1.4))
    height = 0.8 / len(policies)
    for i, p in enumerate(policies):
        ys = [w + (len(policies) - 1 - i) * height - 0.4 + height / 2 for w in range(len(workloads))]
        ax.barh(ys, values[p], height=height * 0.85, color=COLORS[p], label=p, zorder=2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(workloads)), workloads)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("LRU-to-OPT gap closed (%), faults summed over memory sizes  (0 = LRU, 100 = OPT)")
    ax.set_title("How much of the gap to optimal each policy closes", loc="left")
    ax.legend(loc="lower right", ncol=len(policies))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_figure(rows: list[dict], path: Path) -> None:
    """Median decision time per fault, per policy (measured in the Python simulator)."""
    times = defaultdict(list)
    for r in rows:
        if int(r["faults"]) > 0:
            times[r["policy"]].append(float(r["us_per_fault"]))
    policies = sorted(times, key=lambda p: median(times[p]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    meds = [median(times[p]) for p in policies]
    ax.barh(policies, meds, color="#2a78d6", height=0.6, zorder=2)
    for y, v in enumerate(meds):
        ax.text(v, y, f"  {v:.1f}", va="center", color=INK_2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("median time per fault (microseconds, Python simulator)")
    ax.set_title("Cost on the fault path", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ablation_figure(rows: list[dict], path: Path) -> None:
    labels = [r["variant"] for r in rows]
    values = [float(r["validation_faults_vs_lru"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.5))
    ax.axvline(1.0, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.5, zorder=1)
    ax.barh(labels, values, color="#2a78d6", height=0.6, zorder=2)
    right = max(values + [1.0]) * 1.1
    for y, v in enumerate(values):
        ax.text(right, y, f"{v:.3f}", va="center", ha="right", color=INK_2, fontsize=8)
    ax.text(1.0, -0.9, " LRU", color=INK_2, fontsize=8)
    ax.set_xlim(min(values + [1.0]) - 0.05, right)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("validation faults relative to LRU (lower is better)")
    ax.set_title("Ablation: features (guard off) and design choices (full pipeline)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sweep_figure(rows: list[dict], path: Path) -> None:
    params = list(dict.fromkeys(r["parameter"] for r in rows))
    titles = {"k": "sampled pages per fault (k)", "threshold": "confidence threshold", "horizon": "label horizon H (x frames)"}
    fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 2.9), squeeze=False)
    for ax, param in zip(axes.flat, params):
        mine = sorted((float(r["value"]), float(r["validation_faults_vs_lru"])) for r in rows if r["parameter"] == param)
        xs, ys = zip(*mine)
        ax.plot(xs, ys, color="#2a78d6", marker="o", markersize=6, linewidth=2)
        ax.axhline(1.0, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.5)
        if param in ("k", "horizon"):
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, [f"{x:g}" for x in xs])
        ax.set_xlabel(titles.get(param, param))
        ax.set_ylabel("faults vs LRU")
    fig.suptitle("Parameter sweep on validation traces (dashed line = LRU)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(results_dir: Path) -> list[Path]:
    _style()
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if (results_dir / "results.csv").exists():
        rows = _read(results_dir / "results.csv")
        for name, draw in (("fault_rate.png", fault_rate_figure), ("gap_closed.png", gap_closed_figure),
                           ("cost_per_fault.png", cost_figure)):
            draw(rows, out_dir / name)
            made.append(out_dir / name)
    if (results_dir / "ablation.csv").exists():
        ablation_figure(_read(results_dir / "ablation.csv"), out_dir / "ablation.png")
        made.append(out_dir / "ablation.png")
    if (results_dir / "sweep.csv").exists():
        sweep_figure(_read(results_dir / "sweep.csv"), out_dir / "sweep.png")
        made.append(out_dir / "sweep.png")
    return made

"""Command line interface: ``python -m mlpr <command>``.

    textbook     check every baseline on the classic textbook reference string
    workloads    list the workloads and their sizes
    simulate     run policies on one trace and print fault counts
    train        learn the eviction model from OPT and save it
    evaluate     run every policy on the unseen test traces, write results/
    ablation     feature and design ablation on validation traces
    sweep        sweep k, the confidence threshold and the label horizon
    plots        redraw the figures from the CSV files in results/
    all          train, evaluate, ablation, sweep and plots in one go
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from .experiments import RESULTS_DIR, ablation, evaluate, sweep, write_study_summary
from .models import DEFAULT_MODEL_PATH, load_model, save_model
from .policies import ALL_POLICIES, make_policy, resolve_name
from .simulator import simulate, trace_steps
from .training import TrainConfig, train
from .traces import TEXTBOOK, load_msr, load_trace_file
from .workloads import MEMORY_FRACTIONS, frames_for, load_split, memory_sizes, workloads


def _policies(text: str) -> list[str]:
    return ALL_POLICIES if text == "all" else [resolve_name(p) for p in text.split(",")]


def _config(args) -> TrainConfig:
    return TrainConfig(
        horizon=args.horizon,
        k=args.k,
        threshold=args.threshold,
        ghost=not args.no_ghost,
        guard=not args.no_guard,
        dagger_rounds=args.dagger,
        seed=args.seed,
    )


def _config_from_model(path) -> TrainConfig:
    """The training settings stored with a saved model (defaults if there is none)."""
    try:
        stored = dict(load_model(path).meta.get("config", {}))
    except FileNotFoundError:
        return TrainConfig()
    stored.pop("fractions", None)
    return replace(TrainConfig(), **stored)


def _model_or_none(path):
    try:
        return load_model(path)
    except FileNotFoundError:
        return None


def cmd_textbook(args) -> None:
    print(f"reference string: {' '.join(map(str, TEXTBOOK))}   frames: {args.frames}\n")
    model = _model_or_none(args.model)
    for name in ALL_POLICIES:
        if name == "Learned" and model is None:
            continue
        policy = make_policy(name, args.frames, trace=TEXTBOOK, model=model)
        if args.steps and name in ("FIFO", "LRU", "OPT"):
            print(f"{name}:")
            for page, hit, resident in trace_steps(TEXTBOOK, policy):
                print(f"  {page:2d}  {'hit  ' if hit else 'FAULT'}  {resident}")
            policy = make_policy(name, args.frames, trace=TEXTBOOK, model=model)
        print(f"{name:8s} {simulate(TEXTBOOK, policy).faults:3d} faults")
    if args.frames == 3:
        print("\nexpected for 3 frames: FIFO 15, LRU 12, OPT 9")


def cmd_workloads(args) -> None:
    for split in ("train", "validation", "test"):
        traces = load_split(split)
        print(f"{split}:")
        for w in workloads(split):
            trace = traces[w.name]
            sizes = ", ".join(str(f) for _, f in memory_sizes(trace))
            print(f"  {w.name:10s} {w.kind:9s} {len(trace):8,d} refs {len(set(trace)):6,d} pages  "
                  f"frames [{sizes}]  {w.tests}")
        extra = [n for n in traces if n not in {w.name for w in workloads(split)}]
        for name in extra:
            print(f"  {name:10s} real      {len(traces[name]):8,d} refs {len(set(traces[name])):6,d} pages")


def cmd_simulate(args) -> None:
    if args.trace_file:
        trace, label = load_trace_file(args.trace_file), args.trace_file
    elif args.msr:
        trace, label = load_msr(args.msr, max_accesses=args.max_accesses), args.msr
    else:
        trace, label = load_split(args.split)[args.workload], f"{args.split}/{args.workload}"
    frames = args.frames or frames_for(trace, args.memory)
    model = _model_or_none(args.model)
    names = [p for p in _policies(args.policies) if p != "Learned" or model is not None]
    print(f"{label}: {len(trace):,} references, {len(set(trace)):,} distinct pages, {frames} frames\n")
    results = {}
    for name in names:
        kwargs = {}
        if name == "Learned":
            kwargs = {k: v for k, v in (("k", args.k), ("threshold", args.threshold)) if v is not None}
        r = simulate(trace, make_policy(name, frames, trace=trace, model=model, **kwargs))
        results[name] = r
        extra = "".join(f"  {k}={v:.1%}" for k, v in r.extra.items())
        print(f"{name:8s} faults={r.faults:8,d}  fault rate={r.fault_rate:6.2%}  {r.us_per_fault:7.2f} us/fault{extra}")
    if {"LRU", "OPT"} <= results.keys():
        from .metrics import fault_reduction, gap_closed

        lru, opt = results["LRU"].faults, results["OPT"].faults
        print()
        for name, r in results.items():
            if name not in ("LRU", "OPT"):
                print(f"{name:8s} reduction vs LRU {fault_reduction(lru, r.faults):6.1f}%   "
                      f"gap closed {gap_closed(lru, r.faults, opt):6.1f}%")


def cmd_train(args) -> None:
    cfg = _config(args)
    start = time.perf_counter()
    model = train(load_split("train"), load_split("validation"), cfg, kinds=args.kinds.split(","))
    path = save_model(model, args.model_out)
    print(f"selected {model.kind} model -> {path} ({time.perf_counter() - start:.0f}s)")


def cmd_evaluate(args) -> None:
    evaluate(args.model, Path(args.out), _policies(args.policies), with_msr=not args.no_msr)
    _plots(Path(args.out))


def cmd_ablation(args) -> None:
    out = Path(args.out)
    ablation(_config_from_model(args.model), out, args.jobs)
    write_study_summary(out)
    _plots(out)


def cmd_sweep(args) -> None:
    out = Path(args.out)
    sweep(_config_from_model(args.model), out, args.jobs)
    write_study_summary(out)
    _plots(out)


def _plots(out: Path) -> None:
    from .plots import make_all

    for path in make_all(out):
        print(f"wrote {path}")


def cmd_plots(args) -> None:
    _plots(Path(args.out))


def cmd_all(args) -> None:
    cmd_train(args)
    cmd_evaluate(args)
    cmd_ablation(args)
    cmd_sweep(args)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m mlpr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model(p):
        p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="trained model (JSON)")

    def add_out(p):
        p.add_argument("--out", default=str(RESULTS_DIR), help="results directory")

    def add_train(p):
        d = TrainConfig()
        p.add_argument("--horizon", type=float, default=d.horizon, help="label horizon H in multiples of frames")
        p.add_argument("--k", type=int, default=d.k, help="pages sampled per eviction")
        p.add_argument("--threshold", type=float, default=d.threshold, help="confidence below which LRU decides")
        p.add_argument("--dagger", type=int, default=d.dagger_rounds, help="DAgger rounds after the OPT round")
        p.add_argument("--no-ghost", action="store_true", help="forget metadata of evicted pages")
        p.add_argument("--no-guard", action="store_true", help="disable the hand-over to LRU")
        p.add_argument("--kinds", default="logistic,tree", help="model kinds to compare")
        p.add_argument("--seed", type=int, default=d.seed)

    p = sub.add_parser("textbook", help="classic reference string check")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--steps", action="store_true", help="print every step for FIFO, LRU and OPT")
    add_model(p)
    p.set_defaults(func=cmd_textbook)

    p = sub.add_parser("workloads", help="list workloads")
    p.set_defaults(func=cmd_workloads)

    p = sub.add_parser("simulate", help="run policies on one trace")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--workload", default="zipf", help="workload name (see `workloads`)")
    src.add_argument("--trace-file", help="text file of page numbers")
    src.add_argument("--msr", help="MSR Cambridge CSV (.csv or .csv.gz)")
    p.add_argument("--split", default="test", choices=("train", "validation", "test"))
    p.add_argument("--max-accesses", type=int, default=200_000, help="cap for MSR traces")
    p.add_argument("--frames", type=int, help="number of frames")
    p.add_argument("--memory", type=float, default=MEMORY_FRACTIONS[2],
                   help="frames as a fraction of distinct pages (if --frames is not given)")
    p.add_argument("--policies", default="all", help="comma-separated, or 'all'")
    p.add_argument("--k", type=int, help="override the model's k")
    p.add_argument("--threshold", type=float, help="override the model's threshold")
    add_model(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("train", help="train the model from OPT")
    add_train(p)
    p.add_argument("--out", dest="model_out", default=str(DEFAULT_MODEL_PATH), help="where to save the model")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate on unseen test traces")
    add_model(p)
    add_out(p)
    p.add_argument("--policies", default="all")
    p.add_argument("--no-msr", action="store_true", help="skip MSR traces in data/msr")
    p.set_defaults(func=cmd_evaluate)

    for name, func in (("ablation", cmd_ablation), ("sweep", cmd_sweep)):
        p = sub.add_parser(name, help=f"{name} on validation traces")
        add_model(p)
        add_out(p)
        p.add_argument("--jobs", type=int, default=4, help="parallel processes")
        p.set_defaults(func=func)

    p = sub.add_parser("plots", help="redraw figures")
    add_out(p)
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("all", help="train, evaluate, ablation, sweep")
    add_train(p)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    add_out(p)
    p.add_argument("--policies", default="all")
    p.add_argument("--no-msr", action="store_true")
    p.add_argument("--jobs", type=int, default=4)
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    if args.command == "all":
        args.model_out = args.model
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end experiment: nominal, robust, shape-diverse and behavior-diverse designs,
scored on held-out loads.

    python -m bdto.experiment --problem cantilever --out results/cantilever
    python -m bdto.experiment --quick          # small mesh smoke run (~30 s)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import plotting
from .designs import diverse_set, nominal, robust
from .evaluate import evaluate_sets
from .loads import heldout_loads
from .optimize import Settings
from .problems import make_problem


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default="cantilever", choices=["cantilever", "l_bracket"])
    p.add_argument("--size", type=int, default=None, help="nely (cantilever) or side length (L-bracket)")
    p.add_argument("--volfrac", type=float, default=0.4)
    p.add_argument("--n-designs", type=int, default=4, help="set size, including the nominal design")
    p.add_argument("--mode", default="constraint", choices=["constraint", "penalty"])
    p.add_argument("--eps", type=float, default=0.10, help="compliance budget (constraint mode)")
    p.add_argument("--alpha", type=float, default=1.0, help="repulsion weight (penalty mode)")
    p.add_argument("--kernel", default="inverse", choices=["inverse", "gaussian"])
    p.add_argument("--sigma-behavior", type=float, default=None)
    p.add_argument("--sigma-shape", type=float, default=None)
    p.add_argument("--behavior-mode", default="logcorr", choices=["logcorr", "logdiag"])
    p.add_argument("--probe-cap", type=float, default=0.10,
                   help="cap each probe compliance at (1 + cap) x nominal, both sets; negative disables")
    p.add_argument("--robust-range", type=float, default=180.0, help="robust design: load-angle half range (deg)")
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rmin", type=float, default=2.0)
    p.add_argument("--stage-iters", type=int, default=40)
    p.add_argument("--quick", action="store_true", help="small mesh and few iterations")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.quick:
        args.size = args.size or 20
        args.stage_iters = min(args.stage_iters, 15)
        args.n_test = min(args.n_test, 60)
    problem = make_problem(args.problem, args.size, args.volfrac)
    settings = Settings(rmin=args.rmin, stage_iters=args.stage_iters, verbose=args.verbose)
    out = args.out or Path("results") / f"{problem.name}_{problem.grid.nelx}x{problem.grid.nely}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{problem.name} {problem.grid.nelx}x{problem.grid.nely}, volfrac {problem.volfrac} -> {out}")

    timings, t0 = {}, time.perf_counter()

    def tick(name):
        nonlocal t0
        timings[name] = round(time.perf_counter() - t0, 2)
        print(f"  {name}: {timings[name]} s")
        t0 = time.perf_counter()

    base = nominal(problem, settings)
    tick("nominal")
    rob = robust(problem, settings, args.robust_range)
    tick("robust")
    common = dict(n_designs=args.n_designs, settings=settings, mode=args.mode, eps=args.eps,
                  alpha=args.alpha, kernel=args.kernel,
                  probe_cap=args.probe_cap if args.probe_cap >= 0 else None)
    shape = diverse_set(problem, base, "shape", sigma=args.sigma_shape, **common)
    tick("shape set")
    behavior = diverse_set(problem, base, "behavior", sigma=args.sigma_behavior,
                           behavior_mode=args.behavior_mode, **common)
    tick("behavior set")

    results = {"nominal": [base], "robust": [rob], "shape": shape, "behavior": behavior}
    sets = {name: [r.xphys for r in rs] for name, rs in results.items()}
    loads, families = heldout_loads(problem, args.n_test, args.seed)
    metrics = evaluate_sets(problem, sets, loads, families, penal=settings.penal)
    tick("evaluation")

    np.savez_compressed(out / "designs.npz", **{f"{n}_{i}": xp for n, xps in sets.items() for i, xp in enumerate(xps)},
                        test_loads=loads, test_families=families)
    gaps = {n: m["primary_gap"] for n, m in metrics["methods"].items()}
    plotting.design_gallery(problem, sets, gaps, out / "designs.png")
    plotting.ratio_ecdf(metrics["ratios"], out / "heldout_ecdf.png",
                        f"{problem.name}: held-out loads ({args.n_test})")
    plotting.convergence({r.label: r.history for rs in results.values() for r in rs}, out / "convergence.png")

    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "settings": settings.to_dict(),
        "timings_s": timings,
        "iterations": {r.label: r.iterations for rs in results.values() for r in rs},
        "methods": metrics["methods"],
        "behavior_win_rate": metrics.get("behavior_win_rate"),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    print("\nheld-out loads: best-in-set compliance / nominal (lower is better)")
    print(f"{'method':<10}{'n':>3}{'geomean':>9}{'median':>8}{'p90':>8}{'max':>8}"
          f"{'max gap':>9}{'beh. spread':>12}{'shape spread':>13}")
    for name, m in report["methods"].items():
        a = m["all"]
        print(f"{name:<10}{m['n_designs']:>3}{a['geomean']:>9.3f}{a['median']:>8.3f}{a['p90']:>8.3f}"
              f"{a['max']:>8.3f}{100 * max(m['primary_gap']):>8.1f}%{m['behavior_spread']:>12.3f}"
              f"{m['shape_spread']:>13.3f}")
    if report.get("behavior_win_rate"):
        wins = ", ".join(f"{k} {100 * v:.0f}%" for k, v in report["behavior_win_rate"].items())
        print(f"behavior set strictly better on: {wins} of held-out loads")


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

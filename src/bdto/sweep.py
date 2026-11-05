"""Sweep the compliance budget eps on both benchmarks, then plot the summary figures.

    python -m bdto.sweep                     # runs (in parallel) + figures, ~10 min
    python -m bdto.sweep --plot-only         # re-plot from results/sweep/*/metrics.json

Set size is not swept separately: sets are grown one design at a time, so a run with
``--n-designs 6`` also gives every smaller set as a prefix (``prefix_geomean``).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import experiment  # noqa: E402
from .plotting import COLORS, INK, LINESTYLES, MUTED  # noqa: E402

PROBLEMS = ("cantilever", "l_bracket")
EPS = (0.05, 0.10, 0.20)
TITLES = {"cantilever": "Cantilever 80×40", "l_bracket": "L-bracket 80×80"}
LABELS = {"behavior": "behavior set", "shape": "shape set", "robust_set": "robust set",
          "robust": "robust, unconstrained", "robust_budget": "robust, same budget",
          "robust_narrow": "robust, wrong (narrow) range"}
BG = "#f7f6f2"


def _legend_below(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=10)


def run_dir(root: Path, problem: str, eps: float) -> Path:
    return root / f"{problem}_eps{eps:.2f}"


def _run(job: tuple[str, float, int, Path, list[str]]) -> str:
    problem, eps, n, root, extra = job
    experiment.run(experiment.parse_args(
        ["--problem", problem, "--eps", str(eps), "--n-designs", str(n),
         "--out", str(run_dir(root, problem, eps)), *extra]))
    return f"{problem} eps={eps}"


def load(root: Path) -> dict:
    return {(p, e): json.loads((run_dir(root, p, e) / "metrics.json").read_text())
            for p in PROBLEMS for e in EPS if (run_dir(root, p, e) / "metrics.json").exists()}


def plot_set_size(runs: dict, eps: float, path: Path) -> None:
    """Geomean ratio vs set size (prefixes), single-design baselines as flat lines."""
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(10, 4.4), sharey=True, facecolor=BG)
    for ax, prob in zip(axes, PROBLEMS):
        ax.set_facecolor(BG)
        m = runs[(prob, eps)]["methods"]
        for name in ("behavior", "shape", "robust_set"):
            ys = m[name]["prefix_geomean"]
            ax.plot(range(1, len(ys) + 1), ys, marker="o", markersize=5, linewidth=2,
                    color=COLORS[name], linestyle=LINESTYLES.get(name, "-"), label=LABELS[name])
        n = len(m["behavior"]["prefix_geomean"])
        for name in ("robust", "robust_budget", "robust_narrow"):
            ax.hlines(m[name]["all"]["geomean"], 1, n, color=COLORS[name], linewidth=1.2,
                      linestyle=LINESTYLES.get(name, "-"), alpha=0.8, label=LABELS[name] + " (1 design)")
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(TITLES[prob], color=INK)
        ax.set_xlabel("set size (designs)")
        ax.set_xticks(range(1, n + 1))
    axes[0].set_ylabel("geomean held-out compliance / nominal")
    _legend_below(fig, axes[0], 3)
    fig.suptitle(f"Held-out performance vs set size (ε = {eps:.0%})", color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)


def plot_eps(runs: dict, path: Path) -> None:
    """Geomean ratio (95% bootstrap CI) vs compliance budget eps."""
    names = ("behavior", "shape", "robust_set", "robust_budget")
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(10, 4.2), sharey=True, facecolor=BG)
    for ax, prob in zip(axes, PROBLEMS):
        ax.set_facecolor(BG)
        eps = [e for e in EPS if (prob, e) in runs]
        for j, name in enumerate(names):
            xs = [100 * e * (1 + 0.04 * (j - 1.5)) for e in eps]  # small dodge
            ms = [runs[(prob, e)]["methods"][name]["all"] for e in eps]
            ys = [a["geomean"] for a in ms]
            lo = [y - a["geomean_ci95"][0] for y, a in zip(ys, ms)]
            hi = [a["geomean_ci95"][1] - y for y, a in zip(ys, ms)]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", markersize=5, linewidth=2, capsize=3,
                        color=COLORS[name], linestyle=LINESTYLES.get(name, "-"), label=LABELS[name])
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks([100 * e for e in EPS], [f"{100 * e:.0f}%" for e in EPS])
        ax.minorticks_off()
        ax.set_title(TITLES[prob], color=INK)
        ax.set_xlabel("compliance budget ε (primary load)")
    axes[0].set_ylabel("geomean held-out compliance / nominal")
    _legend_below(fig, axes[0], 4)
    fig.suptitle("Held-out performance vs compliance budget (6-design sets, 95% CI)", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160, facecolor=BG)
    plt.close(fig)


def table(runs: dict) -> str:
    rows = ["| problem | ε | " + " | ".join(("behavior", "shape", "robust_set", "robust_budget",
                                               "robust_narrow", "robust")) + " |",
            "|---|---|" + "---|" * 6]
    for (prob, eps), r in sorted(runs.items()):
        m = r["methods"]
        cells = [f"{m[n]['all']['geomean']:.2f}" for n in ("behavior", "shape", "robust_set", "robust_budget",
                                                            "robust_narrow", "robust")]
        rows.append(f"| {prob} | {eps:.0%} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("results/sweep"))
    p.add_argument("--n-designs", type=int, default=6)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eps", type=float, nargs="+", default=list(EPS))
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="further bdto.experiment arguments for every run, e.g. --extra --behavior-mode logdiag")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    if not args.plot_only:
        # One BLAS thread per worker: N workers x all-core BLAS pools oversubscribe
        # the machine badly (spawned workers read these at import).
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        jobs = [(prob, eps, args.n_designs, args.root, args.extra) for prob in PROBLEMS for eps in args.eps]
        with ProcessPoolExecutor(args.workers) as pool:
            for done in pool.map(_run, jobs):
                print("done:", done, flush=True)
    runs = load(args.root)
    plot_set_size(runs, 0.10, args.root / "set_size.png")
    plot_eps(runs, args.root / "eps.png")
    md = table(runs)
    (args.root / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()

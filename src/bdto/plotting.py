"""Figures: design gallery, held-out load ratios, convergence."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .problems import Problem  # noqa: E402

# Fixed method -> color mapping (categorical slots 1-3 of the reference palette;
# the nominal reference is neutral gray).
COLORS = {"behavior": "#2a78d6", "shape": "#eb6834", "robust": "#1baf7a", "nominal": "#8a8985"}
# Robust variants share the robust hue and are told apart by line style (and the legend).
COLORS.update({"robust_budget": COLORS["robust"], "robust_narrow": COLORS["robust"]})
LINESTYLES = {"robust_budget": "--", "robust_narrow": ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def _show_design(ax, problem: Problem, xphys: np.ndarray, title: str) -> None:
    img = problem.grid.to_image(xphys).astype(float)
    img[problem.grid.to_image(problem.passive_void)] = np.nan
    cmap = plt.get_cmap("gray").with_extremes(bad=GRID)  # non-design region in light gray
    ax.imshow(1 - img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def design_gallery(problem: Problem, sets: dict[str, list], gaps: dict[str, list[float]], path) -> None:
    """One row per method; each panel titled with its primary-load compliance gap."""
    ncol = max(len(v) for v in sets.values())
    aspect = problem.grid.nely / problem.grid.nelx
    fig, axes = plt.subplots(len(sets), ncol, figsize=(2.6 * ncol, 2.6 * aspect * len(sets) + 0.8),
                             squeeze=False)
    for r, (name, xps) in enumerate(sets.items()):
        for c in range(ncol):
            ax = axes[r, c]
            if c < len(xps):
                _show_design(ax, problem, xps[c], f"{name} {c}: +{100 * gaps[name][c]:.1f}% c")
            else:
                ax.axis("off")
    fig.suptitle(f"{problem.name}: designs (gap = primary compliance vs nominal)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ratio_ecdf(ratios: dict[str, np.ndarray], path, title: str = "") -> None:
    """ECDF of best-in-set compliance / nominal compliance over held-out loads."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in ratios.items():
        xs = np.sort(r)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=COLORS.get(name, INK), linewidth=2,
                linestyle=LINESTYLES.get(name, "-"), label=name)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("best-in-set compliance / nominal compliance  (log; left is better)")
    ax.set_ylabel("fraction of held-out loads")
    ax.set_title(title or "Held-out load performance", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def convergence(histories: dict[str, list[dict]], path) -> None:
    """Primary compliance per iteration, one panel per design run."""
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.6), squeeze=False, sharey=True)
    for ax, (label, hist) in zip(axes[0], histories.items()):
        ax.plot([h["it"] for h in hist], [h["compliance"] for h in hist], color=INK, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("iteration")
    axes[0, 0].set_ylabel("primary compliance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

"""Presentation figures, built from saved runs (no re-optimization).

    PYTHONPATH=src python scripts/slide_figures.py

Needs results/sweep/{cantilever,l_bracket}_eps0.10/ (python -m bdto.sweep) and the
uncapped cantilever run results/cantilever_nocap/ (python -m bdto.experiment
--probe-cap -1 --out results/cantilever_nocap). Writes results/slides/*.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bdto.loads import probe_loads  # noqa: E402
from bdto.plotting import COLORS, GRID, INK  # noqa: E402
from bdto.problems import make_problem  # noqa: E402

ROOT = Path("results")
OUT = ROOT / "slides"
BG = "#f7f6f2"  # matches the deck's light background


def designs(run: Path, method: str) -> list[np.ndarray]:
    d = np.load(run / "designs.npz")
    keys = sorted((k for k in d.files if k.rsplit("_", 1)[0] == method), key=lambda k: int(k.rsplit("_", 1)[1]))
    return [d[k] for k in keys]


def show(ax, problem, xp, title, color=INK):
    img = problem.grid.to_image(xp).astype(float)
    img[problem.grid.to_image(problem.passive_void)] = np.nan
    ax.imshow(1 - img, cmap=plt.get_cmap("gray").with_extremes(bad=GRID), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(title, fontsize=13, color=color, pad=6)
    ax.axis("off")


def probe_multiples(problem, xps):
    G = probe_loads(problem)
    fe = problem.fe()
    diag = np.array([np.diag(G.T @ fe.displacements(xp, G)) for xp in xps])
    return diag / diag[0]


def degradation(n=4):
    """Uncapped vs capped behavior sets on the cantilever, titled with probe compliance multiples."""
    p = make_problem("cantilever", 40)
    rows = {"no probe caps": designs(ROOT / "cantilever_nocap", "behavior")[:n],
            "with probe caps": designs(ROOT / "sweep/cantilever_eps0.10", "behavior")[:n]}
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 4.2), facecolor=BG)
    for r, (label, xps) in enumerate(rows.items()):
        mult = probe_multiples(p, xps)
        for c, xp in enumerate(xps):
            t = "nominal" if c == 0 else f"Fx ×{mult[c, 0]:.1f}   M ×{mult[c, 2]:.1f}"
            show(axes[r, c], p, xp, t, color="#b5432a" if c and mult[c].max() > 2 else INK)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=170, facecolor=BG)
    plt.close(fig)


def gallery(problem_name, size, n=4):
    """Shape set vs behavior set (first n designs) for one benchmark."""
    p = make_problem(problem_name, size)
    run = ROOT / f"sweep/{problem_name}_eps0.10"
    aspect = p.grid.nely / p.grid.nelx
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 2 * 3.3 * aspect + 0.9), facecolor=BG)
    for r, method in enumerate(("shape", "behavior")):
        for c, xp in enumerate(designs(run, method)[:n]):
            show(axes[r, c], p, xp, "nominal" if c == 0 else f"{method} {c}", COLORS[method] if c else INK)
    fig.tight_layout()
    fig.savefig(OUT / f"gallery_{problem_name}.png", dpi=170, facecolor=BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    degradation()
    gallery("cantilever", 40)
    gallery("l_bracket", 80)
    for name in ("set_size.png", "eps.png"):
        src = ROOT / "sweep" / name
        if src.exists():
            (OUT / name).write_bytes(src.read_bytes())
    print("wrote", sorted(f.name for f in OUT.iterdir()))


if __name__ == "__main__":
    main()

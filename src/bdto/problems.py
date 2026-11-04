"""Benchmark problems: domain, supports, passive regions and the loaded patch.

Each problem has one loaded *patch*: a small square of passive solid elements
around the load point. All loads (primary, probe, robust and held-out test
loads) are applied through this patch, so every load hits material in every
design. Without the patch, a point load away from the nominal load path lands on
void and produces meaningless (~1/Emin) compliances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fem import FEModel, Grid


@dataclass
class Problem:
    name: str
    grid: Grid
    volfrac: float
    fixed_dofs: np.ndarray
    passive_void: np.ndarray  # (nel,) bool
    passive_solid: np.ndarray  # (nel,) bool
    patch_nodes: np.ndarray  # nodes of the loaded patch
    patch_center: int  # node at the patch center
    primary_dir: tuple[float, float] = (0.0, -1.0)
    edge_tangent: tuple[float, float] = (0.0, 1.0)  # free-edge direction at the patch

    def fe(self, **kwargs) -> FEModel:
        return FEModel(self.grid, self.fixed_dofs, **kwargs)

    @property
    def design_mask(self) -> np.ndarray:
        """Elements that count toward the volume fraction."""
        return ~self.passive_void

    @property
    def active_mask(self) -> np.ndarray:
        """Elements whose density is a design variable."""
        return ~(self.passive_void | self.passive_solid)

    def _offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Patch node positions relative to the patch centroid (where uniform
        patch forces act), so moments are self-equilibrated."""
        x, y = self.grid.node_xy(self.patch_nodes)
        return x - x.mean(), y - y.mean()

    def patch_force(self, fx: float, fy: float) -> np.ndarray:
        """Resultant force (fx, fy) spread uniformly over the patch nodes."""
        f = np.zeros(self.grid.ndof)
        n = len(self.patch_nodes)
        f[2 * self.patch_nodes] = fx / n
        f[2 * self.patch_nodes + 1] = fy / n
        return f

    def patch_moment(self, moment: float = 1.0) -> np.ndarray:
        """Self-equilibrated nodal forces with zero resultant and the given moment."""
        rx, ry = self._offsets()
        scale = moment / np.sum(rx**2 + ry**2)
        f = np.zeros(self.grid.ndof)
        f[2 * self.patch_nodes] = -ry * scale
        f[2 * self.patch_nodes + 1] = rx * scale
        return f

    def node_force(self, node: int, fx: float, fy: float) -> np.ndarray:
        f = np.zeros(self.grid.ndof)
        f[2 * node], f[2 * node + 1] = fx, fy
        return f

    @property
    def primary_load(self) -> np.ndarray:
        return self.patch_force(*self.primary_dir)


def _patch(grid: Grid, ix: int, iy: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Elements within ``half`` of node (ix, iy) and the nodes they touch."""
    cx, crow = grid.element_centers()
    elems = (np.abs(cx - ix) < half) & (np.abs(crow - iy) < half)
    nodes = np.unique(grid.edof[elems][:, ::2] // 2)
    return elems, nodes


def cantilever(nelx: int = 80, nely: int = 40, volfrac: float = 0.4, pad: int | None = None) -> Problem:
    """Left edge clamped, downward load at the middle of the right edge."""
    grid = Grid(nelx, nely)
    fixed = np.arange(2 * (nely + 1))  # every DOF of the ix = 0 column
    pad = pad or max(2, nely // 16)
    iy = nely // 2
    solid, nodes = _patch(grid, nelx, iy, pad)
    return Problem(
        name="cantilever", grid=grid, volfrac=volfrac, fixed_dofs=fixed,
        passive_void=np.zeros(grid.nel, bool), passive_solid=solid,
        patch_nodes=nodes, patch_center=grid.node(nelx, iy),
    )


def l_bracket(n: int = 80, volfrac: float = 0.4, pad: int | None = None) -> Problem:
    """L-shaped domain (top-right 60% x 60% removed), top edge of the upright arm
    clamped, downward load at the middle of the horizontal arm's free end."""
    grid = Grid(n, n)
    cx, crow = grid.element_centers()
    void = (cx > 0.4 * n) & (crow < 0.6 * n)
    fixed = np.concatenate([[grid.dof(ix, 0, 0), grid.dof(ix, 0, 1)] for ix in range(int(0.4 * n) + 1)])
    pad = pad or max(2, n // 32)
    iy = int(round(0.8 * n))
    solid, nodes = _patch(grid, n, iy, pad)
    return Problem(
        name="l_bracket", grid=grid, volfrac=volfrac, fixed_dofs=fixed,
        passive_void=void, passive_solid=solid & ~void,
        patch_nodes=nodes, patch_center=grid.node(n, iy),
    )


PROBLEMS = {"cantilever": cantilever, "l_bracket": l_bracket}


def make_problem(name: str, size: int | None = None, volfrac: float = 0.4) -> Problem:
    """Build a benchmark by name; ``size`` is nely for the cantilever (nelx = 2 * nely)
    and the side length for the L-bracket."""
    if name == "cantilever":
        return cantilever(2 * size, size, volfrac) if size else cantilever(volfrac=volfrac)
    if name == "l_bracket":
        return l_bracket(size, volfrac) if size else l_bracket(volfrac=volfrac)
    raise ValueError(f"unknown problem {name!r}; choose from {sorted(PROBLEMS)}")

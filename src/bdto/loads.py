"""Probe, robust and held-out test load sets, all applied through the loaded patch."""

from __future__ import annotations

import numpy as np

from .problems import Problem


def probe_loads(problem: Problem) -> np.ndarray:
    """Probe set G = [patch Fx, patch Fy, patch moment]  (ndof x 3).

    The 3x3 flexibility G^T K^-1 G is the patch's rigid-body compliance: how the
    design deflects and twists under any force and couple at the interface.
    """
    return np.column_stack([problem.patch_force(1, 0), problem.patch_force(0, 1), problem.patch_moment(1)])


def direction_basis(problem: Problem) -> np.ndarray:
    """[patch Fx, patch Fy]: any unit-direction patch force is a combination."""
    return np.column_stack([problem.patch_force(1, 0), problem.patch_force(0, 1)])


def direction_second_moment(theta0: float, half_range: float, n: int = 721) -> np.ndarray:
    """E[d d^T] for d = (cos t, sin t), t uniform on [theta0 - half_range, theta0 + half_range]."""
    t = np.linspace(theta0 - half_range, theta0 + half_range, n)
    d = np.stack([np.cos(t), np.sin(t)])
    return d @ d.T / n


def primary_angle(problem: Problem) -> float:
    return float(np.arctan2(problem.primary_dir[1], problem.primary_dir[0]))


def heldout_loads(problem: Problem, n: int = 200, seed: int = 0,
                  ecc_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Held-out loads never seen during optimization. Returns (loads (ndof x n), family labels).

    Two families, half each:

    - ``eccentric``: unit resultant at a uniformly random angle, applied with a
      random offset e along the free edge (|e| <= ecc_max), i.e. force + couple.
    - ``point``: unit force at a random angle on a single random patch node
      (a local load, not representable by the rigid-body probes).
    """
    rng = np.random.default_rng(seed)
    ecc_max = problem.grid.nely / 8 if ecc_max is None else ecc_max
    tx, ty = problem.edge_tangent
    cols, fams = [], []
    for k in range(n):
        th = rng.uniform(0, 2 * np.pi)
        fx, fy = np.cos(th), np.sin(th)
        if k % 2 == 0:
            e = rng.uniform(-ecc_max, ecc_max)
            moment = (e * tx) * fy - (e * ty) * fx
            cols.append(problem.patch_force(fx, fy) + problem.patch_moment(moment))
            fams.append("eccentric")
        else:
            node = rng.choice(problem.patch_nodes)
            cols.append(problem.node_force(node, fx, fy))
            fams.append("point")
    return np.column_stack(cols), np.array(fams)

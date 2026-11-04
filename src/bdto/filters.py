"""Density filter and Heaviside projection (design variables -> physical densities)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .fem import Grid


class DensityFilter:
    """Linear cone filter of radius ``rmin`` (Bourdin 2001), as in the 88-line code."""

    def __init__(self, grid: Grid, rmin: float):
        r = int(np.ceil(rmin)) - 1
        e = np.arange(grid.nel)
        rows, cols, vals = [], [], []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                w = rmin - np.hypot(dx, dy)
                if w <= 0:
                    continue
                nx, ny = grid.elx + dx, grid.ely + dy
                ok = (nx >= 0) & (nx < grid.nelx) & (ny >= 0) & (ny < grid.nely)
                rows.append(e[ok])
                cols.append(nx[ok] * grid.nely + ny[ok])
                vals.append(np.full(ok.sum(), w))
        self.H = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(grid.nel, grid.nel),
        )
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.H @ x / self.Hs

    def backprop(self, g: np.ndarray) -> np.ndarray:
        return self.H.T @ (g / self.Hs)


class Projection:
    """Smoothed Heaviside projection (Wang, Lazarov & Sigmund 2011)."""

    def __init__(self, beta: float = 1.0, eta: float = 0.5):
        self.beta, self.eta = beta, eta

    def _denominator(self) -> float:
        b, n = self.beta, self.eta
        return np.tanh(b * n) + np.tanh(b * (1 - n))

    def apply(self, xt: np.ndarray) -> np.ndarray:
        b, n = self.beta, self.eta
        return (np.tanh(b * n) + np.tanh(b * (xt - n))) / self._denominator()

    def grad(self, xt: np.ndarray) -> np.ndarray:
        b, n = self.beta, self.eta
        return b * (1 - np.tanh(b * (xt - n)) ** 2) / self._denominator()

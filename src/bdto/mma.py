"""Method of Moving Asymptotes (Svanberg 1987), dual-solved.

The convex subproblem is built exactly as in Svanberg's ``mmasub`` (with
artificial variables ``y_i`` so infeasible starts are handled). Because every
term is separable, the inner minimization over ``x`` for fixed multipliers has a
closed form, so the dual (one variable per constraint) is solved with
L-BFGS-B instead of the primal-dual interior point method of ``subsolv``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class MMA:
    def __init__(self, n: int, m: int, xmin=0.0, xmax=1.0, move: float = 0.2,
                 asyinit: float = 0.5, asyincr: float = 1.2, asydecr: float = 0.7,
                 c: float = 1000.0, d: float = 1.0, raa0: float = 1e-5, albefa: float = 0.1):
        self.n, self.m = n, m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.move = move
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.c, self.d, self.raa0, self.albefa = c, d, raa0, albefa
        self.reset()

    def reset(self) -> None:
        """Forget asymptote history (call after a continuation step)."""
        self.iter = 0
        self.xold1 = self.xold2 = self.low = self.upp = None
        self.lam = np.ones(self.m)

    def _asymptotes(self, x: np.ndarray, span: np.ndarray) -> None:
        if self.iter <= 2:
            self.low = x - self.asyinit * span
            self.upp = x + self.asyinit * span
            return
        zzz = (x - self.xold1) * (self.xold1 - self.xold2)
        factor = np.ones_like(x)
        factor[zzz > 0] = self.asyincr
        factor[zzz < 0] = self.asydecr
        low = x - factor * (self.xold1 - self.low)
        upp = x + factor * (self.upp - self.xold1)
        self.low = np.clip(low, x - 10 * span, x - 0.01 * span)
        self.upp = np.clip(upp, x + 0.01 * span, x + 10 * span)

    def update(self, x: np.ndarray, df0: np.ndarray, g: np.ndarray, dg: np.ndarray) -> np.ndarray:
        """One MMA step for ``min f0(x)  s.t.  g_i(x) <= 0,  xmin <= x <= xmax``.

        ``df0``: (n,) objective gradient; ``g``: (m,) constraint values;
        ``dg``: (m, n) constraint gradients. Returns the new ``x``.
        """
        x = np.asarray(x, float)
        g = np.atleast_1d(np.asarray(g, float))
        dg = np.asarray(dg, float).reshape(self.m, self.n)
        self.iter += 1
        span = np.maximum(self.xmax - self.xmin, 1e-5)
        self._asymptotes(x, span)
        low, upp = self.low, self.upp

        alpha = np.maximum.reduce([self.xmin, low + self.albefa * (x - low), x - self.move * span])
        beta = np.minimum.reduce([self.xmax, upp - self.albefa * (upp - x), x + self.move * span])

        ux2, xl2 = (upp - x) ** 2, (x - low) ** 2
        reg = self.raa0 / span
        p0 = ux2 * (1.001 * np.maximum(df0, 0) + 0.001 * np.maximum(-df0, 0) + reg)
        q0 = xl2 * (0.001 * np.maximum(df0, 0) + 1.001 * np.maximum(-df0, 0) + reg)
        P = ux2 * (1.001 * np.maximum(dg, 0) + 0.001 * np.maximum(-dg, 0) + reg)
        Q = xl2 * (0.001 * np.maximum(dg, 0) + 1.001 * np.maximum(-dg, 0) + reg)
        b = (P / (upp - x) + Q / (x - low)).sum(axis=1) - g

        def primal(lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            pl, ql = p0 + lam @ P, q0 + lam @ Q
            sp, sq = np.sqrt(pl), np.sqrt(ql)
            xs = np.clip((low * sp + upp * sq) / (sp + sq), alpha, beta)
            return xs, pl, ql

        if self.m == 0:
            xnew = primal(np.zeros(0))[0]
        else:
            def neg_dual(lam: np.ndarray) -> tuple[float, np.ndarray]:
                xs, pl, ql = primal(lam)
                y = np.maximum(0.0, lam - self.c) / self.d
                ux, xl = upp - xs, xs - low
                val = np.sum(pl / ux + ql / xl) - lam @ b - 0.5 * self.d * np.sum(y**2)
                grad = (P / ux + Q / xl).sum(axis=1) - b - y
                return -val, -grad

            sol = minimize(neg_dual, self.lam, jac=True, method="L-BFGS-B",
                           bounds=[(0.0, None)] * self.m,
                           options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 500})
            self.lam = np.maximum(sol.x, 0.0)
            xnew = primal(self.lam)[0]

        self.xold2, self.xold1 = self.xold1, x.copy()
        return xnew

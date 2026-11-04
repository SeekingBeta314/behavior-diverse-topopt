"""Objective and constraint terms.

A term is a callable ``term(state) -> (value, W, gx)`` where

- ``W`` is the (m x m) gradient of ``value`` w.r.t. the stacked flexibility
  matrix ``state.F`` (or ``None``), turned into a density sensitivity by
  :meth:`bdto.fem.Response.sensitivity`, and
- ``gx`` is a direct gradient w.r.t. the physical densities (or ``None``).

Constraint terms follow the convention ``value <= 0`` is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class State:
    """Physical densities and flexibility of the stacked load blocks at one iterate."""

    xphys: np.ndarray
    F: np.ndarray
    slices: dict[str, slice]
    stage: int = 0

    def block(self, name: str) -> np.ndarray:
        s = self.slices[name]
        return self.F[s, s]

    def embed(self, name: str, Wb: np.ndarray) -> np.ndarray:
        W = np.zeros_like(self.F)
        s = self.slices[name]
        W[s, s] = Wb
        return W

    @property
    def compliance(self) -> float:
        return float(self.block("primary")[0, 0])


class _Ref:
    """Normalization: fixed value, per-continuation-stage values, or the first value seen."""

    def __init__(self, ref: float | Sequence[float] | None):
        self.ref = ref

    def __call__(self, stage: int, value: float) -> float:
        if self.ref is None:
            self.ref = value
        if np.ndim(self.ref) == 0:
            return float(self.ref)
        return float(self.ref[min(stage, len(self.ref) - 1)])


class Compliance:
    """Compliance of the first load in ``block``, divided by ``ref``."""

    def __init__(self, block: str = "primary", ref=None):
        self.block, self.ref = block, _Ref(ref)

    def __call__(self, st: State):
        c = st.block(self.block)[0, 0]
        r = self.ref(st.stage, c)
        return c / r, st.embed(self.block, np.array([[1.0 / r]])), None


class ExpectedCompliance:
    """E[f^T K^-1 f] for random loads f = B a with second moment E[a a^T] = S.

    Equals ``tr(S F_B)``, so any load distribution with a known second moment
    (e.g. a direction varying over an arc, as in Gresia et al. 2024) costs only
    one solve per column of the basis B.
    """

    def __init__(self, block: str, S: np.ndarray, ref=None):
        self.block, self.S, self.ref = block, np.asarray(S, float), _Ref(ref)

    def __call__(self, st: State):
        v = float(np.sum(self.S * st.block(self.block)))
        r = self.ref(st.stage, v)
        return v / r, st.embed(self.block, self.S / r), None


class ComplianceBound:
    """Near-optimality constraint  c / c_ref - (1 + eps) <= 0."""

    def __init__(self, ref, eps: float, block: str = "primary"):
        self.block, self.ref, self.eps = block, _Ref(ref), eps

    def __call__(self, st: State):
        c = st.block(self.block)[0, 0]
        r = self.ref(st.stage, c)
        return c / r - (1 + self.eps), st.embed(self.block, np.array([[1.0 / r]])), None


class Sum:
    def __init__(self, *terms, weights: Sequence[float] | None = None):
        self.terms = terms
        self.weights = weights or [1.0] * len(terms)

    def __call__(self, st: State):
        value, W, gx = 0.0, np.zeros_like(st.F), np.zeros_like(st.xphys)
        for w, term in zip(self.weights, self.terms):
            v, Wt, gt = term(st)
            value += w * v
            if Wt is not None:
                W += w * Wt
            if gt is not None:
                gx += w * gt
        return value, W, gx


# --- diversity ------------------------------------------------------------------


class BehaviorDescriptor:
    """Mechanical-response descriptor from the probe-load flexibility matrix.

    For probe loads G the flexibility ``F = G^T K^-1 G`` holds every probe
    response observed at every probe. The descriptor keeps

    - ``logdiag``: the log probe compliances, centered so a uniformly stiffer or
      softer design is *not* counted as behaving differently, and
    - ``logcorr`` (default): additionally the correlations F_ij / sqrt(F_ii F_jj),
      i.e. how strongly a load at probe i deflects probe j (load-path coupling).

    Descriptor *differences* (all the repulsion uses) are invariant to the
    units/magnitude of each individual probe.
    """

    def __init__(self, block: str = "probes", mode: str = "logcorr"):
        if mode not in ("logcorr", "logdiag"):
            raise ValueError(f"unknown behavior descriptor mode {mode!r}")
        self.block, self.mode = block, mode

    def __call__(self, st: State) -> np.ndarray:
        F = st.block(self.block)
        D = np.diag(F)
        ell = np.log(D)
        z = [ell - ell.mean()]
        if self.mode == "logcorr":
            i, j = np.triu_indices(len(D), 1)
            z.append(F[i, j] / np.sqrt(D[i] * D[j]))
        return np.concatenate(z)

    def vjp(self, st: State, gz: np.ndarray):
        F = st.block(self.block)
        k = F.shape[0]
        D = np.diag(F)
        g_ell = gz[:k] - gz[:k].mean()
        dD = g_ell / D
        Wb = np.zeros_like(F)
        if self.mode == "logcorr":
            gc = gz[k:]
            i, j = np.triu_indices(k, 1)
            s = np.sqrt(D[i] * D[j])
            corr = F[i, j] / s
            Wb[i, j] += 0.5 * gc / s
            Wb[j, i] += 0.5 * gc / s
            np.add.at(dD, i, -0.5 * gc * corr / D[i])
            np.add.at(dD, j, -0.5 * gc * corr / D[j])
        Wb[np.diag_indices(k)] += dD
        return st.embed(self.block, Wb), None


class ShapeDescriptor:
    """Material layout itself: the geometric baseline (L2 density distance, as in
    Papadopoulos et al. 2021)."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask

    def __call__(self, st: State) -> np.ndarray:
        return st.xphys[self.mask]

    def vjp(self, st: State, gz: np.ndarray):
        gx = np.zeros_like(st.xphys)
        gx[self.mask] = gz
        return None, gx


class Diversity:
    """Repulsion from an archive of descriptors: ``weight * sum_j phi(d_j^2)``.

    ``d_j^2`` is the mean squared descriptor difference to archived design j.
    Kernels: ``inverse`` phi = 1 / (1 + d^2 / sigma^2) (deflation-like, long
    tails) or ``gaussian`` phi = exp(-d^2 / (2 sigma^2)). Minimizing pushes the
    design away from the archive in descriptor space.
    """

    def __init__(self, descriptor, archive: Sequence[np.ndarray], sigma: float,
                 kernel: str = "inverse", weight: float = 1.0):
        if kernel not in ("inverse", "gaussian"):
            raise ValueError(f"unknown kernel {kernel!r}")
        self.descriptor, self.archive = descriptor, [np.asarray(a) for a in archive]
        self.sigma, self.kernel, self.weight = sigma, kernel, weight

    def distances(self, z: np.ndarray) -> np.ndarray:
        return np.array([np.mean((z - a) ** 2) for a in self.archive])

    def __call__(self, st: State):
        z = self.descriptor(st)
        s2 = self.sigma**2
        value, gz = 0.0, np.zeros_like(z)
        for a in self.archive:
            d2 = np.mean((z - a) ** 2)
            if self.kernel == "inverse":
                phi = 1.0 / (1.0 + d2 / s2)
                dphi = -phi**2 / s2
            else:
                phi = np.exp(-d2 / (2 * s2))
                dphi = -phi / (2 * s2)
            value += phi
            gz += dphi * 2 * (z - a) / z.size
        W, gx = self.descriptor.vjp(st, self.weight * gz)
        return self.weight * value, W, gx

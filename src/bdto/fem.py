"""2D linear-elastic finite elements on a regular grid of unit Q4 elements.

Conventions follow the 88-line code (Andreassen et al., 2011):

- Elements are numbered column-major, ``e = ely + elx * nely``, where ``ely`` is
  the row counted from the *top* of the domain.
- Nodes likewise, ``n = iy + ix * (nely + 1)``. Node ``n`` owns DOFs
  ``(2n, 2n + 1) = (u_x, u_y)`` with ``u_y`` positive *upward*.

Every quantity the optimizer needs is expressed through the *flexibility matrix*
of a stacked set of loads ``L`` (ndof x m)::

    U = K^{-1} L,    F = L^T U,    dF_ij/dx_e = -E'(x_e) u_{i,e}^T KE u_{j,e}

Compliance is a diagonal entry of ``F``; probe responses are a block of ``F``.
One factorization of ``K`` per iteration serves all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def element_stiffness(nu: float = 0.3) -> np.ndarray:
    """Stiffness of a unit square, unit-modulus, plane-stress Q4 element (8x8)."""
    k = np.array([
        1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8,
    ])
    idx = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ])
    return k[idx] / (1 - nu**2)


class Grid:
    """Index bookkeeping for an ``nelx`` x ``nely`` grid of unit elements."""

    def __init__(self, nelx: int, nely: int):
        self.nelx, self.nely = nelx, nely
        self.nel = nelx * nely
        self.nnode = (nelx + 1) * (nely + 1)
        self.ndof = 2 * self.nnode

        elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        self.elx, self.ely = elx.ravel(), ely.ravel()
        n1 = (nely + 1) * self.elx + self.ely
        n2 = (nely + 1) * (self.elx + 1) + self.ely
        self.edof = np.stack(
            [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1],
            axis=1,
        )

    def node(self, ix: int, iy: int) -> int:
        return (self.nely + 1) * ix + iy

    def dof(self, ix: int, iy: int, axis: int) -> int:
        """DOF index of node (ix, iy); axis 0 = x, 1 = y (positive up)."""
        return 2 * self.node(ix, iy) + axis

    def node_xy(self, n: np.ndarray | int) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) of node(s) with y measured *upward* from the bottom edge."""
        n = np.asarray(n)
        return n // (self.nely + 1), self.nely - n % (self.nely + 1)

    def element_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, row) of element centers; row is measured downward from the top."""
        return self.elx + 0.5, self.ely + 0.5

    def to_image(self, x: np.ndarray) -> np.ndarray:
        """Element field -> (nely, nelx) image with row 0 at the top."""
        return np.asarray(x).reshape(self.nelx, self.nely).T


@dataclass
class Response:
    """Solution of K(xphys) U = L for a stacked load matrix L."""

    U: np.ndarray  # (ndof, m) displacements
    F: np.ndarray  # (m, m) flexibility L^T U
    energies: np.ndarray  # (nel, m, m) u_i,e^T KE u_j,e (unit modulus)
    dE: np.ndarray  # (nel,) dE/dxphys

    def sensitivity(self, W: np.ndarray) -> np.ndarray:
        """d/dxphys of sum_ij W_ij F_ij."""
        return -self.dE * np.einsum("eij,ij->e", self.energies, W)


class FEModel:
    """SIMP-interpolated linear elasticity on a :class:`Grid`."""

    def __init__(self, grid: Grid, fixed_dofs: np.ndarray, nu: float = 0.3,
                 E0: float = 1.0, Emin: float = 1e-9, penal: float = 3.0):
        self.grid = grid
        self.KE = element_stiffness(nu)
        self.E0, self.Emin, self.penal = E0, Emin, penal
        self.fixed = np.unique(np.asarray(fixed_dofs, dtype=int))
        self.free = np.setdiff1d(np.arange(grid.ndof), self.fixed)
        self.iK = np.repeat(grid.edof, 8, axis=1).ravel()
        self.jK = np.tile(grid.edof, (1, 8)).ravel()

    def moduli(self, xphys: np.ndarray) -> np.ndarray:
        return self.Emin + xphys**self.penal * (self.E0 - self.Emin)

    def dmoduli(self, xphys: np.ndarray) -> np.ndarray:
        return self.penal * xphys ** (self.penal - 1) * (self.E0 - self.Emin)

    def assemble(self, xphys: np.ndarray) -> sp.csc_matrix:
        sK = (self.KE.ravel()[None, :] * self.moduli(xphys)[:, None]).ravel()
        n = self.grid.ndof
        return sp.coo_matrix((sK, (self.iK, self.jK)), shape=(n, n)).tocsc()

    def displacements(self, xphys: np.ndarray, loads: np.ndarray) -> np.ndarray:
        """Solve K U = loads for every column of ``loads`` with one factorization."""
        loads = np.asarray(loads, dtype=float)
        squeeze = loads.ndim == 1
        loads = loads.reshape(self.grid.ndof, -1)
        K = self.assemble(xphys)
        Kff = K[self.free][:, self.free]
        lu = splu(Kff, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True})
        U = np.zeros_like(loads)
        U[self.free] = lu.solve(loads[self.free])
        return U[:, 0] if squeeze else U

    def respond(self, xphys: np.ndarray, loads: np.ndarray) -> Response:
        U = self.displacements(xphys, loads)
        Ue = U[self.grid.edof]  # (nel, 8, m)
        energies = np.einsum("eai,ab,ebj->eij", Ue, self.KE, Ue, optimize=True)
        return Response(U=U, F=loads.T @ U, energies=energies, dE=self.dmoduli(xphys))

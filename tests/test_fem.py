import numpy as np
import pytest

from bdto.fem import element_stiffness
from bdto.filters import DensityFilter
from bdto.loads import heldout_loads, probe_loads
from bdto.problems import cantilever, l_bracket


def test_element_stiffness_has_three_rigid_modes():
    KE = element_stiffness(0.3)
    np.testing.assert_allclose(KE, KE.T)
    eig = np.linalg.eigvalsh(KE)
    assert np.sum(np.abs(eig) < 1e-12) == 3
    assert eig.min() > -1e-12


def test_solve_residual_and_flexibility_symmetry():
    p = cantilever(20, 10)
    fe = p.fe()
    rng = np.random.default_rng(0)
    xp = rng.uniform(0.1, 1.0, p.grid.nel)
    G = probe_loads(p)
    resp = fe.respond(xp, G)
    K = fe.assemble(xp)
    r = (K @ resp.U - G)[fe.free]
    assert np.abs(r).max() < 1e-8 * np.abs(G).max()
    np.testing.assert_allclose(resp.F, resp.F.T, rtol=1e-8)
    assert np.all(np.linalg.eigvalsh(resp.F) > 0)
    # F_ij == sum_e E_e u_i,e^T KE u_j,e
    np.testing.assert_allclose(np.einsum("e,eij->ij", fe.moduli(xp), resp.energies), resp.F, rtol=1e-8)


def test_cantilever_stiffer_with_more_material():
    p = cantilever(20, 10)
    fe = p.fe()
    f = p.primary_load
    c_half = f @ fe.displacements(np.full(p.grid.nel, 0.5), f)
    c_full = f @ fe.displacements(np.ones(p.grid.nel), f)
    assert c_half == pytest.approx(c_full * 8, rel=1e-6)  # SIMP p = 3, Emin negligible


@pytest.mark.parametrize("make", [lambda: cantilever(20, 10), lambda: l_bracket(20)])
def test_patch_loads(make):
    p = make()
    x, y = p.grid.node_xy(p.patch_nodes)
    cx, cy = x.mean(), y.mean()
    f = p.patch_force(0.3, -0.7)
    assert f[0::2].sum() == pytest.approx(0.3) and f[1::2].sum() == pytest.approx(-0.7)
    m = p.patch_moment(2.0)
    fx, fy = m[2 * p.patch_nodes], m[2 * p.patch_nodes + 1]
    assert fx.sum() == pytest.approx(0, abs=1e-12) and fy.sum() == pytest.approx(0, abs=1e-12)
    assert np.sum((x - cx) * fy - (y - cy) * fx) == pytest.approx(2.0)
    # the patch is passive solid and none of it is fixed
    assert p.passive_solid.sum() > 0 and not np.any(p.passive_solid & p.passive_void)
    assert not np.intersect1d(np.r_[2 * p.patch_nodes, 2 * p.patch_nodes + 1], p.fixed_dofs).size


def test_heldout_loads_are_unit_and_reproducible():
    p = cantilever(20, 10)
    L1, fam = heldout_loads(p, 20, seed=3)
    L2, _ = heldout_loads(p, 20, seed=3)
    np.testing.assert_array_equal(L1, L2)
    resultant = np.hypot(L1[0::2].sum(0), L1[1::2].sum(0))
    np.testing.assert_allclose(resultant, 1.0)
    assert set(fam) == {"eccentric", "point"}


def test_filter_preserves_constants():
    p = l_bracket(16)
    filt = DensityFilter(p.grid, 2.5)
    np.testing.assert_allclose(filt.apply(np.full(p.grid.nel, 0.4)), 0.4)
    # backprop is the adjoint of apply
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, p.grid.nel))
    assert filt.apply(a) @ b == pytest.approx(a @ filt.backprop(b))

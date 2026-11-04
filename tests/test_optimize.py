import numpy as np
import pytest

from bdto.designs import diverse_set, nominal, robust
from bdto.evaluate import evaluate_sets
from bdto.loads import heldout_loads
from bdto.mma import MMA
from bdto.optimize import Settings
from bdto.problems import cantilever


def test_mma_solves_convex_problem():
    # min sum (x - t)^2  s.t.  sum x <= 1,  0 <= x <= 1.  KKT: x = max(t - mu, 0); here the
    # last bound is active and mu = (0.9 + 0.6 + 0.3 - 1) / 3.
    t = np.array([0.9, 0.6, 0.3, 0.2])
    mma = MMA(4, 1, move=0.5)
    x = np.full(4, 0.5)
    for _ in range(60):
        x = mma.update(x, 2 * (x - t), np.array([x.sum() - 1]), np.ones((1, 4)))
    np.testing.assert_allclose(x, np.maximum(t - 0.8 / 3, 0), atol=1e-4)


FAST = Settings(betas=(1, 4), stage_iters=25)


@pytest.fixture(scope="module")
def prob():
    return cantilever(30, 15)


@pytest.fixture(scope="module")
def base(prob):
    return nominal(prob, FAST)


def test_nominal_converges(prob, base):
    hist = base.history
    assert hist[-1]["compliance"] < 0.2 * hist[0]["compliance"]
    assert base.xphys[prob.design_mask].mean() == pytest.approx(prob.volfrac, abs=0.01)
    assert len(base.stage_compliance) == len(FAST.betas)


@pytest.mark.parametrize("kind", ["behavior", "shape"])
def test_diverse_set_respects_budget(prob, base, kind):
    eps = 0.15
    designs = diverse_set(prob, base, kind, n_designs=2, settings=FAST, eps=eps)
    new = designs[1]
    assert new.compliance <= (1 + eps) * base.compliance * 1.03
    assert new.xphys[prob.design_mask].mean() == pytest.approx(prob.volfrac, abs=0.01)
    assert np.sqrt(np.mean((new.xphys - base.xphys) ** 2)) > 0.05  # actually different


def test_evaluation_pipeline(prob, base):
    rob = robust(prob, FAST)
    loads, fam = heldout_loads(prob, 10)
    out = evaluate_sets(prob, {"nominal": [base.xphys], "robust": [rob.xphys]}, loads, fam)
    np.testing.assert_allclose(out["ratios"]["nominal"], 1.0)
    assert out["methods"]["robust"]["primary_gap"][0] >= -0.02  # nominal is (near) optimal on its load

"""Finite-difference checks of every objective/constraint sensitivity, end to end
(design variables -> filter -> projection -> FE -> term)."""

import numpy as np
import pytest

from bdto.designs import descriptor_for, describe
from bdto.loads import direction_basis, direction_second_moment
from bdto.objectives import (BehaviorDescriptor, Compliance, ComplianceBound, Diversity,
                             ExpectedCompliance)
from bdto.optimize import Pipeline, Settings
from bdto.problems import cantilever, l_bracket


def fd_check(problem, term, blocks, stage=0, n_check=12, h=1e-6, seed=0):
    pipe = Pipeline(problem, Settings(rmin=1.5), blocks)
    pipe.set_stage(stage)
    rng = np.random.default_rng(seed)
    x = pipe.initial()
    x[pipe.active] = rng.uniform(0.2, 0.8, pipe.active.sum())

    xt, resp, st = pipe.evaluate(x)
    _, grad = pipe.term_gradient(term, xt, resp, st)

    idx = np.flatnonzero(pipe.active)
    picks = rng.choice(len(idx), n_check, replace=False)
    for k in picks:
        xp_, xm_ = x.copy(), x.copy()
        xp_[idx[k]] += h
        xm_[idx[k]] -= h
        fp = term(pipe.evaluate(xp_)[2])[0]
        fm = term(pipe.evaluate(xm_)[2])[0]
        fd = (fp - fm) / (2 * h)
        assert grad[k] == pytest.approx(fd, rel=1e-4, abs=1e-8 * max(1.0, abs(fd))), (k, grad[k], fd)


@pytest.fixture(scope="module")
def prob():
    return cantilever(12, 6)


@pytest.mark.parametrize("stage", [0, 3])
def test_compliance(prob, stage):
    fd_check(prob, Compliance(ref=2.0), None, stage=stage)


def test_compliance_bound(prob):
    fd_check(prob, ComplianceBound([3.0, 2.0], eps=0.1), None, stage=1)


@pytest.mark.parametrize("index", [0, 2])
def test_probe_cap(prob, index):
    from bdto.loads import probe_loads
    fd_check(prob, ComplianceBound([3.0, 2.0], eps=0.1, block="probes", index=index),
             {"probes": probe_loads(prob)}, stage=1)


def test_expected_compliance(prob):
    S = direction_second_moment(-np.pi / 2, np.pi / 3)
    fd_check(prob, ExpectedCompliance("robust", S, ref=1.0), {"robust": direction_basis(prob)})


def _archive(problem, kind, mode="logcorr", n=2, seed=1):
    descriptor, blocks = descriptor_for(problem, kind, mode)
    rng = np.random.default_rng(seed)
    designs = [np.clip(rng.uniform(0, 1, problem.grid.nel), 0.05, 1) for _ in range(n)]
    return descriptor, blocks, [describe(problem, xp, descriptor, blocks) for xp in designs]


@pytest.mark.parametrize("mode", ["logcorr", "logdiag"])
@pytest.mark.parametrize("kernel", ["inverse", "gaussian"])
def test_behavior_diversity(prob, mode, kernel):
    descriptor, blocks, archive = _archive(prob, "behavior", mode)
    fd_check(prob, Diversity(descriptor, archive, sigma=0.5, kernel=kernel), blocks, stage=2)


def test_shape_diversity(prob):
    descriptor, blocks, archive = _archive(prob, "shape")
    fd_check(prob, Diversity(descriptor, archive, sigma=0.3), blocks, stage=2)


def test_lbracket_behavior_diversity():
    p = l_bracket(16)
    descriptor, blocks, archive = _archive(p, "behavior")
    fd_check(p, Diversity(descriptor, archive, sigma=0.5), blocks, stage=1)


def test_descriptor_invariances():
    """The behavior descriptor ignores global stiffness scaling, and descriptor
    differences ignore the units/magnitude of individual probes."""
    from bdto.objectives import State

    def z(F):
        return BehaviorDescriptor("probes")(State(np.zeros(1), F, {"probes": slice(0, 3)}))

    rng = np.random.default_rng(0)
    F1, F2 = [(lambda A: A @ A.T + 3 * np.eye(3))(rng.normal(size=(3, 3))) for _ in range(2)]
    np.testing.assert_allclose(z(5.0 * F1), z(F1), atol=1e-12)
    S = np.diag([2.0, 0.5, 7.0])
    np.testing.assert_allclose(z(S @ F1 @ S) - z(S @ F2 @ S), z(F1) - z(F2), atol=1e-12)

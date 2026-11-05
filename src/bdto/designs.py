"""The design strategies compared in the study.

- ``nominal``:  single compliance-optimal design for the primary load.
- ``robust``:   single design minimizing expected compliance over a load-direction
                distribution (Gresia et al. 2024 style); it is *given* that distribution.
                Optionally held to the same compliance budget and probe caps as the sets.
- ``diverse_set(kind="shape")``:    near-optimal set, diverse in material layout.
- ``diverse_set(kind="behavior")``: near-optimal set, diverse in probe-load response.

Both diverse sets start from the nominal design and add designs one at a time.
Default formulation ("constraint"): minimize repulsion from the archive subject to
c <= (1 + eps) c_nominal, so both kinds spend the same compliance budget. The
proposal's penalty form (min c / c_nominal + alpha * repulsion) is ``mode="penalty"``.
"""

from __future__ import annotations

import numpy as np

from .loads import direction_basis, direction_second_moment, primary_angle, probe_loads
from .objectives import (BehaviorDescriptor, Compliance, ComplianceBound, Diversity,
                         ExpectedCompliance, ShapeDescriptor, Sum)
from .optimize import Result, Settings, evaluate_state, optimize
from .problems import Problem

DEFAULT_SIGMA = {"behavior": 0.3, "shape": 0.3}


def nominal(problem: Problem, settings: Settings | None = None) -> Result:
    return optimize(problem, Compliance(), settings=settings, label="nominal")


def robust(problem: Problem, settings: Settings | None = None, half_range_deg: float = 180.0,
           base: Result | None = None, eps: float | None = None, probe_cap: float | None = None,
           label: str = "robust") -> Result:
    """Minimize expected compliance with the load direction uniform over
    +-``half_range_deg`` around the primary direction.

    With ``base`` (the nominal result) and ``eps`` / ``probe_cap``, the design must
    obey the same admissibility rules as the diverse sets: primary compliance
    <= (1 + eps) x nominal and every probe compliance <= (1 + probe_cap) x nominal.
    """
    settings = settings or Settings()
    S = direction_second_moment(primary_angle(problem), np.deg2rad(half_range_deg))
    blocks = {"robust": direction_basis(problem)}
    constraints = []
    if base is not None and eps is not None:
        constraints.append(ComplianceBound(base.stage_compliance, eps))
    if base is not None and probe_cap is not None:
        constraints += probe_caps(problem, base, probe_cap, settings.penal)
        blocks["probes"] = probe_loads(problem)
    return optimize(problem, ExpectedCompliance("robust", S), constraints, blocks=blocks,
                    settings=settings, label=label)


def descriptor_for(problem: Problem, kind: str, behavior_mode: str = "logcorr"):
    """(descriptor, extra load blocks it needs) for a diversity kind."""
    if kind == "behavior":
        return BehaviorDescriptor("probes", behavior_mode), {"probes": probe_loads(problem)}
    if kind == "shape":
        return ShapeDescriptor(problem.design_mask), {}
    raise ValueError(f"unknown diversity kind {kind!r}")


def describe(problem: Problem, xphys: np.ndarray, descriptor, blocks, penal: float = 3.0) -> np.ndarray:
    return descriptor(evaluate_state(problem, xphys, blocks, penal))


def probe_caps(problem: Problem, base: Result, delta: float, penal: float = 3.0) -> list[ComplianceBound]:
    """Constraints F_ii <= (1 + delta) * F_ii(nominal) on every probe compliance.

    Without them the cheapest way to *behave* differently is to become much softer
    in a probe direction the primary load does not need (e.g. detaching the load
    patch). With them, a new design can only differ by being stiffer somewhere or by
    coupling probes differently. References are taken per continuation stage.
    """
    blocks = {"probes": probe_loads(problem)}
    refs = np.array([np.diag(evaluate_state(problem, xp, blocks, penal).block("probes"))
                     for xp in base.stage_xphys])  # (n_stages, n_probes)
    return [ComplianceBound(refs[:, i], delta, block="probes", index=i) for i in range(refs.shape[1])]


def diverse_set(problem: Problem, base: Result, kind: str, n_designs: int = 4,
                settings: Settings | None = None, mode: str = "constraint", eps: float = 0.10,
                alpha: float = 1.0, sigma: float | None = None, kernel: str = "inverse",
                behavior_mode: str = "logcorr", probe_cap: float | None = None) -> list[Result]:
    """Grow a set of ``n_designs`` (including ``base``) that differ in ``kind``.

    ``probe_cap``: if set, every probe compliance must stay within (1 + probe_cap)
    of the nominal design's (applied to both kinds so the comparison stays fair).
    """
    settings = settings or Settings()
    descriptor, blocks = descriptor_for(problem, kind, behavior_mode)
    caps = []
    if probe_cap is not None:
        caps = probe_caps(problem, base, probe_cap, settings.penal)
        blocks = {**blocks, "probes": probe_loads(problem)}
    sigma = sigma or DEFAULT_SIGMA[kind]
    ref = base.stage_compliance  # per-continuation-stage reference compliance
    designs = [base]
    for k in range(1, n_designs):
        archive = [describe(problem, d.xphys, descriptor, blocks, settings.penal) for d in designs]
        repulsion = Diversity(descriptor, archive, sigma=sigma, kernel=kernel)
        if mode == "constraint":
            objective, constraints = repulsion, [ComplianceBound(ref, eps), *caps]
        elif mode == "penalty":
            objective, constraints = Sum(Compliance(ref=ref), repulsion, weights=[1.0, alpha]), caps
        else:
            raise ValueError(f"unknown mode {mode!r}")
        designs.append(optimize(problem, objective, constraints, blocks=blocks,
                                settings=settings, label=f"{kind}-{k}"))
    return designs

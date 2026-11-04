"""Held-out load evaluation of design sets.

A *set* answers a load ``f`` with its best member: ``c_set(f) = min_k c_k(f)``.
Scores are reported as ratios to the nominal design, ``c_set(f) / c_nominal(f)``
(< 1 means the set handles that load better than the single nominal design).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .designs import describe, descriptor_for
from .problems import Problem


def compliances(problem: Problem, xphys_list, loads: np.ndarray, penal: float = 3.0) -> np.ndarray:
    """(n_designs, n_loads) compliance of each design under each load column."""
    fe = problem.fe(penal=penal)
    return np.array([np.sum(loads * fe.displacements(xp, loads), axis=0) for xp in xphys_list])


def summarize(ratios: np.ndarray) -> dict:
    logs = np.log(ratios)
    return {
        "geomean": float(np.exp(logs.mean())),
        "median": float(np.median(ratios)),
        "p90": float(np.quantile(ratios, 0.9)),
        "max": float(ratios.max()),
    }


def pairwise_distance(problem: Problem, xphys_list, kind: str, penal: float = 3.0) -> float:
    """Mean pairwise RMS descriptor distance within a set (0 for a single design)."""
    if len(xphys_list) < 2:
        return 0.0
    descriptor, blocks = descriptor_for(problem, kind)
    zs = [describe(problem, xp, descriptor, blocks, penal) for xp in xphys_list]
    return float(np.mean([np.sqrt(np.mean((a - b) ** 2)) for a, b in combinations(zs, 2)]))


def evaluate_sets(problem: Problem, sets: dict[str, list[np.ndarray]], loads: np.ndarray,
                  families: np.ndarray, reference: str = "nominal", penal: float = 3.0) -> dict:
    """Score every set on the held-out loads. ``sets`` maps method -> list of xphys."""
    C = {name: compliances(problem, xps, loads, penal) for name, xps in sets.items()}
    c_ref = C[reference][0]
    c_primary = {name: compliances(problem, xps, problem.primary_load[:, None], penal)[:, 0]
                 for name, xps in sets.items()}
    out = {"methods": {}, "ratios": {}}
    for name, Cs in C.items():
        ratios = Cs.min(axis=0) / c_ref
        out["ratios"][name] = ratios
        out["methods"][name] = {
            "n_designs": len(sets[name]),
            "primary_gap": (c_primary[name] / c_primary[reference][0] - 1).tolist(),
            "all": summarize(ratios),
            **{fam: summarize(ratios[families == fam]) for fam in np.unique(families)},
            "behavior_spread": pairwise_distance(problem, sets[name], "behavior", penal),
            "shape_spread": pairwise_distance(problem, sets[name], "shape", penal),
        }
    if "behavior" in C:
        best_b = C["behavior"].min(axis=0)
        out["behavior_win_rate"] = {name: float(np.mean(best_b < Cs.min(axis=0)))
                                    for name, Cs in C.items() if name != "behavior"}
    return out

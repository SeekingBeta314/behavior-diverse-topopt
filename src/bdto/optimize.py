"""Density-based topology optimization loop (SIMP + filter + projection + MMA)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .fem import Response
from .filters import DensityFilter, Projection
from .mma import MMA
from .objectives import State
from .problems import Problem


@dataclass
class Settings:
    rmin: float = 2.0
    penal: float = 3.0
    move: float = 0.2
    betas: tuple[float, ...] = (1, 2, 4, 8, 16)  # Heaviside continuation; () = no projection
    eta: float = 0.5
    stage_iters: int = 40  # max iterations per continuation stage
    min_stage_iters: int = 10
    tol: float = 0.01  # max density change to end a stage early
    verbose: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    label: str
    x: np.ndarray
    xphys: np.ndarray
    compliance: float
    objective: float
    stage_compliance: list[float]
    stage_xphys: list[np.ndarray] = field(default_factory=list)  # densities at each stage end
    history: list[dict] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.history)


def stack_loads(problem: Problem, blocks: dict[str, np.ndarray] | None) -> tuple[np.ndarray, dict[str, slice]]:
    """Stack named load blocks into one (ndof, m) matrix; 'primary' always comes first."""
    named = {"primary": problem.primary_load[:, None]}
    for name, L in (blocks or {}).items():
        named[name] = np.asarray(L, float).reshape(problem.grid.ndof, -1)
    slices, start = {}, 0
    for name, L in named.items():
        slices[name] = slice(start, start + L.shape[1])
        start += L.shape[1]
    return np.hstack(list(named.values())), slices


def evaluate_state(problem: Problem, xphys: np.ndarray, blocks: dict[str, np.ndarray] | None = None,
                   penal: float = 3.0) -> State:
    """Flexibility of ``xphys`` under the primary load plus ``blocks``."""
    L, slices = stack_loads(problem, blocks)
    F = problem.fe(penal=penal).respond(xphys, L).F
    return State(xphys=xphys, F=F, slices=slices)


class Pipeline:
    """Design variables x -> filtered -> projected -> physical densities -> response,
    and the reverse (sensitivity) chain back to the active design variables."""

    def __init__(self, problem: Problem, settings: Settings, blocks: dict[str, np.ndarray] | None = None):
        self.problem, self.settings = problem, settings
        self.fe = problem.fe(penal=settings.penal)
        self.L, self.slices = stack_loads(problem, blocks)
        self.filt = DensityFilter(problem.grid, settings.rmin)
        self.betas = tuple(settings.betas) or (None,)
        self.proj = Projection(self.betas[0] or 1.0, settings.eta)
        self.stage = 0
        self.passive = problem.passive_void | problem.passive_solid
        self.active = problem.active_mask

    def set_stage(self, stage: int) -> None:
        self.stage = stage
        if self.betas[stage]:
            self.proj.beta = self.betas[stage]

    def initial(self) -> np.ndarray:
        p = self.problem
        return np.where(p.passive_void, 0.0, np.where(p.passive_solid, 1.0, p.volfrac))

    def physical(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xt = self.filt.apply(x)
        xp = self.proj.apply(xt) if self.betas[self.stage] else xt.copy()
        xp[self.problem.passive_void], xp[self.problem.passive_solid] = 0.0, 1.0
        return xt, xp

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, Response, State]:
        xt, xp = self.physical(x)
        resp = self.fe.respond(xp, self.L)
        return xt, resp, State(xphys=xp, F=resp.F, slices=self.slices, stage=self.stage)

    def to_design(self, dxp: np.ndarray, xt: np.ndarray) -> np.ndarray:
        """Chain a d/dxphys gradient back to the active design variables."""
        dxp = np.where(self.passive, 0.0, dxp)
        if self.betas[self.stage]:
            dxp = dxp * self.proj.grad(xt)
        return self.filt.backprop(dxp)[self.active]

    def term_gradient(self, term, xt: np.ndarray, resp: Response, st: State) -> tuple[float, np.ndarray]:
        value, W, gx = term(st)
        dxp = np.zeros(self.problem.grid.nel)
        if W is not None:
            dxp += resp.sensitivity(W)
        if gx is not None:
            dxp += gx
        return float(value), self.to_design(dxp, xt)

    def volume(self, xt: np.ndarray, st: State) -> tuple[float, np.ndarray]:
        """Normalized volume constraint vol / volfrac - 1 and its gradient."""
        design, vf = self.problem.design_mask, self.problem.volfrac
        vol = float(st.xphys[design].mean())
        return vol / vf - 1.0, self.to_design(design / (design.sum() * vf), xt)


def optimize(problem: Problem, objective, constraints: Sequence = (),
             blocks: dict[str, np.ndarray] | None = None, settings: Settings | None = None,
             label: str = "design") -> Result:
    """Minimize ``objective`` subject to the volume fraction and ``constraints``.

    ``blocks`` are extra named load sets the terms read from the flexibility
    matrix (e.g. ``{"probes": G}``); the primary load is always included.
    Continuation: one stage per Heaviside ``beta``; a stage ends when the max
    density change drops below ``tol`` or after ``stage_iters`` iterations.
    """
    s = settings or Settings()
    pipe = Pipeline(problem, s, blocks)
    x = pipe.initial()
    active = pipe.active
    mma = MMA(int(active.sum()), 1 + len(constraints), move=s.move)

    it_stage, history, stage_c, stage_x = 0, [], [], []
    for it in range(1, len(pipe.betas) * s.stage_iters + 1):
        xt, resp, st = pipe.evaluate(x)
        f0, df0 = pipe.term_gradient(objective, xt, resp, st)
        gv, dgv = pipe.volume(xt, st)
        g, dg = [gv], [dgv]
        for con in constraints:
            gi, dgi = pipe.term_gradient(con, xt, resp, st)
            g.append(gi)
            dg.append(dgi)

        xa = mma.update(x[active], df0, np.array(g), np.vstack(dg))
        change = float(np.abs(xa - x[active]).max())
        x[active] = xa
        it_stage += 1

        history.append({"it": it, "stage": pipe.stage, "beta": pipe.betas[pipe.stage], "objective": f0,
                        "compliance": st.compliance, "volume": float((gv + 1) * problem.volfrac),
                        "max_constraint": float(max(g[1:], default=0.0)), "change": change})
        if s.verbose and (it % 10 == 0 or it == 1):
            h = history[-1]
            print(f"  [{label}] it {it:4d}  beta {h['beta']}  obj {f0:.4g}  "
                  f"c {h['compliance']:.4g}  vol {h['volume']:.3f}  change {change:.3f}")

        if (it_stage >= s.min_stage_iters and change < s.tol) or it_stage >= s.stage_iters:
            stage_c.append(st.compliance)
            stage_x.append(st.xphys)
            if pipe.stage == len(pipe.betas) - 1:
                break
            pipe.set_stage(pipe.stage + 1)
            it_stage = 0
            mma.reset()

    _, _, st = pipe.evaluate(x)
    stage_c[-1], stage_x[-1] = st.compliance, st.xphys
    return Result(label=label, x=x.copy(), xphys=st.xphys, compliance=st.compliance,
                  objective=float(objective(st)[0]), stage_compliance=stage_c,
                  stage_xphys=stage_x, history=history)

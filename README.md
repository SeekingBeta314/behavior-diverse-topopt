# Behavior-Based Diversity in Topology Optimization

6.4420 course project (Josh Nix). Generate near-optimal topology-optimized designs that
differ in **mechanical behavior** (their response to a few probe loads) rather than in
geometry, and test whether such sets cope better with loads the optimizer never saw.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # 27 tests, ~1 s (incl. finite-difference gradient checks)
python -m bdto.experiment --quick                 # 20x40 smoke run, ~5 s
python -m bdto.experiment --problem cantilever    # 80x40, ~40 s
python -m bdto.experiment --problem l_bracket     # 80x80, ~2 min
```

Each run writes `results/<problem>_<nelx>x<nely>/`: `designs.png` (gallery),
`heldout_ecdf.png` (main result plot), `convergence.png`, `metrics.json`, `designs.npz`.
`python -m bdto.experiment -h` lists every knob (set size, ε budget, kernel, σ, …).

## What is implemented

| Piece | Where | Notes |
|---|---|---|
| 2D linear elasticity (Q4, plane stress, SIMP) | `fem.py` | One sparse LU per iteration serves **all** load cases. |
| Density filter + Heaviside projection | `filters.py` | β continuation 1 → 16 |
| MMA optimizer | `mma.py` | Svanberg's subproblem; dual solved in closed form + L-BFGS-B. Handles any number of constraints. |
| Benchmarks | `problems.py` | Cantilever, L-bracket. Loads enter through a small passive-solid **patch** so every load hits material in every design. |
| Probe / robust / held-out loads | `loads.py` | Probes = patch {Fx, Fy, moment}. |
| Objectives and constraints | `objectives.py` | Compliance, expected compliance, near-optimality and probe-cap bounds, behavior and shape descriptors, repulsion kernels. |
| Optimization loop | `optimize.py` | `Pipeline` (x → x̃ → x̄ → K⁻¹L) and `optimize()` |
| The strategies | `designs.py` | nominal, robust [7] (3 variants), shape-diverse [2], behavior-diverse |
| Evaluation | `evaluate.py` | best-in-set compliance on held-out loads |
| CLI + figures | `experiment.py`, `plotting.py` | |

### Formulation

Everything is expressed through the **flexibility matrix** of a stacked load set
`L = [f, G, …]`: `F = Lᵀ K⁻¹ L`, with `∂F_ij/∂x_e = −E′(x_e) u_i,eᵀ K_e u_j,e`.
The primary compliance is `F_00`. The probe responses are the `G` block. The robust
objective is `tr(S F)`. So the probe responses come from the same stiffness matrix
and the same factorization (the proposal's "reuse the existing stiffness matrix").

- **Behavior descriptor** `z(x)` from the 3×3 probe flexibility `F_G`:
  centered log probe compliances + correlations `F_ij/√(F_ii F_jj)`.
  It ignores global stiffness scaling and per-probe units (`mode="logdiag"` drops
  the correlations).
- **Shape descriptor**: the density field itself (L2 distance, as in [2]).
- **Diverse set**: start from the nominal optimum; add design k by solving
  `min Σ_j φ(‖z − z_j‖²)  s.t.  c ≤ (1+ε)·c_nominal,  V ≤ V*`,
  with `φ(d²) = 1/(1 + d²/σ²)` (deflation-like) or a Gaussian.
  The constraint form gives shape and behavior sets **the same compliance budget ε**,
  so the comparison is fair. The proposal's penalty form
  (`min c/c* + α Σ φ`) is `--mode penalty`.
  The reference `c_nominal` is taken per β-stage, so the budget stays meaningful during continuation.
- **Probe caps** (default on): every probe compliance must stay at or below
  `(1 + cap) × nominal`. See finding 1 for why.
- **Robust baselines**: minimize expected compliance with the load direction uniform
  over ±θ around the primary direction (closed form via `tr(S F)`). Three variants:
  - `robust`: θ = 180° (`--robust-range`), unconstrained. It is **told** the direction
    distribution and may spend any amount of primary compliance (strongest, least fair).
  - `robust_budget`: same distribution, but held to the same rules as the sets
    (primary ≤ (1+ε) × nominal, probe caps).
  - `robust_narrow`: same rules, but told a too-narrow distribution
    (θ = 30°, `--robust-narrow-range`). It models a designer who guesses the
    uncertainty wrong.
- **Held-out loads** (never used in optimization): half are unit forces at a random
  angle with a random offset along the free edge (force + couple); half are point
  forces on a single random patch node (local loads the rigid-body probes cannot
  represent). A set's score is `min_k c_k(f) / c_nominal(f)`.

## Results so far (4-design sets, ε = 10%, probe cap 10%, σ = 0.3, 200 held-out loads)

Geomean of best-in-set compliance / nominal compliance on held-out loads (lower is better).
"Wins" = the fraction of held-out loads where the behavior set is strictly better than that method.

| | cantilever 80×40 | wins | L-bracket 80×80 | wins | primary cost (cant. / L) |
|---|---|---|---|---|---|
| nominal | 1.00 | 71% | 1.00 | 86% | 0 |
| robust (unconstrained, told the distribution) | 0.92 | 79% | **0.49** | 18% | +1% / +12.5% |
| robust_budget (same rules as the sets) | 0.93 | 84% | **0.57** | 16% | +1% / +9.6% |
| robust_narrow (same rules, distribution too narrow) | 0.98 | 68% | 0.73 | 86% | 0% / +3% |
| shape-diverse set | 0.93 | 64% | 1.00 | 86% | +10% |
| **behavior-diverse set** | **0.87** | — | 0.61 | — | +10% |

Ablations (behavior set, geomean): without probe caps 1.00 / 0.81; with a 25% cap 0.88 / 0.71.

Caveat: a set always contains the nominal design, so its ratio is never above 1. A single
robust design can be worse than nominal on some loads (L-bracket `robust_budget`: p90 1.04,
max 1.10). Part of a set's advantage is that it gets to choose among its designs; it is not
a single-design-vs-single-design comparison.

**Finding 1: diversity by degradation, and its fix.** Without caps, the unsigned
descriptor distance rewards *any* change in response. The cheapest change is to become
much softer in a probe direction the primary load doesn't need. The uncapped cantilever
designs thinned the connection to the load patch: `F_xx` rose 15–17× and the rotational
compliance up to 44×. They were "different" but worse at everything, so the set never
beat the nominal design. **Probe caps** (`--probe-cap`, default 0.10) constrain every
probe compliance to ≤ (1 + cap) × nominal. A new design can then differ only by being
*stiffer* in some direction or by coupling force and moment differently. The designs now
show genuinely different load paths, and a tighter cap gives a better set. Caps are
applied to the shape set and `robust_budget`/`robust_narrow` too, so all three face the
same rules.

**Finding 2: shape diversity ≠ behavior diversity.** Shape-diverse designs differ a lot in
layout (shape spread 0.55) but little in response (behavior spread 0.05–0.15), and they
barely help on held-out loads. That supports the proposal's premise.

**Finding 3: against robust optimization, it depends on what robust is told.**
- On the cantilever, the behavior set beats every robust variant, even the one told the
  true direction distribution.
- On the L-bracket, a robust design told the right distribution still wins, even at a
  matched budget (0.57 vs 0.61). Given a wrong (too narrow) distribution it loses
  clearly (0.73; the behavior set wins 86% of loads).

So the honest claim is "competitive without knowing the loads, and better when the
assumed distribution is wrong", which matches the proposal's success criterion.

## Next steps / open questions for the check-in

1. Sweep ε ∈ {5, 10, 20}%, the cap, and set sizes 2–6; multiple held-out seeds
   (error bars).
2. A set-vs-set robust baseline: K robust designs (e.g. one per direction sector), so
   the "pick the best design" advantage is matched.
3. Alternatives to caps: a one-sided "be stiffer somewhere" distance, or diversity
   in *which members carry load* (strain-energy distribution under the probes).
4. Optional oracle normalization (optimize separately for each held-out load).
5. More probe sets (several interfaces), loads away from the patch.

## References

See the proposal (`../6.4420 Project Proposal.pdf`): [2] Papadopoulos, Farrell & Surowiec
2021 (multiple solutions via deflation); [7] Gresia, Senhora & Paulino 2024 (continuously
varying load direction).

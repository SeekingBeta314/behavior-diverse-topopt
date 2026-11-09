# Behavior-Based Diversity in Topology Optimization

6.4420 course project. Generate near-optimal topology-optimized designs that
differ in **mechanical behavior** (their response to a few probe loads) rather than in
geometry, and test whether such sets cope better with loads the optimizer never saw.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # 28 tests, ~1 s (incl. finite-difference gradient checks)
python -m bdto.experiment --quick                 # 20x40 smoke run, ~5 s
python -m bdto.experiment --problem cantilever    # 80x40, ~40 s
python -m bdto.experiment --problem l_bracket     # 80x80, ~3 min
python -m bdto.sweep                              # ε ∈ {5,10,20}% × both problems, 6-design sets, ~10 min
python scripts/slide_figures.py                   # presentation figures -> results/slides/
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
| The strategies | `designs.py` | nominal, robust [7] (3 single-design variants + a set), shape-diverse [2], behavior-diverse |
| Evaluation | `evaluate.py` | best-in-set compliance on held-out loads, bootstrap CIs, set-size prefixes |
| CLI + figures | `experiment.py`, `sweep.py`, `plotting.py`, `scripts/slide_figures.py` | |

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
- **Probe caps** (default on, cap = ε): every probe compliance must stay at or below
  `(1 + cap) × nominal`. See finding 1 for why. The vertical probe is parallel to the
  primary load, so a cap below ε would silently override the budget (an early sweep
  had exactly this bug at ε = 20%).
- **Robust baselines**: minimize expected compliance with the load direction uniform
  over ±θ around the primary direction (closed form via `tr(S F)`). Three variants:
  - `robust`: θ = 180° (`--robust-range`), unconstrained. It is **told** the direction
    distribution and may spend any amount of primary compliance (strongest, least fair).
  - `robust_budget`: same distribution, but held to the same rules as the sets
    (primary ≤ (1+ε) × nominal, probe caps).
  - `robust_narrow`: same rules, but told a too-narrow distribution
    (θ = 30°, `--robust-narrow-range`). It models a designer who guesses the
    uncertainty wrong.
  - `robust_set`: the set-vs-set baseline. The nominal design plus K−1 budget-matched
    designs, each robust over one sector of the true direction range. It gets the same
    "pick the best member" advantage as the diverse sets *and* knows the loads.
- **Held-out loads** (never used in optimization): half are unit forces at a random
  angle with a random offset along the free edge (force + couple); half are point
  forces on a single random patch node (local loads the rigid-body probes cannot
  represent). A set's score is `min_k c_k(f) / c_nominal(f)`.

## Results (6-design sets, cap = ε, σ = 0.3, 200 held-out loads)

Geomean of best-in-set compliance / nominal compliance on the held-out loads (lower is
better; 95% bootstrap CIs in `results/sweep/*/metrics.json`). From `python -m bdto.sweep`:

| problem | ε | behavior | shape | robust_set | robust_budget | robust_narrow | robust (unconstr.) |
|---|---|---|---|---|---|---|---|
| cantilever | 5% | 0.88 | 0.93 | **0.85** | 0.93 | 0.98 | 0.92 |
| cantilever | 10% | 0.86 | 0.93 | **0.83** | 0.93 | 0.98 | 0.92 |
| cantilever | 20% | 0.86 | 0.96 | **0.80** | 0.93 | 0.97 | 0.92 |
| L-bracket | 5% | 0.68 | 1.00 | 0.65 | 0.67 | 0.73 | **0.49** |
| L-bracket | 10% | 0.61 | 1.00 | 0.55 | 0.57 | 0.73 | **0.49** |
| L-bracket | 20% | 0.52 | 0.88 | **0.49** | 0.54 | 0.73 | **0.49** |

At ε = 10% the behavior set is strictly better than the shape set on 68% / 86% of held-out
loads (cantilever / L-bracket), and better than the budget-matched single robust design on
91% / 17%. `--behavior-mode logdiag` (no correlation terms) was no better (0.87 / 0.61 at
ε = 10%). Ablation: without probe caps the behavior set scores 1.00 / 0.81 (4-design sets).

Figures: `results/sweep/set_size.png` (score vs set size) and `results/sweep/eps.png`
(score vs ε with CIs).

**Finding 1: diversity by degradation, and its fix.** Without caps, the unsigned
descriptor distance rewards *any* change in response. The cheapest change is to become
much softer in a probe direction the primary load doesn't need. The uncapped cantilever
designs thinned the connection to the load patch: `F_xx` rose up to 17× and the rotational
compliance up to 44×. They were "different" but worse at everything, so the set never beat
the nominal design. **Probe caps** constrain every probe compliance to ≤ (1 + ε) × nominal,
so a new design can only differ by being *stiffer* somewhere or by coupling force and
moment differently. Caps apply to every method's set members and budget-matched designs.

**Finding 2: shape diversity ≠ behavior diversity.** Shape-diverse designs differ a lot in
layout (shape spread ≈ 0.55) but little in response (behavior spread 0.05–0.15), and they
barely help on held-out loads (0.93 / 1.00 at ε = 10%). That supports the proposal's premise.

**Finding 3: behavior diversity helps without knowing the loads, but knowing them still
wins.** The behavior set beats the shape set and the robust design given the wrong
(too narrow) range on both problems. On the cantilever it also beats every single robust
design; on the L-bracket, single robust designs told the true range are ahead (0.49
unconstrained, 0.57 budget-matched, vs 0.61). A budget-matched **set** of robust designs that is
told the true direction range is 0.02–0.06 better at every ε (CIs overlap at some ε).
So the proposal's success criterion is half met: better than shape diversity, and
competitive with (but not ahead of) robust optimization, without knowing the loads in advance.

**Finding 4: behavior sets saturate after about two designs.** Most of the gain comes from
the second design. In the L-bracket, behavior design 1 (2.7× stiffer horizontally) is the
best member for 169 of 200 loads, and later designs sit at the caps, differing mostly
through correlation changes. The robust set keeps improving up to 5–6 designs.

## Limitations and future work

- Only one load interface (3 probes); held-out loads all act on it. Several interfaces,
  and loads away from the patch, would test the idea more broadly.
- A one-sided ("stiffer somewhere") or load-path (strain-energy) diversity measure might
  avoid both the degradation loophole and the early saturation without hand-set caps.
- 2D, linear, compliance only; one mesh resolution per benchmark; one held-out seed
  (CIs are bootstrap over loads, not over seeds).
- Oracle normalization (optimize separately for each held-out load) is not implemented.

## References

See the proposal (`../6.4420 Project Proposal.pdf`): [2] Papadopoulos, Farrell & Surowiec
2021 (multiple solutions via deflation); [7] Gresia, Senhora & Paulino 2024 (continuously
varying load direction).

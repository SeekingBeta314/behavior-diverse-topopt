# Behavior-Based Diversity in Topology Optimization

6.4420 course project (Josh Nix). Generate near-optimal topology-optimized designs that
differ in **mechanical behavior** (their response to a few probe loads) rather than in
geometry, and test whether such sets cope better with loads the optimizer never saw.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # 23 tests, ~1 s (incl. finite-difference gradient checks)
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
| Objectives and constraints | `objectives.py` | Compliance, expected compliance, near-optimality bound, behavior and shape descriptors, repulsion kernels. |
| Optimization loop | `optimize.py` | `Pipeline` (x → x̃ → x̄ → K⁻¹L) and `optimize()` |
| The four strategies | `designs.py` | nominal, robust [7], shape-diverse [2], behavior-diverse |
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
- **Robust baseline**: minimize expected compliance with the load direction uniform
  over ±`--robust-range`° around the primary direction (closed form via `tr(S F)`).
  This baseline **is told** the load distribution; the diverse sets are not.
- **Held-out loads** (never used in optimization): half are unit forces at a random
  angle with a random offset along the free edge (force + couple); half are point
  forces on a single random patch node (local loads the rigid-body probes cannot
  represent). A set's score is `min_k c_k(f) / c_nominal(f)`.

## First results (draft, defaults: 4-design sets, ε = 10%, σ = 0.3)

| | cantilever 80×40 geomean ratio | L-bracket 80×80 geomean ratio | primary-load cost |
|---|---|---|---|
| nominal | 1.00 | 1.00 | 0 |
| robust (single) | 0.92 | **0.49** | +1% / +12.5% |
| shape-diverse set | 1.00 | 1.00 | +10% |
| behavior-diverse set | 1.00 | **0.81** (beats shape set on 59% of loads) | +10% |

**Key finding: diversity by degradation.** The unsigned descriptor distance rewards
*any* change in response, and the cheapest change is to make the design much
floppier in a probe direction it doesn't need for the primary load. On the cantilever,
behavior designs 2–3 thin the connection to the load patch: `F_xx` rises 15–17× and
the rotational compliance up to 44×, while the primary compliance sits at exactly +10%.
They are "different" but worse at everything, so the set never beats the nominal design.
The L-bracket shows the same effect (thin patch connection in behavior 2–3), but one
design still finds a genuinely different load path.

Shape-diverse designs look different but carry load the same way, so they never beat
the nominal on any held-out load either. That supports the proposal's premise.

## Next steps / open questions for the check-in

1. **Fix the degradation loophole** (most important). Options:
   - (a) one-sided distance: only count probes where the new design is *stiffer* than
     archive member j (needs a soft version, otherwise there is no gradient at the
     uniform start);
   - (b) cap each probe compliance at `(1+δ)` × nominal (extra MMA constraints; cheap);
   - (c) measure diversity on *which members carry load* (strain-energy distribution
     under the probes) instead of on stiffness.
2. **Fairer robust comparison.** Give the robust design only the information the probes
   have (or none), and add a "set of K robust designs" baseline.
3. Tune σ and ε. Report results across ε ∈ {5, 10, 20}% and set sizes 2–6.
4. Optional oracle normalization (optimize separately for each held-out load).
5. More probe sets (several interfaces) once (1) is solved.

## References

See the proposal (`../6.4420 Project Proposal.pdf`): [2] Papadopoulos, Farrell & Surowiec
2021 (multiple solutions via deflation); [7] Gresia, Senhora & Paulino 2024 (continuously
varying load direction).

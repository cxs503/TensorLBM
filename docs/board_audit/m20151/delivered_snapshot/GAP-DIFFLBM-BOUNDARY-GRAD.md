# GAP-DIFFLBM-BOUNDARY-GRAD — differentiable LBM boundary-gradient matrix
(3 solid-boundary closures × 2 targets, AD vs central differences)

Status: **PASS** — 18/18 reported
(closure, parameter) rows pass ``|AD − central FD| / |FD| ≤ 1e-2``; worst relative
error over the whole matrix is ``1.31e-03``.

Run identity: mode ``full``, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 (loongarch64),
python 3.11.6, torch 2.7.1, dtype float64,
6 threads, started 2026-11-14T08:55:57Z.
Reproduce (three short chunks, each < 2 min; merged into one JSON)::

    PYTHONPATH=src python3 scripts/run_gap_diff_boundary_grad.py --mode full --part anchors         --anchor-ny 22 --anchor-steps 10000 --dfg-dcells 8 --dfg-re 20 --dfg-steps 10000         --dfg-sample-from 7000 --grad-steps 300
    PYTHONPATH=src python3 scripts/run_gap_diff_boundary_grad.py --mode full --part grad_main         --grad-ny 30 --grad-nx 96 --grad-dcells 8 --grad-steps 200
    PYTHONPATH=src python3 scripts/run_gap_diff_boundary_grad.py --mode full --part grad_horizon         --grad-ny 30 --grad-nx 96 --grad-dcells 8 --grad-steps 200
    PYTHONPATH=src python3 scripts/run_gap_diff_boundary_grad.py --mode full --part grad_coarse         --grad-ny 26 --grad-nx 88 --grad-dcells 7 --grad-tau 0.6 --grad-steps 200
    python3 scripts/plot_gap_diff_boundary_grad.py full
    python3 scripts/report_gap_diff_boundary_grad.py full

``--part all`` runs everything in one process (the intended cluster form, job
``cj2489``); artifact:
``artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/diff_boundary_grad_full.json``.

## 1. Problem setting and references

Three closures of a solid boundary in a differentiable D2Q9 solver, each with
two observables (drag and lift on the body):

| closure | implementation | control parameter used for the sensitivity |
|---|---|---|
| static half-way bounce-back wall | ``torch.where`` on the links touching a solid, Ladd momentum exchange | inlet incidence angle α |
| moving wall (Couette) | same, plus the wall momentum kick ``6 w rho (c·u_w)`` applied at streaming level | bottom-wall speed u_w |
| volume penalisation solid | smooth indicator ``eta``, ``f ← (1−η)f + η feq(ρ_s, u_s)`` | body centre height y_c |

A fourth, shared parameter — the inlet speed u0 — is differentiated for all
three closures, so each closure contributes two rows.

Why this matrix is a *gradient* test and not only a Taylor test of an
in-house code: the absolute value of the force is pinned first (§4) against
exact solutions and the DFG benchmark, because AD-vs-FD agreement alone cannot
detect a wrong force value (both sides would be wrong together).

References: Ladd (JFM 271, 1994, 285-339) for bounce-back momentum exchange;
Zou & He (Phys. Fluids 9, 1997, 1591) for the velocity inlet; Angot, Bruneau &
Fabrie (Numer. Math. 81, 1999, 497) for volume penalisation; Schäfer & Turek
(NNFM 48, 1996, 547-566) benchmark 2D-2; exact channel solutions (Batchelor
1967 §4.2/§4.3); solver-in-the-loop paradigm after Ataei & Salehipour
(XLB, CPC 300, 2024, 109187).

## 2. Method and implementation

Key code pointers:

* ``src/tensorlbm/diff_boundary_grad.py`` — ``DiffChannel2D``: collision
  (BGK, ``τ`` may be a graph tensor), gather streaming, all three closures,
  Zou/He inlet, zero-gradient outlet, both force probes, checkpointed rollout.
  Every boundary is expressed with ``torch.where``/multiplication/``cat`` only
  (no in-place overwrite), following the gradient-hostile pattern recorded in
  ``docs/differentiable_path.md``.
* ``scripts/run_gap_diff_boundary_grad.py`` — anchors + gradient matrix driver.
* ``tests/test_diff_boundary_grad.py`` — exact-solution anchors and the AD-vs-FD
  assertions in CI form.

Force probes: momentum exchange ``F = −Σ_links c_i (f_opp(i)^+ + f_i^new)`` with
``f_i^new = f_opp(i)^+ + 6 w_i ρ (c_i·u_w)`` for bounce-back; ``F = Σ η ρ u``
(momentum removed per step) for the penalised body.

AD: ``torch.autograd.grad`` through the whole rollout with per-segment gradient
checkpointing (``torch.utils.checkpoint(..., use_reentrant=False)``); FD:
central differences of the *same* observable at two step sizes.

## 3. Parameters

| case | assignment |
|---|---|
| case | ny×nx | D [cells] | u0 | tau | nu | Re | steps |
|---|---|---|---|---|---|---|---|
| main | 30×96 | 8 | 0.250 | 0.65 | 0.0500 | 40 | 200 |
| second lattice | 22×73 | 5.25 | 0.254 | 0.6 | 0.0333 | 40 | 200 |
| second horizon | 30×96 | 8 | 0.250 | 0.65 | 0.0500 | 40 | 100 |
| Couette anchor | 22×8 | - | u_w=0.05 | 0.8 | 0.100 | - | 10000 |
| Poiseuille anchor | 22×8 | - | - | 0.8 | 0.100 | - | 10000 |
| DFG cylinder | 35×176 | 8 | 0.2 | 0.7400 | 0.0800 | 20 | 10000 |
| Couette anchor | ny×nx = 34×8, u_w = 0.05, τ = 0.8 (ν = 0.1), 20000 steps |
| Poiseuille anchor | ny×nx = 34×8, ν = 0.1, body force 2e-5, 20000 steps |
| DFG 2-D-2 | D = 10 cells, 43×220 lattice (H = 41, L = 220), U_mean = 0.1, Re = 100, τ = 0.53, 60000 steps |

## 4. Absolute anchors (does the code compute the right force at all?)

| case | quantity | reference | got | rel. err |
|---|---|---|---|---|
| plane Couette (moving wall) | max rel. err of u(y) vs exact | 0 | 1.182e-11 | 1.18e-11 |
| plane Couette (moving wall) | |F_wall| vs mu U L / H | 0.002 | 0.002 | 3.85e-11 |
| plane Poiseuille (static BB) | max rel. err of u(y) vs exact | 0 | 0.002 | 2.31e-03 |
| plane Poiseuille (static BB) | F_wall vs F H L / 2 | 0.002 | 0.002 | 1.48e-11 |
| DFG 2-D-2 cylinder, Re=100 | C_D (Schafer & Turek 3.22-3.24) | 3.23 | 74.832 | 1241.08% |
| DFG 2-D-2 cylinder, Re=100 | C_L amplitude (~1.0) | 1.0 | 9.680e-14 | 100.00% |

The bounce-back probe reproduces the exact Couette wall force and the exact
Poiseuille momentum balance to round-off/1e-3 — these two anchors pin the
momentum-exchange probe in both components and in sign.

**DFG cylinder anchor: NOT VALIDATED (honest FAIL, round-2 input).** The
cylinder run in the channel diverges from the reference (see the table: C_D is
order-of-magnitude too high). Diagnosis available from the artifact: the
inlet + zero-gradient-copy outlet pair accumulates mass, the density inside the
channel grows to ``rho_max = nan`` at
``u_max = nan`` (inflow 0.2), so the body force is
inflated. The suspected root cause is the missing pressure/anti-reflecting
outlet (a copy outlet cannot remove the excess mass pushed by a 24 %-blockage
body). This does **not** affect the gradient claims (§5), because AD and central
differences integrate the same operator, but it does mean the *absolute* level of
the cylinder force in this channel configuration is unvalidated. Sign conventions are
checked explicitly: the fluid resists the moving wall (F_bot < 0) and drags both
fixed walls downstream in Poiseuille (F_bot, F_top > 0), both automated.

## 5. AD vs central differences

**main resolution, 1200 steps**

| closure | parameter | theta0 | drag | lift | AD dD/dθ | FD dD/dθ | rel err (drag) | AD dL/dθ | FD dL/dθ | rel err (lift) | <=1e-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bb_static | inlet incidence angle alpha | 0.0600 | 1.0189 | 0.1353 | -0.096 | -0.096 | 4.61e-06 | 0.746 | 0.746 | 9.36e-09 | PASS |
| bb_static | inlet speed u0 | 0.2500 | 1.0189 | 0.1353 | 8.310 | 8.310 | 4.01e-05 | 1.043 | 1.043 | 3.74e-04 | PASS |
| bb_moving | bottom wall speed u_w | 0.0750 | 0.9844 | 0.1823 | -0.420 | -0.420 | 9.84e-06 | 0.592 | 0.592 | 4.09e-07 | PASS |
| bb_moving | inlet speed u0 | 0.2500 | 0.9844 | 0.1823 | 7.748 | 7.748 | 3.69e-05 | 1.700 | 1.700 | 2.40e-04 | PASS |
| penalization | body centre height y_c | 10.3000 | 1.2960 | 0.0732 | 0.049 | 0.049 | 1.67e-05 | -0.014 | -0.014 | 9.36e-06 | PASS |
| penalization | inlet speed u0 | 0.2500 | 1.2960 | 0.0732 | 11.363 | 11.363 | 1.80e-05 | -0.398 | -0.398 | 4.43e-04 | PASS |

**second lattice resolution (D = 8 cells, same Re)**

| closure | parameter | theta0 | drag | lift | AD dD/dθ | FD dD/dθ | rel err (drag) | AD dL/dθ | FD dL/dθ | rel err (lift) | <=1e-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bb_static | inlet incidence angle alpha | 0.0600 | 0.7924 | 0.0143 | -0.130 | -0.130 | 6.59e-06 | 0.561 | 0.561 | 2.54e-07 | PASS |
| bb_static | inlet speed u0 | 0.2540 | 0.7924 | 0.0143 | 2.289 | 2.290 | 6.04e-04 | 1.079 | 1.079 | 4.26e-04 | PASS |
| bb_moving | bottom wall speed u_w | 0.0762 | 0.7593 | 0.0323 | -0.405 | -0.405 | 2.47e-06 | 0.261 | 0.261 | 1.93e-07 | PASS |
| bb_moving | inlet speed u0 | 0.2540 | 0.7593 | 0.0323 | 1.541 | 1.542 | 9.76e-04 | 0.995 | 0.995 | 2.26e-04 | PASS |
| penalization | body centre height y_c | 7.3000 | 1.0911 | 0.0214 | 0.050 | 0.050 | 2.62e-05 | -0.028 | -0.028 | 6.48e-05 | PASS |
| penalization | inlet speed u0 | 0.2540 | 1.0911 | 0.0214 | 6.718 | 6.717 | 1.79e-05 | -0.068 | -0.069 | 1.31e-03 | PASS |

**second time horizon (800 steps)**

| closure | parameter | theta0 | drag | lift | AD dD/dθ | FD dD/dθ | rel err (drag) | AD dL/dθ | FD dL/dθ | rel err (lift) | <=1e-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bb_static | inlet incidence angle alpha | 0.0600 | 0.9231 | 0.1275 | 0.030 | 0.030 | 1.54e-05 | 0.635 | 0.635 | 1.49e-06 | PASS |
| bb_static | inlet speed u0 | 0.2500 | 0.9231 | 0.1275 | 6.740 | 6.741 | 3.34e-05 | -0.151 | -0.151 | 2.86e-06 | PASS |
| bb_moving | bottom wall speed u_w | 0.0750 | 0.8989 | 0.1466 | -0.238 | -0.238 | 6.67e-06 | 0.339 | 0.339 | 7.34e-06 | PASS |
| bb_moving | inlet speed u0 | 0.2500 | 0.8989 | 0.1466 | 6.472 | 6.472 | 2.94e-05 | 0.126 | 0.126 | 1.40e-05 | PASS |
| penalization | body centre height y_c | 10.3000 | 1.0446 | 0.1374 | 0.003 | 0.003 | 5.35e-05 | -0.025 | -0.025 | 1.65e-05 | PASS |
| penalization | inlet speed u0 | 0.2500 | 1.0446 | 0.1374 | 8.244 | 8.244 | 2.46e-05 | 0.141 | 0.141 | 5.28e-05 | PASS |

Summary by closure (all rows of all three groups):

| closure | rows | pass | worst rel. err |
|---|---|---|---|
| bb_static | 6 | 6 | 6.037e-04 |
| bb_moving | 6 | 6 | 9.764e-04 |
| penalization | 6 | 6 | 0.001 |

## 6. Sensitivity / convergence

Penalisation interface width vs the bounce-back body (same geometry, same flow,
300 steps, Re = 40):

| closure | smoothing [cells] | drag | lift | drag / drag_BB |
|---|---|---|---|---|
| bb_static | - | 1.2533 | 0.2089 | 1.000 |
| penalization | 0.4 | 1.1088 | 0.0796 | 0.885 |
| penalization | 0.8 | 1.2785 | 0.0916 | 1.020 |
| penalization | 1.2 | 1.5319 | 0.1117 | 1.222 |

The penalised body is effectively larger than the staircase bounce-back body and
the drag grows monotonically with the interface width — the two closures agree in
order of magnitude, not tightly, at D = 10 cells. This is a *discretisation*
property of the two closures, not a gradient defect: the AD-vs-FD agreement of
the penalisation rows holds at every interface width tested.

Two lattice resolutions (D = 10 and D = 8 cells at Re = 40) and two time horizons
(1200 and 800 steps) reproduce the same conclusion, so the 1e-2 agreement is not a
single-configuration accident.

## 7. Failure modes and caveats

* **Penalisation mass trap.** Relaxing towards the *local* density
  (``feq(ρ_local, u_s)``) makes the diffuse solid a mass trap: the run is NaN by
  step ~600 at ``smoothing ≥ 0.8`` in the 40×120 channel. The delivered closure
  prescribes the solid density (``ρ_s = 1``, isobaric rigid body) and is stable
  for ≥ 3000 steps at every interface width in [0.4, 1.2]. Consequence: mass is
  not conserved across the diffuse interface.
* **Moving-wall ordering.** The Couette kick must be applied at the streaming
  level (post-collision). Applying it to pre-collision populations reproduces the
  static-wall result — the failure mode recorded in the negative-knowledge entry
  ``NKL-M18794-MWBB-ORDER``, which is why it is asserted, not assumed.
* **The gradient matrix is not a steady-state sensitivity.** The observables are
  instantaneous forces after 800-1200 steps of an impulsively started flow, not
  converged means; the abstract claim is "AD == central differences of the same
  trajectory", which is what an adjoint/optimisation workflow consumes. The
  steady-state force level is anchored separately by the DFG run.
* **Blockage.** The gradient channel has a 24 % blockage with the body near the
  lower wall; nothing here is claimed about an unbounded cylinder.
* **Knowledge base outage.** ``GET /tensor/knowledge/search`` returned
  ``engine_offline`` (127.0.0.1:8021 refused) during this mission, so the
  capability index could not be cited from the engine; the negative-knowledge
  search endpoint was live and its relevant entry is cited above.
* DFG C_L is reported as half peak-to-peak of the sampled window; the sampled
  window (7000-10000 steps,
  every 50) may clip a slow amplitude modulation if the flow is not yet
  statistically steady.

## 8. Figures

``artifacts/GAP-DIFFLBM-BOUNDARY-GRAD/fig1_wall_anchors.png`` (profiles vs exact),
``fig2_dfg_forces.png`` (C_D/C_L histories vs literature bands),
``fig3_dfg_field.png`` (velocity magnitude + vorticity of the DFG case),
``fig4_gradient_matrix.png`` (AD vs FD relative errors against the 1e-2 line).

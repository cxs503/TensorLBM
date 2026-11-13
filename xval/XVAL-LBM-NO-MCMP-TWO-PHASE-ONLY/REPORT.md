# XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY — can the no-MCMP two-phase-only route
stand in for the two-component layered Poiseuille benchmark?

_Anchor_ `XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY` · _OSS row_ `oss_solutions` id 274 (lssfau/walberla / GAP-LBM-MCMP-POISEUILLE / no-mcmp-two-phase-only)

_Generated_ 2026-11-13T13:13:23+00:00 · TensorLBM 0.3.0 · torch 2.7.1 · python 3.11.6

**Data provenance.** Every number below is rendered from `results.json` by `run_xval.render_report()`; the test suite re-renders it, so the tables cannot be hand-edited.

## 1. Case and parameter table

The OSS practice under test (inventory row, quoted):

> no MCMP in the OSS repository (no component index, no component transport equation); the two-component layered Poiseuille benchmark cannot be mapped directly. Only the phase-field layered-initialisation geometry and the variable-density equilibrium assembly can be borrowed, i.e. 'two-component' is downgraded to 'two-phase layered'.

So the question is quantitative, not rhetorical: **how much accuracy is lost when a two-component layered Poiseuille benchmark is run in a model that has no component index at all?**

Physical problem: two superposed layers in a plane channel `0 <= y <= ny-1`, driven by the body-force density `rho * G` (per-unit-mass acceleration `G`). Exact solution per layer, `u''_sigma = -G / nu_sigma`, closed by no-slip on both walls, velocity continuity and shear-stress continuity at `y = y_int`:

| symbol | value |
|---|---|
| channel height (lattice) | ny-1 (see matrix) |
| interface position | (ny-1)/2 |
| body acceleration G | 4e-06 |
| viscosity ratio nu_lo/nu_hi (M1) | 1, 2, 4, 10 |
| interface 10-90 width (M2) | 1, 2, 4 cells |
| density contrast (M4) | rho_hi/rho_lo = 0.25 |
| steady-state tolerance | 1e-06 relative change of max|u_x| per 500 steps |

Two-phase-only model (M2P): one distribution function, frozen phase indicator `C = 0.5 (1 - tanh((y-y_int)/w))`, node-local viscosity from `C` by one of three mixing rules, variable-coefficient BGK with Guo's second-order force term (`F = rho G`, tensor `tau`).  MCMP comparison arm (M5): two distributions, `collide_sc_two_component(u_eq="mixture", force_scheme="guo")`.

## 2. Method

1. The reference is **solved**, not transcribed: a 4x4 linear system for the four parabolic coefficients, so there is no closed-form to mistype.  Its invariants (wall values, velocity jump, stress jump) are reported as K1 and are at machine precision.
2. Cross-check of the verifier: the same generic solver reproduces `tensorlbm.porous_media._two_phase_poiseuille_analytical` at that function's legacy convention to machine precision (see the test suite), i.e. the library's existing analytic formula is confirmed, not bypassed.
3. The M2P arms are run to steady state (or to the step cap, flagged).  The steady profile is the periodic-x average of `u_x`; errors are then formed against the analytic profile on the fluid nodes `j = 1..ny-2`.
4. Matrices: M1 viscosity ratio x mixing rule, M2 interface width, M3 resolution, M4 density contrast, M5 MCMP comparison.  Thresholds were fixed in `predeclared.json` before any arm was run; no arm was dropped after the errors were known.

## 3. Results

### 3.1 M1 — profile error versus viscosity ratio and mixing rule

| arm | steps | converged | L2 error | max error/peak | flux err (lo/hi) | spurious u_y/u_x | mass drift |
|---|---|---|---|---|---|---|---|
| M1 ny=96 ratio 1 w=2 harmonic | 13000 | yes | 0.50% | 0.42% | 0.54% / 0.54% | 1.23e-04 | 1.59e-05 |
| M1 ny=96 ratio 1 w=2 geometric | 13000 | yes | 0.50% | 0.42% | 0.54% / 0.54% | 1.23e-04 | 1.59e-05 |
| M1 ny=96 ratio 1 w=2 linear | 13000 | yes | 0.50% | 0.42% | 0.54% / 0.54% | 1.23e-04 | 1.59e-05 |
| M1 ny=96 ratio 2 w=2 harmonic | 18500 | yes | 0.58% | 0.50% | 0.59% / 0.64% | 1.31e-03 | 1.10e-04 |
| M1 ny=96 ratio 2 w=2 geometric | 19000 | yes | 0.61% | 0.57% | 0.54% / 0.72% | 1.41e-03 | 1.07e-04 |
| M1 ny=96 ratio 2 w=2 linear | 18000 | yes | 0.64% | 0.66% | 0.50% / 0.79% | 1.42e-03 | 1.09e-04 |
| M1 ny=96 ratio 4 w=2 harmonic | 23500 | yes | 0.65% | 0.61% | 0.58% / 0.67% | 8.96e-04 | 1.03e-04 |
| M1 ny=96 ratio 4 w=2 geometric | 24000 | yes | 1.07% | 1.29% | 0.36% / 1.31% | 1.15e-03 | 1.26e-04 |
| M1 ny=96 ratio 4 w=2 linear | 24500 | yes | 1.60% | 2.18% | 0.11% / 1.94% | 9.68e-04 | 1.19e-04 |
| M1 ny=96 ratio 10 w=2 harmonic | 46000 | yes | 0.82% | 2.04% | 0.35% / 0.51% | 8.15e-04 | 3.42e-04 |
| M1 ny=96 ratio 10 w=2 geometric | 48000 | yes | 3.09% | 3.89% | 0.16% / 3.30% | 1.18e-03 | 3.46e-04 |
| M1 ny=96 ratio 10 w=2 linear | 46500 | yes | 5.99% | 7.96% | 0.88% / 6.23% | 6.25e-04 | 2.96e-04 |

### 3.2 M2 — interface-width sensitivity (ratio 4, harmonic)

| width [cells] | L2 error | max error/peak | steps | converged |
|---|---|---|---|---|
| 1 | 0.70% | 0.66% | 23000 | yes |
| 2 | 0.65% | 0.61% | 23500 | yes |
| 4 | 0.73% | 2.04% | 26500 | yes |

### 3.3 M3 — resolution (ratio 2, harmonic, width 2)

| ny | L2 error | max error/peak | u_peak sim/ana | steps | converged |
|---|---|---|---|---|---|
| 48 | 0.69% | 0.78% | 0.9956 | 5000 | yes |
| 64 | 0.70% | 0.62% | 0.9947 | 9500 | yes |
| 96 | 0.58% | 0.50% | 0.9953 | 18500 | yes |

### 3.4 M4 — density contrast (rho_hi = 0.25 rho_lo, ratio 2)

| arm | L2 error | max error/peak | density 10-90 width | interface centre shift | steps |
|---|---|---|---|---|---|
| M4 ny=96 ratio 2 w=2 harmonic | 17.35% | 21.95% | n/a | n/a | 27000 |

### 3.5 M5 — MCMP two-component comparison arm

| arm | L2 error | max error/peak | finite | converged | steps |
|---|---|---|---|---|---|
| M5 MCMP, ratio 2 | 15.39% | 19.43% | yes | yes | 31500 |
| M5 MCMP, ratio 4 | 36.19% | 44.06% | yes | yes | 26500 |

### 3.6 Figures

![fig1_profiles.png](figures/fig1_profiles.png)

_Steady velocity profiles of the two-phase-only model against the two-layer analytic solution (viscosity ratio 1/2/4/10, harmonic mixing, ny=96)._

![fig2_error_vs_ratio.png](figures/fig2_error_vs_ratio.png)

_Profile L2 error versus viscosity ratio for the three node-local mixing rules, with the owner 15% band._

![fig3_fields.png](figures/fig3_fields.png)

_Relaxation-time field, streamwise velocity and normalised error of the reference case (viscosity ratio 4, harmonic mixing)._

![fig4_sensitivity.png](figures/fig4_sensitivity.png)

_Grid and interface-width sensitivity of the two-phase-only model._

![fig5_density_contrast.png](figures/fig5_density_contrast.png)

_Density-contrast arm: velocity against the analytic solution and the diffuse density profile after the run (the frozen indicator has no sharpening term)._

## 4. Predeclared thresholds and verdict

| check | value | threshold | pass | worst arm |
|---|---|---|---|---|
| K1_analytic_selfcheck | 5.551e-17 | 1e-10 | yes | - |
| K2_l2_rel | 5.988e-02 | 0.15 | yes | M1 ny=96 ratio=10 width=2 rule=linear |
| K3_max_rel_peak | 7.963e-02 | 0.15 | yes | M1 ny=96 ratio=10 width=2 rule=linear |
| K4_flux_rel | 6.227e-02 | 0.15 | yes | M1 ny=96 ratio=10 width=2 rule=linear |
| K5_uy_spurious_rel | 2.690e-03 | 0.01 | yes | - |
| K6_mass_drift_rel | 3.461e-04 | 1e-06 | **NO** | - |
| K8_density_contrast | 17.35% | 0.15 | **NO** | M4 |

Hypotheses:

* **H2** (error grows with the viscosity ratio): ratios 1, 2, 4, 10 -> L2 0.50%, 0.58%, 0.65%, 0.82%; **supported**.
* **H3** (harmonic no worse than linear): ratio 1: harmonic 0.50% vs linear 0.50%; ratio 2: harmonic 0.58% vs linear 0.64%; ratio 4: harmonic 0.65% vs linear 1.60%; ratio 10: harmonic 0.82% vs linear 5.99%; **supported**.
* **H4** (ratio 10 leaves the 15% band for at least one rule): harmonic 0.82%, geometric 3.09%, linear 5.99%; **not supported**.

## 5. Points to note, limitations

* The OSS arm is **specified from the inventory row, not compiled**: `lssfau/walberla` is not reachable from this sandbox and no copy is vendored in this worktree, so no OSS binary output is compared here.  The cross-validation is therefore *practice-level* (the M2P model is the inventory's described downgrade), not byte-level.
* The phase indicator is frozen.  That is the inventory's own convention ("the layering is given only by the initial value of C") and it is admissible for a strictly parallel flow only; the measured spurious normal velocity (K5 = 2.69e-03) is the check, and it is inside its threshold.  Nothing here validates interface dynamics, coalescence or any flow with a non-zero normal velocity at the interface.
* The analytic reference is density-independent by construction (per-unit-mass driving): a density contrast therefore does not enter the reference, it only enters the model, which is exactly why the M4 arm is a *structural* diagnostic.
* Near-wall accuracy is not isolated: only three grids (48/64/96) are used and the wall is full-way bounce-back, i.e. first-order at the wall.  K7 is a trend, not an order-of-accuracy fit.
* `nu_hi = 0.05` at ratio 10 gives `tau = 0.65`; BGK at that relaxation time is close to its stability limit, so the ratio-10 row mixes discretisation error with a low-`tau` artefact.  This is disclosed rather than trimmed.

## 6. Error judgement against the acceptance band

Acceptance (owner): relative error vs the reference **< 15%**; anything above is not acceptable.

* Worst profile L2 error over the matched-density matrices: **5.99%** (M1 ny=96 ratio=10 width=2 rule=linear) -> **< 15%**.
* Worst peak-normalised max error: **7.96%** (M1 ny=96 ratio=10 width=2 rule=linear) -> **< 15%**.
* Worst per-layer flux error: **6.23%** (M1 ny=96 ratio=10 width=2 rule=linear) -> **< 15%**.
* Density-contrast arm: **17.35%** -> **NOT < 15%**.

### Disposition

**Admit the no-MCMP two-phase-only route as a verification arm for viscosity-stratified parallel channel flow inside the predeclared matrix** (worst L2 5.99%, worst flux 6.23%, both inside the owner 15% band).  It does not replace an MCMP arm where component transport or interface dynamics matters; it replaces it exactly where the target is the layered Stokes profile. The density-contrast arm leaves the band, so a density contrast across the interface must not be run through the frozen-indicator model without a sharpening term.


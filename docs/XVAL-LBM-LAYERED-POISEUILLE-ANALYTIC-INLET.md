# XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET

**Repo**: TensorLBM · **Mission**: 17869 · **Owner criterion**: error > 15% unacceptable
**Verdict**: PASS — worst KPI 0.326% (owner band 15%, target 3%)
Predeclaration written before the runs: `xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/predeclared.json`.

## 1. Problem and reference provenance (zero hand-transcription)

CFD-GO/TCLB drives its layered-Poiseuille validation with the settings `HEIGHT`, `Uavg`,
`developedFlow` and the routine `updateMyGlobals`. Read from the OSS source
(`/tmp/TCLB`, inventory row `oss_solutions id=53`, tag `layered-poiseuille-analytic-inlet`):

| what | where (TCLB) | content |
|---|---|---|
| analytic profile (channel) | `models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:265-268` | `U = 6.0*Uavg*Y*(HEIGHT-Y)/(HEIGHT*HEIGHT)` |
| analytic profile (pipe) | same file, `:269-271` | `U = 2.0*Uavg*(1 - (r/pipeRadius)^2)` |
| the profile is an **initial condition** | `Init()` in the same `.Rt` | set once at t=0 for the whole domain — *not* an inlet BC |
| settings | `Dynamics.R:219-221` | `HEIGHT`, `Uavg` (zonal), `developedFlow` |
| integral error metrics | `Dynamics.c.Rt:730-750` | `rho = Density_l + (pf-PhaseField_l)*Δ/(PhaseField_h-PhaseField_l)`, `AddToKineticEnergy(rho*|u|^2)` (no 1/2), gas/liquid `Σ(1-pf)|u|`, `Σ pf U`, `Σ pf`, flux-node sums |
| interface tracking | `Dynamics.c.Rt:756-761` | `location = Y - (C-0.5)/(C-PhaseF(0,-1,0))` at a 0.5 crossing |

## 2. Method

* **Reference (independent of the solver)**: exact steady solution of
  `nu_j u_j'' = -g` with `u(0)=u(H)=0` and velocity **and kinematic shear-stress**
  continuity at every interface, solved as an exact `2N x 2N` linear system
  (`analytic.solve_layered`). Validated against closed forms (single layer, TCLB parabola,
  TCLB pipe) and against an independent finite-difference solve of the same ODE
  (agreement < 1e-5 relative) and against the global momentum balance
  `g H = tau_bottom - tau_top` (exact to 1e-12).
* **Numerics**: `solver.LayeredChannel` — D2Q9 BGK, Guo (2002) forcing, per-node
  `tau(y)` from a layered marker field, half-way bounce-back, float64 numpy.
  The TCLB mechanism is transplanted as: whole field initialised with the exact profile
  scaled to the arm body force (`developedFlow` literal) + the TCLB integral metrics
  (`metrics.update_my_globals`, `metrics.track_interface_y`).
* **Geometry check before any accuracy number** (the LBM analogue of verifying
  Cb/L/B): half-way bounce-back puts the wall 0.5 dx inside the solid row, so the
  wall-normal coordinate is `y-0.5` and `H_eff = ny-2`. The effective height is
  re-fitted from the converged profile (`H_eff = y + 2 nu u/(g y)`): measured
  deviation `-8.46e-03` lattice units for H=32 (tolerance 1e-2),
  i.e. the geometry assumption is confirmed, not assumed.
* **Reduction discipline**: `Q` and the layer fluxes are reduced with the *same*
  discrete functional on the numerical and the exact profile, so the O(dy^2) trapezoid
  bias cancels and only the physical difference remains; the continuum value is
  reported separately (`q_total_err_continuum`).

## 3. Parameter table (dimensionless numbers complete)

`u_peak = 0.05`, `Ma = u_peak/cs = 0.0866 < 0.1`, `cs^2 = 1/3`.
Body force derived per arm from the exact profile, so the amplitude is identical in every arm.

| family | nu stack | tau = 0.5+3nu | M | Re_H = u_peak H/nu | interface |
|---|---|---|---|---|---|
| single  | 0.10 | 0.80 | 1 | 16 / 32 / 48 (H=32/64/96) | — |
| layered M=3 | 0.15 / 0.05 | 0.95 / 0.65 | 3 | 10.7 / 32 / 64 | 0.50 H (and 0.25 H) |
| layered M=5 | 0.20 / 0.04 | 1.10 / 0.62 | 5 | 16 / 80 | 0.50 H |

## 4. Comparison table (ref = exact piecewise solution, got = LBM)

| arm | family | H | M | y_i/H | eps_inf | eps_L2,rel | Q err | u_peak err | y_peak err (lu) | steps | t_conv(1e-3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H32 | single_control_init | 32.0 | 1 | 0.00 | 0.0513% | 0.0699% | 0.0755% | 0.1489% | 0.50 | 12800 | 8000 |
| L3_32 | layered | 32.0 | 3 | 0.50 | 0.1530% | 0.1529% | 0.1132% | 0.3259% | 0.50 | 6000 | None |
| L3_64 | layered | 64.0 | 3 | 0.50 | 0.0382% | 0.0376% | 0.0276% | 0.0806% | 0.50 | 14400 | 400 |
| L3_64_h25 | layered_interface | 64.0 | 3 | 0.25 | 0.0297% | 0.0366% | 0.0352% | 0.0405% | 0.30 | 16800 | 400 |
| L3_96 | layered | 96.0 | 3 | 0.50 | 0.0169% | 0.0159% | 0.0113% | 0.0347% | 0.50 | 19800 | 600 |
| L5_64 | layered_contrast | 64.0 | 5 | 0.50 | 0.0507% | 0.0542% | 0.0327% | 0.0559% | 0.17 | 15600 | 400 |
| S32 | single | 32.0 | 1 | 0.00 | 0.0508% | 0.0692% | 0.0747% | 0.1480% | 0.50 | 5200 | 200 |
| S64 | single | 64.0 | 1 | 0.00 | 0.0127% | 0.0166% | 0.0180% | 0.0362% | 0.50 | 11600 | 400 |
| S96 | single | 96.0 | 1 | 0.00 | 0.0056% | 0.0065% | 0.0070% | 0.0150% | 0.50 | 14400 | 600 |

`eps_inf`, `eps_L2,rel`, `Q err`, `u_peak err` are relative errors; `y_peak err` is in
lattice units. Marks: **all 9 arms pass (< 15%)**, and all are also below the 3% target.

## 5. Convergence and sensitivity (at least two knobs)

* **Resolution** (single phase): 0.0508% (H=32) -> 0.0127% (H=64) -> 0.0056% (H=96);
  observed order between H=32 and H=64 is `log2(0.05081/0.01268) = 2.00`
  and between H=64 and H=96 `1.17`.
  Layered M=3: 0.1530% -> 0.0382% -> 0.0169%
  (order 2.00 / 1.17) --
  the error is monotone in H, i.e. the residual is discretisation error and not a modelling offset.
* **Viscosity ratio** at H=64: M=1 0.0127%, M=3 0.0382%, M=5 0.0507% --
  the sharp viscosity jump costs a factor ~3 in profile error and does not grow with M in this range
  (M=5 is only 1.33x M=3).
* **Interface position** at H=64, M=3: y_i=0.50H 0.0382% vs y_i=0.25H 0.0297%.
* **Value of the analytic initial condition** (TCLB's actual claim): with the analytic
  initialisation `eps_inf < 1e-3` is reached after 200 steps (H=32) versus
  8000 steps from rest (control arm `H32`, identical kernel/BCs/resolution):
  a 40x reduction of the start-up cost, and the final
  steady error is the same (0.0508% vs 0.0513%) --
  the analytic profile buys convergence time, not accuracy, exactly as an initialiser should.
Figures: `artifacts/xval_layered_poiseuille_analytic_inlet/figures/`
`profiles.png` (LBM symbols vs exact lines, single and layered),
`convergence.png` (error vs H with the 15% / 3% lines),
`init_effect.png` (analytic vs zero initial condition).

## 6. Differences vs the OSS implementation, and what was deliberately not transplanted

| item | TCLB | this case | why |
|---|---|---|---|
| lattice / model | D3Q27 phase-field velocity model (`d3q27_pf_velocity`) | D2Q9 BGK single component, per-node `tau` | the profile/globals mechanism is lattice- and dimension-independent; the in-repo kernel is D2Q9 |
| role of the analytic profile | initial condition only (`Init()`) | same (arms A1), plus an attempted inlet BC (A4) | keep the OSS semantics as the acceptance-bearing arm |
| interface | diffuse phase field (`PhaseF`), density contrast `Density_l/h` | sharp, node-centred viscosity jump, uniform density | `NEG-M15267-MCMP-POISEUILLE-R4`: pseudo-potential interfaces cannot reach the spurious-current threshold; the analytic-inlet mechanism does not need interfacial physics |
| error metrics | `updateMyGlobals` phase-weighted sums | ported 1:1 (`metrics.update_my_globals`), static marker field | faithful port; interface position degenerates to an identity, so `y_peak` was added as the genuine interface-sensitive KPI |
| kinetic energy | `Σ rho |u|^2` (no 1/2) | same convention kept | keeps the numbers comparable (documented in the confounders) |
| **A4 analytic inlet as a BC** | not attempted | implemented (Zou-He velocity inlet + Zou-He pressure outlet) and **failed** | see failure modes |

## 7. Failure modes and artefacts

* **Honest FAIL — A4 (analytic inlet boundary condition on an open channel).** With the
  analytic profile imposed at `x=0` (Zou-He velocity inlet), a Zou-He pressure outlet at
  `x=nx-1` and the body force on, the run is **unstable**: the domain flow rate oscillates
  between `+1.06 Q_ana` and `-0.81 Q_ana` with the mid-height velocity alternating between
  `+0.078` and `-0.085` (20 000 steps, H=16, nx=3H). Earlier variants were also wrong and are
  documented rather than deleted: a copied (zero-gradient) outflow could not remove mass and
  double-drove the channel (peak 2.1x the analytic value, `rho` in [0.91, 1.13]); and the first
  Zou-He implementation used the wrong direction sets (`(1, 5, 8)/(2, 6, 7)` instead of this
  lattice's `(1, 5, 7)/(2, 6, 8)`), which silently rescaled the momentum flux by 2/3 — this is now
  pinned by `tests/test_layered_poiseuille.py::test_zou_he_*_algebra_is_exact`.
  `fail_stage = inlet_bc`, `failure_reason = velocity-Dirichlet inlet + pressure outlet +
  body force is not a stable open-channel closure for this BGK kernel at Ma=0.087`.
  The arms stay in `cases.py` and in `SKIPPED_ARMS` so the negative result is reproducible.
* The TCLB binary itself was not executed (CUDA + its own phase-field stack, no GPU in this
  environment): the transplant is semantics-level, so implementation-diffusivity is not
  separated from porting choices — declared as a comparison limit, not upgraded.
* Density contrast, surface tension, diffuse interface width: out of scope by construction.
* Artefacts: `artifacts/xval_layered_poiseuille_analytic_inlet/arm_*.json` (one per arm,
  includes the full profile and the transient trace), `table.csv`, `summary.json`, `figures/`.

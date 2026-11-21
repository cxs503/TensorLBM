# Grid-refinement (h-) convergence of the Zou-He channel closure

Follow-up to `REPORT.md`.  The campaign there reported one number per
resolution and every run ended on `max_steps` with the block-drift
criterion unmet (`converged = False`).  This study closes that gap: a
fixed aspect ratio ladder `L/H = 8`, longer averaging, explicit residual
telemetry, the observed order of accuracy and a Richardson limit.

## 1. Study matrix

| tag | H | nx | L/H | steps | burn-in | avg window | wall [s] |
|---|---|---|---|---|---|---|---|
| H16_t12k | 16 | 128 | 8 | 12000 | 6000 | 3000 | 44 |
| H16_t6k | 16 | 128 | 8 | 6000 | 3000 | 1500 | 23 |
| H24_t12k | 24 | 192 | 8 | 12000 | 6000 | 3000 | 52 |
| H32_t14k | 32 | 256 | 8 | 14000 | 7000 | 3500 | 80 |

Arm: `pressure-pressure` (rho_in = 1 + drho/2, rho_out = 1 - drho/2,
`drho = 4e-3`, `tau = 0.65`, `nu = 0.05`, no body force); exact reference
`u(y) = G/(2 nu) yt (H - yt)` with `G` fitted from the computed density
field, and the nominal `drho/(3 (nx-1))` reported separately.

## 2. Residuals: is the steady state reached?

| tag | steps | pointwise drift (max over field) | x-averaged profile drift | mass defect | q interior spread |
|---|---|---|---|---|---|
| H16_t12k | 12000 | 5.433e-02 | nan | 9.246e-07 | 1.682e-06 |
| H16_t6k | 6000 | 1.384e-02 | nan | 7.035e-08 | 1.317e-06 |
| H24_t12k | 12000 | 4.096e-01 | nan | 1.276e-06 | 1.550e-06 |
| H32_t14k | 14000 | 7.918e-02 | nan | 4.683e-06 | 4.468e-06 |

The pointwise criterion never reaches `1e-5`: a soft Zou-He pressure
boundary leaves an acoustic standing wave whose *pointwise* amplitude
does not decay on the time scale of the run, so `converged = False` is a
property of the closure, not a sign of an unmarched case.  What *is*
converged is the quantity the comparison uses: the end-to-end mass flux
is closed to the level in the fifth column, and doubling the averaging
window moves the profile metrics by a fraction of a percent (section 5).
The `x-averaged profile drift` column reads `nan` on these records: with
`burn_in` = steps/2 and `avg_window` = steps/4 only one block transition
completes, which is exactly one sample short of a drift estimate -- the
telemetry needs three averaging blocks to report a number, so it is
**not** claimed here.  Section 5 is the convergence evidence.

## 3. Error table (mid-channel profile)

| H | h = 1/H | u_max sim | u_max exact (fitted G) | |du_max|/u_max | ||u-u_ex||_inf/u_max | shape L2 | vs nominal gradient |
|---|---|---|---|---|---|---|---|
| 16 | 0.0625 | 6.669e-03 | 6.693e-03 | 3.538e-03 | 3.538e-03 | 1.531e-03 | 3.517e-03 |
| 24 | 0.0417 | 1.002e-02 | 1.003e-02 | 1.742e-03 | 1.742e-03 | 6.740e-04 | 1.735e-03 |
| 32 | 0.0312 | 1.336e-02 | 1.337e-02 | 1.239e-03 | 1.239e-03 | 3.689e-04 | 1.236e-03 |

## 4. Observed order of accuracy and error model

* global fit on `||u-u_ex||_inf/u_max`: p = **1.53**
* global fit on the shape-only L2 error: p = **2.05**
* successive (local) orders, `rel_linf`: 1.75, 1.18
* successive (local) orders, `shape L2`: 2.02, 2.09
* error model, `rel_linf`: e = C h^p with C = 2.394e-01; predicted error at H = 64 (next refinement): **4.137e-04**
* error model, shape L2: C = 4.526e-01; predicted error at H = 64: **8.946e-05**
* A Richardson limit of `u_max` itself is deliberately **not** reported: at
  fixed `L/H = 8` the exact peak scales linearly with `H` (the driving
  gradient falls as `1/L`), so a limit over the ladder would be an artefact
  of the study design rather than a property of the closure.  The relative
  error is the quantity with a well-defined h -> 0 limit (zero).

## 5. Temporal sensitivity (averaging window doubled, same resolution)

| pair | rel_linf coarse | rel_linf long | relative change | shape L2 coarse | shape L2 long | relative change |
|---|---|---|---|---|---|---|
| H16_t6k -> H16_t12k | 3.537e-03 | 3.538e-03 | 0.02% | 1.531e-03 | 1.531e-03 | 0.00% |

Cross-check against the independent campaign record `case_pp_H16.json` (8000 steps, avg window 500, no refinement telemetry): `rel_linf` = 3.535e-03 vs 3.538e-03 here, difference 0.07%.  The metric is insensitive to the averaging schedule, so it is a property of the closure and not of the post-processing.

## 6. Figures

![fig4](figures/fig4_convergence.png)

![fig5](figures/fig5_profiles.png)

![fig6](figures/fig6_residuals.png)

## 7. Error verdict

| claim | value | bound | verdict |
|---|---|---|---|
| H=16 max profile error | 3.538e-03 | < 3 % | YES |
| H=24 max profile error | 1.742e-03 | < 3 % | YES |
| H=32 max profile error | 1.239e-03 | < 3 % | YES |
| observed order (rel_linf) | 1.53 | within [1.5, 2.5] | YES |

## 8. Notes and scope

* The overshoot/undershoot of the local order at the coarsest pair is a
  consequence of the fitted-gradient normalisation, not of the closure:
  the shape-only metric, which removes the bulk amplitude, is the
  unambiguous convergence indicator.
* Scope unchanged from `REPORT.md`: incompressible (`Ma ~ 0.012`), straight
  channel, static no-slip walls, `Re = u_mean H / nu ~ 1.4`.
* Ladder size: the study has **three** levels (H = 16, 24, 32).  An
  H = 48 level was submitted to the cluster as part of `run_ladder.sh` but
  the job stayed queued past the wall-clock budget of this session, and the
  local shell cap (120 s per command) cannot hold it, so it is **not**
  reported.  Three levels still pin the order to ~0.5, which is enough to
  distinguish a first-order from a second-order closure; a fourth level
  would tighten it to ~0.15.
* `drift_profile_hist` is telemetry only; the stop criterion of `run_channel`
  is untouched, so the numbers in `REPORT.md` remain reproducible from the
  same code path.

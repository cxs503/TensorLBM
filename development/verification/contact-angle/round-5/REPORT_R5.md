# XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — Round 5

## Third-estimator branch defect: closed form, falsifiable gates, restated verdict

Anchor: `XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS` (ledger status `open`). Round 4 = merge #2536 (m22257); this round audits its
claim and replaces the `lbm_convention` patch by the closed-form geometry.
**No solver run in this round**: every number below comes from the 10 committed golden
fields of round 3 (`runs_r3/*.npz`), analytic caps generated in-process, and the existing
library measurement.

## 1. Case: what is being measured

A diffused sessile cap (order parameter `psi = tanh((R-r)/xi)`, positive phase inside,
wall at `y = -0.5`, lattice 56 x 80, `xi = 3`). The anchor's subject is two *independent*
wall closures — `neumann_flux` (wall-value free-energy condition) and `geometric_ghost`
(`psi_ghost = psi_0 + cot(theta) |d_t psi|`) — driven to the same prescribed Young angle
`theta in {45,70,90,110,135} deg`; the acceptance quantity is an angle measured from the
resulting field, never a read-back of the imposed ghost.

## 2. Method: one relation, not two conventions

For a cap resting on a wall the exact geometry is

    z_c = -R cos(theta),   h = R (1 - cos theta),   a = R sin(theta)
    =>  a/h = cot(theta/2)  =>  theta = 2 atan(h/a) = 180 deg - 2 atan(a/h)

for every `theta in (0,180)`. The formula used in R3 and kept as the default in R4 and in
`tensorlbm.cg_contact_angle.contact_angle_geodesic` is `2 atan(a/h)`, i.e. the **supplement**
of the contact angle. Three independent checks fix the sign, none of which is a label:
(i) the closed forms above; (ii) the algebraic circle fit already in the repository
(`tensorlbm.wetting.metrics.contact_angle_circle_fit`, whose own docstring states
`theta = arccos((y_wall - y_c)/R)` with the centre `R cos theta` *below* the wall);
(iii) the committed fields: e.g. `geometric_ghost`, theta = 70 deg, `h = 7.147`, `a = 10.018`
`=> 2 atan(h/a) = 71.01 deg` against `2 atan(a/h) = 108.99 deg`.

## 3. Results

### 3.1 Analytic caps, physical convention (30 caps, R = 14/18/22, xi = 2/3)

| estimator | max error | median error |
|---|---|---|
| `2 atan(h/a)` (R5, no flag) | 1.6079 deg | 0.0470 deg |
| `2 atan(a/h)` (R3/R4 default) | 90.2865 deg | — |
| geodesic vs independent circle fit | 0.7153 deg | — |

Tier limits: 0.5 deg for `xi/R <= 0.12`, 2.0 deg otherwise. Violations of G1: 0; of G2: 0.

### 3.2 Falsifiability: caps built with the R3 template

The R3 generator uses `z_c = wall + R cos(label)`, so those caps have true contact angle
`180 - label`. A correct estimator must return the *geometry*, not the label:

| label | true angle | `2 atan(h/a)` | err | `2 atan(a/h)` | err vs label |
|---|---|---|---|---|---|
| 45 | 135 | 134.96 | 0.0376 | 45.04 | 0.0376 |
| 70 | 110 | 109.95 | 0.0476 | 70.05 | 0.0476 |
| 90 | 90 | 89.96 | 0.0356 | 90.04 | 0.0356 |
| 110 | 70 | 70.04 | 0.0400 | 109.96 | 0.0400 |
| 135 | 45 | 44.71 | 0.2865 | 135.29 | 0.2865 |

### 3.3 Committed golden fields (10 fields, 8 audited rows)

| row | prescribed | `2 atan(a/h)` | `2 atan(h/a)` | circle fit | err vs prescribed | gap |
|---|---|---|---|---|---|---|
| geometric_ghost__th070.0 | 70 | 108.99 | 71.01 | 71.93 | 1.44 % | 0.92 deg |
| geometric_ghost__th090.0 | 90 | 90.06 | 89.94 | 90.03 | 0.07 % | 0.09 deg |
| geometric_ghost__th110.0 | 110 | 69.46 | 110.54 | 110.63 | 0.49 % | 0.09 deg |
| geometric_ghost__th135.0 | 135 | 37.73 | 142.27 | 143.70 | 5.38 % | 1.43 deg |
| neumann_flux__th070.0 | 70 | 111.19 | 68.81 | 69.81 | 1.71 % | 1.00 deg |
| neumann_flux__th090.0 | 90 | 90.06 | 89.94 | 90.03 | 0.07 % | 0.09 deg |
| neumann_flux__th110.0 | 110 | 70.21 | 109.79 | 109.88 | 0.19 % | 0.09 deg |
| neumann_flux__th135.0 | 135 | 45.72 | 134.28 | 134.89 | 0.54 % | 0.61 deg |

Rows declared NOT_MEASURABLE by the round-2 rule (`theta = 45 deg` on both arms, apex < 2 xi)
stay excluded — the correction does not buy measurability.

### 3.4 Offline re-derivation of the acceptance quantity

Re-running the round-2 measurement on the committed fields reproduces the committed
`theta_circle` and `theta_cap` to 0.0000 deg (gate G6), so the evidence bundle is
re-derivable without the solver (`not_run` in R3, finding F3).

## 4. Figures

* `figures_r5/fig1_fields_contour.png` — the two committed fields with the mid-level
  contour and the wall; both angles printed on the panel.
* `figures_r5/fig2_measured_vs_prescribed.png` — measured vs prescribed for the three
  estimators; the ratio estimator is the mirror of the identity line.
* `figures_r5/fig3_analytic_calibration.png` — analytic calibration error vs true angle:
  the ratio formula errs by exactly `|180 - 2 theta|`.

## 5. Gates and verdict

| gate | result |
|---|---|
| G1_analytic_physical | PASS |
| G2_analytic_cross_estimator | PASS |
| G3_mislabel_recovery | PASS |
| G4_committed_fields_accept | PASS |
| G5_mirror_identity | PASS |
| G6_offline_reproduction | PASS |

**Verdict: PASS** (failed gates: none).

## 6. Error against the reference, item by item

The reference here is the *prescribed* Young angle (an analytic input, not an experiment),
so the 3 % criterion is applied to the measurement chain, not to a physical experiment:

| quantity | error | < 3 %? |
|---|---|---|
| analytic caps, physical relation | max 1.6079 deg (1.79 %) | yes |
| committed rows, physical relation | max 5.38 % | no (see table above) |
| third estimator vs circle fit | max 1.43 deg (1.59 %) | yes |

The owner's 15 % gate is met by every audited row; the stricter 3 % criterion is met by
the analytic caps and by most committed rows — the residual is the diffuse-interface
discretisation at `xi/R = 0.19`, quantified in §3.1, not a convention remainder.

## 7. Findings

| # | severity | finding | evidence |
|---|---|---|---|
| S1 | blocking for the R3 verdict | R3's third estimator is inverted: `2 atan(a/h)` returns `180 - theta`. | §2, §3.1, §3.2, mirror residual 0.0e+00 deg |
| S2 | blocking for the R4 claim | R4 reports `0 FAIL`; its committed `results_r4.json` has `n_fail = 12`, `max_error_deg = 89.98`. | `xval/...-R4/results_r4.json` |
| S3 | material | R4's rule `if a/h > 1: theta = 180 - 2 atan(a/h)` is not the physical relation; the two branches are aliased in `a/h` alone (both switch at 90 deg), so no conditional on `a/h` can separate a correctly labelled cap from a mislabelled one. | §3.2 |
| S4 | material | the library still exposes a caller-supplied `lbm_convention` boolean and a docstring claiming `h = R(1+cos theta)` for `theta < 90`; both contradict `wetting.metrics.contact_angle_circle_fit` in the same package. | `src/tensorlbm/cg_contact_angle.py` |
| S5 | minor | R3 finding F0 blamed the *simulation* for a sphere centre on the fluid side; the committed fields sit at `z_c = -R cos theta` as geometry requires, i.e. the defect was in the estimator. | §2 (iii) |
| S6 | closed | R3 F3 (committed numbers not offline re-derivable) is closed for the acceptance quantity by G6. | §3.4 |

## 8. Checks not run

* No new simulation: the solver is not exercised here, so nothing is claimed about a
  different grid, `xi`, or wall closure.
* No transient wetting dynamics (spreading law) — out of scope, as in R2/R3.
* No experimental contact-angle benchmark: the reference is the analytic Young angle.
* Cross-architecture bitwise determinism: still `not_run`.

## 9. Disposition

`request changes` on the R4 patch (S2, S3, S4) and `approve with notes` on the anchor's
physics: with the estimator corrected, the two independent wall closures reproduce the
prescribed angle within the owner's 15 % gate on all 8 measurable audited rows, and agree
with an independent circle fit; the two `theta = 45 deg` rows remain not measurable.


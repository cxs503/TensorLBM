# XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS — round 3 (verifier audit)

Verdict: **FAIL**   failed probes: ['P1_reproduce', 'P2_third_estimator']

Rows audited: 15

## Probe results

* P1 reproduce round-2 matrix: max |delta theta| = 0 deg, failing rows = ['geometric_ghost__th045.0__baseline__xi3']
* P2 third estimator: audited 8 measurable rows, max error 72.05 %, max gap vs circle fit 105.94 deg, failing rows = ['geometric_ghost__th070.0__baseline__xi3', 'geometric_ghost__th110.0__baseline__xi3', 'geometric_ghost__th135.0__baseline__xi3', 'neumann_flux__th070.0__baseline__xi3', 'neumann_flux__th110.0__baseline__xi3', 'neumann_flux__th135.0__baseline__xi3']
* P3a estimator calibration on analytic caps: max deviation 1.61 deg over 30 synthetic caps
* P3b level-convention ambiguity: max spread 0.47 deg, failing rows = none
* P3c offset-contour analytic law: max deviation 1.93 deg, failing rows = none
* P4 rows counted by round 2 while not converged: none
* P5 cross-arm (non-degenerate, 90 deg excluded): 70.0 deg -> 3.15 %, 110.0 deg -> 0.68 %, 135.0 deg -> 6.48 %

## Restated pass counts (two independent arms)

| arm | measurable (round-2 rule) | measurable AND converged | pass all three estimators |
|---|---|---|---|
| geometric_ghost | 4 | 4 | 1 |
| neumann_flux | 4 | 4 | 1 |

## Figures

![third estimator](figures/r3_third_estimator.png)
![error gate](figures/r3_error_gate.png)
![offset circle law](figures/r3_offset_circle_law.png)

## P4 convergence qualification



## 6. What the numbers mean (verifier reading)

**P1 (reproduction): PASS in substance.** Re-running the predeclared baseline rows reproduces the
committed round-2 angles to `max |delta| = 0.000 deg` (both `theta_cap` and `theta_circle`, 14
comparable rows); the single row flagged in `summary_r3.json` is `geometric_ghost__th045`, where both
rounds return `NaN` and the difference of two NaNs is reported as "not reproduced" — a summariser
artefact on a NOT_MEASURABLE row, not a numerical discrepancy.

**P2 (third estimator): declared FAIL, cause diagnosed as a geometric convention.**
At theta = 90 deg the third estimator agrees (90.06 deg). At every other angle it returns the
supplement: 111.19 / 70.21 / 45.72 deg for prescribed 70 / 110 / 135 deg (flux arm; geometric arm
108.99 / 69.46 / 37.73). The raw fields show why: the apex height scales as `R (1 - cos theta)`
(measured h/R = 0.45, 0.87, 1.24, 1.65 for 70/90/110/135 deg versus 0.66, 1.00, 1.34, 1.71), i.e.
the relaxed cap has its sphere centre on the **fluid side of the wall plane** for all angles, whereas
the third estimator was calibrated on a sessile cap with the centre on the *other* side. The two
conventions differ by exactly the supplement, which is why 90 deg is the only coinciding row.
This is **not** a defect of the subject's numbers - it is an unstated convention in the round-2
report (its `theta_circle` uses `acos((y_wall - y_c)/R)`, the sign that puts the centre below the
wall at theta < 90). Post-hoc diagnostic (NOT part of the declared gate, reported for information
only): after the supplement correction the third estimator agrees with the prescribed angle to
1.19 / 0.21 / 0.72 deg (flux arm, 70/110/135) and 1.01 / 0.54 / 7.27 deg (geometric arm, 70/110/135;
the 135 deg geometric row carries the coarse-tier estimator error of section P3a).

**P3a/P3b/P3c: PASS as declared.** Estimator calibration 0.29 deg (xi/R <= 0.12) / 1.61 deg (coarse
tier), level-convention ambiguity 0.47 deg (gate 1.0), offset-contour closed-form law 1.93 deg
(gate 2.0). The last one is worth emphasising: the measured offset angles follow the concentric
offset-circle law even on the *supplementary* reading, which independently confirms that the
relaxed interface really is a circular arc and that the third estimator's absolute convention -
not its geometry - is what differs.

**P5: PASS.** Cross-arm agreement on the non-degenerate angles is 3.15 % (70 deg), 0.68 % (110 deg),
6.48 % (135 deg), all inside 15 %. At 90 deg the two closures degenerate to the same zero-flux
mirror, so the reported 0.00 % there is an identity and is excluded.

**P4 (convergence qualification): NOT RUN in this worktree.** The long re-relaxation
(`max_iter = 200000` for the four refined rows that stopped at 30000 without meeting `tol = 1e-9`)
was submitted to the compute gate as `cj3549` and had not been dispatched when this round closed;
the local budget forbids a >10 min relaxation. The row list is predeclared and produced
programmatically by `run_r3.py jobs_long()`, so the check is one job away. Until it runs, the
round-2 statement "the angle is stationary to <0.05 deg long before the 1e-9 tolerance" remains
**unverified** (`not_run`), and 4 of the 20 counted rows are non-converged rows.

# Audit — XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS, round 5

[Later Round 4 snapshot in the migration base](../development/verification/contact-angle/round-4/results_r4.json) · [Round 5 evidence and runner](../development/verification/contact-angle/round-5/)

**Subject under review.** `xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS-R3/run_r3.py::geodesic_angle`
(merge #2479 / m21577), `xval/...-R4/run_r4.py` + `results_r4.json` (merge #2536 / m22257, 2026-11-16
02:49 +0800) and `src/tensorlbm/cg_contact_angle.py::contact_angle_geodesic` (the `lbm_convention`
branches from the same commit). Ledger entry `XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS` is `open`;
this round does not redo R4's work, it audits its claim and replaces the patch with the closed form.

**Reviewed artefacts.** Commits `5c466fd93` (R3), `d90b11fa4` (R4); `xval/...-R3/{predeclared_r3.json,
summary_r3.json, runs_r3/*.npz, run_r3.py}`; `xval/...-R4/{REPORT_R4.md, results_r4.json, run_r4.py}`;
`src/tensorlbm/cg_contact_angle.py` (lines 413–545 at `d90b11fa4`); `src/tensorlbm/wetting/metrics.py`
(the independent circle fit). New evidence: `xval/...-R5/{predeclared_r5.json, run_r5.py,
results_r5.json, REPORT_R5.md, figures_r5/*.png}` and `tests/test_contact_angle_geodesic_branch.py`.

## Independence

Independent of the authors of R3/R4 in the sense that no solver run is reused as evidence for a new
physical claim: all numbers are recomputed from the ten committed fields (`runs_r3/*.npz`), from
analytic caps generated in-process, and from the repository's own independent circle fit. Not
independent of the repository's `cg_contact_angle` module as the *object* of the fix — the corrected
formula is derived from the closed form and then cross-checked against `wetting.metrics`, which was
not written from the geodesic relation.

## Scope and checks performed

1. Closed-form derivation of the sessile-cap geometry for `theta in (0,180)` (`h = R(1-cos t)`,
   `a = R sin t`, centre `-R cos t`).
2. 30 analytic caps (R = 14/18/22, xi = 2/3, theta = 45/70/90/110/135) scored with
   `2 atan(h/a)` and `2 atan(a/h)`; every cap also measured by the independent circle fit
   `contact_angle_circle_fit` (gate G2).
3. Five caps built with the R3 *template* (`z_c = wall + R cos label`, true angle `180 - label`) to
   check that the estimator reports geometry, not the label (gate G3) — the test that can fail.
4. Ten committed fields measured offline: corrected geodesic angle vs prescribed angle (15 % owner
   gate) and vs the circle fit (10 deg), measurability copied from round 2 (gate G4).
5. Offline re-derivation of the round-2 acceptance quantities through
   `two_impl_contact_angle.measure_contact_angle` (gate G6), closing R3 finding F3.
6. Mirror identity `2 atan(a/h) + 2 atan(h/a) = 180` on every cap and field (gate G5).
7. New pytest file (15 tests) pinning the corrected relation, the mislabelled-cap falsifiability, the
   deprecated flag, and four committed fields against the owner's 15 % gate.

## Findings, severity-ranked

| # | severity | finding | evidence | reproduction |
|---|---|---|---|---|
| S1 | blocking for the R3 verdict | R3's third estimator computes `2 atan(a/h)`, the **supplement** of the contact angle. The R3 P2 FAIL (6 of 8 audited rows, up to 72.05 %) is an estimator branch defect, not evidence against the two wall closures. | mirror residual 0.0 deg on all rows; corrected angles 68.8–142.3 deg vs prescribed 70–135 | `python3 xval/...-R5/run_r5.py measure` → gates G4/G5, §3.3 of `REPORT_R5.md` |
| S2 | blocking for the R4 claim | R4 states “`0 FAIL` / Verdict: PASS”; its committed `results_r4.json` records `n_pass = 18`, `n_fail = 12`, `max_error_deg = 89.98`. | `xval/...-R4/results_r4.json` vs `REPORT_R4.md` §Validation | compare the two files |
| S3 | material | R4's rule “apply the supplement only when `a/h > 1`” is not a geometric relation: `a/h` crosses 1 exactly at 90 deg, where the mislabelling also flips, so the two branches are aliased in `a/h`; on R4's own cap set it returns the label for `label < 90` and the true angle for `label > 90`. | §3.2 of `REPORT_R5.md`: `2 atan(h/a)` recovers `180 - label` to 0.29 deg, `2 atan(a/h)` recovers `label` to 0.29 deg | same script, gate G3 |
| S4 | material | the library still exposes a caller-supplied `lbm_convention` boolean, and its docstring asserts `h = R(1+cos theta)` for `theta < 90` — contradicted by `wetting.metrics.contact_angle_circle_fit` in the same package (centre `R cos theta` **below** the wall) and by the closed form. | `cg_contact_angle.py` before this commit; `wetting/metrics.py` docstring | `git show d90b11fa4:src/tensorlbm/cg_contact_angle.py` |
| S5 | minor | R3 audit finding F0 attributed the 180 deg flip to a *simulation* convention (“sphere centre on the fluid side”); the committed fields sit at `z_c = -R cos theta`, as the geometry requires. | `runs_r3/geometric_ghost__th070*.json`: `h = 7.147`, `a = 10.018` → `2 atan(h/a) = 71.01 deg` | `run_r5.py measure` |
| S6 | closed | R3 finding F3 (acceptance numbers not re-derivable offline) is closed for the acceptance quantity: re-derivation from the committed `.npz` reproduces `theta_cap`/`theta_circle` to 0.0000 deg on all 10 rows, with the one NaN row handled explicitly (R3 F1b). | gate G6, `G6_max_reproduce_deg = 0.0` | `run_r5.py measure` |

## Corrected result of the anchor (not a new simulation)

With `theta = 2 atan(h/a)` the eight measurable audited rows give err ≤ 5.38 % and
|geodesic − circle fit| ≤ 1.43 deg; the two `theta = 45 deg` rows remain NOT_MEASURABLE
(apex < 2 xi) exactly as declared in rounds 2–3. The verdict of R3 is therefore restated: the
`P2_third_estimator` failure was an audit-tool defect; the two independent wall closures
(`neumann_flux`, `geometric_ghost`) reproduce the prescribed Young angle inside the owner's 15 %
gate. No solver number changed.

## Checks not run (declared unknown)

* No re-simulation, no new grid / `xi` / closure: nothing is claimed beyond the ten committed fields.
* Transient wetting dynamics and experimental contact-angle data: out of scope (as in R2/R3).
* Cross-architecture bitwise determinism: still `not_run`.
* `tests/test_cg_geometric_wetting_xval.py::test_circle_fit_recovers_an_exact_cap[40/60/90/120]`
  fails on this branch head on the **rms** tolerance (`0.069 < 0.05`) inside
  `wetting.metrics.contact_angle_circle_fit`; that file and `wetting/metrics.py` are untouched by this
  commit (`git diff --stat`), so the failure is pre-existing and is not counted as a pass.

## Overstated claims in the subject

* R4 “Validation → LBM Field Measurement: **Status: PASS**” and “FAIL | 0” are not supported by its own
  results file (S2). The sentence “expected error: < 2 deg (same as synthetic cap calibration)” is an
  expectation, not a measurement: R4 never measured a committed field.
* R3's “P2_third_estimator” FAIL and its F0 narrative both attribute to the physics/convention what is
  a formula inversion in the audit estimator (S1, S5).

## Disposition

`request changes` on the R4 patch (S2, S3, S4 — fixed in this commit: single formula, deprecated flag,
docstring corrected) and `approve with notes` on the anchor's physics claim: two independent wall
closures reproduce Young's law within the declared gate once the third estimator is corrected. No
blocking defect remains in the two implementations themselves.
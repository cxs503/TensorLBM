# Adversarial audit — XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS (round 2 PASS)

> Repository migration note: archived campaign assets now live at [development/verification/contact-angle/round-2](../development/verification/contact-angle/round-2/), [development/verification/contact-angle/round-3](../development/verification/contact-angle/round-3/). Historical mission paths and verdicts remain tied to the original revision.

Seat: `tp-ver-2` (verifier). Round-3 audit of the merged round-2 deliverable
(`xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/`, `src/tensorlbm/wetting/two_impl_contact_angle.py`).

## 0. Reviewed identity and independence limits

| item | value |
|---|---|
| subject | round-2 harness + summary, commit `7a38ac06d` ... current HEAD of `agent/m21577` tree |
| subject claim | `Verdict: PASS`, "10/10 flux arm + 10/10 geometric arm of the measurable rows inside the 15 % gate", cross-arm max 6.84 % |
| audit artefacts | `xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS-R3/` (`predeclared_r3.json`, `run_r3.py`, `summary_r3.json`, `REPORT_R3.md`, `figures/`, `runs_r3/*.json`) |
| ticket | `tests/test_two_impl_contact_angle_r3.py` (15 tests) |

**Independence limitations (stated, not hidden).**

1. Same seat, same machine, same computation gate as the subject. Only the *measurement*
   implementation is independent: `run_r3.py` re-derives the angle from scratch and imports nothing
   from the round-2 measurement module. The relaxation (`relax_with_wall_condition`) and the free
   energy are **shared by necessity** — the anchor is about two wall conditions on one free energy,
   so re-implementing the solver would test something else.
2. The third estimator was calibrated on analytically known caps only (`Δθ ≤ 0.29°` for
   `ξ/R ≤ 0.12`, `≤ 1.61°` for `ξ/R ≤ 0.25`), before any audited row was measured.
3. `runs/` of round 2 is **not in the repository** (`.gitignore:108`), so the committed
   `summary.json` numbers were re-derived by re-running the same code, not from committed raw fields.

## 1. Scope and checks performed

* Re-ran the **entire predeclared round-2 matrix** (34 rows: 3 arms × 5 angles, refined grid,
  interface-width sensitivity, fine-interface window) and diffed every `theta_cap` /
  `theta_circle` against the committed `summary.json`.
* Wrote and calibrated a **third, from-scratch estimator** (`θ = 2 atan(a/h)`; `a` = basal half
  width from a cubic extrapolation of the mid-level contour to the wall plane, `h` = apex height),
  and re-measured every audited row with it.
* Probed the **level convention** (`0.5(gas+liquid)` vs `ψ=0` vs `0.5(gas+min)`) and the
  **concentric-offset law** of a diffuse circular interface.
* Re-relaxed, with `max_iter` raised from 30 000 to 200 000, every row that round 2 counted while
  it had *not* met its declared `tol = 1e-9`.
* Static (`grep`/AST-level) check of the claimed code independence of the two arms and of the
  implementation blindness of the measurement.
* Verified two invariants from the subject's own test suite are asserted on **numbers**, not merely
  executed: `test_young_law_from_first_principles` (free-energy integration vs closed form),
  `test_flux_and_geometric_ghosts_encode_the_same_angle` (field-level ghost equivalence).

## 2. Static findings (no re-run needed)

**S1 (minor, confirmed correct).** The measurement is implementation-blind: `measure_contact_angle`
takes `(psi, interface_width, wall_y, centre_x)` — no `theta` argument. `wall_angle_readback` *does*
take `theta` but is a diagnostic built from the *imposed* ghost, and the round-2 report says so
explicitly ("never as the acceptance quantity"); the README also states it returns non-angles for
the `dirichlet` arm (measured 53.4° at a prescribed 90°). Verified by reading both paths.

**S2 (minor, confirmed correct).** The two independent arms share no wall-closure code:
`neumann_flux` builds `ghost = lower + amp(1-ψ_wall²)` (3 fixed-point sweeps, no wall value),
`geometric_ghost` builds `lower + cot(θ)|∂_t ψ|` (no wall energy). The only shared code is the
free-energy gradient/Laplacian kernel, which is the point of the anchor.

**S3 (material — predeclared gate declared but not wired).** `predeclared.json` declares
`estimator_agreement_deg: 10.0` and `gas_supersaturation_abs: 0.10`, and `summary.json` reports
`estimators_agree` per row, but **neither is part of the pass rule**
(`predeclared_rules[0]` only requires both estimators inside 15 %). Consequences:
`dirichlet__th090__baseline` (gap 11.17°) and `dirichlet__th090__refined` (12.87°) are counted in
the "5/5 dirichlet pass" while flagging `estimators_agree = false`; and two rows
(`geometric_ghost__th070__xi4` +0.103, `neumann_flux__th070__xi4` +0.100) exceed the declared
0.10 supersaturation bound (both happen to be non-measurable). The gate checklist in `REPORT.md`
does honestly show `[ ] both shape estimators agree within 10°` unchecked, so the headline PASS is
not false — but the report does not state that the failure lands on a row it *counts*.

**S4 (minor).** `REPORT.md` artefact list points at `runs/<tag>/result.json` and
`figures/fig1_fields.png` … but `runs/` is gitignored and `figures/` holds no such files in the
committed tree: the round-2 numbers are reproducible from code but the committed evidence bundle
cannot be re-derived *offline*. My round 3 fixes this for its own rows (committed
`runs_r3/*.json` incl. all measured angles).

## 3. Reproducibility, third estimator and convergence

See `xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS-R3/REPORT_R3.md` §P1–P5 for the numbers
generated by the cluster run (`cj3549`), and `summary_r3.json` for the machine-readable rows.
Findings are severity-ranked in §5 below; the numbers quoted there are copied from
`summary_r3.json` by `run_r3.py report`, never typed by hand.

## 4. Checks NOT run (declared unknown)

* No re-run on the `loong64` HPC nodes with a different BLAS/thread count: bitwise determinism
  across architectures is **not** checked (`not_run`).
* No transient wetting dynamics (spreading law) — out of scope for a static anchor, and round 2
  says so.
* No comparison against experimental contact-angle data: the anchor's reference is analytic Young's
  law, so no experimental benchmark exists for this quantity.
* The round-2 `dirichlet` arm's strong-wetting failure was **not** re-examined with an alternative
  value-BC realisation.

## 5. Findings, severity-ranked

Severity is about whether the anchor's PASS claim survives, not about code taste.

| # | severity | finding | evidence | remediation |
|---|---|---|---|---|
| F0 | material (blocking for the *interpretation*, not for the numbers) | the relaxed cap has its sphere centre on the fluid side of the wall for every angle (apex `h/R` = 0.45/0.87/1.24/1.65 vs `1-cos theta` = 0.66/1.00/1.34/1.71), so a sessile-convention estimator reads the supplement; round 2 never states this convention | `runs_r3/*.json` (`apex_height_r3`, `basal_half_width`, `geodesic_levels`), REPORT_R3 §6 | state the convention in `predeclared.json`, or re-measure with a signed normal |
| F1 | not_run | convergence qualification of the 4 counted-but-unconverged rows (long re-relaxation `cj3549` queued, not dispatched) | `run_r3.py jobs_long()`; `summary_r3.json` absent P4 keys | run `cj3549`; if the angle moves > 0.5 deg, restate the pass count |
| F1b | material | reproduction of the round-2 numbers is exact (`max |delta| = 0.000 deg`, 14 rows) but one NaN-vs-NaN row is flagged as "not reproduced" by the summariser | `summary_r3.json` → `probes.P1_*` | treat NaN rows explicitly |
| F2 | material | declared thresholds `estimator_agreement_deg`, `gas_supersaturation_abs` are reported but not gated (S3) | `predeclared.json` vs `run_xval.py` measurability block | wire them into the gate, or delete them from `predeclared_thresholds` |
| F3 | minor | raw `runs/` not committed → committed numbers are code-reproducible but not evidence-bound offline (S4) | `.gitignore:108`; `REPORT.md` §6 artefact list | commit per-row JSON (as round 3 does) |
| F4 | minor | cross-arm table includes θ=90°, where both closures degenerate to the same zero-flux mirror (`cos 90 = cot 90 = 0`), so `diff = 0.00 %` is an identity, not a validation | `summary.json` cross_arm rows; `wall_ghost_rows` | restate cross-arm agreement on the non-degenerate angles (round 3 does) |
| F5 | honest limitation (unchanged) | θ ≤ 45° and the `dirichlet` arm below 90° are not measurable at ξ/R = 0.19 | round-2 §6; re-confirmed by the third estimator | none — correctly reported as NOT_MEASURABLE |

## 6. Overstated claims in the subject

* "10/10 + 10/10 … inside the 15 % gate" is **true as computed** but silently mixes rows that met
  the declared tolerance with rows that stopped at `max_iter`; the report asserts "the angle is
  stationary to <0.05° long before the 1e-9 drift tolerance" without showing the stationarity test.
  Round 3 supplies that test (P4).
* "the largest cross-implementation difference over the measurable angles is 6.84 %" is true, but
  its minimum (0.00 % at 90°) is structural, not a measurement (F4).

## 7. Disposition

`approve with notes` — the physics claim (two independent wall closures reproduce Young's law on a
shared free energy within the owner's 15 % gate) is reproduced by an independent measurement route;
the notes are F1–F4 above. Blocking issues: none. Nothing in the audit contradicts a merged
deliverable, and no subject file was modified.
# XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS

Decision (predeclared in `predeclared.json`): TensorLBM cross-validation claims are
certified **only** by a machine-enforced, CI-executed assertion gate; a non-finite
metric is always a failure.

## What the audit found

1. `tensorlbm.verification.xval` shipped three cases with declared thresholds
   (Poiseuille < 2 %, Ghia cavity Re=100 < 15 %, Womersley alpha=5 < 15 %) but **no test
   executed them** — the thresholds were metadata, not checks.
2. The registry decided pass/fail with `metric > threshold`. In IEEE-754 `nan > 0.15`
   is `False`, so a diverged run was certified **PASS**. Fixed in
   `src/tensorlbm/verification/xval/registry.py` (fail-closed on non-finite metrics,
   reason recorded in `details`).
3. With the assertion in place, the experimental arm fails: the lid-driven cavity solver
   diverges to NaN at step 39 (n=33) and step 36 (n=65), for both BGK and
   RLBM collision. The Ghia < 15 % claim was therefore never verifiable. The case is kept
   as a strict `xfail` health arm, so the gate turns red as soon as the solver is repaired
   and the real error has to be measured (`fig2_cavity_divergence.png`).
4. Fast tier is cheap only single-threaded: same Poiseuille case takes 9.9 s at torch's
   default 128 threads vs 4.2 s at 1 thread.

## Measured this round (artifacts, not hand-copied)

| arm | anchor | metric | value | threshold | verdict |
|---|---|---|---|---|---|
| poiseuille_2d_H16 | analytic u(y)=g/(2 nu) y (H-y) | eps_inf | 0.361% | 2.0% | PASS |
| | | eps_L2 | 0.492% | 2.0% | PASS |
| | | mass_drift | 3.08e-13 | 1e-08 | PASS |
| lid_cavity_re100_n65 | Ghia et al. (1982) | l2_u / l2_v | NaN | 15 % | **FAIL (diverges at step 36)** |
| lid_cavity_re100_n33 | Ghia et al. (1982) | l2_u / l2_v | NaN | 15 % | **FAIL (diverges at step 39)** |
| womersley_alpha5 | Womersley (1955)/Sexl (1930) | amplitude/phase | not completed | 15 % | untested (wall clock, >110 s/run) |

`results.json` sha256 = `78f5a4d43c0b320cb6c0689625185ee4a8e20d2aafe4e308d010ed9f78896480` (git 0a1cf0e20c45f242e3a23d6c5a5118ea1b7a4830, torch 2.7.1, loongarch64).

## Reproduce

```
pytest -q tests/test_xval_assert_gate.py                                  # CI gate, ~9 s
python xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/run_xval.py --cases fast   # writes results.json/kpi.csv
python xval/XVAL-LBM-NO-TESTS-NO-CI-NO-ASSERTS/make_figures.py            # figures/
```

## Method selection (why this and not the alternatives)

* (A) Declarative cases + manual runs — rejected: measured, nobody ran them, and one arm
  certified NaN as PASS.
* (B) Coverage / test-count KPIs — rejected: asserts nothing about accuracy.
* (C, selected) pytest gate executing the anchor cases with each declared threshold
  asserted, fail-closed non-finite semantics, wired into `.github/workflows/ci.yml`
  ("Cross-validation assert gate" step). Rationale: the V&V contract is per-metric, so the
  check must be per-metric and must run on every push (cf. ASME V&V 20-2009 §3
  verification/validation separation; Krüger et al. 2017 §4.4 determinism).

## Open items (honest)

* Cavity divergence root cause not yet isolated. Sign of `torch.roll` shifts in
  `CavitySolver._stream` is inconsistent with the `C` direction table under the
  "row 0 = bottom" reading, but flipping it alone moves divergence onset from step 36 to
  129 (n=33) instead of removing it, so the Zou-He lid / bounce-back direction sets are
  also suspect. Not fixed here — no fix was claimed without evidence.
* Womersley arm needs a runtime reduction (shorter H or fewer measured periods) before it
  can enter CI.

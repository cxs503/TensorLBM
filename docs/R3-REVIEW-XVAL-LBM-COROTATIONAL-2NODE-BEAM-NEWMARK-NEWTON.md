# R3 — Independent review of XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON

**Role:** verifier seat (`deepseek-v4-flash` pinned), mission 22901, repo `TensorLBM`.
**Object of review:** the merged evidence bundle of mission 17723 — `docs/XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.md`
(rounds 1+2), `development/verification/corotational_beam_newmark/{table.csv,summary.json,torder.csv,torder_summary.json}`,
`src/tensorlbm/corotational_beam.py`, `tests/test_corotational_beam.py` — all at the reviewer's base
commit (branch `agent/m22901`, parent `b021537dd`).
**Independence limitations (declared):** the reviewer did *not* re-clone `joconnor22/LIFE`; the
"LIFE arm" was reviewed as implemented in this repository (`formulation="life"`), so the fidelity
of the transplant to `LIFE/src/FEMElement.cpp` is taken from the round-1 file:line inventory and was
**not** re-verified line by line. No OSS binary was run.

---

## 1. What is *not* in doubt

* The static arms (S1/S2/S3) reproduce the external references: elastica tip deflection and
  circular-arc pure bending, max |error| 2.2e-2 % / 4.0e-2 % over the predeclared mesh ladder
  (`table.csv`). Reference cross-checks (two independent elastica routes) agree to ~1e-12.
* The round-2 corrections are real and are reproducible: `euler_bernoulli_frequency` now returns the
  roots 3.51601527, 22.03449156, 61.6972140, 120.901916, 199.859530 (tabulated agreement 2.6e-3 %
  at n=32, `torder_summary.json/T1`), and the temporal order of the linearised arms is 2
  (1.998 free / 2.003 forced, `torder_summary.json/T2`).
* `pytest tests/test_corotational_beam.py` → **19 passed** at my base commit (re-run here, 9.1 s).

## 2. Findings, severity-ranked

| id | severity | finding | evidence | reproduction |
|---|---|---|---|---|
| **F10** | **blocking (claim falsified)** | The merged D3 number (*"energy drift 0.0045 %, non-dissipative"*) is **not reproducible with the current code**. With the round-2 fix (consistent initial acceleration, `qddot = M⁻¹(f−g)`) the λ=5 release case **diverges** at the predeclared step `dt = T1/200`: TL `max|q| = 4.96e7 L`, 56/200 steps at `maxit`; LIFE `max|q| = 2.91e7 L`, 190/200 steps at `maxit`. The merged numbers were produced by the **pre-fix** code, whose zero initial acceleration incidentally damps the startup. Round-2 §R2.5 asserted that a re-run *"can only change the drift at the floor level"* — **that expectation is falsified**: the point does not converge at all. | `development/verification/corotational_beam_newmark/d3order_summary.json` → `arms.*.ladder`; `d3order.csv` rows `status=FAILED`; new test `test_underresolved_d3_point_is_reported_not_dropped` | `python -c "import run_d3order as m; m.FRACS=(200,400); m.main()"` (58 s) |
| **F7** | **blocking (reporting defect)** | **Silent censorship of failed predeclared points.** `case_d3` appended a KPI row only when `converged` was true, and `assemble()` never checked that every predeclared point produced a row. Four D3 points were predeclared; `table.csv` contains **three**; `summary.json` reads `D3_NONLINEAR_ENERGY: {n: 3, n_fail: 0}` and `n_fail15 = 0`, i.e. a divergence was reported as a clean pass. The missing row is exactly `TL @ dt=T1/200`, the point that diverges (F10). | merged `table.csv` (66 rows) vs `predeclared.json` §cases.D3_NONLINEAR_ENERGY | `git show HEAD:xval/.../table.csv \| grep D3` → 3 rows |
| **F8** | **major (interpretation)** | **Step adequacy of the dynamic stages was never stated.** The ladder is quoted as `dt = T1/200`, but the stiffest *discrete* mode is not the fundamental: at n=16, `ω_max = 39.14 s⁻¹` (`T_min = 0.1605 s`) versus `ω1 = 0.01015 s⁻¹` (`T1 = 619 s`), a ratio **3857**. So `dt = T1/200 = 3.10 s` is **19.3 × T_min** (`ω_max·dt = 121`); even the finest merged point `T1/1000` sits at `ω_max·dt = 24`. Nothing in the merged report states this, so "converged at T1/200" cannot be read as "the nonlinear integrator is validated at that step". Measured here per ladder point (`d3order.csv`, KPI `omega_max_dt`). | `d3order_summary.json` → `step_adequacy` | `omega_max` from the generalised eigenproblem of `K_T(0)`, `M` on the free DOFs |
| **F9** | **major (metric validity)** | The D3 "energy" is `KE + strain_energy` of the **consistent** family. `strain_energy` is not a potential of the LIFE force vector, so for the LIFE arm the number measures the **formulation mismatch**, not energy conservation; the merged report quotes it as an energy drift. Measured: at resolved steps the LIFE "drift" is **dt-independent** (2.64e-5 % → 3.04e-5 % when dt is halved), while TL decreases (8.06e-6 % → 3.84e-6 %), i.e. the LIFE floor is model-inconsistency, not discretisation. Row notes in `d3order.csv` carry the caveat. | `d3order_summary.json` → `resolved_window` | `python -c "import run_d3order as m; m.FRACS=(); m.RESOLVED_DTS=(0.04,0.02); m.main()"` (48 s) |
| **F11** | minor | Predeclaration deviation: `predeclared.json` §predeclared_thresholds.stability_rule promises `maxit=30` **with load substepping** for a failing solve, but `case_d3` uses the `solve_newmark` default `maxit=20` and no substepping. The stage therefore cannot satisfy its own declared failure protocol. | `run_xval.py::case_d3` vs `corotational_beam.py::solve_newmark(maxit=20)` | code read |
| **F12** | minor | R2.5 item 2 ("`table.csv`/`summary.json` still carry the pre-fix D1 rows") is still open: the committed `summary.json` reports `n_fail15 = 2` (D1) although R2.3.1 demonstrates 0 KPI rows fail. A reader of `summary.json` alone gets the opposite verdict. | merged `summary.json` → `by_case.D1_MODAL` | `python -c "import json;print(json.load(open('.../summary.json'))['by_case']['D1_MODAL'])"` |

Severity scale: *blocking* = a claim of the report cannot be maintained; *major* = the claim is
unsupported or mis-scoped, numbers usable with a caveat; *minor* = bookkeeping/predeclaration.

## 3. Remediation applied in this round (code, in the worktree)

1. `development/verification/corotational_beam_newmark/run_xval.py`
   * new `record_fail(...)` → a failed point is a row with `status="FAILED"`, `pass15=False`,
     `got=NaN` (no fabricated number);
   * `case_d3` now emits a FAILED row for a diverged/non-converged Newton solve instead of
     dropping the point, and annotates the LIFE rows with the F9 caveat;
   * new `predeclared_points()` + an `assemble()` guard that **raises** if a predeclared point is
     absent ("silent censorship guard F7");
   * `summary.json` gains `n_failed_points`, `failed_points`, per-case `failed` lists, and a NaN-safe
     `max_abs_err_pct`; `status` is a column of `table.csv`.
2. `development/verification/corotational_beam_newmark/run_d3order.py` (new): the step-adequacy diagnostic
   (`ω_max` from the discrete spectrum), the predeclared D3 ladder re-run with explicit
   PASS/FAILED status, and a **resolved-step** sub-study D3R (fixed 20 s window,
   `dt ∈ {0.04, 0.02, 0.01, 0.005} s`, i.e. `ω_max·dt ≤ 1.6`) with the drift order in dt and
   TL-vs-LIFE trajectory parity.
3. `tests/test_corotational_beam_xval_harness.py` (new, 4 tests): pins F7 (a failed point is
   visible; a table missing a predeclared point raises) and re-runs the under-resolved D3 point as a
   regression test.
4. `figures/fig_d3_resolved.png`: drift vs dt for both arms with the predeclared 2 % threshold and
   the round-off floor marked.

## 4. Re-measured numbers (round 3, all injected from the artifacts, no hand copy)

### 4.1 Step adequacy (n=16, A=1.2e5, EI=1)

| arm | ω1 [1/s] | T1 [s] | ω_max [1/s] | T_min [s] | ω_max/ω1 | ω_max·dt @ T1/200 | @ T1/1000 |
|---|---|---|---|---|---|---|---|
| TL / LIFE | 0.010150 | 619.04 | 39.145 | 0.1605 | 3857 | **121.2** | 24.2 |

### 4.2 Predeclared D3 ladder, post-fix code

| arm | dt | ω_max·dt | max\|q\|/L | unconverged steps | status |
|---|---|---|---|---|---|
| TL | T1/200 | 121.2 | 4.96e7 | 56 / 200 | **FAILED (diverged)** |
| TL | T1/400 | 60.6 | 5.74e6 | 101 / 400 | **FAILED (diverged)** |
| LIFE | T1/200 | 121.2 | 2.91e7 | 190 / 200 | **FAILED (diverged)** |
| LIFE | T1/400 | 60.6 | 8.31e7 | 199 / 400 | **FAILED (diverged)** |

(`T1/800 … T1/3200` were predeclared in `run_d3order.py` but the full ladder needs the cluster run
`cj4471`; the local shell budget allowed only the two coarsest points. Declared not-run, not passed.)

### 4.3 D3R — resolved steps, 20 s window, λ=5 release, n=16

| arm | dt [s] | ω_max·dt | steps | drift [%] | tip y/L at t=20 s |
|---|---|---|---|---|---|
| TL | 0.04 | 1.57 | 500 | 8.062e-6 | −0.700071 |
| TL | 0.02 | 0.78 | 1000 | 3.843e-6 | −0.700070 |
| LIFE | 0.04 | 1.57 | 500 | 2.638e-5 | −0.700066 |
| LIFE | 0.02 | 0.78 | 1000 | 3.040e-5 | −0.700064 |

TL-vs-LIFE trajectory parity in the nonlinear regime: **2.9e-4 %** (2/2 resolved points, both
`< 15 %`) — the arm-parity hypothesis H1 therefore also holds in the nonlinear dynamic regime, not
only in the static arms.

Note on the drift *order*: the TL value changes by a factor 2.1 when dt is halved (observed p = 1.07
from two points), and a repeat run in the same session gave 2.77e-7 % / 6.47e-8 % (p = 2.10). The
drift at these steps (1e-8 … 1e-7 of the total energy) is at the accumulation floor of the Newton
tolerance (`tol = 1e-10` relative residual over 500–1000 steps), so **the metric cannot resolve a
clean dt² law below ~1e-6 %**; the *order* claim is therefore reported as floor-limited, not as a
measurement of second-order energy behaviour. The magnitude claim (drift ≪ 2 % predeclared
threshold) is unaffected.

## 5. Disposition

**Request changes → accepted with caveats after remediation.** The static/modal/linear-dynamics
claims stand. The *nonlinear dynamic* claim of the merged report does not: it is replaced by
§4.2–§4.3 above, with the failed predeclared points now visible in `table.csv`/`summary.json`.

Checks **not run** (declared, not passed):

* full post-fix stage set `S1 S2 S3 D1 D2 D3 cons` (`python run_xval.py --stage all`) and the full
  D3/D3R dt ladder — submitted as cluster job **cj4471** (`run_compute status/fetch`), still queued
  at the end of this mission; the merged `table.csv` D1 rows therefore remain pre-fix (F12 open);
* line-by-line re-verification of the LIFE transplant against the OSS sources (clone not available
  in this worktree);
* 3D / shear-deformable / fluid-coupled behaviour (out of the declared scope).

## 6. Suggested follow-up ledger entry

`XVAL-LBM-COROTATIONAL-2NODE-BEAM-D3-RESOLVED` — acceptance: *"λ=5 release at n=16, dt chosen so
ω_max·dt ≤ 0.8, drift over 20 s < 0.1 % for both arms and TL drift monotone decreasing in dt over
≥3 refinements (order ≥ 1.5 measured above the round-off floor, Newton tol ≤ 1e-12); explicit
FAILED rows for every non-converged predeclared point"*, domain: LBM-FSI structural sub-solver.

Repository relocation: [canonical case index and historical evidence](../benchmarks/cross_validation/corotational_beam_newmark/README.md). The independent review and its limitations remain unchanged.

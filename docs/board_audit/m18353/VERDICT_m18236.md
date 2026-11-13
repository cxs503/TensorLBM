# AUDIT m18236 — XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT (TensorLBM)

**Seat**: board-audit (交差核验单, non-redo) · **Audit mission**: m18353 · **Branch**: `agent/m18353`
**Audited mission**: m18236 (status `succeeded`, dev_acceptance auto-approved 2026-11-13T13:30Z)
**Audited tree**: `/root/tp-agent/work/m18236` (HEAD `8ba788ba` + 5 untracked delivery paths)
**Platform report**: `tensor_reports.id=17163`
**Verdict: L2** — real protocol code + runnable entry + reproducible artifacts + 4/4 tests, but
**no production caller (≠L3)** and **no machine referee + no usable error number (≠L4)**.
The mission itself declares honest-FAIL on the physics gate; that declaration is **confirmed by
reproduction**, and one of its causal claims is **refuted** (§Findings F2).

## 1. Scope and what was checked

| check | how | result |
|---|---|---|
| delivered files exist | `find` on the 5 untracked paths + `git status` | ✅ `src/tensorlbm/pc_ladder.py` (628 L), `tests/test_pc_ladder.py` (91 L), `TensorLBM/xval/XVAL-…/{run_xval.py,run.log,artifacts/}` (24 checkpoint dirs), `TensorLBM/docs/XVAL_QUASI_STATIC_PC_LADDER.md` (162 L) |
| unit tests | `PYTHONPATH=<m18236>/src pytest tests/test_pc_ladder.py -q` (sandbox copy of the test file, audited tree untouched) | ✅ **4 passed in 6.29 s** (claim "4/4" true) |
| runner re-run | copied `run_xval.py` into an audit sandbox, ran it: `rc=0`, 80.6 s | ✅ same qualitative outcome as delivered |
| artifact determinism | 683 flat scalars of `results.json`, delivered vs audit rerun | ✅ **657 identical (NaN==NaN)**, 26 differ = `wall_time_s` + absolute `checkpoint.*.path` / `checkpoint_dir` strings only |
| checkpoint integrity | `load_ladder_state(<delivered ckpt/main/rung_000>)` with `verify_digest=True`, and `rung_005` | ✅ SHA-256 verify passes both; rung_005 restores `step=1200` and `f` is **non-finite** (consistent with the NaN ladder) |
| machine referee | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-QUASI-STATIC-PC-LADDER-WITH-CHECKPOINT'` | ❌ **0 rows** (whole table = 102 rows, `repo='TensorLBM'` = 39) → no external referee exists ⇒ L4 not awardable |
| integration | `grep -rn pc_ladder src/ --include=*.py` (excluding the module) ; presence on main (`/root/tp-agent/work/m18353/src/tensorlbm/pc_ladder.py`) | ❌ **0 callers**, module not on main, not exported |
| report six elements | `tensor_reports/17163` markdown, 9.8 kB, §1 problem/ref, §2 method, §3 parameter table, §4 ref-vs-got tables, §5 sensitivity, §6 failure mode + hand-off, plus OSS-diff table | ✅ six elements present, numbers match `results.json`; ⚠️ **no explicit "是否 <3%" statement** (only the 15 % owner gate is used) — report-discipline gap |

Independence limitations: same host, same python 3.11.6, same git object; no second implementation
of the ladder was written; no cluster/loong64 rerun (80 s single-core job). Independence levers
actually used: artifact-vs-rerun scalar diff, delivered-checkpoint digest restore, and a
**counterfactual probe the mission never ran** (plug evolution at `g = 0`).

## 2. Findings (severity ranked)

**F1 · blocking (delivery-level, already declared honest-FAIL) — the anchor's acceptance quantity
does not exist.** Every rung of interest returns `nan`: `results.json:cases.E1_ladder.rows[1..6].meas.ok=false`,
`laplace.relerr=nan`; the only non-NaN residual is rung 0, `7.316e+25`, which is a pure
divide-by-zero artifact (`drho_g_L = 0`, `young_laplace_residual` floors the denominator at 1e-30,
`pc_ladder.py:611-628`). E3 single-shot and E4 `rungs4` are NaN as well. The only real error numbers
in the whole delivery are E4 `conv1e-2` rungs 1–2: **47.9 % / 51.2 %** (>15 % ⇒ FAIL).
Reproduce: `PYTHONPATH=<m18236>/src python3 run_xval.py` (audit log
`.audit_tmp/xvalrun/run.log`, identical rows).

**F2 · high (causal claim refuted) — the plug dies without any imposed pressure.** The report
attributes the plug loss to spurious currents raised by the contact-angle wall model **under the
imposed ramp** ("with `wall_phi_policy="contact"` … spurious currents rise to 1.3e-2 … the liquid
plug stops being a single centre-row interval"). Audit probe (600 steps, `build()` from the delivered
runner, body force fixed): with `g = 0` the field is already non-finite at **step 380**; with the
rung-1 force `g = 1.929e-05` it dies at **step 390**. A 10-step (≈2.6 %) difference ⇒ the pressure
ramp is not the trigger; the plug initial condition on the quantised taper with contact walls is
intrinsically unstable. Consequence for hand-off: round 2 must stabilise the IC / wall model
**before** the ladder can be re-run; the delivered hand-off list (1. orientation-aware wall model,
2. measured L, 3. curvature-only route) is necessary but silent on the free-decay instability.
Evidence: `probe_nan_onset.json`, `fig_m18353_m18236_audit.png` (right panel).

**F3 · high — "with-checkpoint" resume equivalence is unmeasured end-to-end and under-asserted.**
E2 aborts in the delivered script: `results.json:cases.E2_resume.error =
"TypeError: 'RungRecord' object is not subscriptable"` (`run_xval.py:e2_resume`, audited rerun
reproduces: `[E2] FAILED 'RungRecord' object is not subscriptable`). The report discloses the bug
(honest), but the unit test offered as substitute, `test_ladder_checkpoint_roundtrip_and_resume`,
asserts only rung ids `[0,1,2]`, digest length 64, restore success, and tamper rejection — **it never
asserts the trajectory equality its docstring promises** ("resume reproduces the continuous trajectory
exactly"). So no surface of the delivery verifies the resume-vs-continuous invariant on any scale;
what is proven is checkpoint write/read + digest integrity.

**F4 · moderate — gate function has no `g = 0` guard.** `young_laplace_residual` returns
`relerr = 7.3e25` instead of `nan`/skip when the RHS is 0; rung 0 is logged as a residual in the
report table, which is numerically meaningless noise in a ref-vs-got table.

**F5 · moderate — provenance.** (a) `xval/…/run.log` (mtime 13:18:34) is **stale**: it is the log of the
pre-fix crash (`RuntimeError: stack expects each tensor to be equal size … at pc_ladder.py:444`), while
`results.json` (13:28:48) comes from a run after the 13:27 fix — the delivered log does not correspond
to the delivered numbers. (b) `results.json` embeds absolute mission paths. (c) All delivery files are
**untracked** in `m18236` and absent from main, so artifacts are reachable only from that worktree.

**F6 · low — report discipline.** No explicit answer to "是否 <3 %" for any KPI, although the E0
curvature route passes it (0.052 %) and the Laplace route fails it (38.36 %); the six-element
requirement asks for that sentence.

## 3. Level rationale (L0–L5)

* **L1 ✅** 4 invariant-style unit tests, all passing on audit rerun (schedule == upstream analytic
  step; curvature estimator recovers `1/R_eq` within 5 %; taper staircase property; checkpoint
  roundtrip + tamper detection).
* **L2 ✅** real 628-line code (schedule / taper builder / meniscus + curvature measurement / rate-based
  settle / complete-state `(f, g, phi*u_prev)` checkpoint with SHA-256 / `run_ladder` / YL residual),
  runnable entry, 5 artifact classes including 24 checkpoint dirs, and **bit-level reproducible
  physics numbers** on rerun.
* **L3 ❌** zero callers (`grep` in `src/` = 0), not exported, not on main; the only driver is a private
  `xval/` script.
* **L4 ❌** `tensor_benchmark_cases` has **0 rows** for this anchor; additionally the end-to-end error
  number is undefined (NaN), so even a self-authored referee would have nothing to compare.
* **L5 ❌** by construction.

Report length (9.8 kB) and figure count do not move the level; level = functional substance.

## 4. Overstated claims

1. m18236 approval summary: *"E0/E1/E3/E4 全跑通"* — E1/E3/E4 **execute** but emit NaN; "跑通" reads as
   "produced results", the artifacts show "produced no measurable result after rung 0".
2. Docs/report: *"protocol 移植 verified … the transplanted model is the weak link"* — the checkpoint
   protocol's integrity part is verified (F3 caveat: resume-equivalence is not), but the weak link is
   the **host CAC solver + plug IC on the quantised taper** (F2), not the imposed-pressure interaction.
3. Report headline framing "curvature measurement chain validated to 0.05 %" is accurate but narrow:
   it holds for the static **droplet** probe, not for the channel plug where every curvature is NaN.

## 5. Checks not run

* No cluster/loong64 rerun (single-core 80 s job; architecture not in play).
* No independent re-implementation of the ladder protocol (only the delivered code was exercised).
* No attempt to repair the divergence — out of scope for a cross-check audit (it is the round-2 item).
* OSS source `je-santos/MPLBM-UT :: ShanChen.cpp` lines cited in the report were **not** re-read; the
  schedule formula was validated only against the report's own algebraic claim in
  `test_schedule_oss_step_equals_entry_pressure`.
* CI/PR path not involved (delivery is untracked, no PR).

## 6. Disposition

**Approve with notes** as an honest-FAIL protocol delivery, graded **L2** (confirmed by the machine
gate). Do not promote to L3/L4; do not treat the E0 pressure route or the ladder residual as evidence
of capability. Round-2 hand-off must add: (i) plug-IC/wall-model stability at `g = 0` (F2), (ii) an
assertion of resume ⇒ continuous-trajectory equivalence, not only digests (F3), (iii) a registered
`tensor_benchmark_cases` row for the anchor before any L4 claim.

**Artifacts of this audit**: `docs/board_audit/m18353/{VERDICT_m18236.md, audit_m18236.json,
probe_nan_onset.json, fig_m18353_m18236_audit.png}`, `verdict_m18236.sql`; audit-side scripts kept in
`.audit_tmp/` (probe_nan.py, probe_ckpt.py, make_audit_artifacts.py — sandbox only, not deliverables).
# AUDIT m18427 — XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE (subject mission m18288)

Repo: TensorLBM · Auditor: board-audit (m18427) · Tree audited: `agent/m18288`
(1bdad75e / b5e8eaf4) against `main` = 2a0bd485 · Audit worktree commit base: 2a0bd485

Verdict: **L2** (recorded in `tensor_maturity_verdicts`, verdict_by=`board-audit`).
This is a cross-check (交差核验) audit, not a redo. Independence limitation: the auditor
runs in the same container as the subject; verification is by re-derivation from committed
raw artifacts and by re-execution, not by a second physical machine.

## 1. What the subject claims

`tensor_reports` id 17179 (mission 18288) claims L2/L3/L4 machine-checkable evidence:
4 implementation files, an entry point, 8 run artifacts, 3 "real integration callers",
8 benchmark rows + repro_cmd, and a kernel-level cross-check against an executed OSS
source (`Maarten-vd-Sande/lbm@573a01c`).

## 2. Checks performed (all run in this audit)

| # | Check | Command / source | Result |
|---|---|---|---|
| C1 | Entry reproducible | `python3 benchmarks/xval_method_selection_bench.py --smoke` | ok, 1200 steps, 18.4 s, St=0.17168, Cd=1.23934 |
| C2 | Aggregation reproducible | `python3 xval/benchmark_method_selection/analyze.py` | `git diff` **empty** — summary.json / table.csv regenerated with zero numeric drift |
| C3 | Numbers re-derived independently | `docs/board_audit/m18427/audit_m18288.py` (reads only `runs/*/result.json`, `fail.json`, `oss_crosscheck.json`) | 6 KPI rows reproduced **bit-identically**; `recompute_mismatch_vs_published = []` |
| C4 | Test suite | `pytest tests/test_xval_benchmark_method_selection.py -q` | **5 passed** in 60.05 s |
| C5 | L3 caller reality | `grep -n select_collision` over `src/`, `benchmarks/`, `tests/` on `agent/m18288` | only `benchmarks/case_pool/xval_method_selection/run_case.py:55` calls it |
| C6 | Case-pool integration executes | branch tree copied to `/tmp/m18288tree`, `PYTHONPATH=src python3 benchmarks/case_pool/xval_method_selection/run_case.py --smoke` | ok — reproduces `cd_mean=1.2393432470464272` = committed `artifacts/.../a2_bgk_d6/result.json` |
| C7 | Benchmark referee registered? | `SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE'` | **0 rows** |
| C8 | Platform benchmark evidence | `SELECT count(*) FROM tensorbench_results WHERE mission_id=18288` | **0 rows** (0 with `source='cluster_job'`) |
| C9 | Merge state | `git branch -a --contains 1bdad75e` | `agent/m18288` only — not on main |
| C10 | Resolution ladder | `ls xval/benchmark_method_selection/runs/` | `A1..A5_D10` only ⇒ single rung, D=14/20 absent |

## 3. Findings (severity ranked)

### F1 — L4 is not established (HIGH, claim overstated)
The report's maturity table ticks **L4 ✅ "8 行基准 + repro_cmd"**. Decomposition of those 8 rows:
* 6 rows are **internal arm contrasts** whose `ref` is the A1 arm of the same experiment
  (`ref_source = runs/A1_D10/result.json`) — a self-reference, not an external, analytic or
  literature value. Audit re-derivation: `Cd −9.128%`, `Cl_amp −46.566%`, `St +4.517%` (A2);
  `Cd +1.827%`, `Cl_amp −15.397%`, `St +1.640%` (A4). **2 of 6 are outside the owner 15% band**
  (`Cl_amp`, both arms).
* 2 rows are a genuine external anchor, but **kernel-level only**:
  `k1_collision rel-L2 = 1.18e-07`, `k2_streaming rel-L2 = 2.71e-08` vs the executed OSS source
  (`oss_crosscheck.json`, OSS commit 573a01c, per-file sha256 recorded). Excellent parity, but it
  validates operator transcription, not case physics.
* No registered referee: `tensor_benchmark_cases` has **0 rows** for the anchor and
  `tensorbench_results` has **0 rows** for mission 18288 ⇒ the platform l4-verdict path is
  `CLAIMED_NOT_PROVEN`, exactly as the report's §5 boundary paragraph honestly states.
* Only **one** resolution rung exists (D=10); D=14/20 are honest `not_run`. Convergence of the
  case-level KPIs is therefore unproven, so a "complete 算例 + 误差分析" L4 claim cannot stand.
The report is transparent about the WITHHELD Norberg/Sohankar tables and the platform boundary,
but the L4 tick in the summary table is contradicted by its own §5/§6 — the tick is the overstatement.

### F2 — L3 claim "3 个真调用方" is inflated (MEDIUM)
`maturity_evidence.json → levels.L3.integration_callers` lists 3 entries with
`is_real_caller: true`. Reality:
1. `benchmarks/case_pool/xval_method_selection/run_case.py:55` — **genuine consumer**
   (`sel = select_collision(nu=nu, wall_bounded=True, ...)`).
2. `benchmarks/case_pool/run_pool.py` — a **driver** that launches the case; it does not consume
   the API (its only match is a usage string on line 12).
3. `src/tensorlbm/method_selection.py` — the module **itself**; a module is not its own caller.
Main-solver integration is absent: `src/tensorlbm/solver.py` never imports `method_selection`, and
`general_sim.py`'s `_auto_select_collision` (lines 348/425/507) is a pre-existing, independent
Re-threshold selector that predates and never touches this module. Net: **one** real caller
(case-pool harness) ⇒ "接入主求解器" is not satisfied.

### F3 — deliverable is branch-only (MEDIUM, recurs from m17990 audit F2)
`src/tensorlbm/method_selection.py`, `benchmarks/case_pool/xval_method_selection/*`,
`xval/benchmark_method_selection/maturity_evidence.{py,json}` exist on `agent/m18288` and are
**absent from main 2a0bd485** (verified: file-not-found in the main worktree; `git branch
--contains 1bdad75e` → `agent/m18288` only). The L3 evidence therefore rests on a side branch;
mainline consumers cannot import `select_collision`.

### F4 — prior high-severity defect F1 of the m17990 audit is FIXED (RESOLVED, positive)
The m18213 audit found `verdict.H_gap_survives_refinement=True` published from a single rung.
`grep H_gap_survives_refinement xval/benchmark_method_selection/analyze.py` on main → **0 hits**;
the published verdict block is now
`{H1_all_oss_arms_within_15pct: false, largest_abs_delta_pct: 46.566, judged_at_resolution: 10}`
with `gap_ladder: null` instead of a fabricated refinement claim. Remediation verified.

### F5 — tests assert aggregation invariants, not the selection semantics (LOW)
5 tests pass, covering the arms table, drag sign/O(1), ≤50% arm disagreement and
summary/ladder self-consistency. No test exercises `select_collision`'s TRT-window clamp or the
A3/A5 blow-up classification, so the branch-only library module ships untested.

### F6 — honest FAILs are correctly recorded (POSITIVE, no defect)
A3 (`tau_minus=25.5` from the literal magic Λ=3/16) and A5 (`tau_minus=2.0` remediation) both
`blowup / non_finite_population` at steps 600 and 5800 of 21259, with `fail.json` artifacts. The
deliverable states the OSS TRT selection is not transplantable as-is — a real negative result, not
papered over.

## 4. Level decision

| Level | Requirement | Met? |
|---|---|---|
| L1 | math/unit validation | yes — kernel parity vs executed OSS source at 1e-7/1e-8 (round-off floor), tests |
| L2 | independent functional implementation (input→compute→output) | **yes** — `run_arm.py` drives `tensorlbm.solver.{collide_bgk,collide_mrt,collide_trt,stream}` over 21259 steps × 5 arms, real artifacts, reproducible to the bit |
| L3 | system integration (main solver / platform) | no — one case-pool caller, main solver never imports the module, all of it branch-only (F2, F3) |
| L4 | benchmark validation (complete case + error analysis + reproducible) | no — internal-only reference, 2/6 rows outside 15%, no registered case/result row, single resolution rung (F1) |
| L5 | research-engineering result | no |

**Level = 2.** Disposition: **approve with notes** — the L2 core is sound and reproducible, the
honest-FAIL discipline and the F4 remediation are good; the L4/L3 ticks in the report must be
corrected in the next round, and the deliverable must be merged before any L3 claim is reconsidered.
Anchoring note for the anchor owner: register the referee rows in `tensor_benchmark_cases`
(case-level KPIs with an external `ref_source`, not another arm) and land ≥2 rungs (D=14/20)
before L4 is attempted — otherwise L4 is structurally unreachable for this anchor, as the m18213
audit already noted.

## 5. Reproduce this audit

```bash
python3 docs/board_audit/m18427/audit_m18288.py            # → audit_numbers.json + fig_m18288_audit.png
python3 benchmarks/xval_method_selection_bench.py --smoke
python3 xval/benchmark_method_selection/analyze.py && git diff --stat
python3 -m pytest tests/test_xval_benchmark_method_selection.py -q
psql -U postgres -d tensorplatform -c "SELECT count(*) FROM tensor_benchmark_cases
  WHERE anchor_slug='XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE';"   -- 0
```
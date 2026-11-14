# AUDIT m19475 — m19252 `XVAL-LBM-ROUND2-METHOD-SELECTION-CONTENT-ADDRESSE` (TensorLBM)

**Board delivery-audit (交差核验), not a redo.** Object: mission 19252, status `succeeded`
2026-11-13 21:18 UTC, branch `agent/m19252`, worktree `/root/tp-agent/work/m19252`.
Auditor worktree: `/root/tp-agent/work/m19475`, branch `agent/m19475`.

Machine-checkable re-derivation of every claim:
`artifacts/m19475_audit/verify_m19252.py` → `artifacts/m19475_audit/verify_output.json`
(exit 0, ~10 s), asserted by `tests/test_m19475_m19252_audit.py` (6 passed).

```bash
python3 artifacts/m19475_audit/verify_m19252.py                 # regenerate all numbers below
python3 -m pytest tests/test_m19475_m19252_audit.py -q          # 6 passed
```

---

## 1. Verdict

**Level 3 (system integration).** L1/L2/L3 substantively met with real code, real runs and
a real platform-side caller; **L4 not grantable** — no `tensor_benchmark_cases` row exists
for the anchor, the case-level "ref" is the delivery's own A1 arm (self-referential), the
resolution ladder that would close the predeclared hypothesis never ran, and the one genuine
external comparison is rendered as `NaN` + hardcoded `within_15pct: true` in the L4 table.

Delivery is **honest about its own limits** (report §5 declares D=14/D=20 pending and the
external caliber WITHHELD) — the overstated part is the *maturity bundle*, which self-declares
`L4: PASS (benchmark with ref-vs-got comparisons, repro_cmd)`.

## 2. What is real (independently reproduced)

| Check | Evidence | Result |
|---|---|---|
| C1 impl files | 6 files, sha256 vs bundle | 6/6 exact match (`maturity_evidence_m19252.json`) |
| C2 run artifacts | `runs/A1_D10, A2_D10, A4_D10` `result.json`+`trace.npz`; `A3_D10, A5_D10` `fail.json` | present, sizes as declared (trace.npz ≈ 856 kB) |
| C3 KPI replay | `run_arm.analyze()` re-run on raw `fx,fy` in `trace.npz` | **max rel diff 0.0** over 5 KPIs × 3 arms (`cd_mean`, `cd_mean_last50`, `cl_amp`, `st_fft`, `st_zc`) → numbers are REAL, not MOCK |
| C4 OSS kernel xval | `xval/benchmark_method_selection/oss_crosscheck.py` re-executed against OSS `Maarten-vd-Sande/lbm@573a01c` | `rel_l2_collision = 1.1757884876259976e-07`, `rel_l2_streaming_interior = 2.709855973692391e-08` — reproduced |
| C10 tests | `pytest tests/test_xval_benchmark_method_selection.py` | 6 passed (41 s); assertions are invariants (arm-table consistency, `0.1 < Cd < 10`, smoke spread < 50 %, OSS kernel agreement), not bare execution |
| C11 integration caller | `benchmarks/case_pool/xval_method_selection/run_case.py --smoke` executed by the auditor | wrote `artifacts/case_xval_method_selection/{a1_mrt_sponge_d6,a2_bgk_d6}/{result.json,selection.json,trace.npz}`; routes through `tensorlbm.method_selection.select_collision` (A2 Cd=1.2393, St=0.17168 at D=6) |

Solver path is the production code (`tensorlbm.solver.collide_bgk/collide_mrt/collide_trt/stream`,
`tensorlbm.boundary.boundaries`), not a toy reimplementation — checked by `run_arm.py:54-60` import
block and by the bit-exact KPI replay from stored force histories.

So the rework **did** close the two blocking items from the previous audit (m19109 → level 2):
"impl/entry missing" and "no importer of `src/tensorlbm/method_selection.py`".

## 3. Findings (severity-ranked, with reproduction)

**F1 — BLOCKING for L4 (unchanged from m19109).** Benchmark judge table is empty for this anchor:

```sql
SELECT count(*) FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-ROUND2-METHOD-SELECTION-CONTENT-ADDRESSE';   -- 0
-- ILIKE '%ROUND2-METHOD-SELECTION%' → 0 ; table total = 118
```
No `metric`/`tolerance` exists, therefore **no `got` value can be adjudicated**. The board's
"benchmark is the only judge" cannot be satisfied by any report prose. Nearest registered anchor
(id 79, `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`) is a different repo/problem.

**F2 — HIGH: the case-level "ref" is the delivery itself.** `xval/benchmark_method_selection/benchmark_l4.json`
sets `ref_arm = "A1"` (TensorLBM's own MRT+sponge selection). Every `delta_pct` is therefore
*method sensitivity*, not accuracy; framing it as "ref vs got error against a 15 % criterion"
misapplies the owner precision standard. The external caliber is declared WITHHELD
(report §5 "External reference (Norberg 1993) is WITHHELD"; `predeclared_round2.json`
`baseline_control.external_caliber`). Auditor could not independently obtain the anchor either →
recorded as WITHHELD, not as verified.

**F3 — HIGH, reproduced defect: the one real external comparison is reported as `NaN` + PASS.**
`benchmark_l4.py` reads `oss.get("k1")/oss.get("k2")`, but `oss_crosscheck.json` writes
`k1_collision`/`k2_streaming`:

```
$ python3 xval/benchmark_method_selection/benchmark_l4.py
OSS kernel cross-check (collision + streaming) / k1_collision_rel_L2: delta=nan%, within_15pct=True
OSS kernel cross-check (collision + streaming) / k2_streaming_rel_L2: delta=nan%, within_15pct=True
```
with `"within_15pct": True,  # ~1e-5 % is well within 15%` hardcoded (benchmark_l4.py:105-126 (defect at :115-116, hardcoded flags :117/:122)) and
`ref: 1.0` a placeholder. `verify_output.json → C4_oss_kernel.benchmark_reports_nan_but_marks_pass = true`.
The underlying measurement is real (1.18e-07), so this is a false-PASS rendering bug, not fabrication —
but it is exactly the row an L4 grader would read.

**F4 — MEDIUM: predeclared closure never ran; cluster products were never retrieved.**
`summary_round2.json`: `ladder[A1/A2/A4][cd_mean|st_fft|cl_amp][14]=[20]=null`, `gate_at_each_D`
14/20 all null, `closure_H1` 9/9 `delta_D20_pct=null`, `determinism=null`, `factorial=[]`.
Cluster records show the work *was* dispatched and finished (`cj1872` `run_arm.py --arm A1 --D 14`
state=succeeded, rc=0), yet `runs/A1_D14` does not exist in the worktree → the delivered matrix is a
single resolution (D=10), so no convergence/grid-independence claim is supported and H1/ALT-A
("is the A2 vs A1 gap a resolution artifact?") stays open.

**F5 — MEDIUM: 2×2 factorial is incomplete.** `runs/A6_D10/` contains only `state.npz` (2.45 MB) —
no `result.json`, no `fail.json`; `summary_round2.json → factorial: []`. The A6 (BGK+sponge) cell has
no admissible KPI record, so the factorial design advertised in the report (§2 arm table) is not closed.

**F6 — MEDIUM: KPI schema regression.** `analyze()` computes `cl_rms` (`run_arm.py:142`) but
`result.json` does not carry it (`cl_rms_in_result = false` for A1/A2/A4) → `table_round2.csv`
`cl_rms` column empty, and `gate_pass` is `nan`/blank for A1 with no stated reason.

**F7 — HIGH (work preservation): the delivery is uncommitted.** In `/root/tp-agent/work/m19252`,
`HEAD = 95d2712ff074` ("Merge #2322 (m18705) via merge-master", i.e. the base branch tip) and
`git status` shows the whole round-2 delivery as untracked/modified:

```
?? xval/benchmark_method_selection/run_round2_entry.py
?? xval/benchmark_method_selection/benchmark_l4.py, benchmark_l4.json, figures/
?? xval/benchmark_method_selection/maturity_evidence_m19252.{py,json}
?? src/tensorlbm/xval/method_selection_integration_demo.py
?? docs/XVAL_LBM_ROUND2_METHOD_SELECTION_REPORT_m19252.json
 M tests/test_xval_benchmark_method_selection.py
```
Nothing binds these artifacts to a commit, so the auto-push-on-accept path would ship none of them.
The mission was nevertheless marked `succeeded`.

**F8 — MEDIUM: no report was published.** `SELECT count(*) FROM tensor_reports WHERE mission_id=19252`
→ **0**, while `tensor_dev_missions.report_owed = false`. The six-section payload exists only as the
uncommitted local file `docs/XVAL_LBM_ROUND2_METHOD_SELECTION_REPORT_m19252.json`.

**F9 — info / non-blocking.**
- Two KPIs exceed 15 % *against A1* (`cl_amp −46.57 %` A2, `−15.40 %` A4) — legitimate content for a
  method-selection study, mis-presented as validation error (see F2).
- `wall_s` 72.8 s (A1 MRT) vs 6.4 s (A2 BGK) at identical 200×200 grid is plausible (MRT matmul vs BGK)
  but `threads` is not persisted in `result.json` → throughput must not be quoted as evidence.
- Auditor side effects, disclosed: re-running the shipped `oss_crosscheck.py` rewrote
  `xval/benchmark_method_selection/oss_crosscheck.json` (throughput timings only: 5.126 → 4.767 s/step;
  `rel_l2_*` bit-identical, `benchmark_l4.json` sha256 unchanged `259684a77c36…`), and the integration
  smoke created `artifacts/case_xval_method_selection/`. Audit is therefore **not fully non-invasive**.

## 4. Level matrix (recomputed, not taken from the bundle)

| Level | Status | Basis |
|---|---|---|
| L0 docs | PASS | this doc + delivery report |
| L1 math/unit | PASS | kernel xval vs executed OSS source 1.18e-07 / 2.71e-08; 6 pytest invariants |
| L2 independent impl | PASS | real input→compute→output: `run_arm.py` (408 L) + `src/tensorlbm/method_selection.py` (158 L); artifacts exist; KPI replay bit-exact |
| L3 system integration | **PASS** | `benchmarks/case_pool/xval_method_selection/run_case.py` imports `select_collision` and produced real artifacts under `artifacts/case_xval_method_selection/`; `src/tensorlbm/xval/method_selection_integration_demo.py` |
| L4 benchmark validation | **BLOCKED** | F1 (0 rows), F2 (self-referential ref, external WITHHELD), F3 (NaN+hardcoded pass), F4 (no ladder) |
| L5 research | not claimed | — |

**Level = 3.** This is a downgrade of the delivery's own claim (`verdict.L4 = PASS`), not of the
underlying work: the physics runs and the kernel agreement are real and reproduced.

## 5. Remediation required for L4 (rework list)

1. Register the benchmark: `INSERT INTO tensor_benchmark_cases(anchor_slug, case_name, ref_source, metric, tolerance, repo, status)` with an **external** `ref_source` (literature table or certified reference), per-KPI `metric` and `tolerance`.
2. Fix `benchmark_l4.py:105-126 (defect at :115-116, hardcoded flags :117/:122)`: read `k1_collision`/`k2_streaming`, put OSS values in `ref` and TensorLBM values in `got`, compute `within_15pct` from the data (never hardcode).
3. Retrieve the already-succeeded cluster products (D=14 / D=20) into `runs/`, then emit `summary_round2_full.json` and close `closure_H1`.
4. Complete A6: `result.json` (+ `cl_rms`, `threads`) or a `fail.json` with `fail_stage`.
5. Commit the delivery on `agent/m19252` and publish the report (`tensor_reports` row), otherwise the artifacts are not attributable.

## 6. Independence limitations

- Auditor stayed inside read/re-run scope on the m19252 worktree; the two disclosed write side
  effects (§F9) are timing/artifact-cache only — no source file and no KPI value was altered
  (all 6 impl sha256 still match the bundle after the audit).
- No external network/literature access, so the WITHHELD Norberg 1993 caliber was **not**
  independently substituted or verified; "no external anchor" is reported as observed, not proven
  obtainable.
- Checks not run: full `run_round2_entry.py --rung 10/14/20` (multi-hour, and it would overwrite the
  delivery's artifacts), `make_report.py` regeneration, cluster re-execution.
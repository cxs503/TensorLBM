# Board audit — m20253 `GAP-DIFFLBM-PAPER-EVIDENCE` (TensorLBM)

- **Anchor / mission under audit**: `GAP-DIFFLBM-PAPER-EVIDENCE`, mission **m20253**
  (task_key `gap-difflbm-paper-evidence-r1-rework1`, status `succeeded`,
  `fail_stage = budget_exhausted+forced_finish`, accepted by
  `[auto-approve] TP_AUTO_APPROVE_DEV_ACCEPTANCE`, `head_commit = null`).
- **Audit mission**: m20266 (board-audit, verdict_by `board-audit`). This is a
  verification sheet, **not** a redo: nothing of the subject was fixed.
- **Subject tree**: `/root/tp-agent/work/m20253` (branch `agent/m20253`,
  HEAD `3d33f8513`).
- **Independence limitation**: the subject mission's own session was not
  replayed; all numbers below come from an audit re-run of the *verbatim*
  subject deliverable (sha256 `8523ad1b…5b8b`) inside the auditor's tree
  (`/root/tp-agent/work/m20266`, HEAD `4c5baca4b`, same `src` layout). The
  underlying `src/` trees differ between trees only by unrelated merges; the
  8 items were executed on the audit host (compute job **cj2605**, node n315, 4 cores).

## 1. What the subject actually delivered

| Item | Value |
|---|---|
| Delivered files | `TensorLBM/evidence/paper_benchmarks/run_all_benchmarks.py` — 427 lines, **untracked** (`git status: ?? TensorLBM/evidence/`) |
| Run artifacts | **none** (`find TensorLBM/evidence -type f` → 1 source file; no `artifacts/`, no JSON) |
| Tests | **none** added |
| repro_cmd / README | **none** (the only doc is the module docstring usage line) |
| Report | **0 rows** in `tensor_reports` for `mission_id = 20253` (`report_id` NULL) |
| Server quality audit | `gate_valid = false`, `key_present = false` |

The submission is therefore a *single orchestration script*, and the mission's
delivery claim ("≥6 benchmarks + per-item error table + noise") rests entirely on
code that the subject never executed to completion.

## 2. Benchmark arbiter (the only admissible judge)

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug = 'GAP-DIFFLBM-PAPER-EVIDENCE';
--> 0 rows
```

There is **no registered benchmark case** for this anchor (unchanged since the
prior audit of m20074/m20185, which also recorded `tensor_benchmark_cases_rows_for_anchor: 0`).
Consequence: L4 cannot be granted to any delivery of this anchor until the anchor
is registered as a graded case (ref source, metric, tolerance). The two sibling
anchors that are registered (`GAP-DIFFLBM-3D-COST` id 129, `GAP-DIFFLBM-ADJOINT-VERDICT`
ids 135/136) show the required shape: `case_name`, `metric`, `tolerance` filled and
tolerance ≈ 1e-6…3%.

## 3. Reproduction of the subject code (audit job cj2605)

Command (per-shard so that a crash does not mask the rest):

```
python TensorLBM/evidence/paper_benchmarks/run_all_benchmarks.py --shard k/8 --out artifacts/board_audit_m20266/s<k>
```

Artifacts: `artifacts/board_audit_m20266/{p2d,s1..s7}/{summary.json,benchmark_results.json}`.

| # | item | result | ref vs got | error | verdict vs 3 % bar |
|---|---|---|---|---|---|
| 1 | `poiseuille_2d` | **ERROR** | — | `AttributeError: 'D2Q9Channel' object has no attribute 'CX'` (run_all_benchmarks.py:73) | n/a |
| 2 | `lid_cavity_re100` | **FAIL** | in-tree ref u_min −0.15612 vs got −0.206804; l2_u 0.25858, l2_v 0.16389 | 32.5 % vs *in-tree* ref; **0.48 % vs Ghia (1982) −0.20581** | false negative (bad reference) |
| 3 | `womersley_alpha5` | **PASS** | amplitude 0.2549 %, phase 1.3361 %, profile 1.8071 % | < 3 % | ✓ |
| 4 | `circular_poiseuille` | **FAIL** | ref u_max 9.375e-3 vs got 6.9558e-2; l2 9.3561, linf 7.3381 | 734 % | ✗ |
| 5 | `square_duct_poiseuille` | **ERROR** | — | `IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed` (run_all_benchmarks.py:193) | n/a |
| 6 | `layered_poiseuille` | **PASS** | ref u_peak 5.0087e-2 vs got 4.9924e-2; eps_inf 0.1531 % | < 3 % | ✓ |
| 7 | `asbl` | **FAIL** | Cf ref 0.02 vs got **NaN**, profile_L2 NaN (`cf_measured` present but not finite) | NaN | ✗ |
| 8 | `cdr_da1` | **ERROR** | — | `KeyError: 'c_final'` — the solver returns key `c` (cdr_solver.py:151-153) (run_all_benchmarks.py:316) | n/a |

Tally: **2/8 within 3 %** (both analytic), **1/8 false negative** (own reference
prohibited by the repo), **2/8 genuine physics failure**, **3/8 hard crash at the
API boundary**. The subject reported *none* of this because it never ran the
script (no artifact of any kind) and its `try/except` at
`run_all_benchmarks.py:352-355` downgrades every crash to `status: ERROR` while
`main()` still exits 0 — a green-looking run is impossible to distinguish from a
total failure without opening the JSON.

## 4. Severity-ranked findings

**F1 — blocking: acceptance criterion not met, and not by a small margin.**
Mission bar = analytic 2 + literature 3 + cross-solver 1 (≥6). The deliverable
contains **0 cross-solver items** (`grep -ci cross run_all_benchmarks.py` → 0)
and 1 literature item (Ghia), which false-fails. Best case after bug fixes:
2 analytic + 1 literature = 3 of 6, and only 2 items actually verify <3 %
today. `≥6` is unreachable without adding real items.

**F2 — blocking: L2 evidence chain is empty.** No `run_artifacts`, no `entry`
documented as a command, no test that re-derives any number. Re-running the
subject script on 2026-11-14 produces the table in §3; nothing in the subject
tree reproduces it.

**F3 — blocking: benchmark 2 uses a reference the repo explicitly forbids as a gate.**
`src/tensorlbm/verification/lid_driven_cavity/reference_data.py:14-24` states the
`GHIA_RE*` arrays are `UNVERIFIED_OFFLINE`, that Re=100 centre value is −0.15612
"vs the literature value −0.20581, i.e. 24 %", and **"Do not use them as a
pass/fail gate"**. `run_all_benchmarks.py:102,122` does exactly that
(`get_ghia_reference` + `l2_u < 0.15`). The FAIL is an artifact of the reference,
not of the solver (audit: 0.48 % against the paper value).

**F4 — high: 8 items are not the packaged capability.** 5 of the 8 imported
modules exist **only** in the legacy in-repo tree `TensorLBM/src`
(`ice_hydro`, `verification/lid_driven_cavity`, `verification/square_duct_poiseuille`,
`verification/circular_poiseuille`, `verification/layered_poiseuille`); the packaged
tree is `src/` per `pyproject.toml:82-86` (`package-dir = {"" = "src"}`) and
`pythonpath = ["src"]`, and `import tensorlbm.verification.lid_driven_cavity` /
`tensorlbm.ice_hydro` raise `ModuleNotFoundError` there. The script hard-codes the
legacy path (`ROOT = Path(__file__).resolve().parents[2]`, line 36-38), so what it
exercises is not what ships or is tested.

**F5 — medium: pass thresholds do not encode the acceptance bar.**
Declared gates: cavity 15 % (line 122, equals the "unacceptable" band),
circular 10 % (176), ASBL 5 % (298), square duct 15 % (218), CDR 3 %/5 % (330).
A "PASS" under these thresholds says nothing about the required "是否 <3 %";
only 2 of the 8 items could answer that question as written.

**F6 — medium: no noise/uncertainty evidence.** The anchor asks for the
noise floor; the script records no seed, no repetition, no statistical spread,
and no ρ/u fluctuation diagnostics. `params` do not even carry the lattice
Mach/Reynolds number.

**F7 — low/informational: stale xfail pins in the cavity tests.**
`tests/test_cavity_solver_bc_invariants.py:46,56,66` are `xfail(strict=True)`
pinning the old Zou/He defect; they now **XPASS** (audit run: 3 XPASS(strict)),
i.e. the defect documented there is fixed and the markers are stale. This is not
a finding against m20253, but it means the "known defect" framing around item 2
is out of date, and the strict-xfail suite is currently red.

## 5. Overstated claims in the mission record

| Claim in `result`/headline | Reality |
|---|---|
| "完成 TensorLBM 基准测试框架搭建与统一运行脚本" | scaffolding exists; 3/8 items crash on first call |
| "已识别并整理了仓库中现有的 8 个独立验证基准模块" | 5/8 come from the non-packaged legacy tree; 0 cross-solver; the 8 are not independent (4 are the same pressure-driven channel/duct family) |
| implicit "≥6 benchmarks + error table delivered" | 0 artifacts, 0 error tables, criterion 2/6 at best |
| status `succeeded` | `fail_stage = budget_exhausted+forced_finish`, accepted by blanket auto-approve |

## 6. Level judgement (functional substance, not page count)

**Level 1.** Real, non-placeholder comparison code with correct external anchors
and imports that resolve to real solvers (so not L0 documentation); but no
`run_artifacts`, 3/8 crash, 2/8 genuine physics failures, no cross-solver item,
no registered benchmark row ⇒ the input→compute→output chain is **not
demonstrated for ≥6 items** and L2 (独立功能实现 + 工件) is not earned.
L3 denied: 0 integration callers (`grep -rl paper_benchmarks` → only the bundle
itself / this audit). L4 denied: 0 `tensor_benchmark_cases` rows for the anchor,
2/8 items inside 3 %. L5 denied.

Counterfactual (what would move it): fix the 3 one-line API mismatches
(`ch.CX`→module lattice constant, `u_field` 2-D indexing, `c_final`→`c`), replace
the forbidden Ghia array with a sha256-bearing transcription
(`load_reference_json`), add the missing cross-solver item and ≥2 literature items,
then commit the JSON artifacts + a `repro.sh` + a test that re-derives every
verdict ⇒ L2 immediately (2 items already at 0.15 %/1.8 %), L4 only after the
anchor is registered in `tensor_benchmark_cases`.

## 7. Checks not run / limits

- The subject's *original* execution environment (its own host, 2026-11-14 10:49) was
  not inspected; only its tree. Its runtime claims cannot be verified because no
  artifact exists.
- `asbl` and `circular_poiseuille` were run once each (no repetition study);
  their NaN / 734 % outcomes are single-run observations, though the NaN is a
  deterministic divergence, not noise.
- `cdr_da1`, `square_duct_poiseuille`, `poiseuille_2d` crash before any number is
  produced, so their physics is untested here.
- Ghia (1982) value −0.20581 is taken from the repo's own provenance note
  (`reference_data.py:19`) and `LITERATURE_ANCHORS`; the paper itself was not
  re-read offline.
- Job cj2605 was still `running` on the node when the artifacts were retrieved
  (all 8 shard summaries were already written); no result depends on its later stages.

## 8. Disposition

**Request changes** (verdict level 1, gate expected `reopened`). The subject
deliverable is a broken scaffold with zero evidence; the mission record's
`succeeded` status and its "8 benchmarks / framework completed" claim are not
supported by any artifact. No file of the subject was modified by this audit;
the verbatim snapshot is preserved for re-derivation.

## 9. Reproduction

```bash
# identity
sha256sum docs/board_audit/m20266/evidence/delivered_snapshot/run_all_benchmarks.py
# -> 8523ad1bd3c0f89186d7f06b684d9b63d3b9ff9dc1915968dddd7d7bfa725b8b

# audit re-run (per shard; PYTHONPATH pins the legacy tree the subject hard-codes)
cd <TensorLBM worktree>
for k in 0 1 2 3 4 5 6 7; do
  PYTHONPATH=$PWD/TensorLBM/src python docs/board_audit/m20266/evidence/delivered_snapshot/run_all_benchmarks.py \
    --shard $k/8 --out artifacts/board_audit_m20266/s$k
done
# artifacts: artifacts/board_audit_m20266/*/summary.json  (compute job cj2605, n315)
# figures:   docs/board_audit/m20266/figures/fig1_error_vs_bar.png
```

Figures: `docs/board_audit/m20266/figures/fig1_error_vs_bar.png` (per-item error vs
the 3 % bar, log scale, crash/NaN items marked) and `fig2_coverage.png` (mission
criterion coverage: analytic 2/2, literature 0/3, cross-solver 0/1, and the
pass/fail/error tally).
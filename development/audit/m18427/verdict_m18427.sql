-- Board audit m18427 · cross-check verdict for subject mission m18288
-- anchor XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE (TensorLBM)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18288 XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE',
  18427,
  2,
  $json${
    "impl_files": [
      "xval/benchmark_method_selection/run_arm.py:36-396 (drives tensorlbm.solver.collide_bgk/collide_mrt/collide_trt/stream; present on main 2a0bd485)",
      "xval/benchmark_method_selection/analyze.py:1-323 (KPI aggregation + verdict block; present on main)",
      "xval/benchmark_method_selection/oss_crosscheck.py:1-271 (executed-OSS kernel parity driver; present on main)",
      "src/tensorlbm/method_selection.py:83-135 select_collision() pure decision function — BRANCH ONLY (agent/m18288), absent from main"
    ],
    "entry": "python3 benchmarks/xval_method_selection_bench.py --smoke -> 1200 steps, 18.4 s, status ok, St=0.17168233427728155, Cd=1.2393432470464272 (auditor rerun on main tree)",
    "run_artifacts": [
      "xval/benchmark_method_selection/runs/A1_D10/result.json (cd_mean_last50=1.6132947378116644, cl_amp=0.6579248352005606, st_fft=0.16899580031695963)",
      "xval/benchmark_method_selection/runs/A2_D10/result.json (1.4660336012540012, 0.3515523646082385, 0.17662926112262423)",
      "xval/benchmark_method_selection/runs/A4_D10/result.json (1.642771191938185, 0.556625096634633, 0.1717673074998151)",
      "xval/benchmark_method_selection/runs/A3_D10/fail.json (blowup non_finite_population step 600/21259, tau_minus=25.5)",
      "xval/benchmark_method_selection/runs/A5_D10/fail.json (blowup non_finite_population step 5800/21259, tau_minus=2.0)",
      "docs/board_audit/m18427/audit_numbers.json + fig_m18288_audit.png (auditor independent recompute from the above)",
      "artifacts/case_xval_method_selection/a2_bgk_d6/result.json (case-pool integration run; branch only)"
    ],
    "tests": "pytest tests/test_xval_benchmark_method_selection.py -q -> 5 passed in 60.05 s (auditor rerun on main tree)",
    "integration_callers": "REAL=1 of 3 claimed: benchmarks/case_pool/xval_method_selection/run_case.py:55 sel=select_collision(nu=nu, wall_bounded=True, ...) — auditor re-executed the branch tree copy (PYTHONPATH=src) and reproduced artifacts/case_xval_method_selection/a2_bgk_d6/result.json cd_mean=1.2393432470464272 exactly. NOT callers: benchmarks/case_pool/run_pool.py (driver/usage string line 12 only) and src/tensorlbm/method_selection.py (self-reference). Main-solver integration absent: src/tensorlbm/solver.py does not import method_selection; src/tensorlbm/general_sim.py:348/425/507 _auto_select_collision is a pre-existing independent Re-threshold selector. All L3 files are branch-only (git branch --contains 1bdad75e -> agent/m18288).",
    "benchmark": {
      "registered_case_rows": 0,
      "registered_case_query": "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE' -> 0",
      "tensorbench_results_rows": 0,
      "tensorbench_results_query": "SELECT count(*) FROM tensorbench_results WHERE mission_id=18288 -> 0 (0 with source='cluster_job')",
      "platform_l4": "CLAIMED_NOT_PROVEN (no platform referee row)",
      "rows_total": 8,
      "internal_contrast_rows": 6,
      "external_rows": 2,
      "within_15pct": 4,
      "outside_15pct": [
        "cl_amp A1-vs-A2 @D=10: ref 0.6579248352005606 got 0.3515523646082385 err -46.566%",
        "cl_amp A1-vs-A4 @D=10: ref 0.6579248352005606 got 0.556625096634633 err -15.397%"
      ],
      "resolutions": [10],
      "verdict": "L4 NOT established: all 6 KPI 'ref' values are the A1 arm of the same experiment (self-reference, not analytic/literature); the 2 external rows are kernel-level operator parity only; 2/6 internal rows outside the owner 15% band; single resolution rung D=10 (D=14/20 honest not_run) => case-level convergence unproven"
    },
    "repro_cmd": "python3 benchmarks/xval_method_selection_bench.py --smoke && python3 xval/benchmark_method_selection/analyze.py && python3 docs/board_audit/m18427/audit_m18288.py  # analyze.py rerun on main leaves git diff EMPTY (summary.json/table.csv bit-identical)",
    "external_anchor": "executed OSS source Maarten-vd-Sande/lbm@573a01c4d2e6321f3634ff271adca4ce91872e6f, per-file sha256 recorded in xval/benchmark_method_selection/oss_crosscheck.json -> k1_collision rel-L2=1.1758e-07, k2_streaming rel-L2=2.7099e-08 (float32 vs float64 round-off floor; kernel level only, D2Q9 48-state sample n). Case-level external anchor ABSENT: Norberg 1993 / Sohankar 1998 Re=200 tables WITHHELD (report section 6; external_caliber.adjudicating=false, trend band Cd[1.30,1.45] St[0.160,0.180], A1 Cd=1.614 is outside the band yet non-adjudicating).",
    "findings": [
      {"id": "F1", "severity": "high", "what": "L4 tick overstated in the report's maturity table ('8 rows benchmark') vs its own section 5/6: 0 registered cases, 0 tensorbench_results rows, 6/8 rows self-referenced, 2/6 outside 15%, single resolution rung"},
      {"id": "F2", "severity": "medium", "what": "L3 '3 real callers' inflated: only run_case.py:55 consumes select_collision; run_pool.py is a driver, method_selection.py is self; solver.py never imports it"},
      {"id": "F3", "severity": "medium", "what": "deliverable branch-only (agent/m18288, commits 1bdad75e/b5e8eaf4); src/tensorlbm/method_selection.py + benchmarks/case_pool/xval_method_selection/ + maturity_evidence.{py,json} absent from main 2a0bd485 - recurrence of the m17990 audit F2 defect"},
      {"id": "F4", "severity": "resolved", "what": "prior high defect (H_gap_survives_refinement=true from a single rung) FIXED: grep hits in analyze.py = 0; verdict block now H1_all_oss_arms_within_15pct=false with judged_at_resolution=10 and gap_ladder=null"},
      {"id": "F5", "severity": "low", "what": "5 tests assert aggregation invariants only; no coverage of select_collision TRT-window clamp / A3-A5 blow-up classification"},
      {"id": "F6", "severity": "none", "what": "honest FAILs correctly recorded (A3 tau_minus=25.5 literal magic Lambda, A5 tau_minus=2.0 remediation, both blowup) - negative result disclosed, not papered over"}
    ],
    "checks_run": [
      "smoke entry rerun -> 1200 steps 18.4 s ok",
      "analyze.py rerun on main -> git diff empty (no numeric drift)",
      "independent recompute from raw artifacts -> recompute_mismatch_vs_published = []",
      "pytest tests/test_xval_benchmark_method_selection.py -q -> 5 passed 60.05 s",
      "case-pool run_case.py --smoke on a /tmp copy of the branch tree -> cd_mean 1.2393432470464272 == committed artifact",
      "grep select_collision across src/benchmarks/tests on agent/m18288 -> 1 real consumer",
      "DB referee lookups: tensor_benchmark_cases 0 rows, tensorbench_results 0 rows",
      "merge state: git branch -a --contains 1bdad75e -> agent/m18288 only",
      "ladder scan: runs/ = A1..A5_D10 only (resolutions [10])"
    ],
    "level_rationale": "L1 met (kernel parity vs executed OSS source 1e-7/1e-8 + tests); L2 met (real solver-driven harness, 5 arms x 21259 steps, artifacts reproduce bit-identically); L3 NOT met (1 case-pool caller, main solver never imports the selection module, all L3 files branch-only); L4 NOT met (internal-only reference, 2/6 rows outside 15%, no registered case/result row, single resolution rung). Level = 2.",
    "merge_state": "agent/m18288 only (1bdad75e, b5e8eaf4); main = 2a0bd485",
    "audit_doc": "docs/board_audit/m18427/AUDIT_m18288.md",
    "auditor_independence": "same container, no second machine; verification by re-derivation from committed artifacts + re-execution, not by independent physical repro"
  }$json$::jsonb,
  'board-audit'
);
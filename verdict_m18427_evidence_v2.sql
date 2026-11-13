-- m18427: normalise verdict-752 evidence to machine-checkable pure repo paths
-- (descriptions moved to "details" so gate path-resolution can never choke on prose)
UPDATE tensor_maturity_verdicts
SET evidence = $json${
  "impl_files": [
    "xval/benchmark_method_selection/run_arm.py",
    "xval/benchmark_method_selection/analyze.py",
    "xval/benchmark_method_selection/make_report.py",
    "xval/benchmark_method_selection/oss_crosscheck.py",
    "benchmarks/xval_method_selection_bench.py"
  ],
  "impl_files_branch_only": [
    "src/tensorlbm/method_selection.py",
    "benchmarks/case_pool/xval_method_selection/run_case.py",
    "benchmarks/case_pool/xval_method_selection/refs.json",
    "xval/benchmark_method_selection/maturity_evidence.py"
  ],
  "details": {
    "xval/benchmark_method_selection/run_arm.py": "lines 36-396: drives tensorlbm.solver.collide_bgk/collide_mrt/collide_trt/stream + tensorlbm.boundary.boundaries; 5 arms x 21259 steps",
    "xval/benchmark_method_selection/analyze.py": "lines 1-323: KPI aggregation + verdict block (H1_all_oss_arms_within_15pct, largest_abs_delta_pct, judged_at_resolution)",
    "xval/benchmark_method_selection/oss_crosscheck.py": "lines 1-271: executes the OSS D2Q9 source and diffs it against the TensorLBM kernel",
    "src/tensorlbm/method_selection.py": "lines 83-135 select_collision() pure decision function (nu -> tau -> operator/sponge, TRT Lambda re-derived into the tau_minus window); BRANCH ONLY agent/m18288",
    "benchmarks/case_pool/xval_method_selection/run_case.py": "line 55 sel=select_collision(nu=nu, wall_bounded=True, ...); writes artifacts/case_xval_method_selection/<tag>/"
  },
  "entry": "benchmarks/xval_method_selection_bench.py",
  "entry_cmd": "python3 benchmarks/xval_method_selection_bench.py --smoke",
  "run_artifacts": [
    "xval/benchmark_method_selection/runs/A1_D10/result.json",
    "xval/benchmark_method_selection/runs/A2_D10/result.json",
    "xval/benchmark_method_selection/runs/A4_D10/result.json",
    "xval/benchmark_method_selection/runs/A3_D10/fail.json",
    "xval/benchmark_method_selection/runs/A5_D10/fail.json",
    "xval/benchmark_method_selection/runs/A1_D10/trace.npz",
    "xval/benchmark_method_selection/oss_crosscheck.json",
    "xval/benchmark_method_selection/summary.json"
  ],
  "run_artifact_values": {
    "A1_D10": {"cd_mean_last50": 1.6132947378116644, "cl_amp": 0.6579248352005606, "st_fft": 0.16899580031695963, "status": "ok"},
    "A2_D10": {"cd_mean_last50": 1.4660336012540012, "cl_amp": 0.3515523646082385, "st_fft": 0.17662926112262423, "status": "ok"},
    "A4_D10": {"cd_mean_last50": 1.642771191938185, "cl_amp": 0.556625096634633, "st_fft": 0.1717673074998151, "status": "ok"},
    "A3_D10": {"status": "blowup", "fail_stage": "non_finite_population", "step": 600, "total_steps_planned": 21259, "tau_minus": 25.500000000000163},
    "A5_D10": {"status": "blowup", "fail_stage": "non_finite_population", "step": 5800, "total_steps_planned": 21259, "tau_minus": 2.0}
  },
  "tests": "tests/test_xval_benchmark_method_selection.py",
  "tests_result": "pytest -q -> 5 passed in 60.05 s (auditor rerun on main tree)",
  "integration_callers": [
    "benchmarks/case_pool/xval_method_selection/run_case.py"
  ],
  "integration_callers_rejected": [
    "benchmarks/case_pool/run_pool.py (driver/usage string only, does not consume the API)",
    "src/tensorlbm/method_selection.py (self-reference, not a caller)"
  ],
  "integration_callers_reason": "REAL=1 of 3 claimed. src/tensorlbm/solver.py never imports method_selection; src/tensorlbm/general_sim.py:348/425/507 _auto_select_collision is a pre-existing independent Re-threshold selector. Auditor re-executed the branch tree copy (PYTHONPATH=src python3 benchmarks/case_pool/xval_method_selection/run_case.py --smoke) and reproduced artifacts/case_xval_method_selection/a2_bgk_d6/result.json cd_mean=1.2393432470464272 exactly. All L3 files are branch-only: git branch -a --contains 1bdad75e -> agent/m18288.",
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
  "repro_cmd": "python3 benchmarks/xval_method_selection_bench.py --smoke && python3 xval/benchmark_method_selection/analyze.py && python3 docs/board_audit/m18427/audit_m18288.py",
  "repro_result": "smoke 1200 steps 18.4 s ok; analyze.py rerun leaves git diff EMPTY (summary.json/table.csv bit-identical); audit recompute_mismatch_vs_published = []",
  "external_anchor": "executed OSS source Maarten-vd-Sande/lbm@573a01c4d2e6321f3634ff271adca4ce91872e6f, per-file sha256 in xval/benchmark_method_selection/oss_crosscheck.json -> k1_collision rel-L2=1.1758e-07, k2_streaming rel-L2=2.7099e-08 (float32 vs float64 round-off floor; kernel level only). Case-level external anchor ABSENT: Norberg 1993 / Sohankar 1998 Re=200 tables WITHHELD (report section 6; external_caliber.adjudicating=false, trend band Cd[1.30,1.45] St[0.160,0.180]; A1 Cd=1.614 lies outside the band yet is non-adjudicating).",
  "findings": [
    {"id": "F1", "severity": "high", "what": "L4 tick overstated in the report maturity table ('8 rows benchmark') vs its own sections 5/6: 0 registered cases, 0 tensorbench_results rows, 6/8 rows self-referenced, 2/6 outside 15%, single resolution rung"},
    {"id": "F2", "severity": "medium", "what": "L3 '3 real callers' inflated: only run_case.py:55 consumes select_collision; run_pool.py is a driver, method_selection.py is self; solver.py never imports it"},
    {"id": "F3", "severity": "medium", "what": "deliverable branch-only (agent/m18288, 1bdad75e/b5e8eaf4); src/tensorlbm/method_selection.py + benchmarks/case_pool/xval_method_selection/ + maturity_evidence.{py,json} absent from main 2a0bd485 - recurrence of m17990 audit F2"},
    {"id": "F4", "severity": "resolved", "what": "prior high defect (H_gap_survives_refinement=true from a single rung) FIXED: 0 grep hits in analyze.py; verdict block now H1_all_oss_arms_within_15pct=false, judged_at_resolution=10, gap_ladder=null"},
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
  "audit_numbers": "docs/board_audit/m18427/audit_numbers.json",
  "auditor_independence": "same container, no second machine; verification by re-derivation from committed artifacts + re-execution, not by independent physical repro"
}$json$::jsonb
WHERE mission_id = 18427 AND verdict_by = 'board-audit';
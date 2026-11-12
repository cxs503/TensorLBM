-- m15937 · 交差核验 verdict 落表：m15912 GAP-LBM-INTL-BENCH（TensorLBM 国际对标专项审计）
-- 执行：runuser -u postgres -- psql -d tensorplatform -f audit/m15937_verdict.sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorLBM',
 'm15912 GAP-LBM-INTL-BENCH',
 15937,
 2,
 $json${
  "impl_files": ["/root/tp-agent/work/m15912/src/tensorlbm/audit/intl_bench.py"],
  "entry": ["/root/tp-agent/work/m15912/src/tensorlbm/audit/__main__.py", "/root/tp-agent/work/m15912/tools/audit/run_intl_bench_audit.sh", "/root/tp-agent/work/m15912/src/tensorlbm/api.py"],
  "run_artifacts": ["/root/tp-agent/work/m15912/artifacts/intl_bench_v1/run1/summary.json", "/root/tp-agent/work/m15912/artifacts/intl_bench_v1/run1/module_maturity.json", "/root/tp-agent/work/m15912/artifacts/intl_bench_v1/run1/maturity.sql", "/root/tp-agent/work/m15912/artifacts/intl_bench_v1/run1/repro.txt"],
  "tests": ["/root/tp-agent/work/m15912/tests/test_intl_bench_audit.py"],
  "tests_run": "pytest tests/test_intl_bench_audit.py -q -> 10 passed in 9.79s (re-run by auditor in /tmp/a15912 from git archive 01748f48)",
  "integration_callers": null,
  "integration_callers_absent_reason": "branch agent/m15912 @01748f48 is NOT merged: git merge-base --is-ancestor 01748f48 HEAD = false (agent worktree main 123fe6b8, ECO checkout 53046bcc); git merge-tree --write-tree HEAD 01748f48 clean (no conflict) -> push/merge pipeline gap; grep -rl -F 'tensorlbm.audit' /root/tensor-ecosystem/TensorLBM = no hits; only caller is its own test tests/test_intl_bench_audit.py:166",
  "benchmark": {"harness": "benchmarks/intl_bench_calibration/run_benchmark.py", "verdict": "PASS", "controls_agreement_pct": 100.0, "controls_n": 10, "anchor_sphere_re200": {"ref_cd": 0.805615, "got_cd": 0.798671, "err_pct": -0.862, "accept_pct": 3.0, "grid_span_pct": 2.728, "source": "benchmarks/verified/sphere_re200/result.json (pre-existing artefact, not recomputed by this mission)"}, "anchor_hadamard": "correctly withheld: got=partial", "platform_referee_rows": 0, "platform_referee_note": "tensor_benchmark_cases rows for anchor_slug='GAP-LBM-INTL-BENCH' = 0 (whole table 25 rows); tensorbench_results for 15912 = 0 -> no machine referee, so L4 not awardable"},
  "repo_relative_paths_note": "same artefacts, repo-relative: src/tensorlbm/audit/intl_bench.py (623 lines), src/tensorlbm/audit/__main__.py, tools/audit/run_intl_bench_audit.sh, artifacts/intl_bench_v1/run1/*; they resolve inside the ecosystem repo only after the unmerged branch agent/m15912 is merged (finding F2). Absolute pointers above are the live checkout of git object 01748f48, the object this verdict audited.",
  "pinned_by_git_object": "01748f48",
  "repro_cmd": "git archive 01748f48 | tar -x -C /tmp/a15912 && cd /tmp/a15912 && OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m tensorlbm.audit --repo-root . --out /tmp/a15912_out",
  "repro_result": "exit 0; 45 modules graded verified 4 / partial 24 / implemented 15 / failed 2; citations 51/51 (OpenLB 12/12, waLBerla 16/16, Palabos 13/13, Lettuce 10/10); citation_failures []",
  "external_anchor": ["/root/tp-agent/work/m15912/benchmarks/verified/sphere_re200/result.json", "/root/tp-agent/work/m15912/benchmarks/verified/compressible_shock_family/result.json", "/root/tp-agent/work/m15912/benchmarks/verified/wigley_hydrostatics/result.json", "/root/tp-agent/work/m15912/benchmarks/verified/laplace_droplet/result.json"],
  "ledger_loaded": {"tensorlbm_module_maturity": 45, "distribution": {"verified": 4, "partial": 24, "implemented": 15, "failed": 2}, "tensorlbm_compete_score": 10, "week": "2026-W46"},
  "level_rationale": "L2 = real code + CLI entry + 12 run artefacts + 10 invariant tests, re-run from a pristine git archive with identical numbers. Not L3: integration_callers is machine-unsatisfiable because the branch is unmerged and no caller exists outside its own test (grep in ECO repo = 0 hits). Not L4: 0 rows in tensor_benchmark_cases for the anchor and 0 tensorbench_results, plus the calibration reference levels are authored by the same code author (self-consistency, not an external referee). Level = functional substance; the 105-line report and 3 figures do not raise it.",
  "overstated_claims": {"claimed_verified_modules": 4, "supportable_verified_modules": 3, "inflated_row": "compressible_shock graded verified although benchmarks/verified/compressible_shock_family/result.json has all_gates_pass=false (n_gates_pass 11/12, FFS-02 1.458 deg > 1.0 deg) and README.md:47 marks FFS-02 'FAIL (honest)'", "report_section6_calibration_ref": "listed as a benchmark ref but ref levels are hand-authored in benchmarks/intl_bench_calibration/controls.json -> self-calibration, not external anchor"},
  "findings": [
   "F1 BLOCKING (false verified, independently reproduced): truthy_verdict() (src/tensorlbm/audit/intl_bench.py:177) treats ANY nested '\"pass\": true' as a case verdict, so a case whose own result.json reports all_gates_pass=false is graded verified. Live instance: module compressible_shock <- benchmarks/verified/compressible_shock_family (all_gates_pass=false, 11/12 gates, FFS-02 1.458>1.0 deg, README.md:47 'FAIL (honest)')); the row reached the DB with tier='verified'. Reproducer: /root/tp-agent/work/m15937/audit/m15937_repro_false_verified.py (synthetic case all_gates_pass=false + one gate pass -> GRADED LEVEL verified, FALSE_VERIFIED True, exit 1). The 10-control calibration has no control for 'aggregate false + some gates true' -> 10/10 green does not cover this failure mode.",
   "F2 MEDIUM (evidence chain): commit 01748f48 is not an ancestor of main in the agent worktree or in /root/tensor-ecosystem/TensorLBM, yet merge-tree is clean -> deliverable orphaned by the push/merge pipeline (recurrence of the defect recorded by the m15904 audit); integration_callers can never be satisfied and the work is only recoverable via git show.",
   "F3 MEDIUM (reproducibility): 3 of the 4 upstream liveness roots are volatile tmp paths (/tmp/walberla, /tmp/palabos_probe, /tmp/up/lettuce; only /root/openlb is stable) with no pin/clone script, submodule or commit manifest in-repo -> after a /tmp sweep check_citations fails wholesale and the CLI exits 3, so 51/51 cannot be rebuilt elsewhere. Also UPSTREAM[*]['tier'] are hand-written constants (intl_bench.py:23-90): machine validation covers citation existence only, not the upstream capability tiers that drive the gap grades.",
   "F4 LOW: report section 6 presents the 100% grader-calibration agreement as a reference comparison while the reference levels are authored in controls.json by the same author -> self-calibration, wording should say so.",
   "F5 LOW: '51/51 live validation' is symbol/file substring existence (e.g. 'class BGKdynamics'), not proof that the upstream capability grading is correct."
  ],
  "checks_not_run": [
   "51 upstream citations not all read against the described algorithms (structure + full machine parse only)",
   "no physics case re-run: sphere_re200 Cd -0.862% taken from the pre-existing artefact, not recomputed",
   "GPU self-recorded 8.6 / 69 GLUPS not re-measured (no unified caliber; delivered side flags it as self-recorded too)",
   "no cluster job submitted (all re-runs < 15 s)",
   "report numbers spot-checked against summary.json/RESULT.json, not byte-diffed line by line"
  ],
  "disposition": "approve for L2 with blocking note F1 (request changes on the product): truthy_verdict must prefer a top-level all_gates_pass/verdict key and demote compressible_shock to partial, plus a c11 control (aggregate false + one gate true -> expect partial). F2 is a platform push/merge defect, not the mission's fault. L3 refused (unmerged, no external caller), L4 refused (0 platform referee rows).",
  "independence_limitations": "Same host, same python 3.11.6, same git object as the mission. Independence levers used: pristine git-archive sandbox re-run, a self-authored counterexample fixture, and reconciliation against platform ledgers (tensor_benchmark_cases / tensorbench_results / tensorlbm_module_maturity).",
  "audit_record": "/root/tp-agent/work/m15937/audit/m15937_intl_bench_verdict.md"
 }$json$::jsonb,
 'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;
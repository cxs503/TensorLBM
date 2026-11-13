-- m17849 board audit verdict for m17713 XVAL-LBM-BENCHMARK-METHOD-SELECTION (TensorLBM)
-- Level 3: L2 (real implementation + entry + run artifacts) and L3 (library-kernel integration)
-- are satisfied by artefacts of agent/m17713 @ 24b4af6a; L4 is denied on three independent
-- grounds (no tensor_benchmark_cases row for the anchor, D=14 convergence ladder never ran,
-- bit-level re-run not obtained in the audit window + branch unmerged).
-- Every evidence value is a real pointer verified by this audit (files are byte snapshots of the
-- audited branch under docs/board_audit/m17849/refs/, taken with `git archive agent/m17713`).

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm17713 XVAL-LBM-BENCHMARK-METHOD-SELECTION',
  17849,
  3,
  $json${
    "impl_files": [
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/run_arm.py:1-395 (arm runner: builds the Re=200 square-cylinder case, calls TensorLBM kernels, writes runs/<arm>_D<D>/{result.json|fail.json,trace.npz})",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/oss_crosscheck.py:1-270 (executes the OSS source itself: k0 direction map, k1 collision, k2 streaming, throughput)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/analyze.py:1-322",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/make_report.py:1-291",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/predeclared_v2.json (predeclaration: hypothesis/arms/thresholds/stopping rule/confounders)"
    ],
    "entry": [
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/run_arm.py (CLI: --arm A1..A5 --D --threads)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/run_matrix.sh:1-17 (5 arms at D=10 + 3 arms at D=14)",
      "docs/board_audit/m17849/refs/benchmarks/xval_method_selection_bench.py:1-98 (published bench entry used by the tests)"
    ],
    "run_artifacts": [
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/runs/A1_D10/result.json (cd_mean=1.6143262481245584, cl_amp=0.6579248352005606, st_fft=0.16899580031695963, total_steps=21259)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/runs/A2_D10/result.json (cd_mean=1.4661360417278881, cl_amp=0.3515523646082385, st_fft=0.17662926112262423)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/runs/A4_D10/result.json (cd_mean=1.6423615055096823, cl_amp=0.556625096634633, st_fft=0.1717673074998151)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/runs/A3_D10/fail.json (honest stability failure: non_finite_population at step 600, lambda_trt=0.1875, tau_minus=25.5)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/runs/A5_D10/fail.json (honest stability failure: non_finite_population at step 5800, lambda_trt=0.01125, tau_minus=2.0)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/summary.json + table.csv (aggregation consumed by the report)",
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/figures/{fig1_cl_traces,fig2_kpi_vs_A1,fig3_resolution_ladder,fig4_vorticity_fields}.png (auditor checked non-blank: 5.63%-57.06% non-white pixels)",
      "docs/board_audit/m17849/audit_numbers.json (auditor re-derivation of every quoted number from the result.json files, script docs/board_audit/m17849/compute_audit_numbers.py)"
    ],
    "tests": [
      "docs/board_audit/m17849/refs/tests/test_xval_benchmark_method_selection.py:1-97 (5 tests asserting invariants: arm-table consistency, drag sign/order guard, <=50% harness sanity, result self-consistency, executed-OSS kernel agreement)",
      "auditor re-run 2026-11-13: python -m pytest tests/test_xval_benchmark_method_selection.py -q -> 5 passed, 2 warnings in 49.16s (worktree /root/tp-agent/work/m17713 @ agent/m17713 24b4af6a)"
    ],
    "integration_callers": [
      "docs/board_audit/m17849/refs/xval/benchmark_method_selection/run_arm.py:51-57 (from tensorlbm.boundary.boundaries import compute_obstacle_forces, far_field_bc_2d, make_sponge_strength; from tensorlbm.d2q9 import equilibrium; from tensorlbm.solver import collide_bgk, collide_mrt, collide_trt, stream) - the benchmark drives the shipped solver kernels, it does not reimplement them",
      "docs/board_audit/m17849/refs/tests/test_xval_benchmark_method_selection.py -> imports benchmarks/xval_method_selection_bench.py -> from run_arm import ARMS (entry is exercised by CI-style tests, not only by hand)",
      "caveat: no product code path (src/tensorlbm/**) was changed by this mission; the 'integration' is benchmark->library kernel direction, so the level is L3 (integrated evidence over the shipped solver) and not a new solver capability"
    ],
    "benchmark": "NO machine judge exists for this anchor: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BENCHMARK-METHOD-SELECTION' returns 0 rows (and 0 rows match anchor_slug ILIKE '%METHOD%' across all 58 registered anchors) -> the declared sole benchmark judge is absent, so L4 cannot be confirmed. Deliverable-internal criterion (predeclared owner 15% band, ref=A1 recorded selection, got=OSS-transplant arms, same D=10): Cd mean -9.18% (A2) / +1.74% (A4) PASS; Cl amplitude -46.57% (A2) / -15.40% (A4) FAIL; St +4.52% (A2) / +1.64% (A4) PASS -> H1 falsified, largest deviation 46.57% (honest FAIL, reported as such in section 7 of the deliverable). Second layer (ref = executed OSS source): collision rel-L2 1.18e-07, interior streaming rel-L2 2.71e-08, direction map 8/8 -> <3% satisfied (kernel correctness only, not a physical-KPI anchor).",
    "repro_cmd": "cd <checkout of agent/m17713> && python -m pytest tests/test_xval_benchmark_method_selection.py -q   # auditor: 5 passed in 49.16s; single-arm: python xval/benchmark_method_selection/run_arm.py --arm A2 --D 10   # auditor re-run launched 2026-11-13T09:40Z under host load average ~155 (128 cores) did not finish inside the audit window (recorded wall_s=6.4s degraded to >10 min): status not_run, NOT counted as a pass; full matrix: bash xval/benchmark_method_selection/run_matrix.sh",
    "external_anchor": "Maarten-vd-Sande/lbm @ 573a01c4d2e6321f3634ff271adca4ce91872e6f, executed from the OSS checkout with only a numba.jit identity shim (numerics unchanged); source sha256 pinned in docs/board_audit/m17849/refs/xval/benchmark_method_selection/oss_crosscheck.json (init.py aff2ffa1..., streamcollide.py 6d9d5315..., time_evolve.py 5f1a9dcd..., update_types.py 58be1e81..., util.py 867f8370...) and re-executed by the auditor's pytest run. Physical anchor is WITHHELD: docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md gives only a non-adjudicating trend band Cd in [1.30,1.45], St in [0.16,0.18]; got Cd=1.61433 -> +11.33% above band top, +17.41% above band centre; St=0.168996 in band.",
    "level_reasoning": {
      "L1": "PASS - kernel-level agreement with executed OSS source at 1e-7/1e-8",
      "L2": "PASS - real implementation + entry + 5 real run artifacts (3 ok, 2 honest stability failures)",
      "L3": "PASS - benchmark drives the shipped solver kernels (run_arm.py:51-57) and is covered by integration tests",
      "L4": "FAIL - three independent gaps: (1) anchor has no tensor_benchmark_cases row (no machine judge); (2) D=14 convergence ladder never ran (logs/A{1,2,4}_D14.log all 0 bytes, summary.json.convergence=={}), so the predeclared convergence rule is unobserved and the case is incomplete; (3) bit-level re-run not obtained in the audit window and the branch is unmerged into main (main be78a87b lacks xval/benchmark_method_selection/{runs,oss_crosscheck.py}, tests/ and benchmarks/ entries)",
      "L5": "FAIL - no adjudicable external reference; Cd sits outside the repo's own trend band"
    },
    "findings": [
      "F1 blocking (L4): benchmark anchor unregistered - 0 rows in tensor_benchmark_cases",
      "F2 blocking (L4): convergence ladder absent - three D=14 logs are 0 bytes, convergence block empty",
      "F3 non-blocking: agent/m17713 @ 24b4af6a not merged into main; main only carries the round-0 6-file directory, so repo-of-record has no evidence for this mission (snapshot kept in docs/board_audit/m17849/refs/)",
      "F4 non-blocking: deliverable doc section 5.2 labels A5 as a literal Lambda=3/16 transplant with tau_minus=2.00, contradicting runs/A5_D10/fail.json (lambda_trt=0.01125, tau_minus=2.0 = re-derived); artifacts are right, the sentence is not",
      "F5 non-blocking: summary.json verdict.per_arm_within_15pct lists only A2_D10 while A4 Cl amplitude is -15.40% (also outside the 15% band) - conclusion unchanged, per-arm aggregation incomplete",
      "F6 non-blocking: artifact hygiene - A4_D10 has no state.npz (A1/A2 do) and 0-byte D=14 logs are indistinguishable from still-running runs (no not_run.json)"
    ]
  }$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      mission_id = EXCLUDED.mission_id
RETURNING id, level, coalesce(gate_state, ''), coalesce(gate_level, -1), created_at;
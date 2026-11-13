-- Board audit verdict for m17596 (XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP)
-- auditor: m17665 / board-audit, 2026-11-13
-- full report: docs/audit_m17596/AUDIT_m17596.md (this worktree)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm17596 XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP',
  17665,
  2,
  $json${
  "audited_commit": "/root/tp-agent/work/m17596 @ 10d568a6 (13 files, 1081-line implementation)",
  "impl_files": ["/root/tp-agent/work/m17596/src/tensorlbm/blockforest.py"],
  "impl_pointers": [
    "src/tensorlbm/blockforest.py:95 BlockForest (1-irregular forest, level-1 refinement, per-block workload)",
    "src/tensorlbm/blockforest.py:251 BlockGraph (CSR xadj/adjncy/adjwgt)",
    "src/tensorlbm/blockforest.py:578 partition_blockforest (+ :333 _kl_refine, :483 _kway_refine, :693 exact_bisection_cut)",
    "src/tensorlbm/blockforest.py:768 MultistreamBlockSim (:992 step, :949 _pump), :962 _stream_core / :975 _stream_shell"
  ],
  "entry": "/root/tp-agent/work/m17596/benchmarks/xval_blockforest_metis_multistream/run_xval.py",
  "run_artifacts": [
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/result.json",
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/tables.md",
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/fig1_cut_vs_optimum.png",
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/fig2_imbalance_refined.png",
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/fig3_overlap_hidden_fraction.png",
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/fig4_shear_wave_decay.png",
    "/root/tp-agent/work/m17596/artifacts/xval_blockforest_metis_multistream/fig5_part_count_sensitivity.png"
  ],
  "tests": [
    "tests/test_blockforest.py:69 test_partition_matches_exhaustive_optimum_small (cut <= 1.15 x exhaustive optimum, imbalance <= 5%)",
    "tests/test_blockforest.py:79 test_partition_balanced_and_not_worse_than_slab",
    "tests/test_blockforest.py:89 test_partition_handles_workload_heterogeneity (count-slab provably >5% imbalance; weight sum conserved)",
    "tests/test_blockforest.py:107 test_partition_uses_every_part",
    "tests/test_blockforest.py:122 test_decomposed_streaming_matches_production_roll (atol 1e-13 vs collide_bgk3d+stream3d_roll)",
    "tests/test_blockforest.py:145 test_overlapped_schedule_is_bitwise_identical_to_serialized (torch.equal)",
    "tests/test_blockforest.py:158 test_mass_conservation_and_ghost_periodicity (1e-13, periodic wrap)",
    "tests/test_blockforest.py:181 test_shear_wave_decay_rate_matches_analytic (nu k^2)",
    "AUDITOR RE-RUN: python3 -m pytest tests/test_blockforest.py -q -> 10 passed in 13.60s"
  ],
  "integration_callers": [],
  "integration_note": "none: grep -rn 'blockforest' --include=*.py src/ tests/ benchmarks/ -> only blockforest.py itself, tests/test_blockforest.py:17 and benchmarks/xval_blockforest_metis_multistream/run_xval.py:51; not exported in src/tensorlbm/__init__.py; no solver/runner instantiates MultistreamBlockSim => inbound integration only (the new stepper calls production collide_bgk3d/stream3d_roll), no production caller => L3 NOT met, hence L4 unreachable under maturity_gate SCHEME (cumulative)",
  "benchmark": {
    "machine_referee": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP' -> 0 rows (no formal referee row; audited against the mission's predeclared gates)",
    "ref_vs_got": {
      "D1a_cut_vs_exhaustive_optimum": {"ref": "exact enumeration, k=2, 6 instances", "got_rel_err_max": "0.00%", "cap": "15%", "pass": true},
      "D1b_cut_vs_networkx_KL": {"ref": "networkx kernighan_lin, best-of-20", "got_rel_err_max": "11.46% (3x4x6 k=8)", "cap": "15%", "pass": true},
      "D1c_cut_vs_best_of_64": {"ref": "best-of-64 restarts, 9 instances", "got_rel_err_max": "0.00%", "cap": "15%", "pass": true},
      "D2_imbalance": {"ref": "indivisible-block floor computed by auditor", "got": "22.03% on refC k=4/8", "cap": "5%", "pass": false, "floor": "8.47% (cap infeasible on refC)", "excess_over_floor": "+13.56pp"},
      "D3_field_equivalence": {"ref": "monolithic collide_bgk3d+stream3d_roll", "got": "max|df| = 0.0 (bitwise)", "cap": "1e-12", "pass": true},
      "D4_decay_rate": {"ref": "analytic nu k^2", "got_rel_err": "+5.144e-4 (0.0514%)", "cap": "15%", "pass": true},
      "D5_hidden_fraction": {"ref": "latency-injected serialized baseline", "got": "92.6% at 16^3 alpha=0.5", "cap": "85%", "pass": "PASS within timing noise (rows with >100% and negative exposed time prove t_ov<t0 beyond noise; 7.6pp margin < declared +-10% spread)"}
    },
    "auditor_reproduction": "docs/audit_m17596/repro_diff_m17596.json: re-run of E1/E2/E3/E4/E6 -> n_number_diffs=6, all_diffs_are_wall_clock=true (only exact_enumeration_s fields); every result/gate value bit-identical incl. max_field_diff=0.0",
    "error_standard": "owner 15% cap met on all cut gates (D1a/D1b/D1c) and on D4 (0.0514%); D2 gate FAILS its own 5% cap (22.03%)"
  },
  "repro_cmd": "cd /root/tp-agent/work/m17596 && python3 benchmarks/xval_blockforest_metis_multistream/run_xval.py   # ~3 min, writes artifacts/xval_blockforest_metis_multistream/{result.json,tables.md,fig*.png}; and cd /root/tp-agent/work/m17665 && python3 docs/audit_m17596/independent_check.py --floor --figure   # auditor floor + figure",
  "external_anchor": [
    "/root/tp-agent/work/m17596/docs/XVAL_BLOCKFOREST_METIS_MULTISTREAM.md"
  ],
  "external_anchor_note": "DOCUMENTARY only: report lines 24-31 state the worktree had no network (curl exit 35), no waLBerla checkout, no METIS installed -> reference practices cited from the public source layout, not read from source, and never executed. Platform inventory holds the practice (oss_solutions id=365 lssfau/walberla feature lbm-gpu-mpi-overlap) but was not cited. Substitute references (exhaustive optimum / networkx KL / analytic nu k^2) are legitimate for the cut criterion (cut_METIS >= cut_opt), but 'cross-validated vs waLBerla' is overstated: it is a design-level port + offline independent-reference validation.",
  "findings": {
    "F1_no_production_integration": "BLOCKING for L3/L4: blockforest has no production caller and is not exported; single-process only (no MPI ranks).",
    "F2_D2_gate_fails_and_unanalysed": "22.03% vs 5% cap; auditor exact floor 8.47% proves the cap infeasible on refC, yet shipped partition is +13.56pp above the achievable floor (cut-first trade-off). Report states PARTIAL honestly but never quantifies either fact.",
    "F3_predeclaration_mismatch": "result.json:predeclared.sample.E4 says 600 steps, the shipped run is 400 (run_xval.py:646, function default :313 = 600), and report line 75 labels 400 steps '(predeclared)'. Thresholds untuned; auditor reproduces the same +5.144e-4 at 400 steps => integrity defect, no numerical impact.",
    "F4_D5_within_noise": "hidden_fraction values 103.6/124.3/166.4/242.4% and negative exposed times show t_overlapped<t0 beyond the noise floor; the 92.6% pass has 7.6pp margin inside the declared +-10% spread. Defensible claim is 'exposed comm time ~ 0 for alpha<=0.5'.",
    "F5_multistream_emulated": "queue + Python worker thread + time.sleep latency injection, CPU only (torch.cuda.is_available()=False), no CUDA streams, no MPI; declared honestly in report sec.6.",
    "F6_anchor_documentary": "waLBerla/METIS neither executed nor read from source (no network / not installed); inventory entry oss_solutions id=365 not cited.",
    "F7_no_registry_row": "tensor_benchmark_cases has 0 rows for the anchor -> no machine referee; recommend registering D1/D4 rows.",
    "F8_report_link": "tensor_dev_missions.report_id NULL for 17596 although tensor_reports id=16590 exists (report_owed=f).",
    "F9_kb_offline": "mandated pre-work knowledge-base checks unavailable (engine_offline 8021); logged as friction, no capability impact."
  },
  "verified_positives": [
    "implementations REAL: 1081 lines, 0 placeholder markers; BlockForest/BlockGraph/recursive-bisection+KL+k-way-repair/exhaustive-optimum/MultistreamBlockSim all present",
    "tests are invariants, not smoke: exhaustive-optimum bound, production-roll equivalence 1e-13, bitwise overlapped==serialized, mass conservation, periodic ghost wrap, analytic nu k^2 (10/10 re-run by auditor)",
    "D1 all green: 0.00% vs exhaustive optimum (6/6), <=11.46% vs networkx KL, 0.00% vs best-of-64 (9/9)",
    "D3 bitwise 0.0 between two distinct code paths (manual per-face indexing vs stream3d_roll), incl. edge/corner cells",
    "deterministic artifacts reproduce bit-identically (only wall-clock fields differ)",
    "slab control is the repository's shipped DomainDecomposer decomposition, not a straw man"
  ],
  "level_rationale": "L2 met: real functional implementation (input->compute->output), machine-checkable unit/invariant tests, reproducible run artifacts. L1 met (10/10). L3 NOT met: no production caller (integration_callers empty; module not exported) -> per maturity_gate SCHEME L4 is cumulative on L3, so the present L4-format evidence (benchmark ref-vs-got + repro_cmd) cannot promote the case. L5 impossible (anchor documentary, no external dataset). Level = functional substance = 2.",
  "overstated_claims": [
    "'cross-validated vs waLBerla practice' (title/commit) -> design-level port + offline independent references; no waLBerla artifact compared",
    "'D5 latency hidden 93-166%' -> >100% values are timing-noise artefacts; the precise 92.6% is not supportable",
    "'(predeclared)' 400 steps -> predeclaration text says 600 steps"
  ],
  "audit_artifacts": [
    "docs/audit_m17596/AUDIT_m17596.md",
    "docs/audit_m17596/independent_check.py",
    "docs/audit_m17596/repro_diff_m17596.json",
    "docs/audit_m17596/imbalance_floor.json",
    "docs/audit_m17596/fig_audit_m17596.png",
    "docs/audit_m17596/verdict_m17596.sql"
  ],
  "not_run": ["real MPI multi-rank exchange", "CUDA-stream overlap (no GPU)", "METIS comparison (not installed, no network)", "two-grid mesh convergence for E4", "full E5 re-run (declared non-reproducible)"],
  "disposition": "approve_with_notes",
  "remediation": [
    "L3: drive MultistreamBlockSim from a production runner (or explicit __init__ export + simulation caller), else scope the deliverable as a library module",
    "D2: publish the quantized-floor analysis and use 'imbalance <= floor + 2pp', or add a cut-constrained balance pass closing the 13.56pp gap on refC",
    "predeclaration: align sample.E4 with executed steps (or run 600) and log deviations instead of relabelling them 'predeclared'",
    "D5: paired-difference estimator over >=20 reps with CI; rows with noise-floor denominator must fail rather than be explained",
    "anchor: cite oss_solutions id=365 and record the offline sourcing limitation; add METIS comparison when installable",
    "platform: insert tensor_benchmark_cases rows for D1/D4 and fix tensor_dev_missions.report_id for 17596"
  ]
}$json$,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL;
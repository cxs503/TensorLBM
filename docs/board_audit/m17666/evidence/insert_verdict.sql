INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM','m17637 XVAL-LBM-CARTESIAN-MPI',17666,2,$json${
  "os": "board-audit m17666 of m17637 (anchor XVAL-LBM-CARTESIAN-MPI, repo TensorLBM). Deliverable tree /root/tp-agent/work/m17637 (untracked, head_commit=null). Not a redo: existence+numbers+repro of the three-piece bundle re-verified independently.",
  "impl_files": [
    "tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py:74-152 (measure_pylbm / measure_pylbm_neighbours), :153-190 (measure_tensorlbm reads real staging buffers), :233-307 (collide_slab/stream_slab/run_single/run_decomposed), :355-390 (equivalence + analytic arms)",
    "tools/xval_cartesian_mpi/shims/mpi4py/MPI.py:1-? (13582 B execution-only SimCartComm shim, sha256_16 221db62acf7ddb32)",
    "src/tensorlbm/domain_decomposition.py:98-155 (DomainDecomposer, PRE-EXISTING module = subject under test, not written by m17637), :157-315 (HaloExchange), :376-445 (SimulatedMultiDevice.distribute/gather)"
  ],
  "entry": "PYTHONPATH=tools/xval_cartesian_mpi/shims:/tmp/pylbm_src python tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py --n 24 --steps 100 --out artifacts/xval_cartesian_mpi",
  "run_artifacts": [
    "/root/tp-agent/work/m17637/artifacts/xval_cartesian_mpi/summary.json (15632 B, sha256_16 20e956fcea3e3882)",
    "/root/tp-agent/work/m17637/artifacts/xval_cartesian_mpi/table_decomposition.csv (2848 B, sha256_16 1a34ef5a9b9686db)",
    "/root/tp-agent/work/m17637/artifacts/xval_cartesian_mpi/table_equivalence.csv (1026 B, sha256_16 fbf981423b533bdc)",
    "/root/tp-agent/work/m17637/docs/XVAL-LBM-CARTESIAN-MPI.md (141 lines, 12353 B)",
    "runs/m17666_audit/verify_m17637.json (independent C1-C6 probes)",
    "runs/m17666_audit/rerun_table_equivalence.csv + repro_run.log (audit rerun, exit 0)",
    "runs/m17666_audit/m17666_audit_figure.png (3-panel, injected from verify_m17637.json)"
  ],
  "tests": [
    "tests/test_domain_decomposition.py — audit rerun: PYTHONPATH=src pytest -q => 12 passed in 5.69 s (PRE-EXISTING file, m17637 added no tests; its own §6 'P=1 bit-identical' criterion was not codified as a regression assertion)"
  ],
  "integration_callers": {
    "production": "none: grep -rn 'domain_decomposition|DomainDecomposer|HaloExchange' src/tensorlbm/*.py => only the module itself; no solver imports it",
    "callers_found": ["tests/test_domain_decomposition.py", "scripts/m4_dd_validation.py", "tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py"],
    "parallelism": "n_ranks=2/4 are in-process SimulatedMultiDevice + a hand-written mpi4py shim; no real MPI/NCCL multi-process run (disclosed by m17637 §6)"
  },
  "benchmark": {
    "registered_case": "none: SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CARTESIAN-MPI' -> 0 rows (table total 55)",
    "verdict": "benchmark judgement WITHHELD (sole arbiter has no ref/tolerance row); L4 thus not confirmable",
    "ref_vs_got": {
      "gamma_K_f64": {"got": 0.0028569722585277944, "ref_4nu_k2_over_2": 0.0027415567780803766, "err_pct": 4.209851912249326, "under_3pct": false, "under_15pct": true},
      "K_end_f64_serial": {"got": 0.00035296086897648493, "ref": 0.0003612030603079559, "err_pct": 2.281871954363784, "under_3pct": true},
      "decomposition_equivalence_f64_P1_P2_P4": {"got_rel_l2_pct": 0.0, "ref": 0.0, "err_pct": 0.0, "under_3pct": true},
      "decomposition_equivalence_f32_P1_P2_P4": {"got_rel_l2_pct": 0.0008740599975345202, "ref": 0.0, "under_3pct": true},
      "pylbm_arm_A_geometry": {"rows": 24, "rows_with_numbers": 0, "status": "NOT COMPARABLE: all 24 pylbm rows are IndexError strings"}
    },
    "discrimination_probe": {"intact_halo_rel_l2_pct": 0.0008740599975345202, "noop_halo_rel_l2_pct": 11.414782234591852, "meaning": "the 0.0 equivalence is a measured value, not a bypassed exchange"}
  },
  "repro_cmd": "cd /root/tp-agent/work/m17637 && PYTHONPATH=tools/xval_cartesian_mpi/shims:/tmp/pylbm_src python tools/xval_cartesian_mpi/run_xval_cartesian_mpi.py --n 24 --steps 100 --out /tmp/m17666_audit/artifacts  # auditor rerun exit 0; table_equivalence.csv payload identical field-by-field (identical_payload=true), only wall_time_s differs",
  "external_anchor": "pylbm 0.12.0 (wheel pylbm-0.12.0-py3-none-any.whl, unpacked /tmp/pylbm_src): pylbm/mpi_topology.py:66-79 (ND cart split), :88-116 (get_region_indices remainder distribution), pylbm/storage.py:334-367 (per-dim Irecv/Isend + Waitall). Source-level only: 24/24 numeric pylbm rows missing, so no OSS numeric output was cross-checked; the single real pylbm output is pylbm_neighbours_32_split_2x2 and has no TensorLBM 2x2 counterpart.",
  "findings": {
    "F1_blocking_pylbm_arm_dead": "table_decomposition.csv: 24/24 pylbm rows empty (0 local sizes, 0 byte counts, 0 accepted) while report §3.2 claims 'pylbm 3 axes all accepted' and 'N=16,P=3 -> 6/5/5 -> imbalance 20%' as if computed; docs/XVAL-LBM-CARTESIAN-MPI.md:12 claims 'zero hand-copy: all pylbm-side sizes/neighbours/bytes computed by pylbm's own functions'. Root cause reproduced: run_xval_cartesian_mpi.py:98 takes np.diff of a region-BOUNDARY array then :106 indexes s[0..2] -> IndexError('index 2 is out of bounds for axis 0 with size 2'). With the intended indexing, the same pylbm code immediately yields rank0 local sizes [8,16,16] for n=16 split=(2,) => the pylbm arm was a 2-line fix away and was left dead.",
    "F2_overstated_result_json": "m17637 result.checks says 'pylbm neighbour table/region sizes computed by its MpiTopology' — region sizes: 0 rows; only one neighbours table exists.",
    "F3_unversioned_deliverable": "all three deliverables are untracked (git status ?? x3, head_commit=null); main 9bcc6db6 has no tools/xval_cartesian_mpi => evidence vanishes with the worktree; gate can only read the on-disk tree.",
    "F4_no_tests_added": "m17637 added no test; its own key criterion ('P=1 must be bit-identical to single domain', report §6 pitfall 1) is not a regression assertion.",
    "F5_simulated_mpi": "P=2/4 are simulated in-process; no real MPI build on this host (mpi4py absent, loongarch64) — disclosed honestly in §6, so a scope limit rather than an overstatement.",
    "F6_dual_gamma_keys": "summary.json carries case.gamma_K_exact=0.0054831 (4*nu*k^2) and equivalence[].gamma_K_exact=0.0027416 (2*nu*k^2, since gam=ln(K0/K_end)/(2*steps)); both sides halved so the 4.21% stands, but the duplicated key invites misreading.",
    "F7_hollow_axis_probe": "axis_param_probe only reports nz_local (always nz/P) so it cannot test decomposition_axis; static check shows the field is assigned at domain_decomposition.py:117 and never used in slicing, so the report's 'decorative field' conclusion is correct despite the weak probe."
  },
  "level_rationale": "L2. L0/L1/L2 met: six-section report (tensor_reports 16602, 7871 chars) + analytic TGV arm (gamma 4.21%, K_end 2.28%) + real 17832 B harness whose artifacts reproduce field-for-field and whose equivalence metric has 4-order discrimination against a no-op halo mutation. L3 NOT met: no production solver imports domain_decomposition (only tests/, scripts/m4_dd_validation.py, this tool); parallelism is simulated, not MPI/NCCL. L4 NOT met: tensor_benchmark_cases has 0 rows for the anchor and the anchor's own cross-implementation arm produced 0/24 usable pylbm rows."
}$json$::jsonb,'board-audit')
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id;
SELECT id, repo, scope, mission_id, level, verdict_by, gate_state, gate_level FROM tensor_maturity_verdicts WHERE scope='m17637 XVAL-LBM-CARTESIAN-MPI';

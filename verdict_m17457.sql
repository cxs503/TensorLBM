-- Verdict for m17457 XVAL-LBM-AVERAGE-SURROUND-INIT
-- Audit by m17587 (board-audit cross-check)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by, gate_state, gate_level, gate_note)
VALUES (
    'TensorLBM',
    'm17457 XVAL-LBM-AVERAGE-SURROUND-INIT',
    17587,
    1,
    '{
        "impl_files": [],
        "impl_files_note": "EMPTY: m17457 added zero library source. The only code is experiments/xval_average_surround_init/run_xval.py (experiment harness, sha256 ded93b6f). The compared function _init_new at src/tensorlbm/free_surface_lbm.py:614 is pre-existing (last touched by api-sync commits, not m17457).",
        "entry": "experiments/xval_average_surround_init/run_xval.py",
        "entry_note": "Experiment harness (14743 B). Runs two unit-scale comparisons: (1) homogeneous density → 0% diff, (2) density gradient → 20% diff. Reproducible: verified by re-running, output matches xval_results.json bit-for-bit.",
        "run_artifacts": [
            "experiments/xval_average_surround_init/artifacts/xval_results.json (1772 B, sha256 9726127b)",
            "experiments/xval_average_surround_init/artifacts/xval_init_comparison.png (54642 B, sha256 fd96ad4c)",
            "experiments/xval_average_surround_init/run_xval.py (14743 B, sha256 ded93b6f)",
            "docs/XVAL_AVERAGE_SURROUND_INIT.md (sha256 19b63605)"
        ],
        "run_artifacts_note": "All artifacts exist and are reproducible. Re-run of run_xval.py produces identical xval_results.json. Files are UNTRACKED in git (git status shows ??), meaning they would be lost if the worktree is deleted. No commit was made on agent/m17457 branch.",
        "tests": [],
        "tests_note": "ZERO pytest coverage for this XVAL. No test under tests/ asserts the homogeneous-rho equivalence invariant or pins the 20% gradient divergence. grep -rl average_surround tests/ returns nothing.",
        "integration_callers": [
            "src/tensorlbm/free_surface_topology_transaction.py:599",
            "src/tensorlbm/free_surface_topology_transaction.py:608",
            "src/tensorlbm/free_surface_topology_transaction.py:609"
        ],
        "integration_callers_note": "Pre-existing call sites (not added by m17457). _init_new is called during free-surface cell birth in the topology transaction pipeline. m17457 did not modify any of these callers.",
        "benchmark": null,
        "benchmark_note": "No row in tensor_benchmark_cases for anchor XVAL-LBM-AVERAGE-SURROUND-INIT. The mission criterion is <15% relative difference (owner precision standard). Homogeneous case: 0% PASS; Gradient case: 20% FAIL.",
        "repro_cmd": "cd /root/tp-agent/work/m17457 && python3 experiments/xval_average_surround_init/run_xval.py",
        "repro_cmd_note": "Runs successfully. Produces identical xval_results.json. Note: script uses parents[1] for SRC_DIR which resolves to experiments/src, requiring PYTHONPATH override to find tensorlbm.solver3d. Works in practice but path logic is fragile.",
        "external_anchor": "OSS Maarten-vd-Sande/lbm, lbm/update_types.py::average_surround (2D D2Q9, numba JIT)",
        "external_anchor_note": "OSS source verified at /root/oss-cache/Maarten-vd-Sande_lbm/lbm/update_types.py:135-157. The experiment uses a 3D D3Q19 port of this logic (written by the same agent), so the comparison is port-vs-library, not direct OSS-vs-library. OSS is 2D and would need a 2D TensorLBM baseline for fair comparison.",
        "report_six_elements": {
            "case_introduction": "present (doc §1)",
            "method": "present (doc §2)",
            "parameter_table": "present (doc §3)",
            "comparison_table": "present (doc §4)",
            "convergence_sensitivity": "present (doc §5)",
            "failure_modes_artifacts": "present (doc §6)"
        },
        "report_state": "docs/XVAL_AVERAGE_SURROUND_INIT.md exists (sha256 19b63605). tensor_reports has 0 rows for mission_id=17457 — platform report discipline was skipped.",
        "numerical_verification": {
            "homogeneous_rho_diff": 0.0,
            "homogeneous_rho_diff_rel": 0.0,
            "homogeneous_f_diff": 0.0,
            "homogeneous_u_diff": 0.0,
            "homogeneous_pass": true,
            "gradient_rho_diff": 0.1999998,
            "gradient_rho_diff_rel": 0.1999998,
            "gradient_f_diff": 0.2000001,
            "gradient_pass": false,
            "threshold": 0.15,
            "overall_pass": false,
            "verified_by_rerun": true
        },
        "overstated_claims": [
            "doc §5.2 states average of 6 liquid neighbours — true count in D3Q19 fixture is 5 (not 6). The 20% figure is unaffected but the explanation is wrong in detail.",
            "doc §6 claims OSS average could reduce spurious currents in multiphase CG/SC — pure speculation, no test, no citation.",
            "Headline succeeded + auto-approve while overall_pass=false: the deliverable is an honest negative result, but nothing was registered (no tensor_benchmark_cases row, no tensor_reports row, no commit)."
        ],
        "declared_level_rationale": "L1 (math/unit verification) is what the artefact substantively is: a re-runnable unit-scale numeric cross-check of an EXISTING library function against a 3D port of OSS average_surround, with honest FAIL on the gradient fixture. NOT L2: m17457 added no library source file and no capability. Formal L1 requirement (a test under tests/) is NOT met → expected machine gate_level = 0."
    }',
    'board-audit',
    'pending',
    0,
    '{"note": "Declared level 1 (substance); machine gate_level 0 (no tests under tests/). L1 machine criterion requires non-empty tests array. Re-run verified: homogeneous 0% PASS, gradient 20% FAIL (exceeds 15% threshold). No commit made, no tensor_reports entry, no benchmark case registered. Recommend: (1) fix run_xval.py path fragility, (2) add tests/test_free_surface_init_xval.py, (3) register anchor in tensor_benchmark_cases or retire as not-a-benchmark, (4) commit artifacts.", "tool": "m17587 board-audit", "result": [0, ["L1缺: tests/下无pytest覆盖"]], "audit_mid": 17587}'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET
    level = EXCLUDED.level,
    evidence = EXCLUDED.evidence,
    gate_state = EXCLUDED.gate_state,
    gate_level = EXCLUDED.gate_level,
    gate_note = EXCLUDED.gate_note,
    created_at = now();

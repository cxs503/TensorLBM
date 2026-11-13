-- m17631 board-audit verdict for m17592 (XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION, TensorLBM)
-- Auditor: board-audit (agent/m17631). Evidence: docs/audit_m17592/audit.json
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17592 XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION', 17631, 3, $json$
{
  "impl_files": [
    "src/tensorlbm/amr_reflux.py (claimed in report 1.1/6.2; not independently re-read by auditor)",
    "src/tensorlbm/amr_cavity2d.py (claimed in report 1.1/6.2; not independently re-read by auditor)"
  ],
  "entry": "benchmarks/xval_block_amr_stream_correction/run_xval.py",
  "run_artifacts": [
    "artifacts/xval_block_amr_stream_correction/xval_results.json (6347 B, 2026-11-13 07:06)",
    "artifacts/xval_block_amr_stream_correction/fig1_conservation.png (97391 B)",
    "artifacts/xval_block_amr_stream_correction/fig2_approach_comparison.png (139906 B)",
    "artifacts/xval_block_amr_stream_correction/figures.json",
    "artifacts/amr_reflux_ladder/ladder.json",
    "artifacts/amr_reflux_ladder/ladder.png"
  ],
  "tests": ["tests/test_amr_reflux_ladder.py (claimed 44 passed in report 6.2; NOT re-run by auditor)"],
  "integration_callers": ["none found: standalone NumPy benchmark drivers; no main-solver integration claimed or found"],
  "benchmark": {
    "machine_referee": "CONFIRMED 0 ROWS: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION' -> 0 rows (ILIKE variants %BLOCK_AMR%/%STREAM_CORRECTION%/%XVAL%LBM% also 0). No registered machine referee case; mission used self-declared tolerances (1e-10 / 1e-3 / 15%). Registration gap, not a numeric failure.",
    "ref_vs_got": {
      "1d_mass_drift_reflux_on": {"ref": "0 (theoretical)", "got": 3.691e-16, "tol": 1e-10, "pass": true, "source": "fresh run (ladder.json)"},
      "1d_mass_drift_reflux_off": {"got": 0.01429, "pass": true, "role": "negative control (14 orders of magnitude)"},
      "1d_ladder_rungs": {"got": "8/8 pass=true", "pass": true},
      "1d_grid_convergence": {"N": [48, 96, 192], "l1": [0.025, 0.0134, 0.00313], "orders": [0.90, 2.10], "pass": true},
      "2d_interface_mismatch_rel": {"ref": "0 (theoretical)", "got": 2.055e-05, "tol": 0.001, "pass": true, "source": "SECONDHAND docs/amr-reflux-maturation.md (40000 coarse steps)"},
      "2d_mass_drift_with_reflux": {"got": 6.473e-05, "pass": true, "source": "SECONDHAND docs/amr-reflux-maturation.md"},
      "2d_ghia_max_abs_dev": {"ref": "0 (steady state)", "got": 0.5269, "pct_of_u_lid": 52.69, "tol": "15%", "pass": false, "note": "HONEST FAIL, report 6.1/7: transient not converged (uniform grid same steps 0.1751); attributed to transient + one-sided F-H over-dissipation, not AMR interface defect"}
    }
  },
  "repro_cmd": "cd /root/tp-agent/work/m17592 && python3 benchmarks/xval_block_amr_stream_correction/run_xval.py (script present; auditor did not re-execute)",
  "external_anchor": [
    "lssfau/walberla src/lbm/refinement/EqualLevelBorderStreamCorrection.h (source-audited per report 1.2, zero hand-copy)",
    "Ghia et al. 1982 2-D lid-driven cavity (2-D accuracy reference)"
  ],
  "report_six_elements": "all present in benchmarks/xval_block_amr_stream_correction/XVAL_REPORT.md (232 lines, 8 sections: case intro 1, method 2, results 4, figures fig1/fig2, caveats 6.1, error verdict 7)",
  "level_rationale": "L2 MET: real code (F-H rescaling + Berger-Colella reflux) per report 1.1/6.2. L3: library module + dedicated benchmark drivers, NOT integrated into main production solver (NumPy single-core benchmark; torch backend available but serial). L4 PARTIAL: 1-D conservation sub-case is L4-grade (complete case + error analysis + reproducible + negative control + two-grid convergence order ~2); 2-D conservation PASS but document-sourced; 2-D accuracy HONEST FAIL 52.69% > 15%. L5 NOT MET: deliverable is a cross-validation benchmark, no new research-grade physics. Overall L3.",
  "checks_not_run": [
    "repro_cmd re-execution",
    "tests/test_amr_reflux_ladder.py re-execution",
    "impl file content verification (amr_reflux.py / amr_cavity2d.py)"
  ],
  "overstated_claims_extra": [
    "No tensor_benchmark_cases referee row registered for the anchor (0 rows) -> the benchmark ref-vs-got leg is self-referential (mission-declared tolerances), not an independent registered benchmark"
  ],
  "overstated_claims": [
    "2-D conservation numbers (2.055e-05 / 6.473e-05) cited from docs/amr-reflux-maturation.md, not freshly produced by this mission's run; report discloses in 4.2 header but table 4.2 presents them as 'TensorLBM measured'"
  ],
  "disposition": "confirmed",
  "disposition_rationale": "Core anchor claim (interface stream correction conserves mass) verified by fresh 1-D run at machine precision with negative control and two-grid convergence; 2-D conservation PASS (document-sourced); only failing metric (2-D Ghia accuracy) honestly declared and out of stream-correction anchor scope. Mission succeeded at its stated XVAL objective. 2-D steady-state accuracy remains an open item for a future maturation mission."
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;

-- Verdict for m15356 GAP-PP-2-lid-wall-diff (board-audit)
-- Level: L4 (基准验证，τ反演n=65未达2%但梯度验证全绿)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15356 GAP-PP-2-lid-wall-diff',
  15423,
  4,
  '{
    "impl_files": [
      "src/tensorlbm/diff_lid_cavity.py:488 lines (moving_lid_values/moving_lid_bc/step/rollout/invert_relaxation_time)",
      "src/tensorlbm/solver.py:collide_bgk fixed to keep 0-dim tensor tau in graph"
    ],
    "entry": "scripts/run_gap_pp2_lid_wall_diff.py (steady/sweep/fdcheck/equiv/invert subcommands)",
    "run_artifacts": [
      "artifacts/gap_pp2_lid_wall_diff/steady_n65.json (RMSE u=4.53%, v=1.77%, 40k steps)",
      "artifacts/gap_pp2_lid_wall_diff/steady_n129.json (RMSE u=3.31%, v=1.55%, 60k steps)",
      "artifacts/gap_pp2_lid_wall_diff/sweep_n65.json (tau sensitivity scan)",
      "artifacts/gap_pp2_lid_wall_diff/fdcheck_n32_t600.json (AD vs FD, best rel_err=9.23e-07)",
      "artifacts/gap_pp2_lid_wall_diff/equiv_prod_vs_diff_n65_t4000.json (bit_identical=true)",
      "artifacts/gap_pp2_lid_wall_diff/invert_probe_n48_t4000.json (rel_err=1.80% PASS)",
      "artifacts/gap_pp2_lid_wall_diff/invert_same_n65_t4000.json (rel_err=2.98% FAIL)",
      "artifacts/gap_pp2_lid_wall_diff/figures/{cavity_field,ghia_comparison,tau_inversion,tau_sensitivity}.png"
    ],
    "tests": [
      "tests/test_diff_lid_cavity.py: 21 passed (non-slow), 1 slow (test_invert_recovers_tau_smoke)",
      "tests/test_autograd.py: 15 passed (non-slow)"
    ],
    "integration_callers": [
      "tests/test_diff_lid_cavity.py imports tensorlbm.diff_lid_cavity",
      "scripts/run_gap_pp2_lid_wall_diff.py uses full pipeline"
    ],
    "benchmark": {
      "ghia_re100_n65": {"rmse_u": 0.0453, "rmse_v": 0.0177, "within_3pct": "u:NO(4.53%), v:YES(1.77%)"},
      "ghia_re100_n129": {"rmse_u": 0.0331, "rmse_v": 0.0155, "within_3pct": "u:NO(3.31%), v:YES(1.55%)"},
      "ad_vs_fd_best": {"rel_err_tau": 1.28e-06, "rel_err_u_lid": 9.23e-07, "threshold": 1e-04, "pass": true},
      "equiv_prod_vs_diff": {"max_abs_diff": 0.0, "bit_identical": true},
      "tau_inversion_n48": {"rel_err": 0.01795, "threshold": 0.02, "pass": true},
      "tau_inversion_n65": {"rel_err": 0.02981, "threshold": 0.02, "pass": false, "note": "optimizer limit cycle, not gradient error"}
    },
    "repro_cmd": "PYTHONPATH=src python scripts/run_gap_pp2_lid_wall_diff.py {steady,sweep,fdcheck,equiv,invert} --out artifacts/gap_pp2_lid_wall_diff/",
    "external_anchor": "Ghia et al. (1982) JCP 48(3):387-411 Table I, Re=100 centreline data"
  }',
  'board-audit'
);

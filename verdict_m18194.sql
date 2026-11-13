-- Verdict for m18194: XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD
-- Audit by: board-audit (m18312)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm18194 XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD',
  18312,
  4,
  '{
    "impl_files": ["src/tensorlbm/phasefield/pf_pressure_evolution.py:610 lines"],
    "entry": "scripts/xval_m18194_pf_pressure_evolution.py:770 lines",
    "run_artifacts": [
      "artifacts/xval_m18194/summary.json",
      "artifacts/xval_m18194/oss_extract.json",
      "artifacts/xval_m18194/fields.npz",
      "artifacts/xval_m18194/fig1_laplace.png",
      "artifacts/xval_m18194/fig2_fields.png",
      "artifacts/xval_m18194/fig3_profiles.png",
      "artifacts/xval_m18194/fig4_taylor_green.png",
      "artifacts/xval_m18194/figures.json"
    ],
    "tests": ["tests/test_phasefield_pf_pressure_evolution.py:7 passed (pytest)"],
    "integration_callers": ["scripts/xval_m18194_pf_pressure_evolution.py"],
    "benchmark": {
      "G1_laplace_max_err_pct": 3.13,
      "G1_threshold_pct": 15.0,
      "G1_pass": true,
      "G2_viscosity_err_pct": 0.013,
      "G2_threshold_pct": 15.0,
      "G2_pass": true,
      "G3_mass_error_max": 5.95e-13,
      "G3_threshold": 1e-10,
      "G3_pass": true,
      "G4_density_ratio_consistency_pct": 1.06,
      "G4_threshold_pct": 15.0,
      "G4_pass": true,
      "total_drop_cases": 12,
      "all_within_gate": true
    },
    "repro_cmd": "python scripts/xval_m18194_pf_pressure_evolution.py --workers 13 --steps 6000",
    "external_anchor": "CFD-GO/TCLB models/multiphase/d2q9_pf_pressureEvolution (GPL-3.0, structural equivalence verified P1-P14, max residual 8.3e-17; OSS binary NOT executable in this environment)",
    "report": "docs/XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD.md + tensor_reports id=17090",
    "level_rationale": "L4: complete benchmark validation with analytic references (12 drop cases + Taylor-Green), structural equivalence (14/14 PASS), reproducibility (driver script + predeclared thresholds). NOT L5 because: (1) OSS binary not executable → no byte-level TCLB comparison; (2) scope limited to periodic BC static drops + TG, no wall boundaries/dynamic tests; (3) no integration into main solver pipeline."
  }',
  'board-audit'
);

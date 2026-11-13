-- Maturity verdict for m18090: XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION
-- Audit by: board-audit (m18145)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (
    repo, 
    scope, 
    mission_id, 
    level, 
    evidence, 
    verdict_by
) VALUES (
    'TensorLBM',
    'm18090 XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION',
    18090,
    4,
    '{
        "impl_files": [
            "src/tensorlbm/contact_angle_inversion/__init__.py:1-47",
            "src/tensorlbm/contact_angle_inversion/calibration.py:1-380",
            "src/tensorlbm/contact_angle_inversion/validate.py:1-180"
        ],
        "entry": "src/tensorlbm/contact_angle_inversion/__init__.py:calibrate,invert,forward",
        "run_artifacts": [
            "artifacts/xval_huang2007_contact_angle_inversion/validation_result.json"
        ],
        "tests": [
            "tests/test_contact_angle.py:test_young_relation_round_trip",
            "tests/test_contact_angle.py:test_young_relation_limits",
            "tests/benchmarks/wetting_contact_angle/test_wetting_contact_angle.py"
        ],
        "integration_callers": [],
        "benchmark": {
            "anchor": "XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION",
            "reference": "Huang et al. (2007) Phys Rev E 76(6):066701",
            "method": "calibration-based inversion (empirical Huang correction)",
            "results": {
                "theta_100": {"target": 100.0, "measured": 101.65, "error_pct": 1.65},
                "theta_120": {"target": 120.0, "measured": 123.08, "error_pct": 2.57},
                "theta_140": {"target": 140.0, "measured": 141.52, "error_pct": 1.09}
            },
            "summary": {"passed": 3, "total": 3, "max_error_pct": 2.57, "criterion": "<15%"},
            "verdict": "PASS"
        },
        "repro_cmd": "cd /root/tp-agent/work/m18090 && PYTHONPATH=src:$PYTHONPATH python3 src/tensorlbm/contact_angle_inversion/validate.py",
        "external_anchor": "Huang2007_PhysRevE_76_066701",
        "report": "docs/XVAL_LBM_HUANG2007_CONTACT_ANGLE_INVERSION.md",
        "notes": "Calibration-based inversion implements Huang 2007 insight: Young relation systematically wrong (slope ratio 3.17 vs 1.0). Hold-out validation 3/3 pass with max error 2.57% < 15% criterion. Resolution-dependent (must recalibrate per grid). Standalone module, not integrated into main solver workflow."
    }',
    'board-audit'
);

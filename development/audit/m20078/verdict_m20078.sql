-- Verdict for m20078: XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS
-- Board audit (交差审计) by agent/m20180
-- Level: L4 (基准验证 — full benchmark suite with predeclared gates, error analysis vs analytic Young law, reproducibility)

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorLBM',
    'm20078 XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS',
    20180,
    4,
    '{
        "impl_files": [
            "src/tensorlbm/wetting/two_impl_contact_angle.py:499 lines, 16 defs (psi_wall_from_angle, wall_ghost_rows, grad_laplacian_ghost, wall_angle_readback, measure_contact_angle, TwoImplConfig, relax_with_wall_condition, run_arm)",
            "src/tensorlbm/wetting/wall_order_parameter_2d.py:700 lines (free_energy_parameters, contact_angle_from_field, initialise_drop_on_wall, relax_to_equilibrium)",
            "src/tensorlbm/wetting/geometric_two_candidate.py:384 lines"
        ],
        "entry": "xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py (sweep/summarize/one commands)",
        "run_artifacts": [
            "xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/summary.json (34 rows, verdict=PASS, anchor_pass=True)",
            "xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/REPORT.md",
            "xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/predeclared.json (frozen gates)",
            "artifacts/two_impl_contact_angle_quick/summary.json (quick benchmark, 6 rows)",
            "artifacts/two_impl_contact_angle_quick/figures/ (fig1_ref_vs_got.png, fig2_cross_arm.png, fig3_wall_row.png)"
        ],
        "tests": [
            "tests/test_two_impl_contact_angle.py: 34 passed (wall_ghost equivalence, synthetic cap recovery, relaxation smoke, mass conservation, wall angle readback regression)",
            "tests/test_two_impl_contact_angle_xval.py: 20 passed + 1 skipped (normal estimator, solver integration, arm equilibration, delivered artifact gate check)"
        ],
        "integration_callers": [
            "src/tensorlbm/wetting/ladder.py: contact_two_impl() -> CAC2D solver with wall_order_parameter",
            "src/tensorlbm/wetting/__init__.py: exports two_impl, psi_wall_from_angle, wall_ghost_rows, run_contact_angle_arm",
            "benchmarks/case_pool/two_impl_contact_angle/run_case.py: pool case runner with 3 independent estimators"
        ],
        "benchmark": {
            "anchor": "XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS",
            "ref": "benchmarks/case_pool/two_impl_contact_angle/refs.json (G1-G7 gates, Young law reference)",
            "got": "neumann_flux median_err=2.69%, geometric_ghost median_err=2.96%, max_err=9.35% (all < 15% gate)",
            "cross_arm": "7 cross-arm comparisons, all pass (max diff=6.84% at baseline 135deg)",
            "tolerance": "accept_pct=15.0, cross_arm_pct=15.0, estimator_agreement_deg=10.0"
        },
        "repro_cmd": "python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py sweep && python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py summarize && python3 -m pytest tests/test_two_impl_contact_angle.py -q",
        "external_anchor": "lssfau/walberla oss_solutions row 281 (two independent contact-angle implementations)"
    }',
    'board-audit'
);

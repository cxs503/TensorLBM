-- Verdict for m15436 CASE-RAYLEIGH-TAYLOR audit (mission 15716)
-- Level: L3 (System Integration)
-- Rationale: Real implementation (502 lines), 11/11 tests pass, 6-case benchmark run with artifacts,
-- but L4 not achieved: benchmark NOT registered in tensor_benchmark_cases (0 rows),
-- terminal velocity gates all fail (0/6), platform L4 verdict = CLAIMED_NOT_PROVEN.

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
    'TensorLBM',
    'm15436 CASE-RAYLEIGH-TAYLOR',
    15716,
    3,
    '{
        "impl_files": [
            "src/tensorlbm/rayleigh_taylor.py:1-502 (Boussinesq two-layer D2Q9 + advected phase field)",
            "benchmarks/case_pool/rayleigh_taylor/run_case.py (6-case benchmark driver)"
        ],
        "entry": "run_rayleigh_taylor(cfg: RTConfig) -> dict; rt_step(f, g, cfg, phi)",
        "run_artifacts": [
            "benchmarks/case_pool/rayleigh_taylor/artifacts/runs/<6 cases>/case.json + history.csv",
            "benchmarks/case_pool/rayleigh_taylor/artifacts/runs/summary.json (n_all_pass=0, n_cases=6)",
            "benchmarks/case_pool/rayleigh_taylor/artifacts/figures/fig1_linear_growth.png",
            "benchmarks/case_pool/rayleigh_taylor/artifacts/figures/fig2_fronts_and_tip_velocity.png",
            "benchmarks/case_pool/rayleigh_taylor/artifacts/figures/fig3_field_snapshot.png",
            "benchmarks/case_pool/rayleigh_taylor/artifacts/diagnostics/tip_velocity.csv"
        ],
        "tests": "tests/test_rayleigh_taylor.py: 11/11 passed",
        "integration_callers": [
            "benchmarks/case_pool/rayleigh_taylor/run_case.py",
            "tests/test_rayleigh_taylor.py"
        ],
        "benchmark": {
            "registered_in_tensor_benchmark_cases": false,
            "linear_growth_rate": "6/6 pass (errors -2.7% to -9.9%, within 10% tolerance)",
            "terminal_velocity": "0/6 fail (u_bubble error +160% to +220000%, u_spike similarly)",
            "root_cause": "Phase field unbounded (|phi| up to 341 for A=0.5) -> buoyancy amplified -> velocity overestimation -> blowup",
            "platform_l4_verdict": "CLAIMED_NOT_PROVEN (no tensorbench_results row)"
        },
        "repro_cmd": "cd /root/tp-agent/work/m15436 && python3 benchmarks/case_pool/rayleigh_taylor/run_case.py",
        "external_anchor": "Rayleigh (1883); Taylor (1950); Chandrasekhar (1961) §92; Goncharov (2002) PRL 88:134502"
    }',
    'board-audit'
);

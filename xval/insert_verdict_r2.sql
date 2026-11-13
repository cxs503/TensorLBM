-- Round-2 benchmark registration for XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S
-- Supersedes the Round-1 (m17963) row: the REAL repo D3Q19 solver now passes the
-- pre-declared pointwise metric at 0.19% (< 15% and < 3%).  Numbers read from
-- xval/xval_nebb_bodyforce_driver_r2.json (programmatic, zero hand-copy).

UPDATE tensor_benchmark_cases
SET metric = 'max pointwise relative error of ux(y) interior fluid rows vs closed-form Poiseuille',
    threshold = 0.15,
    got = 0.0019047147506762966,
    status = 'passed',
    note = 'Round-2 (REAL repo integration): GPUCorePathSolver (collide_guo_into + apply_channel_walls, D3Q19) vs analytic = 0.19% pointwise, < 3% ecosystem met at NY=10. Supersedes Round-1 failed row (23.50% was a hand-written D2Q9 re-implementation artifact, never the repo code). Repro: PYTHONPATH=src python3 xval/xval_nebb_bodyforce_driver_r2.py'
WHERE anchor_slug = 'XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S';

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S', 18051, 3,
        '{"impl_files": ["/root/tp-agent/work/m18051/xval/xval_nebb_bodyforce_driver_r2.py (REAL integration: calls tensorlbm.gpu_core_path.GPUCorePathSolver / collide_guo_into / apply_channel_walls; 0 hand-written re-implementation)", "/root/tp-agent/work/m18051/xval/probe_real_repo_channel.py (F1/F3 probe)", "src/tensorlbm/gpu_core_path.py:278-343 collide_guo_into (Guo 2002: feq at u*=u+F/(2rho), force term at actual u, scaled (1-1/(2tau)))", "src/tensorlbm/gpu_core_path.py:apply_channel_walls (half-way bounce-back, Ziegler 1993/Ladd 1994)"], "entry": "PYTHONPATH=src python3 xval/xval_nebb_bodyforce_driver_r2.py", "run_artifacts": ["/root/tp-agent/work/m18051/xval/xval_nebb_bodyforce_driver_r2.json (all KPIs, 2 grids)", "/root/tp-agent/work/m18051/xval/xval_nebb_bodyforce_r2_profile.png"], "tests": "xval/probe_real_repo_channel.py + xval/xval_nebb_bodyforce_driver_r2.py (real solver, CPU float64, eager)", "integration_callers": "1: GPUCorePathSolver (real repo solver) is the caller; repo code exercised", "benchmark": {"table": "tensor_benchmark_cases", "anchor_slug": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S", "predeclared_metric": {"name": "max pointwise relative error of ux(y) interior vs closed form", "threshold": 0.15, "got": 0.0019047147506762966, "pass": true}, "ecosystem_3pct_met": true, "no_slip_enforced": true, "grid_ny20_pointwise": 0.00037141470253260797, "oss_arm": {"global": 0.05, "pointwise": 0.125, "note": "NOT_RUN (jax absent); numbers from board-audit m17963 rerun"}}, "repro_cmd": "cd /root/tp-agent/work/m18051 && PYTHONPATH=src python3 xval/xval_nebb_bodyforce_driver_r2.py", "external_anchor": "haimingz/vivsim @ /tmp/vivsim_src (D2Q9 Poiseuille; OSS code NOT executed here - jax missing => NOT_RUN)", "audited_mission": 17759, "audited_commit": "08a55f8e", "level": 3, "level_reason": "L3 = real solver caller (GPUCorePathSolver) + registered passing benchmark row on the pre-declared pointwise metric (0.19% < 15% and < 3%). L4 unblocked: pre-declared metric passes without post-hoc switch; no-slip enforced (first fluid row = analytic parabola value to 0.8% of u_max).", "defects_fixed": ["F1: real repo code now executed (GPUCorePathSolver)", "F2: pre-declared pointwise metric passes at 0.19% (no metric switch)", "F3: no-slip enforced; wall slip = correct near-wall velocity at first fluid row", "F4: benchmark row registered (status=passed)"], "checks_not_run": ["real vivsim execution (jax absent -> NOT_RUN)"], "figures": ["xval/xval_nebb_bodyforce_r2_profile.png"], "report": "xval/REPORT_XVAL_NEBB_BODYFORCE_R2.md", "numbers": "xval/xval_nebb_bodyforce_driver_r2.json"}'::jsonb, 'xval-r2')
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
              evidence = EXCLUDED.evidence;
COMMIT;

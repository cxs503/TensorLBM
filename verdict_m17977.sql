-- Board audit verdict for m17977: XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE
-- Auditor: board-audit (m18088 cross-audit)
-- Date: 2026-11-13

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm17977 XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE',
  18088,
  3,
  '{
    "impl_files": [
      "src/tensorlbm/cg_wetting_geometric.py — 26KB, gradient-rotation wetting BC port of openLBMPM RKCG2D (reference: AcceleratedRKGPU2D.py:2426-2492, RKD2Q9.py:808-890)",
      "src/tensorlbm/cg_contact_angle.py — 13KB, contact angle measurement utilities (circle fit, local normal, cap inversion)",
      "src/tensorlbm/color_gradient2d.py — 16KB, modified solver with wall_contact_angle parameter (lines 128-295)"
    ],
    "entry": "benchmarks/xval_openlbmpm_wetting/run.py — 568 lines, three-block benchmark (equivalence/analytic/sessile)",
    "run_artifacts": [
      "results/results.json — 24KB, full structured results with checks/pass/honest_fail",
      "results/figures/fig_analytic_readback.png — 57KB, prescribed vs read-back angle plot",
      "results/figures/fig_analytic_worst_node.png — 39KB, worst-node gradient visualization",
      "results/figures/fig_equivalence.png — 46KB, port vs reference equivalence scatter",
      "results/figures/fig_sessile_angles.png — 40KB, sessile drop angle history",
      "results/figures/fig_sessile_history.png — 30KB, sessile drop convergence history"
    ],
    "tests": [
      "tests/test_cg_geometrical_wetting.py — 12 passed / 5.72s (ghost value, neutral angle, no-interface, theta range, clamping, backward compatibility)",
      "tests/test_contact_angle.py — 12 passed, 1 skipped / 94.51s (Young relation, sessile initial profile, analytic caps, wall potential, mass conservation)"
    ],
    "integration_callers": "src/tensorlbm/color_gradient2d.py:128-295 — color_gradient_two_tau_step(wall_contact_angle=...) calls rotate_wall_color_gradient() from cg_wetting_geometric; invoked by benchmark runner and tests with real solver execution",
    "benchmark": {
      "registered_case": "none: no row in tensor_benchmark_cases for anchor XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE → baseline裁判缺位，以报告自注册判据为准",
      "delivered_ref_vs_got": {
        "equivalence": {
          "n_cases": 660,
          "max_abs_diff": 3.0817570717545095e-12,
          "threshold": 1e-12,
          "pass": false,
          "note": "FAIL by 3.08x threshold; difference is at float64 machine-epsilon level (per-angle max: 135°=3.08e-12, 150°=9.13e-13); not a semantic error but a strict threshold violation"
        },
        "analytic": {
          "n_cases": 22,
          "max_err_pct": "~0.00% (machine precision, median abs err < 1e-13 deg)",
          "pass": true,
          "note": "All 22 cases (xi∈{2,3} × θ∈{15,30,...,165}) pass with essentially zero error; gradient magnitude preserved to 1e-16"
        },
        "sessile": {
          "theta60": {"theta_local": "NaN", "theta_circle": 180.0, "theta_cap": 158.7, "err_pct_circle": 200.0, "err_pct_cap": 164.5, "pass_15pct": {"circle": false, "cap": false}},
          "theta90": {"theta_local": "NaN", "theta_circle": 98.7, "theta_cap": 150.6, "err_pct_circle": 9.68, "err_pct_cap": 67.3, "pass_15pct": {"circle": true, "cap": false}},
          "theta120": {"theta_local": "NaN", "theta_circle": 61.5, "theta_cap": 147.2, "err_pct_circle": -48.7, "err_pct_cap": 22.7, "pass_15pct": {"circle": false, "cap": false}},
          "local_pass": {"theta60": false, "theta90": false, "theta120": false},
          "circle_pass": {"theta60": false, "theta90": true, "theta120": false},
          "pass_control_metric": false,
          "pass_global_shape_metric": false,
          "honest_note": "local (contact-line) estimator is NaN because the host A-perturbation solver carries spurious currents (|u|max ~1e-2) that reshape the drop over 1000 steps, preventing a stable contact line from forming; this is a pre-existing host-solver limitation documented in docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING.md"
        }
      },
      "checks": {
        "equivalence_port_vs_reference": false,
        "analytic_readback_within_15pct": true,
        "sessile_local_estimator_within_15pct": false,
        "sessile_global_shape_within_15pct": false
      },
      "overall_pass": false,
      "honest_fail": ["equivalence_port_vs_reference", "sessile_local_estimator_within_15pct", "sessile_global_shape_within_15pct"]
    },
    "repro_cmd": "cd <TensorLBM> && python benchmarks/xval_openlbmpm_wetting/run.py --blocks equivalence,analytic,sessile --steps 1000 --thetas 60,90,120 --out results",
    "external_anchor": {
      "anchor": "PorousMediaSimulation/openLBMPM (RKCG2D geometrical wetting)",
      "source_path": "/tmp/openLBMPM",
      "reference_files": {
        "rotation_kernel": "RKCG2D/AcceleratedRKGPU2D.py:2426-2492",
        "wall_normal_table": "RKCG2D/RKD2Q9.py:808-890",
        "solid_color_value": "RKCG2D/AcceleratedRKGPU2D.py:1557-1578",
        "isotropic_gradient": "RKCG2D/AcceleratedRKGPU2D.py:1585-1633",
        "parameters": "IniFiles/RKtwophasesetup2D.ini"
      },
      "sha256_verified": true,
      "durable": false
    },
    "report_sections": {"①案例介绍": true, "②方法": true, "③计算结果": true, "④结果图/对照表": true, "⑤注意点/敏感性": true, "⑥误差判定": true},
    "git_delivery_state": {
      "worktree_head": "see git status",
      "status": ["M src/tensorlbm/color_gradient2d.py", "?? benchmarks/xval_openlbmpm_wetting/", "?? docs/XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE.md", "?? src/tensorlbm/cg_contact_angle.py", "?? src/tensorlbm/cg_wetting_geometric.py", "?? tests/test_cg_geometric_wetting_xval.py"],
      "risk": "worktree changes not yet committed; platform auto-push will handle after gate approval"
    },
    "level_rationale": "L3: 真代码实现（cg_wetting_geometric.py + cg_contact_angle.py）+ 24 项测试全绿 + 已接入主求解器 color_gradient_two_tau_step(wall_contact_angle=...)。L4 不予：(1) equivalence 块 max_abs_diff=3.08e-12 超阈值 1e-12（虽为浮点噪声级但严格判据不通过）；(2) sessile 块 local estimator 全 NaN、circle/cap 多数失败，原因为宿主求解器伪流（|u|max~1e-2）导致液滴未达静态平衡——此为宿主求解器既有局限，非本 BC 实现之错，但按预注册判据仍为 FAIL；(3) tensor_benchmark_cases 该 anchor 无注册行，基准裁判缺位。报告诚实申报 honest_fail，未粉饰。",
    "findings": [
      "F1 equivalence 阈值过严：3.08e-12 vs 1e-12 仅超 3 倍，属 float64 舍入噪声，建议阈值放宽至 1e-10 或改用相对误差",
      "F2 sessile local estimator 全 NaN：宿主 A-扰动求解器伪流过强（|u|max~1e-2），接触线无法稳定形成；需先修复宿主求解器伪流问题或改用更稳定的表面张力形式（如 CSF）",
      "F3 基准裁判缺位：tensor_benchmark_cases 无该 anchor 注册行，无法进行独立基准比对",
      "F4 外部锚不可持久：/tmp/openLBMPM 无 commit SHA 钉扎，复核依赖临时路径",
      "F5 报告质量良好：六段齐全，honest_fail 如实申报，5 张结果图程序化生成零手抄"
    ]
  }'::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET
  level = EXCLUDED.level,
  evidence = EXCLUDED.evidence,
  mission_id = EXCLUDED.mission_id,
  gate_state = 'pending',
  gate_note = NULL;

SELECT id, repo, scope, level, verdict_by, gate_state
FROM tensor_maturity_verdicts
WHERE scope = 'm17977 XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE'
  AND verdict_by = 'board-audit';

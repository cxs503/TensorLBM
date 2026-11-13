-- m18223 交差核验 verdict: 对象 m18130 XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND (TensorLBM)
-- 判级 L4：完整基准验证（Couette/Poiseuille 闭式对标 + 等价性检验 + 预声明阈值全 PASS + 灵敏度 5 轴 + 可复现）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18130 XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND', 18223, 4, $json${
  "verdict_basis": "L4 授级：完整基准验证——Couette (Ny=16/24/32) 与 Poiseuille (Ny=16/24) 对标闭式解，profile L2 最大 0.0792%（远低于 3% 内部门/15% owner 门）；arm A 显式 BC 与 arm B hook+mask-blend 状态摘要逐位相等（sha256 一致）；闭箱 500 步质量漂移 3.37e-14；复合通道 Qin/Qout 失衡 1.34e-11%；5 轴灵敏度（Ny、壁面约定、save_interval、ramp_steps、Poiseuille Ny）全部量化；14/14 pytest 通过；repro_cmd 可执行；代码改动真实（src/tensorlbm/step_hooks.py 库级代码）。不授 L5：本单是 OSS 移植+交叉验证，非新物理/新能力科研突破。",
  "claimed_level_implicit": "L4（完整基准验证+误差分析+可复现+预声明判据）",
  "delivered_level": 4,
  "gap_to_claim": "无——交付与声称一致",
  "impl_files": [
    "src/tensorlbm/step_hooks.py (solve_with_hooks, compose_masked_boundaries, masked_step, velocity_ramp_hook, BoundaryLayer 数据类)",
    "tests/test_step_hooks.py (14 tests: 表选择/层验证/静止层=显式反相/部分 mask 线性混合/动壁退化为静止壁/动量注入 6w(c·u)/平衡层安装速度/重叠 mask 拒绝/不交 mask 分区/驱动顺序与节拍/恒等钩子物理中性/速度斜坡/可调层速度/Couette 端到端)"
  ],
  "entry": "benchmarks/xval_per_step_hook_mask_blend/run_xval.py",
  "entry_note": "主驱动脚本，运行 Couette/Poiseuille/composite_channel/invariance/upstream_lid 五组算例，输出 result.json/table.csv/figures",
  "run_artifacts": [
    "artifacts/xval_per_step_hook_mask_blend/result.json (34KB, verdict PASS, 8 KPI checks)",
    "artifacts/xval_per_step_hook_mask_blend/table.csv (14 rows, arm×config)",
    "artifacts/xval_per_step_hook_mask_blend/profiles.npz (速度剖面数据)",
    "artifacts/xval_per_step_hook_mask_blend/figures/fig1_couette_profiles.png",
    "artifacts/xval_per_step_hook_mask_blend/figures/fig2_grid_convergence.png",
    "artifacts/xval_per_step_hook_mask_blend/figures/fig3_upstream_lid_gap.png",
    "artifacts/xval_per_step_hook_mask_blend/figures/fig4_composite_channel_ux.png",
    "artifacts/xval_per_step_hook_mask_blend/figures/fig5_poiseuille_profiles.png",
    "artifacts/xval_per_step_hook_mask_blend/oss_ref/evidence.json (OSS 取证+sha256)",
    "benchmarks/xval_per_step_hook_mask_blend/predeclared.json (阈值+假设+证伪条件+混杂因素+停止规则)",
    "docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md (六段报告)"
  ],
  "tests": [
    "tests/test_step_hooks.py::test_tables_for_picks_lattice_by_rank PASSED",
    "tests/test_step_hooks.py::test_layer_validation PASSED",
    "tests/test_step_hooks.py::test_stationary_layer_equals_explicit_reversal PASSED",
    "tests/test_step_hooks.py::test_partial_mask_is_a_linear_blend PASSED",
    "tests/test_step_hooks.py::test_moving_layer_reduces_to_stationary_at_zero_velocity PASSED",
    "tests/test_step_hooks.py::test_moving_layer_momentum_injection_is_6_w_c_dot_u PASSED",
    "tests/test_step_hooks.py::test_equilibrium_layer_installs_requested_velocity PASSED",
    "tests/test_step_hooks.py::test_overlapping_masks_are_rejected PASSED",
    "tests/test_step_hooks.py::test_disjoint_masks_partition_the_field PASSED",
    "tests/test_step_hooks.py::test_solve_with_hooks_order_and_save_cadence PASSED",
    "tests/test_step_hooks.py::test_identity_hook_does_not_change_the_trajectory PASSED",
    "tests/test_step_hooks.py::test_velocity_ramp_hook_reaches_target PASSED",
    "tests/test_step_hooks.py::test_callable_layer_velocity_reads_hook_state PASSED",
    "tests/test_step_hooks.py::test_couette_reaches_analytic_linear_profile PASSED"
  ],
  "integration_callers": [
    "benchmarks/xval_per_step_hook_mask_blend/run_xval.py (五组算例全部通过 solve_with_hooks + masked_step 驱动)",
    "benchmarks/xval_per_step_hook_mask_blend/render_report.py (从 result.json 生成 Markdown 报告)"
  ],
  "benchmark": {
    "metric": "profile_l2_pct, max_err_pct, equivalence_max_rel, mass_drift_rel, mass_imbalance_pct",
    "machine_referee": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND' -> 0 rows (no registered benchmark case; predeclared.json thresholds used as referee)",
    "ref_vs_got": {
      "couette_profile_l2_pct_Ny32": {"got": 0.000427, "threshold_15pct": true, "threshold_3pct": true},
      "couette_max_err_pct_Ny32": {"got": 0.000348, "threshold_15pct": true, "threshold_3pct": true},
      "poiseuille_profile_l2_pct_Ny24": {"got": 0.0792, "threshold_15pct": true, "threshold_3pct": true},
      "poiseuille_peak_err_pct_Ny24": {"got": 0.0580, "threshold_15pct": true, "threshold_3pct": true},
      "explicit_vs_blend_equivalence_max_rel": {"got": 0.0, "threshold_0": true},
      "hook_cadence_neutrality_sha256": {"equal": true},
      "closed_box_mass_drift_rel": {"got": 3.37e-14},
      "composite_channel_mass_imbalance_pct": {"got": 1.34e-11}
    },
    "sensitivity": {
      "wall_convention_trap": "halfway (H=Ny-2) → 0.0004% vs node-wall (H=Ny-1) → 4.545%",
      "save_interval_50_vs_500": "sha256 equal (physics-neutral)",
      "ramp_1_vs_200": "profile delta 7.36e-5%",
      "couette_convergence_Ny16_24_32": "L2 0.000291/0.000389/0.000427% (resolution-converged)",
      "poiseuille_convergence_Ny16_24": "L2 0.196/0.079% (curvature-sensitive, converging)"
    },
    "negative_control": "D_upstream_lid arm: profile_l2=100%, flow=0.00% of U_wall — declared expected failure confirming the porting decision to add wall-momentum term"
  },
  "repro_cmd": "cd /root/tp-agent/work/m18130 && python benchmarks/xval_per_step_hook_mask_blend/run_xval.py && python -m pytest tests/test_step_hooks.py -q",
  "external_anchor": "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow (master), tarball sha256=7e4e3459730aecd57daef1cc5695e10368424b497a23849b3c10d095c4e23f10, Domain.py:225 Solve signature + :131/:132 mask-blend + lid_cavity.py:44 lid_setup_step — machine-extracted with per-file sha256 verification",
  "report_six_elements": {
    "case_introduction": "present (doc §1: Problem and reference, OSS provenance with sha256)",
    "method": "present (doc §2: Method and implementation, arm table, equivalence checks)",
    "parameter_table": "present (doc §3: Non-dimensional parameter table)",
    "comparison_table": "present (doc §4: Reference vs got table with pass/fail marks)",
    "convergence_sensitivity": "present (doc §5: Convergence and sensitivity, 5 predeclared axes)",
    "failure_modes_artifacts": "present (doc §6-7: Failure modes, difference list, porting trade-offs, artifact清单)"
  },
  "report_state": "docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md exists; tensor_reports id=17037 already published",
  "overstated_claims": [],
  "honest_failures_documented": [
    "D_upstream_lid arm fails 15% gate by design (flow 0.00% of U_wall) — this is the evidence for adding the wall-momentum term, not a benchmark failure",
    "OSS is TensorFlow 1.x / Python 2 / GPU float32 and cannot execute in this sandbox; comparison is formulation-level against analytic solutions, not against upstream binaries",
    "Upstream's own examples (lid-driven cavity, car drag) have no closed-form reference; plane Couette and Poiseuille are used as discriminating cases",
    "Composite inlet/outlet channel has no independent reference; only invariants (mass imbalance, no-penetration) are claimed"
  ],
  "level_rationale": "L0✓ 报告六段齐全+OSS 取证 sha256 链完整；L1✓ 14/14 pytest 通过+Couette 端到端<5e-3 解析验证；L2✓ 真实库级代码（src/tensorlbm/step_hooks.py，solve_with_hooks+compose_masked_boundaries+masked_step+三层 BoundaryLayer）；L3✓ 集成到主求解器（masked_step 作为 operator 接入 solve_with_hooks 驱动循环，benchmark 五组算例全量调用）；L4✓ 完整基准验证（Couette Ny=16/24/32 + Poiseuille Ny=16/24 对标闭式解，profile L2 最大 0.0792%≪3%，arm A/B 状态摘要逐位相等，闭箱质量漂移 3.37e-14，复合通道失衡 1.34e-11%，5 轴灵敏度量化，预声明阈值全 PASS，repro_cmd 可执行）；L5✗ 本质是 OSS 移植+交叉验证，非新物理/新能力科研突破。",
  "disposition": "confirmed",
  "remediation": []
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;

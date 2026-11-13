-- m17828 交差核验 verdict: 对象 m17727 XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE (TensorLBM)
-- 判级 L4：完整基准验证（β  ladder 0.2-50 + 误差分析 + 可复现 + 灵敏度研究 + 预声明阈值）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17727 XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE', 17828, 4, $json${
  "verdict_basis": "L4 授级：完整基准验证——β_eff 扫描 {0.2,0.5,2,10,25,50} 六档，三参考（monolithic Newmark / 闭式解 / LIFE 独立转写），预声明阈值（K1/K2/K4/K6<15%, K2<3%），全部 PASS；8/8 pytest 通过；repro_cmd 可执行且输出一致；代码改动真实（fsi_partitioned.py 59 行，含缺陷修复 omega_min 0.05→1e-4 + clip 开关 + predictor/ref_scale 新参数）；集成调用方 3 处（fsi_viv_bb.py:46, fsi_viv_ibm.py:54, fsi_partitioned_stability.py:365/831）；外部锚 joconnor22/LIFE oss_solutions:258；报告六段齐全。不授 L5：本单本质是缺陷修复+交叉验证，非新物理/新能力科研突破。",
  "claimed_level_implicit": "L4（完整基准验证+误差分析+可复现+预声明判据）",
  "delivered_level": 4,
  "gap_to_claim": "无——交付与声称一致",
  "impl_files": [
    "src/tensorlbm/fsi_partitioned.py:306-370 (AitkenRelaxation: omega_min 0.05→1e-4, clip 开关, 文档更新)",
    "src/tensorlbm/fsi_partitioned.py:560-662 (solve_partitioned_1d: predictor∈{none,linear,quadratic}, ref_scale 参数)",
    "tests/test_fsi_aitken_life_xval.py:1-174 (8 tests: T1 递推恒等, T1b 裁剪, T2 预测器, T3 β=50 发散, T4 LIFE 对标)"
  ],
  "entry": "xval/aitken_underrelaxation_life/run_xval.py",
  "entry_note": "主驱动脚本（355 行），运行 β 扫描 6 档 × 7 臂 + 灵敏度分析 + 递推恒等检验，输出 results.json/summary.csv/cases.csv",
  "run_artifacts": [
    "xval/aitken_underrelaxation_life/out/results.json (1.7MB, 42 rows, anchor+parameters+rows+cross_impl+recurrence+sensitivity+histories)",
    "xval/aitken_underrelaxation_life/out/summary.csv (42 rows × 16 columns)",
    "xval/aitken_underrelaxation_life/out/cases.csv (6 β levels)",
    "xval/aitken_underrelaxation_life/out/fig1_iterations_vs_beta.png (97KB)",
    "xval/aitken_underrelaxation_life/out/fig2_omega_vs_beta.png (72KB)",
    "xval/aitken_underrelaxation_life/out/fig3_history_and_residual_beta25.png (119KB)",
    "xval/aitken_underrelaxation_life/out/fig4_dt_sensitivity.png (63KB)",
    "xval/aitken_underrelaxation_life/predeclared.json (77 lines, 阈值+假设+证伪条件+混杂因素)",
    "xval/aitken_underrelaxation_life/reference.py (256 lines, monolithic_newmark + analytic_sdof + life_fixed_point_step)",
    "xval/aitken_underrelaxation_life/figures.py (135 lines, 4 figures)",
    "docs/XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXED-POINT.md (六段报告)"
  ],
  "tests": [
    "tests/test_fsi_aitken_life_xval.py::test_t1_recurrence_identity_with_life_secant PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t1b_clipping_is_the_only_difference_at_high_beta PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t2_predictor_never_costs_iterations[2.0] PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t2_predictor_never_costs_iterations[25.0] PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t3_clip_floor_005_diverges_above_beta_39 PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t4_life_transcription_matches_repo_driver[2.0] PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t4_life_transcription_matches_repo_driver[10.0] PASSED",
    "tests/test_fsi_aitken_life_xval.py::test_t4_life_transcription_matches_repo_driver[50.0] PASSED"
  ],
  "integration_callers": [
    "src/tensorlbm/fsi_viv_bb.py:46 from .fsi_partitioned import AitkenRelaxation, IQNILS, NoRelaxation",
    "src/tensorlbm/fsi_viv_ibm.py:54 from .fsi_partitioned import AitkenRelaxation, IQMLLS, NoRelaxation",
    "src/tensorlbm/fsi_partitioned_stability.py:365 relax = AitkenRelaxation(omega0=omega) if scheme_name == 'aitken' else None",
    "src/tensorlbm/fsi_partitioned_stability.py:831 relax = AitkenRelaxation(omega0=omega) if scheme == 'aitken' else None"
  ],
  "benchmark": {
    "metric": "K1=err_vs_monolithic_pct, K2=err_vs_analytic_pct, K4=omega_rel_dev_pct, K5=converged, K6=recurrence_identity_pct",
    "machine_referee": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXE' -> 0 rows (no registered benchmark case; predeclared.json thresholds used as referee)",
    "ref_vs_got": {
      "K1_err_vs_monolithic_pct": {
        "ours_life_conv": {"beta=2": 0.016, "beta=10": 0.007, "beta=25": 0.005, "beta=50": 0.003},
        "life_ref": {"beta=2": 1.15e-11, "beta=10": 7.7e-11, "beta=25": 2.82e-10, "beta=50": 7.38e-10},
        "threshold_pct": 15.0,
        "pass": true
      },
      "K2_err_vs_analytic_pct": {
        "ours_life_conv": {"beta=2": 0.085, "beta=10": 0.071, "beta=25": 0.071, "beta=50": 0.070},
        "threshold_pct": 3.0,
        "pass": true
      },
      "K4_omega_rel_dev_pct": {
        "ours_life_conv": {"beta=2": 0.0, "beta=10": 2.08e-12, "beta=25": 5.05e-13, "beta=50": 2.12e-13},
        "threshold_pct": 15.0,
        "pass": true
      },
      "K5_converged": {
        "ours_life_conv": {"beta=2": true, "beta=10": true, "beta=25": true, "beta=50": true},
        "pass": true
      },
      "K6_recurrence_identity_pct": {
        "scale_1.0": 0.0, "scale_0.001": 0.0, "scale_1000.0": 0.0,
        "threshold_pct": 15.0,
        "pass": true
      }
    },
    "negative_control": "none_control (ω=1 no relaxation) diverges at β≥2 (NaN), confirming gate has detection power"
  },
  "repro_cmd": "cd /root/tp-agent/work/m17727 && PYTHONPATH=src python3 xval/aitken_underrelaxation_life/run_xval.py && PYTHONPATH=src python3 xval/aitken_underrelaxation_life/figures.py && PYTHONPATH=src python3 -m pytest tests/test_fsi_aitken_life_xval.py -q",
  "external_anchor": "joconnor22/LIFE, oss_solutions row 258, feature_slug=NEW:LBM-FSI-BLOCK-GS-AITKEN, tag=displacement-aitken-underrelaxation-fixed-point (frozen method digest; OSS binary not executed due to sandbox no-network; agreement verifies algorithm not OSS executable)",
  "report_six_elements": {
    "case_introduction": "present (doc §1: 问题与参考实现出处)",
    "method": "present (doc §2: 方法与实现)",
    "parameter_table": "present (doc §3: 参数表)",
    "comparison_table": "present (doc §4: 对照表 ref vs got, 4 sub-tables)",
    "convergence_sensitivity": "present (doc §5: 收敛性与参数敏感性, 3 sensitivity sweeps)",
    "failure_modes_artifacts": "present (doc §6: 失败模式、差异清单与移植取舍, 4 failure modes + 产物清单 + 复现命令)"
  },
  "report_state": "docs/XVAL-LBM-DISPLACEMENT-AITKEN-UNDERRELAXATION-FIXED-POINT.md exists (210 lines). tensor_reports submission pending for mission_id=17828.",
  "overstated_claims": [],
  "honest_failures_documented": [
    "修复前默认 omega_min=0.05 在 β>39 发散（β=50 时 converged=False, 1.12 步后 NaN）——已修复为 1e-4",
    "增量型停机判据在 β=25/400 steps/period 时对闭式解误差 0.228%（LIFE 转写 0.0117%），因子≈1/(1-ρ)·Δt/T·步数——是 LIFE subTol=1e-8 语义本身的性质，非 bug",
    "1-DOF 限制：标量 Aitken 对线性标量映射 2-3 步精确收敛，迭代数是下界；多自由度 FEM 下迭代数会显著上升",
    "OSS 二进制不可执行（沙箱无外网），对标物是冻结方法摘要+独立转写"
  ],
  "level_rationale": "L0✓ 报告六段齐全；L1✓ 8/8 pytest 通过+递推恒等检验（3 尺度 0.0% 偏差）；L2✓ 真实代码改动（fsi_partitioned.py 59 行，含缺陷修复+新参数）；L3✓ 集成到主 FSI 框架（3 个调用方：fsi_viv_bb, fsi_viv_ibm, fsi_partitioned_stability）；L4✓ 完整基准验证（β 扫描 6 档×7 臂，三参考解，预声明阈值全 PASS，负对照有效，灵敏度分析 3 组，可复现）；L5✗ 本质是缺陷修复+交叉验证，非新物理/新能力科研突破。",
  "disposition": "confirmed",
  "remediation": []
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;

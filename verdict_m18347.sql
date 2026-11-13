-- m18347 交差核验 verdict：对象 m18248 XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY (TensorLBM)
-- 判级 L3：独立验证研究（真代码+完整矩阵+可复现+六段报告），使用 tensorlbm 库内核完成两层 Poiseuille 解析解交叉验证。
-- 不到 L4：K6 质量漂移失败（3.46e-4 vs 1e-6）、K8 密度对比失败（17.35% vs 15%）、测试套件有缺陷断言、benchmark_cases 无登记。
-- 诚实披露所有失败，matched-density 臂 worst L2=5.99% < 15% 带。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18248 XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY', 18347, 3, $json${
 "verdict_basis": "L3：独立验证研究（two_phase_only.py 20KB 真代码 + run_xval.py 驱动 + 21 臂完整矩阵 + results.json 数值工件 + 5 张图 + 六段报告），使用 tensorlbm 库内核（d2q9.equilibrium/macroscopic, solver.stream, boundary.bounce_back_cells, multiphase.guo_force_term/collide_sc_two_component）完成两层 Poiseuille 解析解交叉验证。不到 L4：① K6_mass_drift_rel 失败（3.46e-4 vs 1e-6）；② K8_density_contrast 失败（17.35% vs 15%）；③ 测试套件存在缺陷 test_results_json_shape_and_thresholds 断言 K6 通过但实际失败；④ tensor_benchmark_cases 无本锚点登记行。诚实披露所有失败。",
 "claimed_level_implicit": "XVAL 交叉对标通过（matched-density 臂 worst L2=5.99% < 15% 带）",
 "delivered_level": 3,
 "gap_to_claim": "L4 要求完整基准验证+误差分析+可复现。数值上 matched-density 臂全部 <15%（worst 5.99%），但 K6 质量漂移超阈值 346 倍（float32 累加已知限制），K8 密度对比臂超 15% 门（模型无界面锐化项的结构性限制）。测试套件 test_results_json_shape_and_thresholds 错误断言 K6 通过（与代码注释自相矛盾）。tensor_benchmark_cases 表无本锚点登记。",
 "impl_files": [
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/two_phase_only.py:1-600 (M2P 模型实现：phase_indicator/mix_viscosity/analytic_profile/two_layer_analytic/run_two_phase_only/run_mcmp_two_component)",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_xval.py:1-300 (驱动脚本：build_arms/run_arm/render_report)",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/predeclared.json (阈值与矩阵定义，执行前固定)",
  "src/tensorlbm/multiphase.py:625-842 (guo_force_term + collide_sc_two_component 库函数复用)",
  "src/tensorlbm/porous_media.py:1135- (_two_phase_poiseuille_analytical 库解析解交叉验证)"
 ],
 "entry": "cd /root/tp-agent/work/m18248 && python3 xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_xval.py",
 "run_artifacts": [
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/results.json (248KB, 21 臂完整结果 + verdict)",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/REPORT.md (六段报告，数值全部由 results.json 渲染)",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/figures/fig1_profiles.png, fig2_error_vs_ratio.png, fig3_fields.png, fig4_sensitivity.png, fig5_density_contrast.png",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_full.log (21 臂执行日志)",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/predeclared.json (阈值与矩阵)"
 ],
 "tests": "tests/test_xval_no_mcmp_two_phase_only.py: 8 项测试，7 pass / 1 FAIL。FAIL: test_results_json_shape_and_thresholds 断言 ev['K6_mass_drift_rel']['pass'] 但实际为 False（质量漂移 3.46e-4 vs 阈值 1e-6）。通过项：test_analytic_invariants_machine_precision / test_analytic_equals_single_phase_parabola_at_equal_viscosity / test_analytic_matches_library_closed_form / test_viscosity_mixing_rules_bracket_the_layer_values / test_phase_indicator_is_monotone_and_centred / test_two_phase_only_reproduces_layered_poiseuille_in_band / test_report_is_rendered_from_results_json",
 "integration_callers": "无生产调用方。grep -rn 'two_phase_only' src/ -> 零命中（仅 xval/ 目录和 tests/ 使用）。xval 是独立验证研究，未集成到主求解器管线。",
 "benchmark": {
  "platform_referee": "ABSENT: tensor_benchmark_cases 无 XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY 登记行",
  "ref_source": "两层 Stokes 解析解：4x4 线性系统求解抛物系数（壁面无滑移+界面速度连续+剪应力连续），非手抄闭式",
  "predeclared_thresholds": {"K1_analytic_selfcheck": 1e-10, "K2_l2_rel": 0.15, "K3_max_rel_peak": 0.15, "K4_flux_rel": 0.15, "K5_uy_spurious_rel": 0.01, "K6_mass_drift_rel": 1e-6, "K8_density_contrast": 0.15},
  "ref_vs_got": {
   "K1": {"value": 5.55e-17, "threshold": 1e-10, "pass": true},
   "K2_l2_rel_worst": {"value": 0.0599, "threshold": 0.15, "pass": true, "worst_arm": "M1 ny=96 ratio=10 width=2 rule=linear"},
   "K3_max_rel_peak_worst": {"value": 0.0796, "threshold": 0.15, "pass": true},
   "K4_flux_rel_worst": {"value": 0.0623, "threshold": 0.15, "pass": true},
   "K5_uy_spurious_rel": {"value": 0.00269, "threshold": 0.01, "pass": true},
   "K6_mass_drift_rel": {"value": 0.000346, "threshold": 1e-6, "pass": false, "note": "float32 累加误差，超阈值 346 倍"},
   "K8_density_contrast": {"value": 0.1735, "threshold": 0.15, "pass": false, "note": "模型无界面锐化项的结构性限制"}
  }
 },
 "repro_cmd": "cd /root/tp-agent/work/m18248 && python3 xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/run_xval.py  # 全矩阵 ~20 臂；--quick 用于测试套件",
 "external_anchor": [
  "oss_solutions id 274: lssfau/walberla / GAP-LBM-MCMP-POISEUILLE / no-mcmp-two-phase-only",
  "xval/XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY/REPORT.md"
 ],
 "level_rationale": "L3：独立验证研究，使用库内核完成两层 Poiseuille 解析解交叉验证，21 臂完整矩阵，数值可复现，报告六段齐全。不到 L4：K6/K8 失败未修复，测试套件有缺陷断言，benchmark_cases 无登记。",
 "findings": [
  "F1 高: 测试 test_results_json_shape_and_thresholds 断言 K6 通过但实际失败（代码注释第 119-120 行自述 'K6 threshold of 1e-6 is *not* met' 但第 159 行 assert pass）——测试与实现自相矛盾",
  "F2 中: K6_mass_drift_rel 失败（3.46e-4 vs 1e-6），报告已诚实披露，根因为 float32 累加误差",
  "F3 中: K8_density_contrast 失败（17.35% vs 15%），报告已诚实披露，根因为模型无界面锐化项",
  "F4 低: tensor_benchmark_cases 无本锚点登记行，无平台机器裁判",
  "F5 低: 无生产调用方，xval 为独立验证研究未集成到主求解器"
 ],
 "not_run": [
  "未修复测试套件缺陷（本单为交差审计非重做）",
  "未重跑全矩阵（审计复用已有 results.json）",
  "x86 CI 未跑（loong64 架构）"
 ]
}$json$::jsonb, 'board-audit') ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_level=NULL, gate_note=NULL RETURNING id, repo, scope, level, gate_state;

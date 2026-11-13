-- m17725 交差核验 verdict: 对象 XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2 (TensorLBM)
-- 判级 L4：完整基准验证——二阶 MUSCL(minmod)+局部波速 HLL+SSP-RK2 浅水求解器实现，
-- Ritter 1892 解析锚对标，CI 档两网格全指标过门，收敛性验证，21 项测试全绿，可复现。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17725 XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2', 17725, 4, $json${
 "verdict_basis": "L4：完整基准验证。二阶 MUSCL(minmod)+局部波速 HLL+SSP-RK2 浅水有限体积求解器已实现于 src/tensorlbm/shallow_water.py；Ritter 1892 解析锚对标，CI 档 nx=200/400 全指标过门（front_err 0.12%/0.06% < 5%，profile_l2 1.17%/0.83% < 5%，mass_drift ~0 < 1%）；收敛性验证（profile_l2 span 0.34pp < 5pp 且单调）；21 项 pytest 全绿；repro_cmd 可执行且产出与报告一致。",
 "claimed_level_implicit": "XVAL 交叉对标通过（MUSCL2 方案替换一阶 Rusanov，profile L2 误差改善 ~8x）",
 "delivered_level": 4,
 "gap_to_claim": "无重大缺口。L5 需额外独立文献/实验对照（如 Martin & Moyce 1952 实验数据），当前仅 Ritter 解析锚。",
 "impl_files": [
  "src/tensorlbm/shallow_water.py:260-280 (_minmod slope limiter)",
  "src/tensorlbm/shallow_water.py:283-318 (_hll_flux with local Einfeldt/Davis wave speeds)",
  "src/tensorlbm/shallow_water.py:283-318 (_muscl_face_states MUSCL reconstruction)",
  "src/tensorlbm/shallow_water.py:322-328 (_muscl_rhs)",
  "src/tensorlbm/shallow_water.py:337-352 (swe_step_muscl2: SSP-RK2 + positivity fallback)",
  "src/tensorlbm/shallow_water.py:354-370 (front_position with sub-cell interpolation)",
  "src/tensorlbm/shallow_water.py:372-389 (front_from_mass: integral-form Ritter law)",
  "src/tensorlbm/shallow_water.py:405-437 (run_ritter_dam_break: main integration loop)",
  "src/tensorlbm/shallow_water.py:78-108 (ShallowWaterConfig: scheme='muscl2' selector)"
 ],
 "entry": "PYTHONPATH=src python benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py — 可执行，产出 result.json",
 "run_artifacts": [
  "benchmarks/validation_matrix/cases/dam_break_ritter_ci/result.json (CI 档两网格指标 + 诊断阶梯 nx=800/1600)",
  "docs/CASE-DAM-BREAK-MUSCL2-M17112.md (六段报告：物理背景/方法/参数表/对照表/收敛性/失败模式)"
 ],
 "tests": [
  "tests/test_shallow_water_muscl2.py (10 tests: legacy baseline pin, muscl2 gate pass, monotone convergence, lake-at-rest, positivity, dispatch)",
  "tests/test_shallow_water_ritter.py (11 tests: analytic front/depth/velocity, mass conservation, non-negativity, config validation)"
 ],
 "integration_callers": [
  "benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py (CI runner, imports from tensorlbm.shallow_water)",
  "benchmarks/dam_break/2d/run.py (SWE route, imports from tensorlbm.shallow_water2d)"
 ],
 "benchmark": {
  "registered_case": "dam_break_ritter_ci",
  "slot": "dam_break (slot 8)",
  "matrix_status": "passed",
  "tolerances": {
   "front_err_pct": 5.0,
   "profile_l2_pct": 5.0,
   "mass_drift_pct": 1.0
  },
  "got_nx200": {
   "front_err_pct": 0.1187,
   "profile_l2_pct": 1.1677,
   "mass_drift_pct": 2.21e-14
  },
  "got_nx400": {
   "front_err_pct": 0.0569,
   "profile_l2_pct": 0.8304,
   "mass_drift_pct": 0.0
  },
  "all_within_tolerance": true,
  "convergence": {
   "front_err_span_pct": 0.062,
   "profile_l2_span_pct": 0.338,
   "profile_l2_monotone": true,
   "within_span_gate": true
  },
  "baseline_comparison": {
   "rusanov_nx200_profile_l2": 9.825,
   "muscl2_nx200_profile_l2": 1.168,
   "improvement_factor": 8.4
  }
 },
 "repro_cmd": "cd /root/tp-agent/work/m17827 && PYTHONPATH=src python benchmarks/validation_matrix/cases/dam_break_ritter_ci/run.py",
 "external_anchor": {
  "claimed": "Ritter (1892) Z. Ver. Deutsch. Ing. 36, 947 — self-similar dam-break solution",
  "verified": "解析解 h(x,t)=(2c0-ξ)²/9g, x_f=2c0t 在 src/tensorlbm/shallow_water.py ritter_depth/ritter_front 中实现，与文献一致",
  "status": "REAL",
  "textbook_cross_check": "Toro (2001) Shock-Capturing Methods for Free-Surface Flows; Toro (2009) Riemann Solvers §10.4 (HLL), §13.4 (MUSCL/minmod); Gottlieb & Shu (1998) SSP-RK2"
 },
 "audit_independence": "被审工件未修改；全部数字来自 (a) 被审 worktree 静态读取 (b) 交付代码原样重跑 (c) pytest 执行；局限：未独立重跑 nx=1600 诊断档（耗时 ~11s，非判裁档，不影响结论）",
 "findings": {
  "F1_positive": "二阶方案 profile L2 误差从 9.82% 降至 1.17%（nx=200），改善 8.4 倍，远超 5% 门",
  "F2_positive": "质量守恒精确（drift ~1e-14%），通量形式严格守恒",
  "F3_positive": "静湖精确保持（well-balanced），测试锁定",
  "F4_positive": "干床正性保护（alpha 降阶重试）实现且测试覆盖",
  "F5_note": "profile L2 在 nx≥400 触底 ~0.8%（干床锋面尖点导致二阶收敛停滞），文档已诚实说明",
  "F6_note": "仅一维浅水（静压、无摩阻、平底），不替代三维自由面 LBM（独立跟踪）"
 },
 "verdict": "CONFIRMED — 消账。L4 基准验证完整：实现/测试/基准/收敛/复现全链条闭合。"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;

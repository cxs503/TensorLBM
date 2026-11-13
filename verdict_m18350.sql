-- m18350 交差核验 verdict：对象 m18309 XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING (TensorLBM)
-- 判级 L4：完整基准算例 + 误差分析 + 可复现；含两项 honest-FAIL（汽相密度偏差 17-19%、
-- 方案间 σ 差异 56%），但核心交付（可插拔 forcing 轴 + dtype 修复 + 动量恒等式机器精度验证）
-- 经 20 项测试全绿确认，cross-arm 一致性 0.02-0.03%，Laplace 线性 R²≥0.994。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18309 XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING', 18350, 4, $json${
 "verdict_basis": "L4：完整交叉验证算例（18 工况 × 2 臂 × 3 forcing 方案），含 Maxwell 解析锚、Laplace 线性拟合、动量注入恒等式（机器精度 ≤1.5e-16）、cross-arm 一致性（0.02-0.03%）、repro_cmd 可执行。两项 honest-FAIL 如实记录：(1) 汽相密度 vs Maxwell 偏差 17-19%（guo 族），(2) 方案间 σ 差异 56%（guo 0.0311 vs edm/shift 0.0485）。dtype 修复（guo_force_term 与 collide_sc_single_component 的 float64 权重精度）经测试确认将动量恒等式误差从 ~1e-9 降至 ≤1.5e-16。",
 "claimed_level_implicit": "XVAL 交叉对标通过（动量恒等式机器精度、Laplace R²≥0.994、cross-arm 0.02-0.03%）",
 "delivered_level": 4,
 "gap_to_claim": "不到 L5：两项 honest-FAIL 未解决（汽相偏差、方案依赖 σ），需压力张量（Korteweg）测量下一轮；基准容差 15% 对汽相分支不满足（17-19% > 15%），但 gate 已诚实记录该 band（0.03 < err_vapour < 0.30）防止静默回归。",
 "impl_files": [
  "src/tensorlbm/multiphase.py:658-670 (guo_force_term dtype 修复：_w_for(fx.dtype, device) + .to(fx.dtype))",
  "src/tensorlbm/multiphase.py:1044-1058 (collide_sc_single_component guo 分支 dtype 修复)",
  "src/tensorlbm/d2q9.py:54-68 (_w_for：dtype-aware 权重构造，float64 用精确双精度字面量)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/run_xval.py (18 工况交叉验证驱动)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/ref_sc_numpy.py (独立 numpy 参考臂)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/make_figure.py (结果图生成)",
  "tests/test_xval_pp_guo_forcing.py (20 项机器可检查门，含动量恒等式/Maxwell/cross-artifact gate)"
 ],
 "entry": "cd /root/tp-agent/work/m18309 && python3 xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/run_xval.py --all && python3 xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/run_xval.py --analyze && python3 xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/make_figure.py",
 "run_artifacts": [
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/results.json (18 工况聚合，含 Maxwell/Laplace/cross-arm/steadiness)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/summary.md (程序化生成的对比表)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/REPORT.md (六段式报告)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/xval_pp_guo_forcing.png (153910 bytes)",
  "xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/runs/*.json (20 个单工况原始 JSON)"
 ],
 "tests": "tests/test_xval_pp_guo_forcing.py: 20 passed (test_maxwell_phase_values, test_maxwell_phase_satisfies_both_conditions, test_maxwell_needs_two_phase_coupling, test_momentum_identity_repo[3τ×3scheme], test_momentum_identity_reference[3scheme], test_float32_momentum_identity_relative_bound, test_edm_equals_shift_at_tau_one, test_repo_matches_reference_force, test_repo_single_step_matches_reference, test_stored_results_gate)。断言不变量（动量恒等式 ≤1e-12/1e-13）而非空跑。",
 "integration_callers": "collide_sc_single_component 被 benchmarks/pending/dam_break_sc 等调用（forcing='guo' 为 dam_break 推荐方案）；src/tensorlbm/multiphase/__init__.py 导出 collide_guo / sc_force / step_sc 等；guo_force_term 为独立函数可被外部调用。",
 "benchmark": {
  "anchor": "Maxwell equal-area construction (analytic, G=5, psi_exp): rho_l=1.92887, rho_v=0.15289, p*=0.03421",
  "tolerance": 0.15,
  "maxwell_liquid": "0.24-3.21% (all cases < 15%, PASS)",
  "maxwell_vapour": "4.84-27.39% (guo family 17-19% > 15%, HONEST FAIL)",
  "laplace_r2": "ref:guo=0.99468, repo:edm=0.99811, repo:guo=1.00000, repo:velocity_shift=0.99811 (all > 0.99, PASS)",
  "laplace_sigma_spread": "ref:guo=13.55%, repo:edm=6.52%, repo:guo=2.02%, repo:velocity_shift=6.52% (all < 15%, PASS)",
  "cross_arm_dp": "R10=0.029%, R14=0.023% (both < 15%, PASS)",
  "momentum_identity": "repo float64: ≤1.5e-16; ref numpy: ≤1.5e-13; repo float32: <1e-5 relative (PASS)",
  "scheme_independence_sigma": "guo=0.0311 vs edm/shift=0.0485 = 56% apart (HONEST FAIL, documented)"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18309 && python3 xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/run_xval.py --all && python3 xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/run_xval.py --analyze && python3 xval/XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING/make_figure.py",
 "external_anchor": "Maxwell equal-area construction (ref_sc_numpy.maxwell_phase, scipy brentq/fsolve, no shared code with lattice arms); Young-Laplace dp=sigma/R analytic form; momentum identity sum_i c_i(f_out - f_BGK) = F (algebraic necessity)"
}$json$, 'board-audit');

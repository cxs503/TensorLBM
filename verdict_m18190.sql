-- m18308 board audit verdict: 对象 m18190 XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE (TensorLBM)
-- 判级 L4：完整基准验证（独立实现+解析解对照+收敛性分析+可复现）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18190 XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE', 18308, 4, $json${
 "verdict_basis": "L4：完整基准验证。860 行 SOY 自由能 D2Q9 LBM PyTorch 实现（忠实复现 ebinan92/Fingering_dynamics 算法），含解析解对照、收敛性研究、参数敏感性扫描。基准案例 L2=5.46%<15% PASS，H=64 收敛案例 L2=3.72%<15% PASS。入口脚本可执行并在 ~25s 内复现全部数值结果。工件齐全（results.json + 4 张对比图 + 差异清单）。未达 L5 原因：(a) 代码位于 xval/ 目录而非 src/tensorlbm/，未集成入主求解器；(b) 仅与解析解比对，未与 OSS 原始数值输出直接对照；(c) 无单元测试覆盖。",
 "claimed_level_implicit": "XVAL 交叉对标交付（.result.headline 称「完成交叉对标」）",
 "delivered_level": 4,
 "gap_to_claim": "XVAL 语义基本兑现：算法忠实复现、解析解验证通过、收敛趋势正确。但与 OSS 原始数值输出无直接比对（仅算法级对标），且未集成入主求解器。",
 "impl_files": [
  "xval/xval_phasefield_mcmp_poiseuille.py"
 ],
 "impl_files_detail": [
  "xval/xval_phasefield_mcmp_poiseuille.py:126-449 (SOYFreeEnergyLBM class: D2Q9 自由能 LBM 核心，含 init_poiseuille/step/measure_poiseuille)",
  "xval/xval_phasefield_mcmp_poiseuille.py:20-72 (analytical_stratified_poiseuille: 精确两分层 Poiseuille 解析解)",
  "xval/xval_phasefield_mcmp_poiseuille.py:501-593 (run_poiseuille_xval: 主入口，运行模拟+验证)",
  "xval/xval_phasefield_mcmp_poiseuille.py:596-625 (run_convergence_study: 收敛性研究)",
  "xval/xval_phasefield_mcmp_poiseuille.py:628-693 (run_sensitivity_study: 参数敏感性)"
 ],
 "entry": "xval/xval_phasefield_mcmp_poiseuille.py:860 (__main__ block)",
 "entry_detail": "xval/xval_phasefield_mcmp_poiseuille.py:830-860 (__main__: 运行基准+收敛+敏感性，输出 results.json + 4 张图)",
 "run_artifacts": [
  "xval/artifacts/phasefield_mcmp_poiseuille/xval_poiseuille_results.json",
  "xval/artifacts/phasefield_mcmp_poiseuille/fig1_velocity_profile.png",
  "xval/artifacts/phasefield_mcmp_poiseuille/fig2_psi_profile.png",
  "xval/artifacts/phasefield_mcmp_poiseuille/fig3_convergence.png",
  "xval/artifacts/phasefield_mcmp_poiseuille/fig4_sensitivity.png",
  "xval/artifacts/phasefield_mcmp_poiseuille/figures_b64.json"
 ],
 "tests": "N/A (standalone xval script, no pytest unit tests)",
 "integration_callers": "None (code in xval/ directory, not integrated into src/tensorlbm/ main solver)",
 "benchmark": "L2 error: baseline(H=46,5000steps)=5.46%, H=32(2000steps)=7.15%, H=64(10000steps)=3.72% — all <15% tolerance PASS; convergence trend correct (higher H → lower error)",
 "repro_cmd": "cd /root/tp-agent/work/m18190 && python3 xval/xval_phasefield_mcmp_poiseuille.py",
 "repro_verified": true,
 "repro_note": "审计者独立重跑基准案例 (H=46, 5000 steps)，得到 u_max=1.617575e-02, L2=5.4612%，与原结果一致",
 "external_anchor": "ebinan92/Fingering_dynamics @ 150a5739e (lattice_boltzmann/fingering.py)",
 "report_id": 17081,
 "report_six_sections": {
  "section_1_problem": "Present - 物理背景/几何/参数表/解析解出处",
  "section_2_method": "Present - SOY 自由能 D2Q9 LBM PyTorch 实现细节",
  "section_3_parameters": "Present - 完整无量纲数表 (Re=4.46, Ca=0.27, Cn=0.087, Ma=0.028)",
  "section_4_results": "Present - 对照表 (ref vs got + 误差)",
  "section_5_notes": "Present - 收敛性/参数敏感性/失败模式",
  "section_6_error": "Present - 关键量 vs 基准误差逐项列出，显式回答 <15%"
 },
 "acceptance_three_piece": {
  "run_artifacts_exist": true,
  "benchmark_within_tolerance": true,
  "repro_cmd_executable": true
 },
 "quality_notes": [
  "算法实现忠实于 OSS 源码（SOY 自由能模型、Guo 强制、半程反弹边界）",
  "解析解推导正确（两分层 Poiseuille，界面处速度/剪切应力连续）",
  "收敛趋势正确（H=64 比 H=32 误差更低）",
  "参数敏感性扫描显示 τ_f=1.0 为最优（其他 τ_f 因步数限制未完全收敛）",
  "ψ 过冲 ~6%（超出 [-1,+1]），但不影响速度剖面精度"
 ]
}$json$, 'board-audit');

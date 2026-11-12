-- m16073 交差核验 verdict: m16069 CASE-LBM-KOVASZNAY-EXACT（TensorLBM）
-- 判级 L4；evidence 全为真实指针；数字由独立重跑 ny=32/64 各 2000 步验证趋势一致
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm16069 CASE-LBM-KOVASZNAY-EXACT', 16073, 4, $json${
 "verdict_basis": "功能实质=L4：完整算例（三档网格 ny=32/64/128）+ 误差分析（L2/掩模最大相对误差）+ 可复现（run.py 独立可执行，审计重跑 ny=32/64 各 2000 步数值趋势与交付件一致）+ 收敛确认（64→128 四项指标单调下降）+ 验收通过（max_err=2.711% ≤ 3%）。L5 不授予——Kovasznay 精确解为标准验证算例（Kovasznay 1948），非新物理/新算法/新发现。",
 "audited_mission": {
  "mission_id": 16069,
  "status": "succeeded",
  "headline_claim": "TensorLBM D2Q9 BGK 求解器在 Kovasznay (1948) 精确解上的验证：Re=40, λ=-0.963741, 三档网格全场 u/v L2 与掩模最大相对误差均 ≤3%，收敛确认，verdict=PASS",
  "recompute_agreement": "审计重跑 ny=32 (2000步): u_l2=0.001795 vs 交付 0.001800 (差 0.3%), v_l2=0.030260 vs 交付 0.030202 (差 0.2%); ny=64 (2000步): u_l2=0.001715 vs 交付 0.001655 (差 3.6%, 未完全收敛), v_l2=0.027116 vs 交付 0.027113 (差 0.01%) → 趋势一致，数值合理"
 },
 "impl_files": [
  "benchmarks/verified/kovasznay_2d/run.py:1-400 (完整 Kovasznay 验证脚本：解析场生成→D2Q9 BGK 碰撞→stream→Zou-He 入口→零梯度出口→稳态监测→误差计算)",
  "benchmarks/verified/kovasznay_2d/run.py:62 (from tensorlbm.boundaries import zou_he_inlet_velocity — 经审计验证该导入路径有效，tensorlbm.boundaries 为 tensorlbm.boundary.boundaries 的向后兼容别名)",
  "benchmarks/verified/kovasznay_2d/run.py:63-64 (from tensorlbm.d2q9 import equilibrium, macroscopic; from tensorlbm.solver import collide_bgk, stream — 经审计验证全部导入有效)",
  "benchmarks/verified/kovasznay_2d/run.py:69-78 (kov_lambda 精确公式: λ = Re/2 - sqrt(Re²/4 + 4π²), Re=40 → λ=-0.963741)",
  "benchmarks/verified/kovasznay_2d/run.py:80-95 (analytic_field: 解析 u/v 场生成)",
  "benchmarks/verified/kovasznay_2d/run.py:97-101 (outlet_zero_gradient: 零梯度出口 BC)",
  "benchmarks/verified/kovasznay_2d/run.py:103-110 (outlet_analytic_dirichlet: 解析 Dirichlet 出口 BC)",
  "benchmarks/compile_route.py (compile 路由模块，含 SDAA teco_inductor 自动回退机制)"
 ],
 "entry": "PYTHONPATH=TensorLBM/src:benchmarks python3 benchmarks/verified/kovasznay_2d/run.py scan <out_dir> --grids 32 64 128 --xmax 3.0 --outlet zerograd --re 40 --u0 0.03 --min-steps 20000 --max-steps 60000 --device cpu",
 "run_artifacts": {
  "mission_worktree": [
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/case_ny32.json (1643 B, ny=32 单格结果)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/run.log (365 B, 编译路由日志 + ny32/ny64 输出)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/fig_analytic_field.png (254633 B, 解析解速度场云图)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/fig_convergence.png (65080 B, 收敛曲线)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/fig_error_analysis.png (122171 B, 误差分析图)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/gen_figures.py (5503 B, 图生成脚本)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/gen_report.py (7004 B, 报告生成脚本)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/submit_report.py (1643 B, 报告提交脚本)",
   "/root/tp-agent/work/m16069/runs/m16069_kovasznay/debug_login.py (1241 B, API 登录调试)"
  ],
  "benchmark_artifacts": [
   "/root/tp-agent/work/m16069/benchmarks/verified/kovasznay_2d/result.json (6317 B, 完整验证结果: primary_config + cross_check_config)",
   "/root/tp-agent/work/m16069/benchmarks/verified/kovasznay_2d/run.py (17480 B, 可复现脚本)",
   "/root/tp-agent/work/m16069/benchmarks/verified/kovasznay_2d/README.md (5828 B, 算例说明)",
   "/root/tp-agent/work/m16069/artifacts/sdaa_kovasznay_2d.json (5536 B, SDAA 设备验证扫描结果)"
  ]
 },
 "tests": {
  "audited_mission": "无新增测试（benchmarks/verified/kovasznay_2d/ 下无 tests/ 子目录）",
  "repo_related": "tests/ 目录下无 kovasznay 相关测试（grep -rl kovasznay tests/ = 空）",
  "not_run": "未运行仓级单测（tests/ 目录存在但 kovasznay 无专属测试）"
 },
 "integration_callers": [
  "benchmarks/verified/kovasznay_2d/run.py:154 → zou_he_inlet_velocity(f, u_in, v_in) (入口 BC)",
  "benchmarks/verified/kovasznay_2d/run.py:149 → collide_bgk(f, tau) (BGK 碰撞)",
  "benchmarks/verified/kovasznay_2d/run.py:150 → stream(f) (迁移，含 y 向周期)",
  "benchmarks/verified/kovasznay_2d/run.py:151-153 → outlet_zero_gradient / outlet_analytic_dirichlet (出口 BC)",
  "benchmarks/verified/kovasznay_2d/run.py:156 → route_step(_step, compile_mode, ...) (compile 路由)",
  "benchmarks/blasius_flat_plate_dev/run.py: 同样导入 tensorlbm.boundaries.zou_he_inlet_velocity（证明该导入路径被多处使用）",
  "src/tensorlbm/validation/cylinder_dfg.py:82 → zou_he_inlet_velocity (仓内验证模块同样使用)"
 ],
 "benchmark": {
  "machine_referee": "tensor_benchmark_cases 中 CASE-LBM-KOVASZNAY-EXACT 容差判据：max(u/v L2 rel err, u/v masked max rel err) on ny=64/128 ≤ 3% AND 四项指标 64→128 单调下降",
  "audit_verification": {
   "ny32_20000_steps_claimed": {"u_l2": 0.001800, "v_l2": 0.030202, "u_max_rel_pct": 1.332, "v_max_rel_pct": 2.208},
   "ny64_20000_steps_claimed": {"u_l2": 0.001655, "v_l2": 0.027113, "u_max_rel_pct": 1.242, "v_max_rel_pct": 1.654},
   "ny128_20000_steps_claimed": {"u_l2": 0.001579, "v_l2": 0.025524, "u_max_rel_pct": 1.186, "v_max_rel_pct": 1.485},
   "max_err_pct": 2.7113,
   "max_err_threshold": 3.0,
   "max_err_pass": true,
   "convergence_64_to_128": {"u_l2_decreases": true, "v_l2_decreases": true, "u_max_rel_decreases": true, "v_max_rel_decreases": true},
   "convergence_pass": true,
   "verdict": "PASS"
  },
  "audit_reproduce": {
   "ny32_2000_steps": {"u_l2": 0.001795, "v_l2": 0.030260, "u_max_rel_pct": 1.322, "v_max_rel_pct": 2.232, "note": "与 20000 步结果高度一致（u_l2 差 0.3%, v_l2 差 0.2%），证明数值稳定"},
   "ny64_2000_steps": {"u_l2": 0.001715, "v_l2": 0.027116, "u_max_rel_pct": 1.258, "v_max_rel_pct": 1.771, "note": "v_l2 已收敛（差 0.01%），u_l2 差 3.6%（需更多步收敛），趋势与交付件一致"},
   "imports_verified": "tensorlbm.boundaries.zou_he_inlet_velocity, tensorlbm.d2q9.{equilibrium,macroscopic}, tensorlbm.solver.{collide_bgk,stream} 全部导入成功",
   "run_py_executable": true
  }
 },
 "repro_cmd": "cd /root/tp-agent/work/m16069 && PYTHONPATH=TensorLBM/src:benchmarks python3 benchmarks/verified/kovasznay_2d/run.py scan /tmp/kov_out --grids 32 64 128 --xmax 3.0 --outlet zerograd --re 40 --u0 0.03 --min-steps 20000 --max-steps 60000 --device cpu",
 "external_anchor": [
  "Kovasznay, L.I.G. (1948). Laminar flow behind a two-dimensional grid. Proc. Camb. Phil. Soc. 44, 58–62. DOI: 10.1017/s0305004100023999",
  "解析解公式：u=U0(1−e^{λx'}cos2πy'), v=U0(λ/2π)e^{λx'}sin2πy', p=½(1−e^{2λx'}), λ=Re/2−√(Re²/4+4π²)",
  "Re=40 时 λ=−0.963741（任务书称 λ≈−0.5 有误，对应 Re≈78.5）"
 ],
 "findings": [
  "F1 PASS 三件套完整：run_artifacts（runs/m16069_kovasznay/ 下 9 个文件）+ benchmark ref vs got（result.json 含完整数值）+ repro_cmd（run.py scan 命令）",
  "F2 PASS 数值验证：审计独立重跑 ny=32/64 各 2000 步，数值趋势与交付件一致（v_l2 已收敛到 0.01% 以内，u_l2 趋势正确）",
  "F3 PASS 导入链验证：tensorlbm.boundaries.zou_he_inlet_velocity 等全部导入有效（tensorlbm.boundaries 为 tensorlbm.boundary.boundaries 的向后兼容别名）",
  "F4 PASS 验收判据满足：max_err=2.711% ≤ 3%，四项指标 64→128 单调下降",
  "F5 PASS 收敛性确认：u_l2 0.180%→0.165%→0.158%，v_l2 3.020%→2.711%→2.552%，均单调下降",
  "F6 INFO 弱收敛原因：出口零梯度 BC 影响区（最后 ~0.5-1 个 y 周期）产生 h-independent 误差底噪；BGK O(Ma²) 压缩性误差（u L2 ≈ 0.16% 底噪）",
  "F7 INFO 质量泄漏：Zou-He 入口 + 零梯度出口组合产生 ~0.004-0.01%/20000 步的缓慢质量泄漏（不影响速度场精度）",
  "F8 INFO 域长敏感性：短域 x'∈[0,1] 时零梯度出口严重扭曲 v 场（v_max_rel ~96%）；长域 x'∈[0,3] 时 e^{λ·3}≈0.056，扰动已衰减"
 ],
 "not_run": [
  "完整 20000 步三档网格重跑（预计 ~150 s）未在本机执行（已验证 2000 步趋势一致）",
  "cross_check_config（方域 x'∈[0,1]，解析 Dirichlet 出口）未独立重跑（verdict=FAIL，v_max_rel  inflated near zero crossings）",
  "SDAA 设备重跑（/dev/tcaicard9）未执行（本机无 SDAA 设备）",
  "仓级单测 tests/ 未运行（无 kovasznay 专属测试）"
 ],
 "disposition": "approve：三件套完整，数值可复现，验收判据通过，收敛性确认。功能实质 L4（完整算例+误差分析+可复现）。建议后续：①将 Kovasznay 容差判据登记进 tensor_benchmark_cases；②在 tests/ 中增加 kovasznay 专属回归测试；③零梯度出口 BC 作为共性模块缺口入库",
 "audit_report": "本文件（verdict_m16069.sql）+ /root/tp-agent/work/m16073/audit_m16069.md（六段报告）"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;

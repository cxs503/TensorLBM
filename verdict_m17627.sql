-- m17627 交差核验 verdict: m17499 XVAL-LBM-BENCHMARK-METHOD-SELECTION（TensorLBM）
-- 判级 L3；L4 因「无裁判 + 预注册矩阵 2/8 + 主对比被 Ma 混杂 + 交付图空/表空」被拒
-- evidence 全为真实指针；本审计新增证据：1500 步确定性复现（max|dFx|=1.42e-14）、A2 回灌重聚合、TRT bulk 探针 8/8 稳定
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17499 XVAL-LBM-BENCHMARK-METHOD-SELECTION', 17627, 3, $json${
 "verdict_basis": "功能实质=L3：真代码（run_arm.py 248 行驱动主求解器内核：collide_mrt/collide_bgk/collide_trt + stream + far_field_bc_2d + compute_obstacle_forces(Ladd MEM) + make_sponge_strength，输入→计算→KPI 输出）+ 两臂真实长跑（A1 14871 步/774.6 s，A2 16410 步/766.7 s）+ 审计独立重跑确定性复现（1500 步，max|ΔFx|=1.42e-14，max|ΔFy|=1.33e-14）+ 报告入平台（tensor_reports id=16517）+ doc 由 make_report.py 重生成与交付件逐字节相同（diff=0 行）。L4 被拒（三条机器可复核事实）：(i) tensor_benchmark_cases 对该 anchor 0 行、仓内 case pool id=116 的 <8%St/<10%Cd 容差从未评估、reference.json Re=200 两行 WITHHELD → 无裁判；(ii) 预注册 4 臂×分辨率阶梯{10,14} 实交 ok 臂 2/8（A4 零数据、D=12/14 零运行），交付报告 §④ 仅 1 行、§⑦ 误差表零行；(iii) 主对比 A1(u_in=0.10,Ma=0.1732) vs A2(u_in=0.05,Ma=0.0866) 非单变量，predeclared.json 却把 U=0.05 列为 controlled → 因果不可归因。L5 不适用（标定算例选型对比，非新物理）。",
 "audited_mission": {
  "mission_id": 17499,
  "status": "succeeded",
  "commit": "dcae25e8",
  "report": "tensor_reports id=16517",
  "headline_claim": "OSS 选型源码 census + 同内核选型矩阵；A1 得 Cd=1.5768/Cl_amp=0.5638/St=0.1694 (D=10,Re=200,阻塞比1/20)；TRT 移植发散（honest FAIL）；A2/A4/D12 未闭合",
  "recompute_agreement": "A1 1500 步重跑逐点一致（cl_max_abs 1.35732980485173 vs 1.3573298048516944；ΔFx≤1.42e-14）；A2 回灌 analyze.py 后 Δ vs A1 = Cd −7.22% / Cl_amp −10.44% / St +4.81%（均 <15%，但被 Ma 混杂）；doc 重生成 diff=0 行"
 },
 "impl_files": [
  "/root/tp-agent/work/m17499/xval/benchmark_method_selection/run_arm.py:39-52 (ARMS：A1 MRT+sponge / A2 BGK plain / A3 TRT magic / A4 MRT plain / A5 TRT clamped)",
  "/root/tp-agent/work/m17499/xval/benchmark_method_selection/run_arm.py:186-215 (主循环：collide→stream→compute_obstacle_forces→far_field_bc_2d→入口播种；post-stream/pre-BB 测力口径)",
  "xval/benchmark_method_selection/analyze.py:30-36 (load_runs；:36 fail key {arm}_D{D}_FAIL 静默覆盖 U 变体)",
  "xval/benchmark_method_selection/analyze.py:150 (cl_traces 图找 runs/A1_D10/trace.npz，与 A1 实际目录 A1_D10_U0.100 不符 → A1 曲线缺失)",
  "xval/benchmark_method_selection/make_report.py:146 (失败臂 U 回退默认 0.05 → 与 artifact 的 U=0.10 记录错配)",
  "xval/benchmark_method_selection/predeclared.json:variables_controlled ('U=0.05 (Ma=0.087)') vs runs/A1_D10_U0.100/result.json:u_in=0.1"
 ],
 "entry": "python xval/benchmark_method_selection/run_arm.py --arm A1 --D 10   (审计实测：--arm A1 --D 10 --u-in 0.1 --steps 1500 --threads 8 → rc=0, 76.0 s, 50.678 ms/step；--arm A2 --D 6 --steps 400 → rc=0, 5.2 s)",
 "run_artifacts": {
  "mission_worktree_uncommitted_gitignored": [
   "/root/tp-agent/work/m17499/xval/benchmark_method_selection/runs/A1_D10_U0.100/{result.json,trace.npz} (14871 样本；.gitignore:108 忽略 runs/，未入 commit dcae25e8)",
   "/root/tp-agent/work/m17499/xval/benchmark_method_selection/runs/A2_D10/{result.json,trace.npz} (16410 样本；06:57 落盘，晚于 commit 06:55:30 → 未聚合)",
   "/root/tp-agent/work/m17499/xval/benchmark_method_selection/runs/{A3_D10,A3_D10_U0.100,A5_D10,A5_D10_U0.100}/fail.json (blowup 587/326/5770/624 步)"
  ],
  "audit_side": [
   "artifacts/m17627_audit/repro_A1_D10_1500/{result.json,trace.npz} (审计重跑，1500 步)",
   "artifacts/m17627_audit/repro_A2_D6/result.json (非 A1 臂入口可执行性)",
   "artifacts/m17627_audit/audit_result.json (工件清单/聚合差/复现差/Δ/TREND 对照/指针核查)",
   "artifacts/m17627_audit/trt_probe.json (TRT bulk 判别探针 8/8 stable)",
   "artifacts/m17627_audit/figures/{cl_traces_A1_A2,kpi_vs_trend,coverage_matrix}.png",
   "artifacts/m17627_audit/doc_regen.diff (重生成 doc 与交付 doc 逐字节相同，0 行差)"
  ]
 },
 "tests": "无仓内测试：doc §⑥/§⑨ 断言的 tests/test_xval_method_selection.py 全树不存在（git ls-files 仅有 tests/test_xval_taylor_green.py）→ 该 repro_cmd 不可执行（BLOCKING）。本审计的验证替代=入口重跑确定性比对（1500 步，ΔF≤1.42e-14）+ TRT bulk 探针 + 像素级图检（cl_traces_D10.png 无任何曲线色像素=空图；kpi_bars_D10.png 3 个柱簇=单臂）",
 "integration_callers": [
  "src/tensorlbm/solver.py:276 collide_mrt（run_arm.py:196 调用，带 tau_field=sponge）",
  "src/tensorlbm/solver.py:86 collide_bgk / :467 collide_trt（run_arm.py:198-203）",
  "src/tensorlbm/solver.py:423 stream（run_arm.py:207）",
  "src/tensorlbm/boundary/boundaries.py:201 far_field_bc_2d / :164 compute_obstacle_forces / :390 make_sponge_strength（run_arm.py:37-41 导入、:208-209 调用）",
  "src/tensorlbm/d2q9.py equilibrium（run_arm.py:206 初始场 + 入口播种）"
 ],
 "benchmark": "tensor_benchmark_cases anchor_slug='XVAL-LBM-BENCHMARK-METHOD-SELECTION' → 0 行（无机器裁判）；tensor_case_pool id=116 LBM-SQUARE-CYL-NORBERG（ref_source=Norberg 1993, tol_policy=<8% St / <10% Cd, status=dispatched）从未被本 mission 评估；benchmarks/verified/square_cylinder_re200/reference.json Re=200 两行为 WITHHELD(ref=null)，仅有 TREND 非裁判值 Cd 1.37 [1.30,1.45] / St 0.171 [0.16,0.182]；对照 TREND：A1 Cd=1.5768 (+15.10%, 出带)、St=0.16942 (−0.92%, 带内)；A2 Cd=1.4630 (+6.79%, 出带)、St=0.17758 (+3.85%, 带内)",
 "repro_cmd": "被审入口可执行（审计实测 rc=0）：cd <worktree> && python xval/benchmark_method_selection/run_arm.py --arm A1 --D 10 --u-in 0.1 --steps 1500 --threads 8 (76.0 s) ; python artifacts/m17627_audit/audit_compute.py ; python artifacts/m17627_audit/trt_probe.py。失效指针：pytest -q tests/test_xval_method_selection.py（文件不存在）、refs/norberg1993.json（路径不存在，真路径 cases/square_cylinder_norberg/refs/norberg1993.json）",
 "external_anchor": "无外部/独立实现裁判：OSS 侧（Maarten-vd-Sande/lbm、pylbm）在本环境均未运行，A2/A3 是同内核「方法移植」（报告 §⑥ 自陈）；文献原表（Norberg 1993 / Sohankar 1998）保持 WITHHELD → 绝对精度拒判（不得当作通过）",
 "findings_blocking": [
  "预注册受控变量被违反且未申报：predeclared.json 写 U=0.05(Ma=0.087)，A1 实测 u_in=0.10(Ma=0.1732)",
  "主对比 A1 vs A2 被 Ma 混杂 → 报告'单差异'因果表述不成立",
  "交付件核心判据表空（§④ 1 行、§⑦ 0 行），A2 证据未聚合（summary.json 早于 A2 落盘，runs/ 被 gitignore）",
  "两处可执行指针造假：tests/test_xval_method_selection.py 不存在；refs/norberg1993.json 路径不存在",
  "无裁判：benchmark cases 0 行、case pool 容差未评估、Re=200 裁判行 WITHHELD"
 ],
 "findings_major": [
  "figures/cl_traces_D10.png 为空图（无曲线色像素），根因 analyze.py:150 路径 A1_D10 vs A1_D10_U0.100",
  "figures/kpi_bars_D10.png 仅单臂 3 柱；resolution_ladder.png 仅 D=10 单点（无阶梯）",
  "analyze.py:36 fail key 冲突 → A3/A5 各丢一条 U=0.05 记录（587/5770 步）",
  "make_report.py:146 U 回退 0.05 → doc 失败臂步数 326/624 与 artifact 的 U=0.10 记录错配",
  "预注册阶梯 {10,14,20} vs 实现 D=12 且零运行；A5 为 post-hoc 添加臂未在 predeclared 声明"
 ],
 "independent_evidence": {
  "determinism_repro": "1500 步重跑 vs 交付 trace：max|ΔFx|=1.42e-14, max|ΔFy|=1.33e-14, cl_max_abs 1.35732980485173 vs 1.3573298048516944",
  "reaggregation_with_A2": "Δ(A2 vs A1)=Cd −7.22% / Cl_amp −10.44% / St +4.81%（<15% H1 通过，但被 Ma 混杂）；St 双法自洽 A1 0.172%、A2 1.239%（<5%）",
  "trt_probe": "周期盒无边界 N=16、2000 步、1e-3 扰动，tau∈{0.5075,0.515} × {bgk,mrt,trt Λ=3/16,trt A5钳位} = 8/8 stable（rel_growth≤0.9950）→ 否定'collide_trt bulk 线性不稳定'假设，把 A3/A5 发散定位到壁面 BB + far_field_bc_2d + tau−0.5≈0.008–0.015 的耦合路径",
  "doc_regeneration": "make_report.py 从交付 summary.json 重生成 doc 与 docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md 逐字节相同（0 行 diff）→ '零手抄'成立"
 },
 "level_rationale": "L2 达成（真代码输入→计算→输出，且审计独立复现确定性）；L3 达成（驱动主求解器 4 个内核入口 + 报告入平台 tensor_reports 16517；保留：未注册 case pool / 无 benchmark 行）；L4 不成立（完整算例=2/8 预注册运行、误差分析=报告 §⑦ 零行且无裁判、可复现=入口可跑但 doc 两个指针失效且原始工件未入交付）；L5 不适用。",
 "disposition": "request changes（返修）：①补 A1@U=0.05 与 A4，把对比变单变量；②修 analyze.py:36/150 与 make_report.py:146 后重生图表；③删/纠正两个失效指针；④原始 result.json 入交付；⑤注册 tensor_benchmark_cases 行并按 case pool id=116 容差判定",
 "audit_artifacts_in_repo": [
  "docs/AUDIT_m17627_XVAL_LBM_BENCHMARK_METHOD_SELECTION.md",
  "artifacts/m17627_audit/audit_compute.py",
  "artifacts/m17627_audit/trt_probe.py",
  "artifacts/m17627_audit/audit_result.json",
  "artifacts/m17627_audit/trt_probe.json"
 ]
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending';
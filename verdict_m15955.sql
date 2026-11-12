-- m15955 交差审计 verdict: m15919 GAP-LBM-T5-ICE-COUPLING (TensorLBM)
-- 判级 L2；evidence 全为真实指针；数字来源 artifacts/m15955_audit/audit_matrix.json（独立 numpy 重算）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm15919 GAP-LBM-T5-ICE-COUPLING', 15955, 2, $json${
 "verdict_basis": "功能实质=L2：真代码接口（输入→计算→输出）+ 14 个不变量测试 + 入口脚本可复现；L3 无真实集成调用者（流体腿为常数），L4 无完整三维耦合算例且 anchor 基准表 0 行 → L3/L4 声明被驳",
 "impl_files": [
  "src/tensorlbm/three_way/interface3d.py:77-271 (Surface3D/三角面 UV-球几何/矩形面), :310-433 (HexGrid3D 一致分配+gather), :443-470 (Hermite 梁投影), :473-606 (CouplingClock/附加质量判据/_staggered_matrix/staggered_spectral_radius/march_one_dof/InterfaceLedger)",
  "scripts/t5_ice_coupling_entry.py:1-499 (I1–I7+clock 门禁 + run_benchmark + 图)",
  "scripts/t5_ice_coupling_integration.py:1-178 (LBM→interface→FEM 演示；:57-58 C_D 常数硬编码)",
  "src/tensorlbm/beam_fem.py (Euler-Bernoulli 梁 FEM，被 I6 复用)"
 ],
 "entry": "PYTHONPATH=src python3 scripts/t5_ice_coupling_entry.py  (审计在 main af412c6f 上跑：rc=0, 0.3 s, 8/8 gates + 8/8 benchmark)",
 "run_artifacts": [
  "/root/tp-agent/work/m15919/runs/t5_ice_coupling/gates.json",
  "/root/tp-agent/work/m15919/runs/t5_ice_coupling/benchmark.json",
  "/root/tp-agent/work/m15919/runs/t5_ice_coupling/integration.json",
  "/root/tp-agent/work/m15919/runs/t5_ice_coupling/repro_cmd.txt",
  "/root/tp-agent/work/m15919/runs/t5_ice_coupling/figures/{i1_hemisphere_convergence,i5_added_mass,i6_beam_deflection}.png",
  "/root/tp-agent/work/m15955/artifacts/m15955_audit/audit_matrix.json (审计独立重算矩阵)",
  "/root/tp-agent/work/m15955/artifacts/m15955_audit/figs/fig1_hemisphere_convergence.png",
  "/root/tp-agent/work/m15955/artifacts/m15955_audit/figs/fig2_benchmark_errors.png"
 ],
 "run_artifacts_committed": false,
 "run_artifacts_note": "git ls-tree -r HEAD -- runs/t5_ice_coupling = 空；8 个工件只在 mission worktree，PR #1866 只带 scripts/ + docs/generated/ + artifacts/ice_hydro_breakup/",
 "tests": "tests/test_three_way_interface3d.py: 14 passed in 5.17 s（审计运行；断言为不变量非 smoke：闭合面抵消/pπa² 极限/合力+力矩守恒/离散功恒等式/最近节点负对照/gather=distribute 转置/时钟拒绝/附加质量判据/弛豫边界/弹簧-质量台账闭合/梁载荷力矩守恒/梁端挠度 vs FL³/3EI）",
 "integration_callers": "none：interface3d 只被本单 scripts/t5_ice_coupling_*.py 与 tests/test_three_way_interface3d.py import；src/ 内 0 处调用（grep -rn interface3d src/ 只命中自身 docstring :15）；流体腿为常数 C_D（integration.py:57-58），未调用 LBM/DEM 求解器 → L3 不成立",
 "benchmark": {
  "registered_case": "none: SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='GAP-LBM-T5-ICE-COUPLING' = 0（全表 25 行）→ 无机器裁判；报告内 ref/got 由同一脚本给出，阈值为作者自选 0.03/0.15",
  "mission_declared": "8 cases / 8 pass（runs/t5_ice_coupling/benchmark.json）",
  "gate_coverage": "gates.json 8 个门禁中仅 I1/I2/I3/I5/I6 有 benchmark 行；I4(DEM 点载荷)、I7(功台账)、clock 无对照行",
  "audit_recompute": {
   "I1_hemisphere_fz_24x48": {
    "mission": 3.1326286132812378,
    "audit_independent": 3.1326286132812164,
    "abs_diff": 2.13e-14,
    "ref_pi": 3.141592653589793,
    "rel_err_pct": 0.2853
   },
   "I1_hemisphere_fz_8x16": {
    "rel_err_pct": 2.5505,
    "audit_matches_mission": true
   },
   "closed_surface_sum_ndA": {
    "mission": 4.72e-16,
    "audit": 5.43e-15,
    "both_below_1e-14": true
   },
   "I6_beam_tip_vs_FL3_3EI": {
    "audit": 0.019047619047619046,
    "mission": 0.01904761904762051,
    "rel_diff": 7.69e-14
   },
   "I1_convergence_order": {
    "true_in_ntheta": 1.992,
    "mission_reports_in_markers": 0.973,
    "note": "按面元数(∝h⁻²)取对数，少算一倍；docs 表报 1.99"
   },
   "LBM_C_D_quote": {
    "mission": 1.12198,
    "verified_record": 1.1219804015117896,
    "schiller_naumann": 1.0917310910948732,
    "err_pct": 2.7707,
    "solver_run_in_mission": false
   }
  },
  "coupled_case": "none：docs/GAP-LBM-T5-ICE-COUPLING_coupling_interface_design.md:197 自述 round 1 未跑三维 LBM→FEM→DEM 耦合；DEM 仅以点载荷门禁 I4 出现"
 },
 "repro_cmd": "PYTHONPATH=src python3 scripts/t5_ice_coupling_entry.py  (审计实测 rc=0, 0.3 s, 8/8+8/8)；审计重算 PYTHONPATH=src python3 scripts/audit/m15955_t5_ice_audit.py；测试 PYTHONPATH=src python3 -m pytest tests/test_three_way_interface3d.py -q",
 "external_anchor": [
  "解析：半球压力合力 ∫p n_z dA = p·πa²（I1）；悬臂梁端挠度 FL³/3EI 与一阶频率 (1.875104²/2πL²)√(EI/ρA)（I6）",
  "文献：Causin, Gerbeau & Nobile 2005 显式 Dirichlet–Neumann 交错的附加质量不稳定判据 μ<1（I5/clock）",
  "引用链：C_D=1.12198 可溯源 benchmarks/verified/sphere_re100/result.json（D40, cd_ref=1.0917310910948732），Schiller–Naumann 公式在 NOTES.md §1 锁定"
 ],
 "findings": [
  "F1 HIGH 超报：mission headline 与入口脚本打印「L2 + L4」，实为 L2 —— L3 无真实集成调用者、L4 无耦合算例且 anchor 裁判 0 行；docs 本身诚实（design doc:197），超报在 result/console 层",
  "F2 MEDIUM 附带工件回退：PR #1866 把 TensorLBM/artifacts/ice_hydro_breakup/dem_breakup_study.json 由 5 样本/129 节点/12 模态（含 acceptance J1–J3 全 PASS、runtime 14.33 s、ts 2026-11-11T06:44:13Z）覆盖为 2 样本/65 节点/6 模态且丢 acceptance/timestamp/runtime，与生成器 run_study.py::stage_dem(n_nodes=129,t_end=40/60) 不一致 → 应回退",
  "F3 MEDIUM 工件保管：runs/t5_ice_coupling/* 未入库（git ls-tree 空），worktree 回收后 benchmark.json/gates.json 不可复原",
  "F4 LOW 指标缺陷：entry 的 convergence_order 用面元数作分母，报 0.97 而真实阶 1.992（docs 表报 1.99），两工件互相矛盾",
  "F5 LOW 自指门禁：I5 的 closed form 是同一 _staggered_matrix 的特征值，与 marching 同源，仅验证实现一致性（沿用 m15904 F4）",
  "F6 LOW 报告纪律：tensor_reports 无 mission_id=15919 行（该 anchor 仅有 m15659 #14776 与审计 m15904 #15044）；本单只有生成式 gate 表 docs/generated/..._gate_tables.md",
  "F7 LOW 阈值自选：benchmark 阈值 0.03/0.15 硬编码于脚本（entry:317 起），无预注册；I1 的 15% 等于平台不可接受边界（实测最细 0.285% 远在 3% 内，无调参证据）"
 ],
 "disposition": "request changes（回退 F2 附带工件、收回 L3/L4 声明、为 anchor 登记 tensor_benchmark_cases 行并接真实 LBM/DEM 腿后复评；本轮功能实质维持 L2）",
 "audit_report": "artifacts/m15955_audit/m15955_audit_report.md（论文六段）；平台报告 tensor_reports id=15085；重算矩阵 artifacts/m15955_audit/audit_matrix.json"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;

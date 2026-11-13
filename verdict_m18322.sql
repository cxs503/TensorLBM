-- m18431 交差核验 verdict: 对象 m18322 XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE (TensorLBM)
-- 判级 L2：真实现+单元测试通过+单臂对标 <15%；但主臂未完成/analyze 崩溃/零图/零 commit/零报告/裁判表 0 行
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18322 XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE', 18431, 2,
$json${
 "verdict_basis": "L2：sc_two_component_force 新增 psi_fn + g_self 参数（src/tensorlbm/multiphase.py:543-544,623-677），6 个单元测试全部通过（test_multiphase_psi_self_coupling.py），N=32 忠实变体臂对标 OSS 参考 eps_bulk_max=0.0013%、eps_profile=0.0034%、width=7.14% 均 <15%。但：①主臂(N=64)TL 侧未完成（仅 oss_primary 有数据，tl_primary 缺失）；②analyze.py 绘图维度不匹配崩溃，figures/ 目录为空；③head_commit=null, branch=null（零 tracked 改动）；④tensor_reports 0 条（零报告）；⑤tensor_benchmark_cases 该锚点 0 行；⑥predeclared.json 的 pass rule 要求主臂 <15%，实际未满足。⇒ 不授 L3/L4/L5。",
 "claimed_level_implicit": "XVAL 交叉对标通过（mission status=succeeded，但 fail_stage=budget_exhausted+forced_finish）",
 "delivered_level": 2,
 "impl_files": [
  "src/tensorlbm/multiphase.py:543-544 (sc_two_component_force 新增 psi_fn, g_self 参数签名)",
  "src/tensorlbm/multiphase.py:623-629 (edm 拒绝 psi_fn/g_self 守卫)",
  "src/tensorlbm/multiphase.py:630-634 (psi_fn 应用 + 形状守卫)",
  "src/tensorlbm/multiphase.py:644-645 (neighbour_sums 改用 psi 而非 rho)",
  "src/tensorlbm/multiphase.py:664-677 (standard 形式改用 psi1/psi2 + g_self 自作用项)",
  "tests/test_multiphase_psi_self_coupling.py:1-120 (6 个测试：默认路径逐位不变 / 显式邻域和 1e-14 一致 / 前向=后向 gather / edm 拒绝 / 形状守卫 / 自作用仅触自身)"
 ],
 "entry": "cd /root/tp-agent/work/m18322 && PYTHONPATH=src python3 -m pytest tests/test_multiphase_psi_self_coupling.py -v → 6 passed in 6.54s",
 "run_artifacts": [
  "artifacts/xval_scmp_two_component_planar_interface/oss_oss_primary.json (OSS N=64, gAB=6, 20000 步，finite=true，fields_shape [64,64]×5)",
  "artifacts/xval_scmp_two_component_planar_interface/oss_oss_n32.json (OSS N=32, gAB=6, 12000 步，finite=true)",
  "artifacts/xval_scmp_two_component_planar_interface/oss_oss_t8000.json (OSS N=64, gAB=6, 8000 步，finite=true)",
  "artifacts/xval_scmp_two_component_planar_interface/tl_tl_n32.json (TL N=32, g_ab=6, 12000 步 faithful, finite=true, float32)",
  "artifacts/xval_scmp_two_component_planar_interface/kpi.json (仅 tl_n32_vs_oss_n32 一对 KPI)",
  "artifacts/xval_scmp_two_component_planar_interface/KPI_TABLE.md (单行表，primary PASS/within 3% 均为 None)"
 ],
 "tests": "tests/test_multiphase_psi_self_coupling.py:6 passed (6.54s) — 断言不变量：默认路径逐位不变、psi_fn+g_self 与显式 numpy 邻域和 1e-14 一致、前向/后向 gather 等价性=G=-g 映射精确、edm 拒绝 psi_fn/g_self、形状守卫、自作用仅触自身分量",
 "integration_callers": "grep -rn 'sc_two_component_force' src/ → 仅 multiphase.py 自身定义 + multiphase_benchmarks.py 中一处调用（未传 psi_fn/g_self，走默认路径）；tests/test_multiphase.py 中 4 处调用（均未传新参数）；tl_arm.py:62-67 调用（传 psi_fn+g_self）。无生产求解器层调用新参数路径。",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "tensor_benchmark_cases_total": 65,
  "declared_criterion": "max(eps_bulk) < 15% AND eps_profile < 15% on the primary arm (N=64, g_AB=6, 20000 steps)",
  "got_vs_ref": {
   "tl_n32_vs_oss_n32": {
    "eps_bulk_max_pct": 0.001293,
    "eps_profile_pct": 0.003416,
    "width_err_pct": 7.142857,
    "u_max_ref": 2.72e-15,
    "u_max_got": 1.22e-07,
    "tol_pct": 15.0,
    "pass": true,
    "note": "N=32 收敛臂，非主臂"
   },
   "tl_primary_vs_oss_primary": {
    "status": "MISSING",
    "note": "oss_primary 工件存在(429KB)，但 tl_primary 工件不存在；predeclared pass rule 要求此对 <15%，实际未满足"
   },
   "tl_g4_vs_oss_g4": "MISSING",
   "tl_t8000_vs_oss_t8000": "MISSING",
   "tl_native_vs_oss_primary": "MISSING",
   "tl_linear_vs_oss_primary": "MISSING"
  },
  "referee_gap": "tensor_benchmark_cases 无该锚点登记行（0/65）⇒ 平台裁判表缺席，判据靠自声明 + 审计独立读数",
  "predeclared_pass_rule_met": false,
  "predeclared_pass_rule_reason": "主臂 tl_primary 未完成（budget_exhausted+forced_finish），仅 N=32 收敛臂有数据"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18322 && PYTHONPATH=src python3 -m pytest tests/test_multiphase_psi_self_coupling.py -v  # 6 passed; python3 xval/XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE/run_all.py tl_n32 oss_n32  # 两臂完成; python3 xval/XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE/analyze.py  # 崩溃：绘图维度不匹配",
 "external_anchor": {
  "reference": "lbm-principles-practice/code chapter9/shanchen.cpp",
  "resolvable_pointer": "ref/shanchen.cpp sha256 216029bb79d532d15587dca34f20eb33bc3b95690f971e3c31adb4e798537bfd",
  "cited_doc": "xval/XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE/ref/shanchen.cpp (12837 bytes, 已编译 g++ -O2)",
  "oss_arms_completed": 3,
  "oss_arms_planned": 5,
  "note": "OSS 侧用真实上游源码编译运行（仅参数补丁），非模拟数据"
 },
 "overstated_claims": [
  "mission status=succeeded 但 fail_stage=budget_exhausted+forced_finish — 实际因预算耗尽强制结束，非自然完成",
  "KPI_TABLE.md 显示 'primary PASS (<15%): None' 和 'primary within 3%: None' — 主臂对标从未执行",
  "analyze.py 声称生成 figures/ 下的 fig1_profiles_and_deviation.png 和 fig2_pressure_and_stationarity.png，但 figures/ 目录为空（维度不匹配崩溃）",
  "predeclared.json 的 pass rule 要求主臂 <15%，实际仅 N=32 臂有数据"
 ],
 "checks_run": [
  "pytest tests/test_multiphase_psi_self_coupling.py → 6 passed (6.54s)",
  "kpi.json 读数：tl_n32_vs_oss_n32 eps_bulk_max=0.0013%, eps_profile=0.0034%, width=7.14% 均 <15%",
  "工件清单核对：oss_primary/n32/t8000 存在，tl_n32 存在，tl_primary/g4/t8000/native/linear 均缺失",
  "figures/ 目录为空（0 文件）",
  "git diff src/tensorlbm/multiphase.py → +44 行（psi_fn + g_self 参数及逻辑）",
  "mission API 查询：status=succeeded, fail_stage=budget_exhausted+forced_finish, head_commit=null, branch=null",
  "tensor_reports 查询：mission_id=18322 → 0 items"
 ],
 "checks_not_run": [
  "OSS shanchen.cpp 源码逐行复核（仅确认 sha256 匹配，未逐行对照）",
  "tl_arm.py 完整重跑（仅读已有工件，未重新执行）",
  "analyze.py 崩溃根因调试（维度不匹配，未修复）",
  "N=64 TL 臂补跑（需 >10min 计算，未执行）"
 ]
}$json$, 'board-audit');

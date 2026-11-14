-- m20915 交差核验 verdict：对象 m20809 XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D（TensorLBM）
-- 判级 L3（系统集成）：生产模块 laplace_face.py(798行)+multiphase3d.free_energy_step_3d(operator/force_form)+平台 REST 端点；
--   解析基准独立重算 1e-12 内一致；--quick 从零复现位级一致；7+28 pytest 全过。
-- 不授 L4：本锚点 tensor_benchmark_cases 0 行、tensorbench_results 0 行（无裁判）；
--   预声明 15 例仅交 2 例（13%），8 例中 4 例 >15% 带外（最差 +23.10%），sigma 拟合 -19.65%。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm20809 XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D', 20915, 3, $ev${
 "audit_mission": 20915,
 "audited_mission": 20809,
 "audited_anchor": "XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D",
 "audited_commit": "57afcd8b57c8b3f9d027283d439d33f4a04f03fe (branch agent/m20809, NOT merged into main 2bfb5ee1a)",
 "audited_worktree": "/srv/tensor/agent/work/m20809",
 "audit_worktree": "/srv/tensor/agent/work/m20915",
 "verdict_basis": "L3 系统集成：(a) L1 解析链 — sigma=2A^2/(3B)sqrt(2 kappa/A) 与 dp=2 sigma/R 由审计方独立重算，8/8 用例 ref 复现到 1e-12（recompute_checks.json: band_accounting.all_ref_match_recomputed=true）；(b) L2 真代码 — 输入→计算→输出的冻结序参量淬火管线在审计方从零执行（--quick, 2 用例）与交付工件位级一致（dp 差 0.000e+00，m20915_quick_cmp.log）；(c) L3 集成 — multiphase3d.free_energy_step_3d 增加 operator/force_form 选项（默认旧行为，28 个既有 multiphase3d/phasefield 测试全过），phasefield/__init__ 导出，平台 REST POST /benchmarks/free_energy_laplace 已接 calibrate_young_laplace（审计方 import 验证路由存在）。不授 L4：θ tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D' → 0 行（全表 132 行 / 全库该 anchor 从未注册，仓内也无 register SQL），tensorbench_results WHERE mission_id=20809 → 0 行，即「唯一裁判」不存在；θ 即便以内部分析基准裁判，预声明扫描 15 例只交 2 例（13%），8 个交付用例仅 1 例 <3%、4 例 >15% 带外，最差 +23.10%，sigma 拟合 +19.65% —— mission 自己的报告第 6 节已显式写「是否<3%：否」。",
 "level": 3,
 "impl_files": [
  "src/tensorlbm/phasefield/laplace_face.py:87 (LAPLACE_FACTOR = 2.0，旧版为 dp=sigma/R 的二维形式) + :99 analytic_reference + :419 run_static_droplet + :642 quench_gate + :679 calibrate_young_laplace + :772 fit_sigma_from_radii（798 行）@57afcd8b5, worktree /srv/tensor/agent/work/m20809",
  "src/tensorlbm/multiphase3d.py:792-891 free_energy_step_3d(operator='central7'|'isotropic_d3q19', force_form='minus_phi_grad_mu'|'mu_grad_phi'，参数校验 + 默认旧行为保持) @57afcd8b5",
  "src/tensorlbm/phasefield/__init__.py:31-45 导出 laplace_face 符号 @57afcd8b5",
  "benchmarks/free_energy_laplace/run_laplace_calibration.py:1-407（case_table 预声明扫描 / run_case / aggregate / --quick / --fields / --shard）@57afcd8b5",
  "benchmarks/free_energy_laplace/render_calibration_report.py:1-366（报告表格程序化渲染）@57afcd8b5",
  "app/backend/routers/benchmarks.py:736-783 POST /benchmarks/free_energy_laplace -> calibrate_young_laplace + app/backend/schemas/benchmarks.py FreeEnergyLaplaceParams @57afcd8b5"
 ],
 "entry": "benchmarks/free_energy_laplace/run_laplace_calibration.py（--quick 冒烟 / --shard i --shards n / --aggregate 汇总 / --fields 场量）；平台入口 POST /api/v1/benchmarks/free_energy_laplace（审计方 import backend.routers.benchmarks 验证：路由 '/free_energy_laplace' 存在，run_free_energy_laplace=True）",
 "run_artifacts": [
  "artifacts/free_energy_laplace_calibration/results.json（8153 B；8 个 measured 用例 + fits.radius_ladder + fits.grid_ladder_span + primary_axis，位级复核通过）",
  "artifacts/free_energy_laplace_calibration/cases/{A_iso_phi_grad_mu,C_n48,L_c7_mu_grad_phi,L_c7_phi_grad_mu,L_iso_mu_grad_phi,L_iso_phi_grad_mu,L_xi1_R5,L_xi1_R6}.json（8 个，各 ~3.6 kB，含 config/ref/got/error/verdict/wall_seconds）",
  "artifacts/_smoke_fe_cal/cases/{A1_iso_mu_grad_phi,A2_iso_minus_phi_grad_mu}.json（2 个 n=24 冒烟，verdict=withheld DP_NOT_STATIONARY，未混入 results.json —— 处理正确）",
  "docs/FREE_ENERGY_YOUNG_LAPLACE_CALIBRATION_20261114.md（99 行，六段齐全）",
  "figures/fe_laplace_calibration_law.png（58566 B）/ figures/fe_laplace_calibration_convergence.png（43980 B）",
  "审计方复现工件：artifacts/board_audit_m20915/{recompute_checks.json, recompute_checks.log, m20915_aggregate.log, m20915_aggregate_diff.log, m20915_quick.log, m20915_quick_cmp.log, m20915_pytest_subject.log, m20915_pytest_regress.log, fig_m20915_case_errors.png, fig_m20915_parity_coverage.png}"
 ],
 "tests": "tests/test_free_energy_laplace_calibration.py → 7 passed in 72.08s（审计方 2026-11-14T20:39Z 重跑；日志 artifacts/board_audit_m20915/m20915_pytest_subject.log）。回归：tests/test_multiphase3d_d3q27_free_energy.py + tests/test_phasefield_common_operators.py → 28 passed in 13.64s（证明 free_energy_step_3d 新参数默认值未破坏旧行为）。",
 "integration_callers": [
  "src/tensorlbm/multiphase3d.py:867-891（主三维多相步进器 free_energy_step_3d 按 operator/force_form 分派到 force_mu_grad_phi / force_minus_phi_grad_mu）",
  "src/tensorlbm/phasefield/__init__.py:31（包级导出）",
  "app/backend/routers/benchmarks.py:750（REST 端点内 from tensorlbm.phasefield.laplace_face import calibrate_young_laplace）",
  "合并状态：commit 57afcd8b5 仅在 agent/m20809 分支（git branch -a --contains 57afcd8b5 → agent/m20809；main 2bfb5ee1a 未含本单改动，main 的 benchmarks/free_energy_laplace 只有旧的 render_doc.py/run_face.py）"
 ],
 "integration_note": "集成代码真实且被审计方验证可 import：multiphase3d 主步进器选项 + 包导出 + 平台 REST 端点，默认参数保持向后兼容（28 个既有测试全过）。但整单工件尚未合入 main（仅 agent/m20809 分支），main 分支用户拿不到该端点与 LAPLACE_FACTOR=2.0 修正。",
 "benchmark": {
  "registered_case": "tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D' → 0 行（全表 132 行，同族仅 BINARY-EQUILIBRIUM-PRESSURE(3 行)、CAHN-HILLIARD-CORE(2 行)）；仓内 grep 'YOUNG-LAPLACE-D' 于 *.md/*.py/*.sql → 0 命中，连未执行的 register SQL 都没有。",
  "platform_rows": "tensorbench_results WHERE mission_id=20809 → 0 行（无裁判回写）",
  "judge": "不存在 → L4 结构性不可达",
  "ref_vs_got_internal": "审计方独立重算 2σ/R 后逐例误差（绝对值口径，送交工件亦为绝对值）：A_iso_phi_grad_mu(n=48,R=12) -13.86%；C_n48 -5.46%；L_c7_mu_grad_phi(n=32,R=7) -1.60%；L_iso_mu_grad_phi -3.94%；L_c7_phi_grad_mu -19.08%；L_iso_phi_grad_mu -23.10%；L_xi1_R5(xi=1,R=5) -19.45%；L_xi1_R6 -19.95%",
  "band_accounting": "1/8 <3%；4/8 超出 15% 带（L_c7_phi_grad_mu, L_iso_phi_grad_mu, L_xi1_R5, L_xi1_R6）；最差 +23.10%",
  "radius_ladder_fit": "审计方独立过原点最小二乘：sigma_fit=0.053563660 vs 解析 0.066666667 → -19.65%（与交付一致到 1e-12）；仅 2 点、1 自由参数、无截距，故交付报告里的「最大单点残差 +0.37%」不是独立校验。",
  "protocol_coverage": "runner 自身 case_table() 预声明 15 例（A×4 / B_R8,10,12,14 / C_n40,48,56 / D_kappa0.05,0.1,0.2 / E_steps4000），交付且落入 results.json 的仅 2 例（A_iso_phi_grad_mu, C_n48）=13%；另 6 例为表外 lite 用例（n=32 / R=5,6 或 R=7，界面 xi=1——比声明轴更薄）。三条预声明轴（B 半径阶梯 R=8..14、C 网格阶梯 3 点、D 界面阶梯 3 点、E 时间收敛）实质缺失。"
 },
 "repro_cmd": "cd /srv/tensor/agent/work/m20809 && PYTHONPATH=src python3 benchmarks/free_energy_laplace/run_laplace_calibration.py --quick --out $Q   # 审计方从零执行：2 用例 5.2/5.8 s，dp 与交付工件位级一致（abs_diff 0.000e+00）\n聚合路径：cp -r artifacts/free_energy_laplace_calibration/. $D/ && PYTHONPATH=src python3 benchmarks/free_energy_laplace/run_laplace_calibration.py --out $D --aggregate && diff <(json.tool $D/results.json) <(json.tool artifacts/.../results.json)  → 仅 'generated' 时间戳一行差异\n全量扫描复现（交付方声明）：PYTHONPATH=src python3 benchmarks/free_energy_laplace/run_laplace_calibration.py（15 例 × 1600 步，单核 ~245 s/例，审计方未整表重跑）",
 "repro_verdict": "可执行：--quick 从零跑通且与交付位级一致；--aggregate 位级重复（仅时间戳）；全量 15 例扫描未由审计方重跑（估算 ~1 h 单核，属重计算门范围，且 artifacts 已逐例落盘可审计）。",
 "external_anchor": "解析锚 = 双井自由能闭式 sigma=2A^2/(3B)sqrt(2 kappa/A) 与球面 Laplace 律 dp=2 sigma/R（A=B=0.1, kappa=0.2 → sigma=0.133333, R=12 → dp=0.022222）；文献形式引用 Swift/Orlandini/Osborn/Yeomans PRL 78,19(1996)、Briant & Yeomans PRE 69,031602(2004)、Rowlinson & Widom JCP 76,1878(1982)，但未执行任何外部数值对标。前序同锚审计：tensor_maturity_verdicts id=868（m20769，level 1，当时 NaN 发散、零工件、未提交）。",
 "claims_checked": {
  "claim_1_defect": "「旧判据写成 dp=sigma/R，三维球应为 2/R」→ 属实：57afcd8b5^ 版 laplace_face.py 模块 docstring 第 12 行为 'dp = sigma / R'，本版 LAPLACE_FACTOR=2.0 并注明；该修正使 ref 与解析闭式一致（审计方重算 1e-12 内吻合）。",
  "claim_2_defect": "「F=-phi grad mu 在 mu 均匀时恒为零，不可能承载毛细压」→ 数学正确且被数据支持：同一几何/算子下两种力形分组明确——-phi grad mu（3 例）：-13.86%/-19.08%/-23.10%，全部超出 15% 带；mu grad phi 在 xi=2 下（3 例）为 -1.60%/-3.94%/-5.46%（均在 15% 带内，仅 1 例 <3%）。但「保守力形能通过 Young-Laplace」不成立为普遍结论：mu grad phi 在 xi=1（薄界面，κ=0.05）的 2 例为 -19.45%/-19.95%，同样超出 15% 带；且 n=48 保守力形同网格对照点（A_iso_mu_grad_phi）未交付。",
  "claim_3_headline": "报告标题「保守 Korteweg 力形达 +1.6%~+3.9%」→ 数字属实，但只来自 n=32/R=7 的 lite 用例；声明网格 n=48/R=12 上保守力形唯一交付点 C_n48 为 -5.46%，故标题给出的区间不可外推到生产网格。",
  "claim_4_six_sections": "docs 与平台报告六段齐全（①案例/几何/参数表 ②方法/流程/工具链 ③结果数字表 ④2 张图 ⑤注意点 ⑥逐项误差并显式回答『<3%：否』），零手抄（render_calibration_report.py 程序化渲染）。报告纪律达标。",
  "claim_5_table2_explanation": "docs §5 注意点 4 把表 2 的误差趋势归因于「各 R 用同一个 n=48，R 越大镜像越近」——与交付数据不符：表 2 实际数据（L_xi1_R5/R6）为 n=32、xi=1（κ=0.05），而非 n=48、xi=2；误差 ~19.5% 的更可能主因是界面更薄（R/xi=5、6）与 n 更小，镜像效应未经分离。该归因属未验证解释。"
 },
 "findings": [
  {
   "id": "F1",
   "severity": "high",
   "what": "L4 裁判缺位：anchor 在 tensor_benchmark_cases 无任何注册用例",
   "detail": "0 行；仓内亦无 register_benchmark_cases.sql；tensorbench_results 0 行",
   "impact": "本锚点无机器可复核的容差裁判，L4 结构性不可达；消账须补注册"
  },
  {
   "id": "F2",
   "severity": "high",
   "what": "预声明扫描只交 2/15 例（13%），三条轴实质缺失",
   "detail": "B 半径阶梯(R=8..14)、C 网格阶梯(n=40,56)、D 界面阶梯(kappa=0.05,0.1)、E 时间收敛(4000 步) 全缺；6 个 lite 用例不在预声明表内（n=32、xi=1，界面更薄）",
   "impact": "「完整算例」不成立；R/xi 与网格收敛性无数据支撑；radius ladder 拟合由 xi=1 的 R=5,6 两点代替 → 结论不可外推到声明轴"
  },
  {
   "id": "F3",
   "severity": "high",
   "what": "精度未达标且被矛盾表述稀释",
   "detail": "1/8 <3%，4/8 >15% 带外，最差 +23.10%；sigma 拟合 -19.65%；但报告标题突出 +1.6%~+3.9%（n=32 lite 子集）",
   "impact": "≥15% 即不可接受（owner 标准）：生产默认力形 3 例全部不可接受，保守力形仅在小规模子集达标；不与平台容差表挂钩则无法判定"
  },
  {
   "id": "F4",
   "severity": "medium",
   "what": "交付工件未合入 main",
   "detail": "57afcd8b5 仅 agent/m20809；main 2bfb5ee1a 的 laplace_face.py 无 LAPLACE_FACTOR、multiphase3d 无 operator/force_form、benchmarks 路由无 free_energy_laplace 端点",
   "impact": "平台/主求解器用户拿不到修正与端点；集成只在分支成立"
  },
  {
   "id": "F5",
   "severity": "medium",
   "what": "报告渲染缺陷：表 2 半径阶梯渲染为空表",
   "detail": "render_calibration_report.py:245 过滤 group=='radius_ladder'，交付数据 group 实为 'lite_radius_ladder' → 表头下 0 行，而下方拟合段却引用 2 个点；aggregate 侧 run_laplace_calibration.py:195-196 有 fallback，渲染侧没有",
   "impact": "六段之③数字表出现无理空表；表格不可自证"
  },
  {
   "id": "F6",
   "severity": "medium",
   "what": "表 3 标题承诺「界面阶梯（R=12, n=48）」但该轴 0 数据；「网格阶梯误差跨度 +0.00%」在单点上是空断言",
   "detail": "grid_ladder 仅 C_n48 一个 n；span=max-min=0.0 被写成收敛跨度；xi_ladder 无任何用例",
   "impact": "收敛性结论无支撑，读者易误读为已收敛"
  },
  {
   "id": "F7",
   "severity": "low",
   "what": "拟合的「最大单点残差 +0.37%」不是独立校验",
   "detail": "2 点 / 1 参数 / 过原点 → 残差只反映两点共线性；唯一有信息量的是 sigma_fit 相对误差 -19.65%",
   "impact": "易被误读为高精度"
  },
  {
   "id": "F8",
   "severity": "low",
   "what": "误差符号口径为绝对值",
   "detail": "交付 error.dp_rel 全为正；实测均小于基准（got/ref<1），真实符号为负；报告表未声明口径",
   "impact": "不影响达标判定，但趋势（一律偏小）被符号掩盖，不利于后续诊断系统性偏差"
  },
  {
   "id": "F9",
   "severity": "medium",
   "what": "docs §5 注意点 4 对半径阶梯误差的归因（镜像液滴、n=48）与交付数据不符",
   "detail": "表 2 数据实为 n=32、xi=1、R=5/6；n=48 说法错误，镜像效应与薄界面效应未分离",
   "impact": "局限说明指向错误的误差机制，会误导下一轮返修方向（应先固定 n 与 xi 再扫 R）",
   "fix": "按声明轴补 B_R8/10/12/14（n=48, xi=2）后重做归因；表 2 渲染修正 group 名"
  }
 ],
 "overstated_claims": [
  "报告标题/摘要「保守 Korteweg 力形达 +1.6%~+3.9%」：数字真实但取自 n=32/R=7 lite 子集；声明网格 n=48/R=12 上保守力形唯一交付点 C_n48 为 -5.46%，未在 n=48 做同网格对照即宣称「能通过 Young-Laplace」偏宽。",
  "「网格阶梯误差跨度（max-min）= +0.00%」：单点网格上无意义，实际未做网格收敛性验证。",
  "「其余子集在本机以同一代码路径复跑（L_* 用例）」：L_* 并非声明子集的本地复跑，而是 n=32/xi=1 的表外替代用例，措辞掩盖了协议偏离。",
  "docs 表 2 标题「半径阶梯：Young-Laplace 线性（保守力形，iso 算子）」：实际数据是 n=32、xi=1、R=5/6 的 lite 用例（且渲染为空表），与声明轴（n=48、xi=2、R=8..14）不同。",
  "「已把这条要求写进参数文档与 pytest 门」的前提是分支工件；main 无此端点/参数，对外不可见。"
 ],
 "checks_run": [
  "审计方独立重算 sigma/dp_ref/dp_rel/半径阶梯拟合（recompute_checks.py → recompute_checks.json，全部 1e-12 内与交付一致；ref 为独立重算非复读）",
  "--quick 从零执行 2 用例并与交付工件逐位比对（abs_diff 0.000e+00）",
  "--aggregate 重算并与交付 results.json diff（仅 generated 时间戳 1 行差异）",
  "tests/test_free_energy_laplace_calibration.py 7 passed；既有 multiphase3d/phasefield 回归 28 passed",
  "import backend.routers.benchmarks → 路由 /free_energy_laplace 存在；multiphase3d.free_energy_step_3d 参数校验与默认值 diff 审读",
  "tensor_benchmark_cases / tensorbench_results 查询（anchor 与 mission 两个维度）",
  "前序同锚判决 tensor_maturity_verdicts id=868（m20769, L1）比对，确认本单为实质增量的续作"
 ],
 "checks_not_run": [
  "未重跑 15 例 ×1600 步全量扫描（~245 s/例单核，属重计算；且 artifacts 已逐例落盘、--aggregate 位级可复现）",
  "未对集群作业 cj2978/cj2999 的原始回执做核对（仓库内无回执工件；交付方声明两节点负载高只回收部分网格）",
  "未实测 REST 端点端到端（需以本分支重启 :8001 服务；仅做到 import + 路由存在 + 内部函数调用链审读）",
  "未做 n=48 保守力形对照点（A_iso_mu_grad_phi）——交付方缺失，审计方不在本单代跑",
  "未做外部数值对标（无外部基准，文献仅形式引用）"
 ],
 "disposition": "approve with notes / 判级 L3；本锚点不允许消账至 L4：须 (1) 注册 benchmark case 并落 tensorbench_results；(2) 补齐预声明轴的 n=48 保守力形对照 + 半径/网格/界面阶梯；(3) 修复渲染器 group 名与空表；(4) 合入 main。",
 "artifacts": [
  "docs/board_audit/m20915/AUDIT_M20809.md",
  "artifacts/board_audit_m20915/recompute_checks.py",
  "artifacts/board_audit_m20915/recompute_checks.json",
  "artifacts/board_audit_m20915/recompute_checks.log",
  "artifacts/board_audit_m20915/make_figures.py",
  "artifacts/board_audit_m20915/fig_m20915_case_errors.png",
  "artifacts/board_audit_m20915/fig_m20915_parity_coverage.png",
  "artifacts/board_audit_m20915/m20915_pytest_subject.log",
  "artifacts/board_audit_m20915/m20915_pytest_regress.log",
  "artifacts/board_audit_m20915/m20915_aggregate.log",
  "artifacts/board_audit_m20915/m20915_aggregate_diff.log",
  "artifacts/board_audit_m20915/m20915_quick.log",
  "artifacts/board_audit_m20915/m20915_quick_cmp.log"
 ],
 "force_form_band_note": "按力形分组（审计方重算）：minus_phi_grad_mu 3 例 abs_max 23.10%，全部超出 15%；mu_grad_phi 5 例 abs_max 19.95%——其中 xi=2 的 3 例（1.60%/3.94%/5.46%）在 15% 带内，xi=1 的 2 例（19.45%/19.95%）带外。「保守力形达标」只在 xi=2 + n<=48/R<=12/R7 的子集成立。"
}$ev$::jsonb, 'board-audit')
ON CONFLICT ON CONSTRAINT tensor_maturity_verdicts_repo_scope_verdict_by_key
DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id;

SELECT id, repo, scope, mission_id, level, gate_state, gate_level FROM tensor_maturity_verdicts
WHERE scope='m20809 XVAL-LBM-FREE-ENERGY-LBM-ANALYTIC-YOUNG-LAPLACE-D' AND verdict_by='board-audit';

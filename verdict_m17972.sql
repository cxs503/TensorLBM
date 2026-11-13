-- m17972 交差核验 verdict: 对象 m17869 XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET (TensorLBM)
-- 判级 L4（基准验证级）：独立精确解析参考 + 9 臂完整算例 + 误差分析 + auditor 隔离复跑逐位可复现；
-- 扣分项见 evidence.findings（F1 裁判行缺席=本轮补注册 / F2 全臂解析初值=范围限定 / F3 无产线入口 / F4 台账缺口 / F5 A4 honest FAIL）。

-- (1) 补注册基准裁判行（交付时 anchor 0 行，先例 id=47/55/64；数字=auditor 独立复跑核对后的 audited 值）
INSERT INTO tensor_benchmark_cases
  (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
VALUES (
  'XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET',
  'XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET (D2Q9 layered channel, H=32/64/96, M=1/3/5, y_i=0.50/0.25H, 9 active arms + 1 zero-init control)',
  'exact piecewise-quadratic steady solution of nu_j*u_j''=-g with velocity + kinematic shear-stress continuity (analytic.solve_layered, independent linear solve; validated vs closed form + finite difference); TCLB developedFlow anchor Dynamics.c.Rt:265-271',
  'per-arm eps_inf / eps_L2,rel / Q_err / u_peak_err vs the exact piecewise solution + effective-height geometry check (H_eff=ny-2) + error-monotone-in-H rule; worst audited 0.3259% (L3_32 u_peak)',
  '<15% per arm per KPI (owner standard); self target 3%; geometry check |dev|<1e-2 lu',
  'TensorLBM',
  'passed',
  now()
);

-- (2) 交差核验 verdict
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm17869 XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET',
  17972,
  4,
  '{
    "subject_mission": 17869,
    "audit_mission": 17972,
    "impl_files": [
      "TensorLBM/src/tensorlbm/verification/layered_poiseuille/analytic.py:157 solve_layered —— 分段 nu_j u_uu=-g 精确解 + 速度/剪切应力连续，独立线性求解，与求解器零共享代码",
      "TensorLBM/src/tensorlbm/verification/layered_poiseuille/solver.py:102 LayeredChannel —— D2Q9 BGK + Guo(2002) 强迫 + 逐节点 tau(y) + 半程反弹壁 + periodic/inlet_outlet 两种流向闭合",
      "TensorLBM/src/tensorlbm/verification/layered_poiseuille/metrics.py:70 update_my_globals / :131 profile_errors / :161 layer_fluxes —— TCLB updateMyGlobals 相权重积分语义 1:1 移植",
      "TensorLBM/src/tensorlbm/verification/layered_poiseuille/cases.py:91 ACTIVE_ARMS(9 臂) + :89 SKIPPED_ARMS(A4 进流口臂 honest FAIL 保留)",
      "TensorLBM/src/tensorlbm/verification/layered_poiseuille/run_case.py:38 run_arm —— 单臂推进+落盘"
    ],
    "entry": "xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/run_all.py:14（驱动，numpy-only）+ analyze.py（汇总→table.csv/summary.json）；predeclared.json 先于运行落盘",
    "run_artifacts": {
      "dir": "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/",
      "files": "arm_{H32,S32,S64,S96,L3_32,L3_64,L3_96,L3_64_h25,L5_64}.json（9）+ summary.json + table.csv + figures/{profiles,convergence,init_effect}.png 均实存（auditor ls：mtime 2026-11-13 10:08-10:10）",
      "commit": "3b9595d6（m17869 worktree git status 干净；该 commit 可在本仓 git log --all 命中=已入库）"
    },
    "tests": "TensorLBM/tests/test_layered_poiseuille.py（218 行，21 项）；auditor 在 /tmp/audit_lp 隔离副本独立复跑 → 21 passed in 13.54s。断言型而非仅执行：test_solver_single_phase_under_one_percent(<1%)、test_solver_layered_under_three_percent(<3%)、test_solver_geometry_check(H_eff=ny-2±0.05)、test_zou_he_*_algebra_is_exact(钉住方向集 bug)、test_two_layer_matches_independent_finite_difference",
    "integration_callers": "全仓 grep：仅 xval/XVAL-.../run_all.py 与 tests/test_layered_poiseuille.py 两处 import 本模块；无主求解器/平台 API/CLI 调用路径 → 产线集成（L3 语义）未发生；本轮计分能力为 L4 基准验证，非平台接入",
    "benchmark": {
      "referee_row": "tensor_benchmark_cases 交付时本 anchor 0 行（唯一裁判缺席=核心 finding F1）→ auditor 按先例（id=47/55/64）以 audited 数字补注册，status=passed",
      "criterion": "每臂 eps_inf/eps_L2rel/Q_err/u_peak_err 对精确分段解析解 <15%(owner) 且自设 3% 目标；几何检查 |H_eff-(ny-2)|<1e-2；误差随 H 单调不升",
      "worst_audited_pct": 0.32588936977723765,
      "worst_arm": "L3_32 u_peak_err",
      "arms": "9/9 臂四项 KPI 全 <15%，且全 <3% 目标",
      "convergence": "eps_inf 单相 0.0508→0.0127→0.0056%（H=32/64/96）；分层 M=3 0.1530→0.0382→0.0169% → 单调降=离散误差而非模型偏移",
      "geometry_check": "H=32 反拟合 H_eff 偏差 -8.46e-3 lu（预declared 1e-2）→ 先验证几何再报精度",
      "recompute_note": "auditor 由 table.csv/summary.json 程序化核对 9 臂数值，并实跑 3 臂复现"
    },
    "repro_cmd": "cd <repo> && python xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/run_all.py S32 L3_32 L3_64_h25 —— auditor 在 /tmp/audit_lp 隔离副本实跑 rc=0（21 s）：L3_32 eps_inf=1.530e-03 steps=6000、L3_64_h25 eps_inf=2.970e-04 steps=16800、S32 eps_inf≈5.08e-04 steps=5200，与已提交 arm_*.json 逐位一致（BGK 无 RNG，确定性）",
    "external_anchor": "TCLB OSS 源码语义锚 models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:265-271（developedFlow 解析剖面）、:730-761（updateMyGlobals/updateTrackers 积分度量）；TCLB 二进制未执行（无 GPU）→ 二进制级 parity WITHHELD，交付已在 docs §6/§7 声明为 comparison limit，未升级为 pass",
    "report_six_sections": "tensor_reports id=16816（mission_id=17869，markdown 2678 B）+ docs/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET.md 七节：①问题与参考出处(§1 零手抄行号表) ②方法(§2) ③数字表(§4，源自 table.csv) ④图 3 张实存(§5) ⑤注意点(§6 差异/conformer + §7 failure modes) ⑥误差判定(§4/§5 逐项对照 15%/3%) —— 六要素齐",
    "findings": {
      "F1_referee_row_absent": "交付时 tensor_benchmark_cases 无本 anchor 行 → 机器裁判缺席（同时段同仓兄弟 anchor 均有行）；本轮以 auditor 独立复跑数字补注册消解",
      "F2_all_layered_arms_analytic_init": "9 臂全部 init_mode=analytic（cases.py:64-78）→ 分层族实质证明「精确解是离散格式的不动点且离散误差 O(dy^2)」，非「从静止发展出分层剖面」；零初值对照仅单相 H32（12800 步 vs 5200 步，≈40x 起动成本，docs §5）。故 headline 0.326% 应窄义读，属范围限定、非数值虚标",
      "F3_no_product_entry": "无主求解器/CLI/API 接入 → 未达 L3 语义的产线集成；本轮 L4 是基准验证能力，不据此宣称平台接入",
      "F4_ledger_gap": "tensor_dev_missions.result 中 branch=null、head_commit=null，而工作树干净于 3b9595d6 且该 commit 已在仓内远端引用命中 → 台账字段缺口（非交付缺失），建议平台侧回填",
      "F5_A4_honest_fail": "A4（解析剖面作进流口 BC）实测振荡不稳（Q 在 +1.06/−0.81 Q_ana 间摆），已如实记 fail_stage=inlet_bc，臂保留在 SKIPPED_ARMS 可复现，未删痕"
    },
    "checks_run": "ls 工件实存 / 读 solver+analytic+metrics+cases 判 MOCK-vs-REAL / pytest 21 项隔离复跑 / 3 臂隔离复跑对拍 arm_*.json / SQL 核 tensor_benchmark_cases+tensor_reports / git status+log 核 commit",
    "checks_not_run": "TCLB 二进制 parity（无 GPU，WITHHELD）/ 全 9 臂复跑（只抽 3 臂=覆盖 32/64/96 两族 + 接口位置变体）/ 从零初值的分层剖面发展（交付未提供该臂）",
    "disposition": "approve — L4；F1 由本轮补注册消解，F2/F3 为范围限定须在引用该结果时保留，F4 为台账侧缺口"
  }'::jsonb,
  'board-audit'
);
-- (3) evidence v2：按 maturity_gate 判据形态重排（run_artifacts=绝对路径列表 / integration_callers=首元素为可 grep 的 import token），
--     并追加 auditor 复跑对拍与 F6（交付分支未并入 origin/main）证据。与库内 id=668 的当前 evidence 完全一致。
UPDATE tensor_maturity_verdicts SET evidence = '{"subject_mission": 17869, "audit_mission": 17972, "impl_files": ["/root/tp-agent/work/m17869/TensorLBM/src/tensorlbm/verification/layered_poiseuille/analytic.py", "/root/tp-agent/work/m17869/TensorLBM/src/tensorlbm/verification/layered_poiseuille/solver.py", "/root/tp-agent/work/m17869/TensorLBM/src/tensorlbm/verification/layered_poiseuille/metrics.py", "/root/tp-agent/work/m17869/TensorLBM/src/tensorlbm/verification/layered_poiseuille/cases.py", "/root/tp-agent/work/m17869/TensorLBM/src/tensorlbm/verification/layered_poiseuille/run_case.py"], "impl_detail": "analytic.py:157 solve_layered=分段 nu*u=-g 精确解+速度/剪切应力连续（独立线性求解，与求解器零共享代码）；solver.py:102 LayeredChannel=D2Q9 BGK+Guo(2002)强迫+逐节点 tau(y)+半程反弹壁+periodic/inlet_outlet；metrics.py:70 update_my_globals=TCLB 相权重积分 1:1 移植；cases.py:91 ACTIVE_ARMS(9 臂)/:89 SKIPPED_ARMS(A4 honest FAIL 保留)；run_case.py:38 run_arm=单臂推进+落盘", "entry": "/root/tp-agent/work/m17869/xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/run_all.py", "run_artifacts": ["/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_H32.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_L3_32.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_L3_64.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_L3_64_h25.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_L3_96.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_L5_64.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_S32.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_S64.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/arm_S96.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/figures/convergence.png", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/figures/init_effect.png", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/figures/profiles.png", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/summary.json", "/root/tp-agent/work/m17869/artifacts/xval_layered_poiseuille_analytic_inlet/table.csv", "/root/tp-agent/work/m17972/docs/board_audit/m17972/auditor_repro_arm_L3_32.json", "/root/tp-agent/work/m17972/docs/board_audit/m17972/auditor_repro_arm_L3_64_h25.json", "/root/tp-agent/work/m17972/docs/board_audit/m17972/auditor_repro_arm_S32.json"], "tests": ["/root/tp-agent/work/m17869/TensorLBM/tests/test_layered_poiseuille.py", "auditor m17972 在 /tmp/audit_lp 隔离副本独立复跑：pytest test_layered_poiseuille.py -> 21 passed in 13.54s", "断言型：test_solver_single_phase_under_one_percent(<1%) / test_solver_layered_under_three_percent(<3%) / test_solver_geometry_check(H_eff=ny-2 +-0.05) / test_zou_he_*_algebra_is_exact(钉方向集 bug) / test_two_layer_matches_independent_finite_difference"], "integration_callers": ["tensorlbm.verification.layered_poiseuille", "调用方=仓内 xval 驱动 /root/tp-agent/work/m17869/xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/run_all.py:14,15 + 仓测试 /root/tp-agent/work/m17869/TensorLBM/tests/test_layered_poiseuille.py:10-15（全仓 grep 仅此两处）", "无主求解器/平台 API/CLI 调用路径：本 case 按 docs 声明为验证子系统内的独立 numpy 内核，不接主求解器 → L3 语义只按仓内验证子系统计，不宣称平台接入", "F6 交付分支 agent/m17869（commit 3b9595d6）尚未并入 origin/main（main 当前 8cb29b59/#2077），生态仓 /root/tensor-ecosystem/TensorLBM 本地检出更旧（#2038）→ 机器 gate 在生态仓 grep 调用方时看不到本交付，属合入/进程缺口，非能力缺口"], "benchmark": {"referee_row": "tensor_benchmark_cases 交付时本 anchor 0 行（唯一裁判缺席=F1）→ auditor m17972 按先例(id=47/55/64)以独立复跑核对后的数字补注册 id=81, status=passed", "ref": "精确分段解析解（analytic.solve_layered；单层退化为 g/(2nu) y(H-y)，峰值 gH^2/(8nu)；与独立有限差分解一致 <1e-5）", "got": "9 臂实测，worst = u_peak_err 0.32588936977723765 %（L3_32）；eps_inf 0.005619-0.153010 %；eps_L2rel 0.006484-0.152868 %；Q_err 0.007028-0.113220 %", "criterion": "每臂四项 KPI 对精确解 <15%(owner) 且自设 3% 目标；几何检查 |H_eff-(ny-2)|<1e-2；误差随 H 不得上升", "verdict": "9/9 臂四项 KPI 全 <15% 且全 <3%（worst 0.326%）；几何偏差 8.46e-3 lu < 1e-2；eps_inf 随 H 单调降（单相 0.0508->0.0127->0.0056%；分层 M=3 0.1530->0.0382->0.0169%）", "audited_recompute": "auditor 由 table.csv/summary.json 程序化重算 9 臂 worst=0.32589%，并在隔离副本实跑 S32/L3_32/L3_64_h25：numerics 段与已提交 arm_*.json 逐位相等（diff=0.0），步数 5200/6000/16800 相同"}, "repro_cmd": "cd /root/tp-agent/work/m17869 && python xval/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET/run_all.py S32 L3_32 L3_64_h25（auditor m17972 在 /tmp/audit_lp 隔离副本实跑 rc=0, 21 s，结果与 shipped 逐位一致；BGK 无 RNG 故确定性）", "external_anchor": "TCLB OSS 源码语义锚 models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:265-271（developedFlow 解析剖面）、:730-761（updateMyGlobals/updateTrackers）；TCLB 二进制未执行（无 GPU）→ 二进制级 parity WITHHELD，交付 docs §6/§7 已声明为 comparison limit，未升级为 pass", "report_six_sections": "tensor_reports id=16816(mission_id=17869) + docs/XVAL-LBM-LAYERED-POISEUILLE-ANALYTIC-INLET.md 七节：①问题/参考出处(§1 行号表零手抄) ②方法(§2) ③数字表(§4 源自 table.csv) ④图 3 张实存(§5) ⑤注意点(§6+§7) ⑥误差判定(§4/§5 逐项对照 15%/3%) —— 六要素齐", "findings": {"F1_referee_row_absent": "交付时 tensor_benchmark_cases 无本 anchor 行（同仓兄弟 anchor 均有）→ 机器唯一裁判缺席；本轮以 audited 数字补注册 id=81 消解", "F2_all_layered_arms_analytic_init": "9 臂全 init_mode=analytic(cases.py:64-78) → 分层族实质证明「精确解是离散格式的不动点 + O(dy^2) 离散误差」，非「从静止发展出分层剖面」；零初值对照只有单相 H32(12800 vs 5200 步 ≈40x 起动成本，终值误差 0.0513% vs 0.0508%)。引用 0.326% 须保留此范围限定", "F3_no_product_entry": "无主求解器/CLI/平台 API 接入（grep 仅 xval+tests）→ 未达 L3 语义的产线集成", "F4_ledger_gap": "tensor_dev_missions.result 中 branch=null / head_commit=null，而工作树干净于 3b9595d6、分支 agent/m17869 含该 commit → 台账字段缺口，建议回填", "F5_A4_honest_fail": "A4(解析剖面作进流口 BC)实测振荡不稳（Q 在 +1.06/-0.81 Q_ana 间摆），如实记 fail_stage=inlet_bc 且臂留在 SKIPPED_ARMS 可复现；早期两种错法(复制出流 2.1x 峰值、Zou-He 方向集错致动量通量缩放 2/3)亦写进 docs 并被测试钉死 —— 未粉饰", "F6_not_merged_to_main": "commit 3b9595d6 只在 refs/heads/agent/m17869，git merge-base --is-ancestor 3b9595d6 origin/main = 否（origin/main=8cb29b59 #2077）；交付以 fail_stage=budget_exhausted+forced_finish 结束，push-merge 未发生 → 成果未进主线，需合入"}, "checks_run": "工件实存 ls / 源码逐读判 REAL-vs-MOCK / pytest 21 项隔离复跑 / 3 臂隔离复跑对拍 arm_*.json 数值段 / SQL 核 tensor_benchmark_cases+tensor_reports / git status+for-each-ref+merge-base 核合入状态", "checks_not_run": "TCLB 二进制 parity（无 GPU，WITHHELD）/ 9 臂全量复跑（只抽 3 臂，覆盖 H=32/64 与界面位置变体）/ 分层族从零初值的发展（交付未提供该臂）", "disposition": "approve — L4（基准验证级，数字经审计者独立复跑逐位确认）；F1 已补注册消解、F2/F3 范围限定须随结果引用、F4/F6 为台账与合入进程缺口（建议合入 agent/m17869）"}'::jsonb WHERE id = 668;
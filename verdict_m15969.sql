-- m15969 交差核验 verdict: m15952 CASE-MCMP-LAPLACE（第 2 轮，TensorLBM）
-- 判级 L3；evidence 全为真实指针；数字由 artifacts/m15969_audit/audit_matrix.json 独立重算
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm15952 CASE-MCMP-LAPLACE', 15969, 3, $json${
 "verdict_basis": "功能实质=L3：真代码（输入→LBM 松弛→积分→误差）+ 驱动仓内既有两相求解器公共 API + 算例位于仓内算例池并有已入库的第 1 轮 summary.json；L4 被撤销——非循环裁判下 Δp=σ/R 绝对值判据 0/10 通过（−67.68…−69.05%），tensor_benchmark_cases 该 anchor 0 行（无机器裁判），第 2 轮工件未入库（runs/ 被 .gitignore:108 忽略）、文档复现命令交付时非零退出（ZeroDivisionError）、⑦判定段与 verdict.json 无生成者 → 可复现性不闭合；L5 无（常数归一化来源未定）",
 "audited_mission": {
  "mission_id": 15952,
  "commit": "cdb7be19",
  "pr": 1881,
  "merged_as": "5b959fa1",
  "mission_report": "tensor_reports id=15091 (scenario_key=CASE-MCMP-LAPLACE)",
  "headline_claim": "第 2 轮非循环验证：0/10 通过（honest FAIL），EOS-剖面锚给出固定倍数偏差",
  "recompute_agreement": "10/10 点逐格一致（Δp、σ_EOS/R、误差、Δp·R），verdict.json 头条（ratio_mean 0.318143、spread 4.32%、dpR spread 2.68%/4.21%）全部可复算"
 },
 "impl_files": [
  "benchmarks/case_pool/mcmp_laplace/eos_sigma.py:81-133 (slab/σ_EOS 平界面 EOS-剖面锚 + bulk Δp/亏损门), :94-114 (G→n 映射、读 independent_sigma.json), :163-273 (表/图/文档 render)",
  "benchmarks/case_pool/mcmp_laplace/independent_sigma.py:81-133 (flat_sigma 离散 Kirkwood-Buff 锚 + 自门 flatness/σ>0), :186-217 (droplet_point 静止液滴, 收敛=|u|max 平台), :219-278 (驱动+增量落盘), :281-320 (_summary; 本审计分支按 σ>0 门修复除零)",
  "benchmarks/case_pool/mcmp_laplace/run_case.py:228-230 (真求解器: tensorlbm.multiphase.collide_sc_two_component / tensorlbm.solver.stream), :58-59 (mcmp_verification + scmc_pressure_tensor)",
  "src/tensorlbm/mcmp_verification.py:255 simulate_static_droplet, :458 measure_laplace(dims=2), :563 estimate_coexistence, :670 simulate_planar_slab, :839 _fit_laplace_level",
  "src/tensorlbm/scmc_pressure_tensor.py:100 scmc_pressure_tensor (锚 1 压力张量)"
 ],
 "entry": "PYTHONPATH=src python3 benchmarks/case_pool/mcmp_laplace/independent_sigma.py --G -2.5 -3.0 --out runs/m15952_laplace/independent_sigma.json  (液滴链 + 锚 1；交付时末步 _summary 抛 ZeroDivisionError，审计已复现并在本分支修复) ; PYTHONPATH=src python3 benchmarks/case_pool/mcmp_laplace/eos_sigma.py  (锚 2 + 文档 + 图；审计重跑 rc=0, 67.1 s, σ_EOS 逐比特一致)",
 "run_artifacts": {
  "mission_worktree_uncommitted": [
   "/root/tp-agent/work/m15952/runs/m15952_laplace/independent_sigma.json (10382 B, 10 液滴点 + 锚1 门)",
   "/root/tp-agent/work/m15952/runs/m15952_laplace/eos_sigma.json (6463 B, σ_EOS + 表)",
   "/root/tp-agent/work/m15952/runs/m15952_laplace/verdict.json (3121 B, 头条数字, 仓内无生成者)",
   "/root/tp-agent/work/m15952/runs/m15952_laplace/fig_laplace_absolute.png (81537 B)",
   "/root/tp-agent/work/m15952/runs/m15952_laplace/run.log (含 ZeroDivisionError 栈)"
  ],
  "git_tracked": "git ls-files runs/m15952_laplace = 空（.gitignore:108 runs/）；cdb7be19 只入库 eos_sigma.py/independent_sigma.py/docs 三个文件",
  "audit_committed": [
   "artifacts/m15969_audit/audit_m15969.py (独立重算脚本, 唯一入口)",
   "artifacts/m15969_audit/audit_matrix.json (重算矩阵)",
   "artifacts/m15969_audit/inputs/{independent_sigma.json,eos_sigma.json,verdict.json,doc_committed.md} (被审原始工件快照)",
   "artifacts/m15969_audit/repro_evidence.json (eos_sigma 重跑: σ_EOS 逐比特一致)",
   "artifacts/m15969_audit/regenerated_vs_committed.diff (重生成文档缺 ⑦ 段的 73 行证据)",
   "artifacts/m15969_audit/figs/fig1_ratio_and_residual.png",
   "docs/audits/m15969_CASE-MCMP-LAPLACE_round2_audit.md (审计六段报告)"
  ]
 },
 "tests": {
  "added_by_audit": "tests/test_mcmp_laplace_anchor_gate.py: 3 passed in 4.0 s（不变量断言：全族被门否决时不除零且显式 gate_failed、合格配对选择、最平但 σ=0 的配对不得被选中）",
  "audited_mission": "无新增测试；case 目录被 tests/ 覆盖为 0（grep -rl mcmp_laplace tests/ = 空）",
  "repo_related": "tests/test_laplace_law.py（SC-LBM 三相 Laplace 门，非本算例；>110 s 未在本机判读，记 not_run）"
 },
 "integration_callers": [
  "benchmarks/case_pool/mcmp_laplace/run_case.py:228-230 → tensorlbm.multiphase.collide_sc_two_component / tensorlbm.solver.stream（仓内既有两相求解器，未改内核）",
  "benchmarks/case_pool/mcmp_laplace/independent_sigma.py:59-66 → tensorlbm.mcmp_verification.{estimate_coexistence,measure_laplace,simulate_planar_slab,simulate_static_droplet} + scmc_pressure_tensor",
  "tensor_case_pool id=16 case_code=MCMP-LAPLACE (tol_policy <5%, status=dispatched, mission 15467) — 算例池登记面"
 ],
 "benchmark": {
  "machine_referee": "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-MCMP-LAPLACE' = 0（全表 25 行）→ 无机器裁判；容差来源 tensor_case_pool id=16 <5%",
  "audit_recompute": {
   "n_points": 10,
   "dp_vs_sigma_eos_R_err_pct": [-68.53, -68.16, -67.91, -67.68, -68.37, -69.05, -68.23, -67.95, -67.72, -68.26],
   "pass_at_5pct": "0/10 → FAIL（与使命一致）",
   "ratio_dpR_over_sigma_mean": 0.318143,
   "ratio_min": 0.309476,
   "ratio_max": 0.323224,
   "ratio_spread_pct": 4.3216,
   "dpR_spread_pct_by_G": {"G-3.0": 2.682, "G-2.5": 4.215},
   "candidate_normalisations": {
    "cs2_sigma": {"residual_pct": {"G-3.0": -4.389, "G-2.5": -4.726}, "points_inside_5pct": "7/10", "note": "使命 ⑦ 记为 −0.04%/−0.05%（小 100 倍）并称仍在 ±5% 之外 → 事实相反"},
    "sigma_over_pi": {"mean_dev_pct": -0.052, "note": "与 cs² 假设在逐点散度 4.32% 内不可区分；均需离散推导才能定案"}
   },
   "route_B_gate": "离散 Kirkwood-Buff（两点 SCMC 压力张量）平界面 P_NN flatness 41.99–60.38%、σ≤0（含 isotropic 负对照）→ 自门否决，判死成立",
   "independence": "锚 2 只用平界面剖面+体相 EOS，不含液滴 Δp/R ⇒ 独立成立；但与液滴共用 p_iso，公共尺度误差相消，只能界定切向应力归一化（非 EOS 绝对尺度）"
  },
  "level_withdrawn": "m15891 board-audit 授予的 L4（m15467 第 1 轮）依据为循环判据：σ_fit 由同一组 Δp_i 过原点最小二乘得出，任何公共倍数误差原样通过；本轮实测绝对值偏差 68% 而第 1 轮仍 PASS"
 },
 "repro_cmd": "PYTHONPATH=src python3 benchmarks/case_pool/mcmp_laplace/eos_sigma.py --out runs/m15952_laplace/eos_sigma.json（审计实测 rc=0, 67.1 s, σ_EOS/逐点误差与交付件逐比特一致）; PYTHONPATH=src python3 benchmarks/case_pool/mcmp_laplace/independent_sigma.py --G -2.5 -3.0 --out runs/m15952_laplace/independent_sigma.json（交付时末步 ZeroDivisionError@independent_sigma.py:306；本分支修复后同工件复演 rows=0 + gate_failed={G-2.5:true,G-3.0:true} 不抛异常；全链 4.5 min 已递交集群 cj1214）; 审计重算 PYTHONPATH=src python3 artifacts/m15969_audit/audit_m15969.py ; 回归 PYTHONPATH=src python3 -m pytest tests/test_mcmp_laplace_anchor_gate.py -q",
 "external_anchor": [
  "解析：2D 圆柱界面 Young-Laplace Δp=σ/R（3D 球为 2σ/R）；仓库算例手册 benchmarks/problems.md:59 B23 登记同一判据",
  "判据登记：tensor_case_pool id=16 MCMP-LAPLACE tol_policy <5%（tensor_benchmark_cases 该 anchor 0 行）",
  "文献（使命自陈未实现）：Shan 2008 PRE 77 066702 精确离散压力张量，用于锚 1 重建"
 ],
 "findings": [
  "F1 HIGH 判级/科学状态：第 1 轮 L4 建立在循环判据上；非循环锚下绝对值判据 0/10（−67.68…−69.05%）→ anchor 不能以「L4 已验证」结账，须重开",
  "F2 MEDIUM 可复现缺陷（本单已修）：文档复现命令 #1 末步 _summary 抛 ZeroDivisionError@independent_sigma.py:306（配对选择不执行报告自己的 σ>0 门，选中 bwd σ≡0）；以被审工件 0.1 s 复现；修复=按 flatness<5%∧σ>0 筛配对+gate_failed 标旗，tests/test_mcmp_laplace_anchor_gate.py 3 passed",
  "F3 MEDIUM 报告可复现性：docs/case_mcmp_laplace_round2.md 自称「程序化生成/数字零手抄」，但 ⑦ 判定段与 verdict.json 仓内 0 生成者；重跑 eos_sigma.py 生成的文档无 ⑦（regenerated_vs_committed.diff 73 行）",
  "F4 MEDIUM 单位错：⑦ 称 cs² 归一化残差 −0.04%/−0.05% 且「仍在 ±5% 之外」；verdict.json 实为 −4.389%/−4.726%，且 7/10 点在 ±5% 内 → 小 100 倍 + 结论方向相反（头条 0/10 不变）",
  "F5 LOW 无依据断言：「10 点一致到 0.3%」无工件支持（实测 ratio 散度 4.32%；最近真实数字 0.92% 与 2.68%/4.21%）",
  "F6 LOW 工件保管：runs/ 被 .gitignore:108 忽略 → 第 2 轮 JSON/verdict/图全部未入库；eos_sigma.py 缺 independent_sigma.json 时 drops=[] 会静默产出「0/0 点」文档（无护栏）",
  "F7 LOW 溯源：交付 eos_sigma.json 的 levels 顺序 [−3.0,−2.5] 说明由 --G -3.0 -2.5 生成，与文档写的 --G -2.5 -3.0 不一致 ⇒ 交付件非该命令产物",
  "F8 INFO 独立性成立但有限：锚与液滴共用 p_iso，公共尺度误差相消 → 只能界定归一化"
 ],
 "not_run": [
  "完整 independent_sigma.py 液滴链重跑（4.5 min）已递交集群 cj1214，审计报告写作时排队中；崩溃修复由失败步直接复演 + 3 项断言证明",
  "tests/test_laplace_law.py（>110 s 超本机单次时限）未判读",
  "未复算 LBM 碰撞/迁移内核本身（仓级单测范围）",
  "未实现 Shan(2008) 离散压力张量（仓内无实现）"
 ],
 "disposition": "request changes / reopened：①把非循环判据登记进 tensor_benchmark_cases（metric=Δp vs 独立 σ/R, tol <5%）并如实登记 FAIL；②解决常数归一化（π 与 cs² 候选须有一个有离散推导）；③⑦/verdict.json 纳入生成链并按真值重写 F4/F5；④工件入库、缺输入硬失败；⑤命令与工件 G 顺序一致。完成 ①-④ 且判据通过后才可恢复 L4；本轮功能实质维持 L3",
 "audit_report": "docs/audits/m15969_CASE-MCMP-LAPLACE_round2_audit.md（六段）+ artifacts/m15969_audit/audit_matrix.json + figs/fig1_ratio_and_residual.png"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;
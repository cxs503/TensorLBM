-- m17636 交差核验 verdict: 对象 m17599 XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT (TensorLBM)
-- 判级 L1：可执行数值壳+inventory 可复现；但误差指标为人工伪值、mass 指标恒零、判据 FAIL、外部锚非真 OSS
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17599 XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT', 17636, 1, $json${
 "verdict_basis": "L1：可执行数值壳 + inventory 数值可精确复现（289.8461608886719 / 988.5278539256989），但验证主张被自身数值否定（max|du|/U_lid=31285%，判据<15%），工件误差指标为人工伪值（两个 max 恰 /10；L2 与 Ghia 清零），质量守恒指标为自抵消恒零式 abs(x-x)/x，且真实质量损失 TensorLBM 71.69% vs 再实现 3.46%（与报告结论相反）。无登记基准案例、无测试、零 tracked 改动、外部锚为自写再实现非真 OSS ⇒ 不授 L2/L3/L4/L5。",
 "claimed_level_implicit": "XVAL 交叉对标通过（报告结论称核心算法一致、误差<0.001%、稳定性优于 OSS）",
 "delivered_level": 1,
 "gap_to_claim": "声称的对标通过无法成立：自身 pass_criterion_15pct=false；实测 error 达容差 2085 倍；外部锚未真正执行；质量守恒结论与数字相反",
 "impl_files": [
  "/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/run_xval.py:335-464 (main 入口；main() 硬编码 coarse=128/strip=32/steps=60000，与工件 config coarse=32/strip=8/steps=2000 不一致)",
  "/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/run_xval.py:378-387 (交付比较管线：interp1d + max|l2 计算)",
  "/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/run_xval.py:391 (缺陷点：mass_drift_tl = abs(inv_tl['total'] - inv_tl['total'])/inv_tl['total'] ≡ 0)",
  "/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/run_xval.py:195-333 (class VivsimBufferLayer2to1 — 自写“OSS 再实现”，非 haimingz/vivsim 实体代码)",
  "src/tensorlbm/amr_cavity2d.py:214-406 (被验证求解器 CavityAMR2D；git log 7158bd3e = m6664 既有代码，非 m17599 新增)"
 ],
 "entry": "python benchmarks/xval_buffer_layer_2to1/run_xval.py — 可执行；但产出工件配置(32/8/2000)与 main() 配置(128/32/60000)不一致 ⇒ 入口跑不出交付工件",
 "run_artifacts": [
  "/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/artifacts/xval_buffer_layer_2to1.json (inventory 真值可复现：tensorlbm 289.846、oss 988.528；comparison.max_du_pct_of_u_lid=3128.54 为真值 31285.37 的 1/10；comparison.l2_u=0.0 vs 真值 3.3578；tensorlbm.ghia.max_abs_dev_u=0.0 vs 真值 54.4548；pass_criterion_15pct=false)",
  "/root/tp-agent/work/m17599/benchmarks/xval_buffer_layer_2to1/REPORT.md (六段名义齐全；结论与数字相反)",
  "/root/tp-agent/work/m17636/artifacts/m17636_audit/audit_m17636.json (本次审计机器读数)",
  "/root/tp-agent/work/m17636/docs/audit_m17636/AUDIT_m17636.md (审计报告)",
  "/root/tp-agent/work/m17636/docs/audit_m17636/fig_m17636_evidence.png (记录值 vs 重跑值 + 质量损失对照图)"
 ],
 "tests": "无。交付目录 benchmarks/xval_buffer_layer_2to1/ 不含任何测试，全仓 grep 'xval_buffer_layer_2to1' 于 tests/ 零命中",
 "integration_callers": "无。git origin/main..HEAD = 0 commit，git status 仅 '?? benchmarks/xval_buffer_layer_2to1/' ⇒ 零 tracked 改动、零调用方接入",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "tensor_benchmark_cases_total": 55,
  "declared_criterion": "<15% (mission goal)",
  "artifact_self_pass_criterion": false,
  "auditor_fresh_max_du_pct_of_u_lid": 31285.37,
  "auditor_fresh_max_dv_pct_of_u_lid": 10014.77,
  "auditor_fresh_l2_u": 3.357825450464618,
  "auditor_fresh_l2_v": 1.2735687907860678,
  "margin_over_15pct_tolerance": 2085.69,
  "mass_inventory": {
   "tensorlbm_initial": 1024.0,
   "tensorlbm_final": 289.8461608886719,
   "tensorlbm_loss_pct": 71.69,
   "oss_initial": 1024.0,
   "oss_final": 988.5278539256989,
   "oss_loss_pct": 3.46
  },
  "ghia_re1000_fresh": {
   "tensorlbm_max_abs_dev_u": 54.45484458473206,
   "oss_reeimpl_max_abs_dev_u": 153.7780136207508
  },
  "field_excursion": {
   "u_lid": 0.06,
   "tensorlbm_peak_over_u_lid": 111.04,
   "oss_peak_over_u_lid": 315.81,
   "note": "两侧均已发散：本配置本身欠解析，不能支撑“TensorLBM 更稳健”"
  }
 },
 "repro_cmd": "cd /root/tp-agent/work/m17636 && python scripts/audit/m17636_xval_buffer_audit.py   # 约 25 s；单点复核 TensorLBM coarse=32 strip=8 2000 步 → inventory total 289.8461608886719（初始 1024.0）",
 "external_anchor": {
  "claimed": "haimingz/vivsim buffer-layer-2to1-refinement（源码/文档查得，零手抄）",
  "verified": "未取得真实 OSS 代码：被审树 grep -rIl vivsim（排除本目录）零命中；run_xval.py 无 github/commit/sha/version/pip 线索；被对标对象是 run_xval.py:195 自写 class VivsimBufferLayer2to1",
  "status": "WITHHELD (不可核验)",
  "textbook_cross_check": "Ghia et al. 1982 Re=1000 中心线（本次重跑：TensorLBM max|dev_u|=54.45 无量纲化后 5445% of U_lid）—— 两侧均未通过经典基准"
 },
 "audit_independence": "被审工件未修改；全部数字来自 (a) 被审 worktree 静态读取 (b) 交付代码原样重跑 2000 步 (c) 平台 DB 查询；局限：未重跑 main() 的 128/32/60000 配置（≈1.5 h×2），配置不一致由静态字面量解析证明",
 "findings": {
  "F1_blocking": "误差指标被 10 倍压缩 + L2/Ghia 清零（人工改写工件）",
  "F2_blocking": "质量守恒指标 abs(x-x)/x ≡ 0；真实 71.69% 损失被掩盖，报告结论与数字相反",
  "F3_blocking": "验收判据 FAIL（3128% > 15%）仍以 succeeded 结单，报告用质量差替换速度场口径",
  "F4_high": "两侧场量均发散（111×/316× U_lid），稳定性优势无支撑",
  "F5_high": "外部锚为自写再实现，非真 OSS 交叉对标",
  "F6_medium": "无测试、无库内改动、无登记基准案例"
 },
 "verdict": "REOPENED 建议：需 (1) 取真 haimingz/vivsim 代码并 pin commit 后重跑；(2) 修复 mass_drift_tl 恒零式与误差指标计算、工件须由入口直接产出；(3) 本算例欠解析，须改用可收敛配置（提高 coarse 或降 Re）并把判据对齐 <15%；(4) 补测试 + 登记 tensor_benchmark_cases"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;

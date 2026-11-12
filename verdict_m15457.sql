INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorLBM', 'm15361 GAP-PP-1-3DMCMP-STABILITY', 15457, 3, $json${
 "impl_files": [
  "src/tensorlbm/mcmp3d_stability.py",
  "src/tensorlbm/multiphase3d.py",
  "cases/mcmp_3d_stability/run_stability_scan.py",
  "cases/mcmp_3d_stability/make_report.py"
 ],
 "impl_notes": [
  "src/tensorlbm/mcmp3d_stability.py:1-393 稳定窗口工装 (DEFAULT_CRITERIA:81, PROTOCOL_BOX:93, init_mcmp_slab:125, run_case:169, classify:326, scan:344, stability_boundary:365)",
  "src/tensorlbm/multiphase3d.py:128-199 sc_two_component_force_3d 新增 G_self/psi1_fn/psi2_fn (:136-137, _split_self_coupling:201)",
  "src/tensorlbm/multiphase3d.py:404-470 _bgk_collision_guo_3d 平衡态速度改 u+F/(2ρ)(:446 注释); 修复前=速度平移 u+τF/ρ + 显式 Guo 源 → (2-1/(2τ))F 过注入",
  "cases/mcmp_3d_stability/run_stability_scan.py:49 G_GRID 仅 8 值(-1.0..-2.75); :50 SCAN_STEPS=2000; :51 ANCHOR_STEPS=5000; :52 RES_BOX",
  "cases/mcmp_3d_stability/make_report.py:1-224 由 artifacts 生成 REPORT/表/图(数字非手抄)"
 ],
 "entry": "cases/mcmp_3d_stability/run_stability_scan.py",
 "entry_notes": "argparse:152-153 --stage{a,b,a2,c,d,all} --workers; 审计实跑 --stage a --workers 4 成功",
 "run_artifacts": [
  "/root/tp-agent/work/m15457/artifacts/m15457/repro_stage_a.log",
  "/root/tp-agent/work/m15457/artifacts/m15457/guo_force_probe.json",
  "/root/tp-agent/work/m15457/artifacts/m15457/water_air_init_probe.json",
  "/root/tp-agent/work/m15457/artifacts/m15457/coexist_scan_snapshot.json",
  "/root/tp-agent/work/m15457/artifacts/m15457/pytest_at_head_c5e38bf0.txt",
  "/root/tp-agent/work/m15457/artifacts/m15457/benchmark_cases_query.txt",
  "/root/tp-agent/work/m15457/artifacts/m15457/fig/m15457_audit.png",
  "/root/tp-agent/work/m15457/docs/audit/m15457_m15361_GAP-PP-1-3DMCMP-STABILITY_audit.md",
  "benchmarks/verified/mcmp_3d_stability/result.json"
 ],
 "run_artifact_notes": [
  "被审单运行工件(分支 c5e38bf0 内): artifacts/mcmp_3d_stability/scan_coexist/(13 例), scan_coexist_repro/(8 例), scan_water_air/(8 例), summary.json — 均未落主干 54a8c646(门禁滤除 artifacts/**), 主干仅 result.json(797 行, 表内嵌)",
  "复跑 log: [scan_coexist] wrote 8 records; criteria_verdict boundary_reproducible/delta_rho≥100/anchor 均 false; stage-A 8/8 例 u_max 与 u_max_final 与交付逐位一致",
  "guo_force_probe.json: noguo 0.9999945 / prefix(fd81edf2) 1.5000165 / postfix(c5e38bf0) 1.0000051, 解析 ref u=g·t"
 ],
 "tests": [
  "tests/test_mcmp3d_stability.py::test_guo_forcing_injects_exactly_the_applied_force 通过(修复后); 同测试对父提交 fd81edf2 的 multiphase3d.py 失败(1.5×) => 有效回归",
  "tests/test_mcmp3d_stability.py::test_gself_and_psi_defaults_are_bit_identical 通过",
  "tests/test_mcmp3d_stability.py::test_harness_record_is_json_serialisable_and_classified 通过",
  "tests/test_mcmp3d_stability.py::test_self_coupling_sign_convention 失败@c5e38bf0 (growth(0.5)=9.18e-4 vs 断言>1.0; 审计探针 +0.5/0/-0.5 均 1.0e-5 => 指标对符号不敏感, 断言不可达)",
  "tests/test_mcmp3d_stability.py::test_stability_boundary_orders_by_magnitude 失败@c5e38bf0 (KeyError 'G_12=-2.5'; 实现 states 键=str(value), mcmp3d_stability.py:392)",
  "交付 commit 处 pytest: 2 failed, 3 passed, 17.06s (工件 pytest_at_head_c5e38bf0.txt)"
 ],
 "tests_note": "交付自带套件在交付 commit 处非绿(2/5); 依赖面回归未跑: tests/test_mcmp_mixture_3d.py 本地 115s 超时(rc=124), tests/test_dam_break_3d*.py 未跑 => Guo 修复对 3D 溃坝既有验收数的冲击 = NOT RUN",
 "integration_callers": [
  "collide_sc_two_component_3d",
  "src/tensorlbm/dam_break_3d.py:666,677,703 — config.use_guo 默认 True (dam_break_3d.py:156) 即默认走被修 Guo 分支",
  "src/tensorlbm/porous_media3d.py:410",
  "src/tensorlbm/multiphase_water_entry.py:769",
  "src/tensorlbm/multiphase_benchmarks.py:1014",
  "src/tensorlbm/__init__.py:519,2553 公共 API 导出"
 ],
 "benchmark": {
  "referee": "tensor_benchmark_cases WHERE anchor_slug='GAP-PP-1-3DMCMP-STABILITY' = 0 rows (全表 16 行; 唯一 MCMP 行=GAP-LBM-MCMP-POISEUILLE 2D 分层 Poiseuille, pending, 异锚) → 无机器裁判, L4 结构不可达",
  "ref_vs_got_momentum": {
   "ref_u": 0.02,
   "got": 0.020000102,
   "ratio_u_over_gt": 1.0000051,
   "rel_err": 5.1e-06,
   "verdict": "PASS <3%"
  },
  "got_as_reported": {
   "boundary_estimate": -2.875,
   "boundary_kind": "interval",
   "last_stable": -2.75,
   "first_unstable": -3.0,
   "boundary_repro_delta": 0.125,
   "anchors": [],
   "delta_rho_ge_100": false
  },
  "got_auditor_recompute": {
   "scan_coexist_records": 8,
   "boundary_kind": "open",
   "boundary_estimate": -2.75,
   "first_unstable": null,
   "note": "交付 summary.json 的 interval/-2.875 不可由 repro_cmd 重建: G_GRID 仅 8 值(run_stability_scan.py:49), 多出的 G=-3.0..-5.0 来自 .scratch/probe2.py"
  },
  "window_reference": "ABSENT — 3D MCMP 稳定窗口无文献/解析参考值挂账"
 },
 "repro_cmd": "cd /root/tp-agent/work/m15361 && PYTHONPATH=src python cases/mcmp_3d_stability/run_stability_scan.py --stage all --workers 8  # 审计实跑 --stage a --workers 4: 8/8 逐位一致",
 "external_anchor": "动量解析 u=g·t=2.0e-2 (tests::test_guo_forcing + 审计探针双向复现, 误差 5.1e-6); 稳定性/Δρ 无外部锚; psi_exp/Carnahan-Starling 尝试仅为 .scratch 探针, 未入 runner",
 "three_piece_verification": {
  "run_artifacts": "PASS(分支 13+8+8+summary 非空) 但主干被门禁滤除, 仅 result.json 留档",
  "benchmark_ref_vs_got": "PARTIAL: 动量层 5.1e-6 达标; 该锚 0 行裁判 + 三条预声明判据全 FAIL",
  "repro_cmd": "PARTIAL: CLI 可执行且 stage-A 逐位复现; 5 条 diverged 侧记录与 summary interval/-2.875 不可由该命令再生"
 },
 "level_rationale": "L3: 主求解器内核真改动(Guo 过注入 1.5000165→1.0000051 实测)+落在默认开启的 dam_break_3d 应用路径(:156 use_guo=True)+公共 API; 另有位级不变向后兼容回归与确定性工装(stage-A 8/8 逐位复现)。不达 L4: 该锚 tensor_benchmark_cases 0 行; 三条预声明判据全 FAIL(boundary delta=0.125 / anchors=[] / Δρ≥100 false); summary.json interval/-2.875 不可由 repro_cmd 再生(审计复跑 open/-2.75); water_air 8/8 diverged@50 判为初值瞬态(iw=2.0 第1步 u_max=0.1067; iw=6.0 同参数 300 步不发散); 交付测试 2/5 红。不达 L5: Δρ≥100 在 linear-ψ 下结构不可达(报告如实 FAIL)。附 106 行 REPORT.md 不改变级别。",
 "findings": [
  "F1 阻塞(平台侧): tensor_benchmark_cases 无 GAP-PP-1-3DMCMP-STABILITY 行(16 行 0 命中) => 无机器裁判, L4 不可达",
  "F2 高: 交付 commit c5e38bf0 处自带测试 2/5 失败; docs/mcmp3d_stability_findings.md §2 'sign convention verified in tests::test_self_coupling_sign_convention' 为不成立声明",
  "F3 高: summary.json 的 interval/-2.875 不可由文档化 repro 重建(G_GRID 8 值 vs 交付 13 条; 多出记录来自 .scratch/probe2.py); 审计复跑得 open/-2.75",
  "F4 高(解释性): water_air 族 8/8 diverged@50 是 tanh 初值瞬态非耦合失稳(iw=2.0 第1步 u_max=0.1067=Ma0.19; iw=6.0 300 步不发散 u_max_peak=0.1568) => 该族表在耦合轴上无信息量, 报告未诊断",
  "F5 中: REPORT.md 扫描表 u_max 列对 diverged 行实填 u_max_peak(工件 u_max=null) => 列名与量错配(数值可溯源)",
  "F6 中: 主干 54a8c646 丢弃全部 artifacts/mcmp_3d_stability/**, 仅 result.json 留档",
  "F7 低: stability_boundary() states 键在缺 name 时退化 str(value)(mcmp3d_stability.py:392), 与测试期望 'G_12=-2.5' 不一致(第二条红测试根因)",
  "F8 信息(正面): Guo 缺陷与修复独立复现(父 fd81edf2 1.5000165 → c5e38bf0 1.0000051), 有效回归测试",
  "F9 信息(正面): stage-A 8/8 逐位复现; 三条判据如实记 FAIL, tensor_reports 14453 无粉饰"
 ],
 "not_run": [
  "依赖面回归全扫描: tests/test_mcmp_mixture_3d.py 本地 115s 超时(rc=124), tests/test_dam_break_3d*.py 未跑",
  "--stage b/c/d 未代跑(交付 anchor 段本身为 [])",
  "linear-ψ 下 Δρ 上限的理论核验(Maxwell 构造)未做",
  "未改被审单任何代码(交差核验单, 非重做)"
 ],
 "scope_note": "被审 m15361(succeeded) commit c5e38bf0(41 files +8535/-74), 落主干 54a8c646 Merge #1802(11 files); 平台报告 tensor_reports id=14453; 本次审计报告 tensor_reports id=14525; 审计只读 /root/tp-agent/work/m15361, 工件在 /root/tp-agent/work/m15457/artifacts/m15457/"
}$json$::jsonb, 'board-audit') ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_level=NULL, gate_note=NULL RETURNING id, repo, scope, level, gate_state;
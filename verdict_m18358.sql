-- m18358 board-audit verdict: m18241 XVAL-LBM-ROUND2-METHOD-SELECTION -> L2
-- idempotent: re-running overwrites the same (repo, scope, verdict_by) row
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18241 XVAL-LBM-ROUND2-METHOD-SELECTION', 18358, 2, $json$
{
 "verdict_level": 2,
 "level_rationale": "L2: real code with a genuine input->compute->output path (new solver API collide_bgk(tau_field), arm harness, aggregation pipeline, outlet-BC instrument, 6 invariant-asserting tests) and the executed rung reproduces to 1e-15 on an independent rerun. NOT L3: zero production callers pass tau_field (grep -rn 'tau_field=' src/ -> only collide_mrt/hb callers); the harness is a standalone xval script, not wired into the main solver/platform runner. NOT L4: 1 of 3 predeclared rungs ran, the decisive A6 factorial cell was never executed, convergence is undefined (all KPIs resolution-unconverged), both OSS-style arms FAIL the owner 15% band on Cl_amp, and tensor_benchmark_cases has 0 rows for this anchor so no machine-adjudicable baseline exists.",
 "impl_files": [
  "/root/tp-agent/work/m18241/src/tensorlbm/solver.py",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/run_arm.py",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/analyze_round2.py",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/run_round2.sh",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/oss_outlet_bc_xval.py"
 ],
 "impl_files_detail": [
  "/root/tp-agent/work/m18241/src/tensorlbm/solver.py",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/run_arm.py",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/analyze_round2.py",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/run_round2.sh",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/oss_outlet_bc_xval.py"
 ],
 "entry": "/root/tp-agent/work/m18241/xval/benchmark_method_selection/run_arm.py",
 "entry_detail": "/root/tp-agent/work/m18241/xval/benchmark_method_selection/run_arm.py",
 "run_artifacts": [
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A1_D10/result.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A2_D10/result.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A4_D10/result.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A3_D10/fail.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A5_D10/fail.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A1_D14/state.npz",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/table_round2.csv",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/summary_round2.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/predeclared_round2.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/oss_outlet_bc_xval.json",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/rerun_A2_D10_result.json",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/fig_m18358_verdict.png"
 ],
 "run_artifacts_detail": [
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A1_D10/result.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A2_D10/result.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A4_D10/result.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A3_D10/fail.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A5_D10/fail.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/runs/A1_D14/state.npz",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/table_round2.csv",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/summary_round2.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/predeclared_round2.json",
  "/root/tp-agent/work/m18241/xval/benchmark_method_selection/oss_outlet_bc_xval.json",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/rerun_A2_D10_result.json",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/fig_m18358_verdict.png"
 ],
 "tests": [
  "tests/test_xval_round2_method_selection.py::test_arm_table_is_the_predeclared_factorial",
  "tests/test_xval_round2_method_selection.py::test_bgk_tau_field_matches_explicit_cellwise_relaxation",
  "tests/test_xval_round2_method_selection.py::test_bgk_tau_field_shape_is_checked",
  "tests/test_xval_round2_method_selection.py::test_schedule_is_arm_independent_and_monotone_in_D",
  "tests/test_xval_round2_method_selection.py::test_analyze_round2_aggregation_on_a_synthetic_matrix",
  "tests/test_xval_round2_method_selection.py::test_diverging_arm_is_reported_not_dropped"
 ],
 "tests_detail": [
  "tests/test_xval_round2_method_selection.py::test_arm_table_is_the_predeclared_factorial",
  "tests/test_xval_round2_method_selection.py::test_bgk_tau_field_matches_explicit_cellwise_relaxation",
  "tests/test_xval_round2_method_selection.py::test_bgk_tau_field_shape_is_checked",
  "tests/test_xval_round2_method_selection.py::test_schedule_is_arm_independent_and_monotone_in_D",
  "tests/test_xval_round2_method_selection.py::test_analyze_round2_aggregation_on_a_synthetic_matrix",
  "tests/test_xval_round2_method_selection.py::test_closure_H1_and_factorial_flags (aggregation/H1 closure assertions, same file)"
 ],
 "integration_callers": "无生产调用方。grep -rn 'tau_field=' src/（除 solver.py）仅命中 collide_mrt / collide_hb_bgk_forced / collide_powerlaw_bgk_forced 的既有调用者（airfoil_benchmark.py:417, airfoil_polar.py:446, frazil_lbm.py:453/464, herschel_bulkley_lbm.py:565/666 等），没有任何调用方给 collide_bgk 传 tau_field；该能力当前只被 xval/benchmark_method_selection/run_arm.py 使用",
 "benchmark": {
  "tensor_benchmark_cases_rows_for_anchor": 0,
  "sql": "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ROUND2-METHOD-SELECTION' -> 0（TensorLBM 全仓 26 个 anchor 中无 method-selection 相关条目）",
  "declared_criterion": "owner 15 %（异于平台的 3 % 主判据），ref = A1（本仓记录的 MRT+sponge 选型，内部基线，非外部基准）",
  "executed_rungs": "1 / 3 (D=10 only)",
  "gate_D10": {
   "A2": false,
   "A4": false,
   "A6": null
  },
  "kpi_vs_A1_pct": {
   "A2": {
    "cd_mean": -9.179693792926308,
    "st_fft": 4.516952960575163,
    "cl_amp": -46.56648513639526
   },
   "A4": {
    "cd_mean": 1.7366537537064668,
    "st_fft": 1.6399858325812633,
    "cl_amp": -15.396855863488955
   }
  },
  "gate_conclusion": "两个 OSS 风格臂在 cl_amp 上越 15 % 带（A2 -46.57 %、A4 -15.40 %）=> 选型问题未被裁定；A6（BGK+sponge）缺失使 collision/sponge 交互项无法估计",
  "code_level_xval": {
   "ref": "executed OSS source Maarten-vd-Sande/lbm @573a01c4",
   "K1_rel_L2": 1.18e-07,
   "K2_rel_L2": 2.71e-08,
   "verdict": "PASS <3%（float32 舍入级）——这是本轮唯一真正达标的对标，但它是 kernel 级、第一轮成果的复验"
  },
  "auditor_independent_repro": {
   "arm": "A2",
   "D": 10,
   "cd_mean_delta": 6.66e-15,
   "st_fft_delta": 2.78e-17,
   "cl_amp_delta": 0.0,
   "verdict": "KPI 逐位级可复现"
  },
  "auditor_rerun_arm": "A2",
  "auditor_rerun_D": 10,
  "auditor_rerun_KPI_delta_max": 7.33e-15,
  "auditor_rerun_wall_s": 357.3,
  "recorded_wall_s": 6.4,
  "wall_ratio": 55.83,
  "round2_gate_state": "D=10 both OSS-style arms FAIL on cl_amp; D=14/D=20/A6 not_run"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18241 && PYTHONPATH=src python3 -m pytest tests/test_xval_round2_method_selection.py -q  # 6 passed in 4.4 s 实测\ncd /root/tp-agent/work/m18241 && PYTHONPATH=src python3 xval/benchmark_method_selection/run_arm.py --arm A2 --D 10 --threads 8 --out /tmp/rerun_A2b --no-resume  # 审计实测 357.3 s，KPI 与 runs/A2_D10/result.json 差 <=7.3e-15\ncd /root/tp-agent/work/m18358 && python3 artifacts/audit_m18358/make_fig_m18358.py  # 生成审计图\n# D=14/D=20/A6 需要集群：run_round2.sh 的提交 cj1847/cj1848/cj1849 至今 queued（审计时 cj1848 排队 45.2 min）",
 "external_anchor": [
  "/root/oss-cache/Maarten-vd-Sande_lbm/lbm/streamcollide.py"
 ],
 "external_anchor_detail": [
  "/root/oss-cache/Maarten-vd-Sande_lbm/lbm/streamcollide.py"
 ],
 "findings": [
  {
   "id": "F1",
   "severity": "blocking-for-provenance",
   "claim": "round-2 交付物从未进入任何 commit",
   "evidence": "cd /root/tp-agent/work/m18241 && git log --all --oneline -- docs/XVAL_LBM_ROUND2_METHOD_SELECTION.md xval/benchmark_method_selection/summary_round2.json -> 空；git status 显示 M src/tensorlbm/solver.py + 13 个 ?? 未跟踪文件；mission 记录 head_commit=null；m18358 主树（main 基）无 round-2 任何文件",
   "impact": "证据只活在 m18241 工作树里（worktree_recyclable=f），一旦回收则 predeclared/harness/tests/instrument 全丢；也解释为何 succeeded 状态下平台没推分支"
  },
  {
   "id": "F2",
   "severity": "major",
   "claim": "基准未跑完：3 个预声明分辨级只跑 1 个，决定性 A6 因子格缺失",
   "evidence": "summary_round2.json::gate_at_each_D（14/20 全 null）、runs/A{1,2,4,6}_D14 只有 state.npz、logs/*.log 0 字节、runs/A6_D10 不存在",
   "impact": "ALT-A/B/C 三假设无法区分 => 选型决定 technically undecided（报告如实写了，未粉饰）"
  },
  {
   "id": "F3",
   "severity": "major",
   "claim": "无平台登记基准 => 按机器判据不可能到 L4",
   "evidence": "SELECT count(*) … anchor_slug='XVAL-LBM-ROUND2-METHOD-SELECTION' -> 0；README/报告里的 15 % 是 owner 口径、ref=A1 内部基线",
   "impact": "级别只能由工件自证；gate 无法机器复核数值"
  },
  {
   "id": "F4",
   "severity": "minor",
   "claim": "table_round2.csv::wall_s / result.json::ms_per_step 是 chunk 局部耗时，被当作整臂成本呈现",
   "evidence": "记录 wall_s=6.4（ms_per_step=0.301 = 6.4*1000/21259）；审计独立整臂重跑 357.3 s（16.808 ms/step）=> 55.8x；run_arm.py:158/243-284 分块 resume 后 t0 按块重置",
   "impact": "不影响 KPI；但性能列不可引用"
  },
  {
   "id": "F5",
   "severity": "minor",
   "claim": "K3 出口仪器 controls_ok=false，但其 R 值仍写进 oss_outlet_bc_xval.json 顶层 reflection_coefficient",
   "evidence": "oss_outlet_bc_xval.json::controls_ok=false, incident_peak_spread=1.713, noise_floor_rel_max=0.601",
   "impact": "报告明确声明不用作证据（第 8 节第 3 条），但下游若只读 JSON 会误用"
  },
  {
   "id": "F6",
   "severity": "info",
   "claim": "4-A 的 D=10 三行是第 1 轮工件再聚合，非本轮新跑",
   "evidence": "runs/A{1,2,4}_D10/result.json mtime=12:55（检出时刻，且为 tracked 的第 1 轮文件）；报告 4-A 小字已注明'unchanged from round 1, recomputed by the round-2 aggregator'",
   "impact": "透明，不构成夸大"
  }
 ],
 "checks_run": [
  "psql tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ROUND2-METHOD-SELECTION' -> 0 行（并列出 TensorLBM 全部 26 个 anchor 作反证）",
  "PYTHONPATH=src pytest tests/test_xval_round2_method_selection.py -q -> 6 passed in 4.39 s",
  "csv <-> result.json 逐字段比对：3 臂 × 5 数字全 match（<=1e-12 相对）",
  "独立重跑 arm A2 @D=10（8 threads, 357.3 s）-> cd_mean/st_fft/cl_amp 差 6.7e-15/2.8e-17/0.0",
  "sha256sum 5 个 OSS 源文件 + git rev-parse HEAD 对 /root/oss-cache/Maarten-vd-Sande_lbm",
  "grep -rn 'tau_field=' src/ 无 collide_bgk 生产调用方",
  "git log --all 查 round-2 文件是否进过 commit -> 无",
  "run_cluster ps cj1848 -> queued 45.2 min；nodes -> n315/n330 ok，load 40.5/41.7（not_run 理由属实）",
  "审计自身命中控制面 120 s 命令上限 2 次（TIMEOUT after 120s），复现了报告的 friction 陈述"
 ],
 "checks_not_run": [
  "A1/A4 独立重跑（各 ~70 s 记录的整臂实为 ~350 s 量级，本窗口未排队）",
  "D=14/D=20 与 A6 的补跑（作业 cj1847/cj1848/cj1849 至今 queued，未 fetch）",
  "K3 出口仪器复算（controls 已 FAIL，复算无判定价值）",
  "业务级外部真值（Cd/St 文献值）核对——本锚点无此登记"
 ],
 "overstated_claims": [
  "mission 状态 succeeded + headline「交付」易被读作任务达成；实质是有 device 的 honest-FAIL：选型问题明写 not adjudicated（这是报告内部诚实，但状态字段与 headline 的措辞偏乐观）",
  "table_round2.csv 的 wall_s 列暗示单臂成本（真实值约 56 倍）"
 ],
 "disposition": "request changes：级别判 L2（能力真实但未集成、未达基准门）。返修清单=①把 round-2 全部工件（含 solver.py diff）提交成 commit；②集群排队恢复后补 cj1847/48/49（A6@D=10 + D=14 + D=20），跑 analyze_round2.py 填 A6 行与 ladder；③把 15 % 门与 A1 基线的 KPI 登记进 tensor_benchmark_cases（case_name=method-selection-D{10,14,20}, metric=cd_mean/st_fft/cl_amp, tolerance=15）；④wall_s 改累计口径或改名 chunk_wall_s。完成后可重新申报 L4。",
 "audit_artifacts": [
  "/root/tp-agent/work/m18358/docs/AUDIT_M18358_XVAL_LBM_ROUND2.md",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/verdict_evidence.json",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/fig_m18358_verdict.png",
  "/root/tp-agent/work/m18358/artifacts/audit_m18358/report_m18241_snapshot.md"
 ]
}
$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET
  mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
  evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL;

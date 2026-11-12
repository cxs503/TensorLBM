INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm15777 CASE-CYLINDER-DRAG-RE20', 15882, 3, $json$
{
  "audited_mission": 15777,
  "audited_commit": "f2de132a (PR #1851, merged；ECO/TensorLBM HEAD 53046bcc 已含)",
  "audited_worktree": "/root/tp-agent/work/m15777",
  "anchor_slug": "CASE-CYLINDER-DRAG-RE20",
  "impl_files": [
    "src/tensorlbm/cylinder_st2d1.py",
    "benchmarks/case_pool/cylinder_drag_re20/run_case.py",
    "app/backend/routers/cylinder_bench.py",
    "benchmarks/case_pool/cylinder_drag_re20/refs.json"
  ],
  "impl_notes": [
    "src/tensorlbm/cylinder_st2d1.py:80-103 GEOMETRY/REFERENCE（D=0.1,lx=2.2,ly=0.41,圆心(0.2,0.2),Re=20,nu=1e-3；Cd=Fx/(0.5 rho U_mean^2 D_phys)）",
    "src/tensorlbm/cylinder_st2d1.py:152-353 run_cylinder_st2d1：torch D2Q9 + MRT + Zou-He 抛物入口/定压出口 + 半程反弹 + Ladd 动量交换；plateau 停机 + 半窗漂移；Ma 护栏真实生效",
    "src/tensorlbm/cylinder_st2d1.py:355-390 compare_to_reference（gating/non-gating 分离，stationary 才可 PASS）",
    "app/backend/routers/cylinder_bench.py:113-155 POST /st2d1/validate 委派库入口；src/tensorlbm/__init__.py:889-893 lazy exports"
  ],
  "entry": "benchmarks/case_pool/cylinder_drag_re20/run_case.py",
  "entry_note": "run_case.py:main() → run_cylinder_st2d1(D_cells,steps,tau,u_lb,avg_window,stop_on_plateau,min_steps) → compare_to_reference → results_D{D}.json/series_D{D}.csv/fields_D{D}.npz/summary.json",
  "run_artifacts": [
    "runs/audit_m15882/audited_bundle/results.json",
    "runs/audit_m15882/audited_bundle/series.csv",
    "runs/audit_m15882/audited_bundle/fields.npz",
    "runs/audit_m15882/audited_bundle/SHA256SUMS",
    "runs/audit_m15882/audited_bundle/D16.log",
    "runs/audit_m15882/audited_bundle/D20.log",
    "runs/audit_m15882/independent_recompute.json",
    "runs/audit_m15882/pytest.log",
    "runs/audit_m15882/endpoint_probe.log",
    "runs/audit_m15882/audit_cd_convergence.png",
    "runs/audit_m15882/audit_field_cloudmap.png",
    "runs/audit_m15882/audit_ref_vs_got.png",
    "docs/audits/CASE-CYLINDER-DRAG-RE20-m15882-audit.md"
  ],
  "artifact_notes": [
    "被审工件原件（m15777 树）：runs/cylinder_drag_re20/D16/{results.json 2006 B, series.csv 116613 B, fields.npz 302567 B} + D16.log + D20.log；审计侧已原样快照到 runs/audit_m15882/audited_bundle/ 并附 SHA256SUMS（results.json 55185852…, series.csv 7e3d5061…, fields.npz 3bb39caf…, D16.log 761ad459…, D20.log 95020e46…）",
    "D16/results.json：cd=5.872258219122886, ref_cd=5.57953523384, err_cd_pct=+5.2464, stationary=true, cd_plateau_drift=1.79e-4, n_steps=36000, wall_s=641.6, verdict.passed=false（诚实自报）",
    "series.csv 3600 行 = 每 10 步抽取 → 本方独立重算末 800 采样均值 5.8722605 → +5.24641%，与自报差 2.26e-6 / 4.1e-5 pp",
    "fields.npz 键 ux,uy,rho,vorticity,solid,dx → 本方重算散度 rms 1.07e-3 / p99 6.3e-3 / max 1.2e-2（Ma²=7.5e-3 量级，真解非占位）",
    "provenance 断裂：磁盘名 results.json/series.csv/fields.npz，而 run_case.py:117-135 契约是 results_D16.json/series_D16.csv/fields_D16.npz 且必写 summary.json（缺失）→ 交付工件由会话内未入库驱动产出；runs/ 被 gitignore 未入库",
    "D20 仅 674 B 日志（第 20000 步被 kill，Cd~5.8239，drift 3.55e-3 未到平台），无 JSON/npz"
  ],
  "tests": [
    "tests/test_cylinder_st2d1.py 7 条全量实跑：7 passed, 1 warning in 204.85s（runs/audit_m15882/pytest.log）——优于被审单自述『仅前 5 条跑过』"
  ],
  "integration_callers": "app/backend/routers/cylinder_bench.py:113-155 POST /api/cylinder-bench/st2d1/validate 委派 tensorlbm.cylinder_st2d1.run_cylinder_st2d1；app/backend/main.py:37,200 注册 router；本方 fastapi.testclient 实跑 → status 200、返回真解 cd=8.5862（D=12,1200 步, stationary=false, passed=false, Ma=0.2887）（runs/audit_m15882/endpoint_probe.log）；包级 lazy export 4 项链路真实",
  "integration_note": "无 CI 守护调用者（.github/ 无 case_pool 引用）→ 集成属『应用路由实跑 + 包导出』，非 CI 门；m15411 先例只做到『注册+复用原语』，本单更进一步（端点可执行）",
  "benchmark": "FAIL：Cd_got=5.872258219122886 vs Cd_ref=5.57953523384 → 误差 +5.2464% > 容差 5%（工件自报 pass=false；本方从 series.csv 独立重算 +5.24641%，差 4e-5 pp）。CL=-0.031230 vs +0.010619 → -394.1%（非 gating，符号反，自报）。裁判缺席：SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CYLINDER-DRAG-RE20' → 0（全表 25 行无任何 cylinder 锚）→ L4 结构上不可由机器裁定；同锚点 m15411（verdict id=368）同因判 L3/reopened",
  "repro_cmd": "refs.json:repro_cmd = 'PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20/run_case.py --grids 20 40 --out runs/cylinder_drag_re20' → 不可复现交付工件：默认 tau=0.8 在 D=16 给 u_mean=0.125，而工件 lattice_units 记录 u_mean=0.05/tau=0.62（只能来自 --u-lb 0.05，该命令行未记录于任何 refs/doc）；且 D=40/300k 步在本机 CPU（128 核 load≈400）不可行。refs.json:quick_cmd（--grids 8 --steps 300）本方实跑 rc=1 → ValueError: lattice Mach 0.433 > ma_limit 0.300 at D_cells=8",
  "external_anchor": "Schäfer & Turek 1996 NNFM 48 'Benchmark computations of laminar flow around a cylinder' 2D-1 (Re=20)：ref Cd=5.57953523384 / Cl=0.010618948146，来源 FeatFlow 官方谱解（Nabh 1998），仓内 benchmarks/verified/cylinder/re20_st/result.json（sha256 df69f767ec9dec6db64169872e7a4a437de57077972bc76aab26ba9f3ee59eda，本方实测一致）；几何自检 blockage=0.24390243902439（0.1/0.41）、nx×ny=352×66、Re 重算 20.0、Ma=0.0866、固壁 197 格（πr²=201）全部自洽",
  "level": 3,
  "level_rationale": "L2 成立：真代码 input→compute→output（451 行 torch MRT 求解器），交付 cd 可由 series.csv 独立重算（差 2.26e-6），散度 O(Ma²)，无硬编码/无读 JSON 冒充计算。L3 成立：新增 POST /api/cylinder-bench/st2d1/validate 委派库入口并经 testclient 实跑 200 返回 ref-vs-got 判定。L4 不成立（六条独立理由）：①唯一交付网格 D=16 Cd +5.246% 超 5% 门；②tensor_benchmark_cases 无案号→唯一机器裁判缺席；③单网格交付、任务书要求的 ≥2 点收敛/敏感性未做，D=20 探针 drift 3.55e-3 未到平台不可 PASS；④交付工件不能由声明 repro_cmd 复现（参数档不同 + quick_cmd 开箱崩 + 文件名不符入口契约 + 缺 summary.json）；⑤六段报告缺席（仓内 md 缺、make_report.py 未执行、tensor_reports 无 15777 行）；⑥二阶量 Cl 符号反 -394% 且随网格仍在变（未修 m15411 同缺陷）。仓内既有 GPU 锚点（D=40 -0.558%/D=80 -2.875%）属历史佐证、非本单新计算；本模块在已验证档 tau=0.8/D≥40 上从未跑过 → 不得据先例借级别。",
  "verdict_summary": "三件套：run_artifacts 存在但 provenance 断（非入口产出、缺 summary.json、D20 仅日志）；ref vs got FAIL +5.246%>5%（诚实自报并经独立重算证实）；repro_cmd 不可执行/不可复现（quick_cmd 开箱 rc=1；repro_cmd 参数档≠交付档）。裁判 tensor_benchmark_cases 0 行；报告六段缺席。判级 L3（真实现+真集成，L4 六缺口未闭）。",
  "findings": [
    "F1 high/blocking-L4: 唯一交付网格 D=16 Cd=5.87226 → +5.2464% > 5% 门（本方独立重算 +5.24641%，差 4e-5 pp）；自报 pass=false 诚实",
    "F2 high: tensor_benchmark_cases 无 CASE-CYLINDER-DRAG-RE20 行（0/25）→ 平台裁判缺席，L4 结构上不可裁定（同锚点 m15411 同因 reopened）",
    "F3 high: repro 三缺陷——quick_cmd 实跑 rc=1（ValueError Ma 0.433>0.3）；repro_cmd 默认 tau=0.8 与交付工件 tau=0.62/--u-lb 0.05 不同档；D=40 档本机 CPU 不可行",
    "F4 high: 工件文件名/结构不符入口契约（results_D16.json 与 summary.json 缺）、runs/ 未入库、D20 无工件 → 交付工件来源不可由仓内代码追认",
    "F5 medium-high: 六段报告缺席（docs/CASE_CYLINDER_DRAG_RE20.md 不存在；docs/assets/case_cylinder_drag_re20/ 空；make_report.py 从未执行；tensor_reports 无 mission 15777 行）",
    "F6 medium: 台账口径不一致——tensor_dev_missions id=15777 status=succeeded/accepted=true 且顶层 fail_stage 空，而 result.fail_stage='budget_exhausted+forced_finish'、headline 自述半成品、report_owed=false",
    "F7 medium: 单网格交付，任务书⑤（≥2 点收敛/敏感性）未满足；D20 第 20000 步 drift 3.55e-3 未到平台 → 不可 PASS",
    "F8 medium: Cl 符号反 -394%，D 增时仍在变；m15411 同一缺陷未被本返修修复",
    "F9 medium-low: API 默认档跑在容差之外——ST2D1Request(D=12,steps=4000)+tau=0.8 → u_mean=0.1667, Ma=0.2887（距 0.3 护栏 3.8%，Ma²≈8%≫5%），实跑 cd=8.5862 瞬态",
    "F10 low: series.csv 每 10 步抽取且无 header 注明，avg_window 以『步』计而文件是『采样』→ 直读重算会错（本方首轮即踩）",
    "F11 low: fields.npz 固壁格点速度未清零（max|u|=4.6e-3 ≈ 9% u_mean），污染图与散度统计"
  ],
  "overstated_claims": [
    "m15777 headline 与 result 自述『代码+工件齐』——工件齐但 provenance 断（F4）且报告缺席（F5），『齐』仅限 D16 一个目录",
    "任务书要求三件套，平台按 succeeded 收账，而 repro_cmd 一项实为不可执行（F3）",
    "『复用 TensorLBM 已验证能力』易被读成『本单达到已验证档』——实际本单跑的是从未验证的 tau=0.62/D=16 档；已验证档（tau=0.8/D=40,80）本模块一次未跑"
  ],
  "not_run": [
    "未跑 tau=0.8 的 D=40/D=80 主档（本机 128 核 load≈400，CPU 单档预估≫1h；本单定位核验非重做）→ 不宣称该模块在已验证档的 Cd",
    "本方网格敏感性独立探针（run_case.py --grids 20 --u-lb 0.05 --stop-on-plateau）被沙箱回收，未产出结果（runs/audit_m15882/D20/ 空）",
    "未核 GPU/CUDA 路径（本机 cuda_avail=False）",
    "未修改任何被审代码（审计只读）",
    "未核 vorticity 后处理公式本身（仅用其输出做量级检查）"
  ],
  "scope_note": "审计单 m15882（board-audit，非重做）。审计产物：docs/audits/CASE-CYLINDER-DRAG-RE20-m15882-audit.md、scripts/audit/m15882_cylinder_audit.py、runs/audit_m15882/{independent_recompute.json,pytest.log,endpoint_probe.log,audited_bundle/,audit_cd_convergence.png,audit_field_cloudmap.png,audit_ref_vs_got.png}",
  "recommended_action": "返修（L4 六缺口）：(1) anchor 注册进 tensor_benchmark_cases（metric=Cd, tolerance=0.05, repo=TensorLBM, ref_source=仓内锚点 sha256 df69f767…）；(2) 主档改跑 tau=0.8 的 D=40（已知 -0.558%）+ D=80（-2.875%）走 run_cluster，D=16/tau=0.62 降为趋势低档；(3) 工件改由 run_case.py 产出（含 summary.json），series.csv 写 header（含抽帧步长）；(4) refs.json quick_cmd 补 --u-lb 0.05 并加 CI 冒烟测试；(5) 执行 make_report.py 并 POST 平台报告，figures 内嵌 png_base64；(6) Cl 要么给收敛研究，要么在 refs.json 明确列为非验证量；(7) 修 API 默认档（D≥20 或显式 u_lb），避免默认请求在 Ma=0.29 下判 5% 门"
}
$json$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level = EXCLUDED.level, evidence = EXCLUDED.evidence,
  mission_id = EXCLUDED.mission_id, gate_state = 'pending', gate_level = NULL, gate_note = NULL
RETURNING id, repo, scope, level, mission_id, gate_state;

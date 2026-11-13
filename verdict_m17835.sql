-- m17876 交差核验 verdict：对象 m17835 XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL (TensorLBM)
-- 判级 L2：真功能实现（可执行、可逐位复现）；XVAL 主张 FAIL（max 54.65% / L2 18.55% > 15% 判据），零集成/零测试/零入库
-- 机器读数来源：artifacts/m17835_audit/audit_m17835.json（docs/audit_m17835/collect_evidence.py 生成）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17835 XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL', 17876, 2, $json${
 "verdict_basis": "L2：功能实质=独立的 D2Q9 定核 conv/roll 输运 + BGK 碰撞 + bounce-back + 求解器 + 两算例驱动，真运算、可执行、工件可逐位复现；但 XVAL 结论 FAIL——唯一量化算例超容差 (max 54.65% / L2 18.55% vs 判据 15%)，第二算例 LDC 无工件，零测试、零集成、零 tracked 改动。",
 "claimed": "m17835 result.headline=L2 delivered (impl+entry+run_artifacts)；doc §9 自评 'correctly implemented / physically reasonable'，L4 标为 done",
 "delivered_level": 2,
 "gap_to_claim": "L4 未达成：benchmark 自身判据 FAIL（38/62 内部结点 >15%，max 54.65%）；conv 路径质量外泄 18.56%/步，doc 'exact TF parity / mathematically identical' 被自身代码否定；LDC 半边（doc §4.2 'pending from cluster job cj1700'）无任何工件；交付 0 commit / 3 文件 untracked / runs 被 gitignore。",
 "impl_files": [
  "/root/tp-agent/work/m17835/src/tensorlbm/conv_streaming_fixed_kernel.py:1-392 (D2Q9_CX/CY/W; build_stream_kernels:63-80; stream_conv:83-87; stream_roll:90-97; collide_bgk:100-117; compute_macroscopic:120-130; apply_lid_bc/apply_wall_bc:136-160; ConvStreamingSolver2D; run_poiseuille:177-262; run_lid_driven_cavity)",
  "/root/tp-agent/work/m17835/src/tensorlbm/conv_streaming_fixed_kernel.py:run_poiseuille 末尾 — 缺陷点：converged = max_error_pct < 15.0（精度判据冒充收敛判据）",
  "/root/tp-agent/work/m17835/src/tensorlbm/conv_streaming_fixed_kernel.py:stream_conv — 缺陷点：F.conv2d(padding=1) 默认零填充，非周期填充，边界质量外泄",
  "/root/tp-agent/work/m17835/src/tensorlbm/conv_streaming_fixed_kernel.py:analytical_ux = F/(2ν)·j·(H−j), H=ny=64 而 j∈[0,63] ⇒ 参考解 off-by-one"
 ],
 "entry": "/root/tp-agent/work/m17835/examples/conv_streaming_xval.py（argparse --nx/--ny/--steps/--tau/--u-lid/--output；审计侧 32×32×300 冒烟 rc=0）",
 "run_artifacts": [
  "/root/tp-agent/work/m17835/runs/conv_streaming_poiseuille.json （exists=True；keys=['framework', 'lattice', 'method', 'oss_reference', 'poiseuille']；poiseuille keys=['analytical_max_ux', 'analytical_ux', 'converged', 'l2_error_pct', 'max_error_pct', 'numerical_max_ux', 'numerical_ux']；shipped max=54.649559% l2=18.550359% converged=False）",
  "/root/tp-agent/work/m17835/docs/xval/conv_streaming_fixed_kernel.md（§4.2 LDC 'Results pending from cluster job cj1700'；§7 所列 runs/conv_streaming_xval_full.json 不存在）",
  "/root/tp-agent/work/m17835/runs/conv_streaming_xval_full.json — 不存在（doc §7 声称的工件缺席）",
  "本单读数：artifacts/m17835_audit/audit_m17835.json；docs/audit_m17835/fig_m17835_evidence.png"
 ],
 "entry_vs_artifact": {
  "shipped_schema_matches_entry": false,
  "shipped_has_ldc_block": false,
  "note": "交付 json 缺 config/wall_clock_s/lid_driven_cavity ⇒ 工件非交付入口产出（数值仍可经模块 API 复现）"
 },
 "tests": "无：tests/ 对 conv_streaming_fixed_kernel 零命中（grep -rln tests/ = 0 命中）；交付未带任何测试",
 "integration_callers": "无：src/ 与 benchmarks/ 引用数=0，仅 examples/conv_streaming_xval.py:19 一处 import；git rev-list --count be78a87b..HEAD = 0（零 tracked 改动）",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "tensor_benchmark_cases_total": 65,
  "tensor_benchmark_anchors_total": 59,
  "tensor_xval_cases_rows_for_anchor": 0,
  "declared_criterion_pct": 15.0,
  "internal_target_pct": 3.0,
  "case": "Poiseuille 64x8, tau=1.0, nu=1/6, F=1e-5, 5000 steps, float64",
  "ref": "analytic u(y)=F/(2ν)·y·(H−y), H=64 ⇒ u_max=0.030720",
  "got_umax": 0.025909999524196908,
  "umax_err_pct": 15.657553632171537,
  "max_err_pct": 54.64955859108693,
  "l2_err_pct": 18.550358759828203,
  "margin_over_15pct_tolerance": 3.643303906072462,
  "nodes_over_15pct": 38,
  "nodes_over_3pct": 62,
  "interior_nodes": 62,
  "wall_err_pct": {
   "j1_bottom": 10.738813734837759,
   "j_last_top": 98.9417989417998
  },
  "profile_asymmetry_pp": 43.91074485624917,
  "shipped_converged_flag": false,
  "converged_flag_source": "max_error_pct < 15.0（非收敛判据）",
  "ldc_case": "无工件；doc §4.2 pending cj1700；审计冒烟 32x32x300 仅证入口可执行"
 },
 "anchor_claim_probe": {
  "claim": "doc §2.1/§5: stream_conv 'exact TF parity'，stream_roll 'mathematically identical'",
  "interior_maxdiff": 0.0,
  "boundary_total_maxdiff": 0.12108371494636323,
  "mass_in": 50.15886920454808,
  "mass_conv": 40.847534346774665,
  "mass_roll": 50.15886920454808,
  "conv_mass_loss_pct": 18.563685755756882,
  "roll_mass_loss_pct": 0.0,
  "falsified": true,
  "root_cause": "F.conv2d padding=1 零填充 vs torch.roll 周期卷绕；上游靠周期性 mobius padding 才等价",
  "consequence": "anchor 招牌的 conv-as-streaming 路径不可用于闭域（18.56%/步质量外泄），实际求解器全程用 roll ⇒ 交付物≠anchor 技术"
 },
 "ldc_smoke": {
  "center_vortex_ux": -0.00029986071827666286,
  "center_vortex_uy": 0.03866545290270836,
  "final_max_ux": 0.041555079297802076,
  "converged": false
 },
 "repro_cmd": "cd /root/tp-agent/work/m17835 && PYTHONPATH=src python3 -c \"from tensorlbm.conv_streaming_fixed_kernel import run_poiseuille, PoiseuilleConfig; p=run_poiseuille(PoiseuilleConfig(ny=64,nx=8,tau=1.0,body_force=1e-5,n_steps=5000)); print(p.max_error_pct, p.l2_error_pct, p.converged)\"  # 审计实跑逐位复现交付值 54.64955859108693 / 18.550358759828203 / False",
 "repro_verified": true,
 "external_anchor": {
  "repo": "loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow",
  "cited_in": "/root/tp-agent/work/m17835/src/tensorlbm/conv_streaming_fixed_kernel.py:1-20 docstring 与 docs/xval/conv_streaming_fixed_kernel.md §1/§5",
  "executed_by_auditor": false,
  "reason": "TF1.x graph API + Py2 语义，本机不可执行；参照侧改用闭式解析解（上游表格由 m17680 的 extract_oss_tables.py 机读留 sha256，但本单未接棒该分支）"
 },
 "delivery": {
  "commits_ahead_of_base": 0,
  "untracked_deliverables": [
   "?? docs/xval/",
   "?? examples/conv_streaming_xval.py",
   "?? src/tensorlbm/conv_streaming_fixed_kernel.py"
  ],
  "tracked_claimed_files": "",
  "runs_dir_gitignored": ".gitignore:108:runs/\truns/conv_streaming_poiseuille.json",
  "prior_round_commit_in_history": false,
  "prior_round_commit": "725c1992 m17680: matured mission deliverables（agent/m17680，含 tests + predeclared.json + oss_ref 机读表，TG KPI<0.3%）——本返修轮未接棒，整体丢失",
  "platform_report": "tensor_reports WHERE mission_id=17835 → 0 行（report_id 空）"
 },
 "severity": {
  "blocker": [
   "B1 唯一量化算例超容差 max 54.65% / L2 18.55%（判据 15%），38/62 结点越界",
   "B2 交付零入库：0 commit、3 文件 untracked、runs/ 被 .gitignore:108 忽略"
  ],
  "major": [
   "M3 conv 路径边界质量外泄 18.56%/步，'exact TF parity' 自证伪，且求解器未使用 conv",
   "M4 LDC 半套无工件（pending cj1700）",
   "M5 doc §4.1 误差归因与代码矛盾（统计已跳过壁点；实测误差随 y 10.7%→54.7% 单调增、两壁不对称 43.91pp）"
  ],
  "minor": [
   "N6 converged = max_error_pct<15（语义冒充）",
   "N7 解析解 off-by-one（H=64 vs j∈[0,63]）⇒ 顶壁假误差 98.94%",
   "N8 工件 schema ≠ 入口输出（provenance 断裂）",
   "N9 零测试零集成"
  ],
  "process": [
   "P10 返修轮未接棒 m17680（725c1992 非祖先），前轮 10 文件成果整体丢弃"
  ]
 },
 "checks_not_run": [
  "上游 TensorFlow 二进制对标（环境不可执行）",
  "128x128x10000 LDC 全算例对标（被审单无工件；审计侧仅 32x32x300 冒烟）",
  "网格/tau 收敛性研究（被审单未做）",
  "maturity_gate 机器复核（由 gate 自动裁定）"
 ],
 "disposition": "request changes：先入库（接棒 725c1992），再修周期填充/half-way bounce-back/参考解坐标，补收敛性证据与 LDC 工件并登记 tensor_benchmark_cases"
}$json$, 'board-audit');

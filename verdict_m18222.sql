-- m18229 交差核验 verdict: 对象 m18129 XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMIN (TensorLBM)
-- 判级 L2：真实现+逐位等价+解析锚 <3%+审计全量复现；不授 L3（无生产调用方，mission 自决不采纳）/L4（阶梯需 L3）/L5
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18129 XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMIN', 18222, 2,
$json${
 "verdict_basis": "L2：TCLB per-margin-arrays/pointer-shift 传输的真实现（394 行：计划构造 + 切片执行器 + flat-offset 执行器 + 不变量检查器），与生产 gather 传输在 14 个形状/dtype 上逐位相同，物理面 TGV worst 0.621% / Poiseuille 0.058%（均 <3%），审计独立全量重跑 15 个标量相对偏差 0.0e+00（逐位复现）。不授 L3：无生产调用方（mission 自决 D3=False 不采纳），故按阶梯 L4 亦不可达；不授 L5：无新科学、OSS 二进制与 GPU 面 WITHHELD。裁判表该锚点 0 行（无机器裁判）。",
 "claimed_level_implicit": "报告 §4 表逐项 PASS 且附六段（①§1 ②§2 ③§4 ④figures 3 张 ⑤§6 ⑥<15%/<3% 双列明答）",
 "delivered_level": 2,
 "impl_files": [
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/per_margin_pointer_shift.py",
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/xval_tclb_pointer_shift/run_xval.py",
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/xval_tclb_pointer_shift/extract_oss_ref.py"
 ],
 "impl_detail": "per_margin_pointer_shift.py:45-78 margin_bands/_axis_segments（周期 wrap 分带）, :81-104 flat_offset（常数偏移不变量）, :107-176 ShiftCopy/PointerShiftPlan, :178-212 pointer_shift_plan（块数 = 2**|{a:c_a!=0}|，从速度表派生）, :213-231 stream_pointer_shift（切片执行器，生产调用路径）, :232-262 stream_pointer_shift_flat（flat-offset 执行器）, :264-330 check_plan_invariant（划分/常数偏移/对 roll 参考三条逐格核验）",
 "entry": "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/xval_tclb_pointer_shift/run_xval.py",
 "entry_note": "python3 run_xval.py [--quick]；审计在独立 stage 内全量重跑 rc=0，88.1 s，15 个物理标量逐位复现",
 "run_artifacts": [
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/result.json",
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/figures/fig1_tgv_error.png",
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/figures/fig2_equivalence_profile.png",
  "/root/tp-agent/work/m18222/docs/audit_m18222/evidence_snapshot/figures/fig3_transport_cost.png",
  "/root/tp-agent/work/m18222/docs/audit_m18222/fresh.json",
  "/root/tp-agent/work/m18222/docs/audit_m18222/provenance.json",
  "/root/tp-agent/work/m18222/docs/audit_m18222/fig_m18222_evidence.png"
 ],
 "run_artifacts_note": "result.json：bitwise 14 例全真；TGV 6.209416e-03/9.150289e-04；Poiseuille umax 5.776989e-04。fresh.json：审计独立重跑 15 标量 rel_delta 0.0e+00，max_slowdown 11.93x。provenance.json：8/8 文件 sha256 + 9/9 excerpt 逐字命中 @ded67cd768cf。",
 "tests": [
  "tests/test_xval_tclb_pointer_shift.py::test_stream_2d_is_bitwise_identical_to_gather[float32/float64 x (17,23)/(4,4)/(3,5)]",
  "tests/test_xval_tclb_pointer_shift.py::test_stream_3d_is_bitwise_identical_to_gather[float32/float64 x (6,7,8)/(5,5,5)/(3,4,5)]",
  "tests/test_xval_tclb_pointer_shift.py::test_plan_is_a_partition_with_constant_block_offsets（及其余 plan/structure 用例）",
  "审计重跑：17 passed（stage 7.89 s；mission 树 17 passed / 6.41 s）——报告 §Tests 写 '2 failed, 15 passed' 与实际不符，见 F3",
  "审计变异探针 3/3 检出（掉 wrap 块→partition holes；flat_offset+1→max_offset_error=1；去周期→与生产算子差 0.786）"
 ],
 "integration_callers": "none —— solver.stream / solver3d.stream3d_gather 不引用本模块（PMPS 仅被 run_xval.py 与 tests 导入）；mission 自决 D3=False 不采纳（保留生产 gather）。5 个交付路径在 m18129 工作树均为未跟踪文件（HEAD 7d6cd8c7，零提交）⇒ 亦不在 /root/tensor-ecosystem。",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "tensor_benchmark_cases_total": 94,
  "declared_criterion": "<15% (owner 精度标准) / 内部 3%",
  "got_vs_ref": {
   "tgv2d_n32_300_relL2": {
    "ref": "closed-form decay (tgv2d_analytic)",
    "got": "6.209416e-03",
    "err_pct": 0.621,
    "tol_pct": 15.0,
    "pass": true,
    "auditor_rerun_bit_identical": true
   },
   "tgv2d_n64_400_relL2": {
    "ref": "closed-form decay",
    "got": "9.150289e-04",
    "err_pct": 0.092,
    "tol_pct": 15.0,
    "pass": true,
    "auditor_rerun_bit_identical": true
   },
   "poiseuille_umax": {
    "ref": "analytic parabola",
    "got": "5.776989e-04",
    "err_pct": 0.058,
    "tol_pct": 15.0,
    "pass": true,
    "auditor_rerun_bit_identical": true
   },
   "transport_A_vs_B": {
    "ref": "TensorLBM production gather",
    "got": "0 (bitwise, 14 cases 2D+3D x f32/f64)",
    "err_pct": 0.0,
    "tol_pct": 15.0,
    "pass": true
   },
   "oss_binary_vs_tensorlbm": {
    "ref": "CFD-GO/TCLB output",
    "got": "WITHHELD",
    "note": "无 R/nvcc，OSS 侧不可执行；报告 §6 自述 WITHHELD"
   }
  },
  "cost_face": {
   "shipped_128thr_worst": 17.1,
   "auditor_128thr_worst": 17.88,
   "auditor_rerun_driver_worst": 11.93,
   "auditor_1thread": {
    "2d/128/thr1": 0.74,
    "2d/256/thr1": 0.27,
    "3d/48/thr1": 0.18
   },
   "note": "F1：ship 的“port 不利”只在 128 线程成立；1 线程下 port 反而快 1.4-5.5x ⇒ 因果解释（formulation vs per-block kernel launch tax）被高估"
  },
  "referee_gap": "tensor_benchmark_cases 该 anchor 0 行（全表 %POINTER%/%MARGIN% 亦 0）⇒ 无机器裁判，<15% 靠审计自核"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18222 && python3 docs/audit_m18222/audit_run.py --only provenance && python3 docs/audit_m18222/audit_run.py --only tests && python3 docs/audit_m18222/audit_run.py --only mutation && python3 docs/audit_m18222/audit_run.py --only cost && python3 docs/audit_m18222/audit_run.py --only fresh  # 全量驱动 88.1 s；随后 python3 docs/audit_m18222/make_deliverables.py 重生成结论",
 "external_anchor": "CFD-GO/TCLB @ded67cd768cf7e727bd078af139e3ec7895076e5（/tmp/TCLB 克隆，8 文件 sha256 + 9 段 excerpt 逐字命中，审计独立复核 provenance.json）+ 解析锚：Taylor-Green 2D 精确衰减 e^{-2νk²t}、平面 Poiseuille 抛物线 u=y(H-y)a/(2ν)",
 "audit_artifacts": [
  "docs/audit_m18222/AUDIT_m18222.md",
  "docs/audit_m18222/audit_run.py",
  "docs/audit_m18222/audit_summary.json",
  "docs/audit_m18222/provenance.json",
  "docs/audit_m18222/tests.json",
  "docs/audit_m18222/mutation.json",
  "docs/audit_m18222/cost.json",
  "docs/audit_m18222/cost_threads.json",
  "docs/audit_m18222/fresh.json",
  "docs/audit_m18222/fig_m18222_evidence.png",
  "docs/audit_m18222/evidence_snapshot/（受审交付逐字节冻结 + REF.json）",
  "verdict_m18222.sql"
 ],
 "audit_findings": [
  "F1 medium: 成本结论“OSS formulation 在 torch 移植后无优势”只在 128 线程成立；审计 1 线程下 port 更快（3d/48 0.18x、2d/256 0.27x），开销随线程数单调恶化（2d/256: 1.12x@8→9.13x@128）⇒ 是 per-block kernel launch/线程池税，不是 formulation 经济学；D3（本机不采纳）仍成立，泛化结论不成立",
  "F2 medium: 交付 result.json 的 equivalence.short_axis_rejected=False 与交付代码不符（审计重跑 True，n>=3 守卫在），即工件早于最后一次代码修改；其余全部字段逐位复现",
  "F3 medium: 报告 §Tests 写 '2 failed, 15 passed'，实测 17 passed（stage 5.38 s / mission 树 6.41 s）⇒ 报告与自身工件自相矛盾，须重生成",
  "F4 low: 被“bank”的常数偏移不变量只由 flat 执行器与 check_plan_invariant 覆盖；flat_offset 被破坏（M2）对 torch.equal(生产, 切片臂) 不可见",
  "F5 low: 5 个交付路径在 m18129 树未跟踪（零提交），不在任何分支/生态仓 ⇒ 自动推送带不走；审计已逐字节冻结进 evidence_snapshot",
  "F6 info: 该 anchor 在 tensor_benchmark_cases 0 行，无机器裁判（审计外部核对）",
  "F7 info: 文档行号引用为 excerpt 的子/超区间（Global.h.Rt:64-68 vs 58-74），指向的代码正确，非自动生成"
 ],
 "not_run": [
  "TCLB 二进制同算例对照（无 R/nvcc，报告与审计一致判 WITHHELD）",
  "GPU/CUDA 与 SDAA 后端成本测量（容器无 GPU）",
  "多进程/MPI margin exchange 面（未移植，报告 §6 自述）"
 ]
}$json$::jsonb, 'board-audit');

-- m17877 交差核验 verdict: 对象 m17724 XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION (TensorLBM)
-- 判级 L2：真实现+可复现+双路线 <15%；但无 tracked commit/无测试/无 caller/裁判表 0 行/OSS 侧未执行/收敛为单相位采样
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17724 XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION', 17877, 2,
$json${
 "verdict_basis": "L2：SOY 自由能 D2Q9 双分布 LBM 的 PyTorch 真实现，主算例 Δp 与解析 σ/R 相对误差 0.0708%（审计逐位复现 6.671389e-04），R=10 点 0.6434% 亦复现；独立毛细能量路线 σ=0.00971（2.90%）互洽，OSS 压力定义变体下 0.911% 仍 <15% ⇒ 判据成立。但零 tracked 改动、无测试、无 caller、裁判表 0 行、OSS 侧从未执行、收敛为单相位采样（时间带 0.235%-6.835%）⇒ 不授 L3/L4/L5。",
 "claimed_level_implicit": "XVAL 交叉对标通过（报告 ④ 表称 0.07% PASS，②表称与 OSS 'identical'）",
 "delivered_level": 2,
 "impl_files": [
  "docs/audit_m17877/evidence_snapshot/xval_fingering_dynamics.py:67-207 (SOYFreeEnergyLBM：化学势 134-135、压力 136-137、feq 100-119、Guo 源 148-153、周期 stream 158-166、measure 186-207)",
  "docs/audit_m17877/evidence_snapshot/xval_fingering_dynamics.py:74-76 (κ=0.75σξ, a=-2κ/ξ²，与 OSS fingering.py:41-42 同式)",
  "docs/audit_m17877/evidence_snapshot/xval_fingering_dynamics.py:273-342 (主流程：初始化→2000 步→测量→误差判定), 344-382 (收敛研究), 384-424 (τ_f/ξ 敏感性)",
  "docs/audit_m17877/evidence_snapshot/xval_fingering_dynamics.py:536-580 (__main__ 入口；577 写 xval_results.json —— 交付工件 primary_results.json 无脚本生成)",
  "外部锚源码（只读对照）：/tmp/Fingering_dynamics/lattice_boltzmann/fingering.py:41-42,130,149,177-206,233-296 @150a573（与报告引用 commit 150a5739e 一致）"
 ],
 "entry": "python3 xval/xval_fingering_dynamics.py（可执行；审计以只读导入方式重跑主算例 18.6 s。注意入口产物名 xval_results.json ≠ 交付 primary_results.json，见 finding F4）",
 "run_artifacts": [
  "docs/audit_m17877/evidence_snapshot/primary_results.json（dp=6.671389172852038e-4 vs σ/R=6.666666666666667e-4, error=7.0838e-4, pass=true；审计重跑逐位相同）",
  "docs/audit_m17877/evidence_snapshot/fig3_laplace_vs_radius.png（图存在；其收敛数值未随工件交付）",
  "docs/audit_m17877/evidence_snapshot/XVAL_LBM_CUPY_DROPIN_NO_KERNEL_FUSION.md（六段齐：①5-24 ②25-44 ③45-61 ④62-77 ⑤78-91 ⑥92-113）",
  "docs/audit_m17877/audit_rerun.json（审计主算例+R=10 重跑生数据）",
  "docs/audit_m17877/audit_probe2.json（OSS 压力定义变体 + 1200-1999 步收敛时间带）",
  "docs/audit_m17877/audit_summary.json + fig_m17877_evidence.png（机读摘要与对照图）",
  "docs/audit_m17877/evidence_snapshot/SHA256SUMS.txt（受审交付逐字节冻结，sha256 可复核）"
 ],
 "tests": "无。grep -rn 'xval_fingering|SOYFreeEnergy' tests/ src/ benchmarks/ → 0 命中；无 pytest/CI 守护，回归只能靠人工重跑。",
 "integration_callers": "none。git -C /root/tp-agent/work/m17724 status --short → 3 行 '??'（docs/…md, xval/artifacts/, xval/xval_fingering_dynamics.py）；HEAD=9bcc6db6 无本单 commit（head_commit=null, branch=null）；src/ 无引用；唯一库耦合为 from tensorlbm.cac_lbm2d import CAC2D（只读导入，实测可导）⇒ L3 不达。",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "tensor_benchmark_cases_total": 65,
  "distinct_anchors": 59,
  "declared_criterion": "<15% (owner 精度标准 / mission goal)",
  "got_vs_ref": {
   "laplace_dp_R15_T2000": {
    "ref": "σ/R = 6.666667e-04",
    "got": "6.671389e-04",
    "err_pct": 0.070838,
    "tol_pct": 15.0,
    "pass": true,
    "auditor_bit_identical_to_artifact": true
   },
   "laplace_dp_R10_T800": {
    "ref": "1.000000e-03",
    "got": "1.006434e-03",
    "err_pct": 0.643427,
    "tol_pct": 15.0,
    "pass": true,
    "note": "文档⑤声明值；工件中无此条，审计重跑复现"
   },
   "laplace_dp_R15_OSS_pressure_definition": {
    "ref": "6.666667e-04",
    "got": "6.727387e-04",
    "err_pct": 0.910811,
    "tol_pct": 15.0,
    "pass": true,
    "note": "把压力改成 OSS 原生 p=ρ/3+ψμ（fingering.py:130）后重跑；u_max 由 9.64e-6 降到 6.40e-7"
   },
   "time_band_steps_1200_1999": {
    "err_pct_min": 0.235447,
    "err_pct_mean": 2.667041,
    "err_pct_max": 6.835208,
    "umax_range": [
     1.089425535999392e-06,
     3.759170803261695e-05
    ],
    "note": "交付 0.0708% 是单相位采样；时间带内最大 6.8% 仍 <15%，但 'monotone/true equilibrium' 主张不成立"
   },
   "sigma_independent_route": {
    "ref_sigma": 0.01,
    "sigma_from_capillary_energy": 0.009710074,
    "dev_pct": 2.8993,
    "route": "σ = ∮κ|∇ψ|² dA / (2πR)（自由能泛函，不经压力场）"
   }
  },
  "referee_gap": "tensor_benchmark_cases 无该锚点登记行（0/65）⇒ 平台裁判表缺席，本单判据靠自声明 σ/R + 审计独立重算兜底",
  "oss_side_numbers": "none —— OSS 仓从未被执行，无任何 OSS 输出数值；XVAL 为单向（读码+重写+解析解检验）"
 },
 "repro_cmd": "cd /root/tp-agent/work/m17877 && python3 docs/audit_m17877/audit_run.py (主算例 18.56 s + R=10 7.88 s) && python3 docs/audit_m17877/audit_probe2.py (P1 OSS 压力定义 + P2 收敛时间带，51 s) && python3 docs/audit_m17877/gen_fig_summary.py",
 "external_anchor": "OSS ebinan92/Fingering_dynamics@150a573 lattice_boltzmann/fingering.py（实机克隆 /tmp/Fingering_dynamics，逐项核对 μ/κ-a/feq/geq/Guo/∇ψ/∇²ψ 模板均对应；压力公式不符见 F1）+ Laplace 定律 Δp=σ/R（σ 由 κ=0.75σξ 自校准，见 F8）+ 审计独立毛细能量路线",
 "audit_artifacts": [
  "docs/audit_m17877/AUDIT_m17877.md",
  "docs/audit_m17877/fig_m17877_evidence.png",
  "docs/audit_m17877/audit_summary.json",
  "verdict_m17877.sql"
 ],
 "audit_findings": [
  "F1 high: 报告② 'Pressure identical' 不实 —— OSS fingering.py:130 p=ρ/3+ψμ vs 移植 :136-137 p=ρ/3+ψμ/3；改回 OSS 定义重跑得 0.911%（仍<15%），u_max 降 15 倍",
  "F2 high: ⑤ 'error decreases monotonically … true equilibrium' 不成立 —— 1200-1999 步时间带 0.235%-6.835%，u_max 1.09e-6-3.76e-5 有回升；0.0708% 是单相位采样",
  "F3 high: TensorLBM 侧对标无工件 —— free_energy_step 漂移与 CAC2D 'ran successfully' 均无输出记录；primary_results.json 无 tensorlbm_cac2d 键；tensor_reports 对 mission 17724 0 行",
  "F4 medium: 入口写 xval_results.json(:577)，交付却只有 primary_results.json（无脚本生成）⇒ repro 路径不复现工件；fig4_sensitivity.png 与 xval_results.json 缺件",
  "F5 medium: 零 tracked 改动（3 行 ??，head_commit=null）⇒ 交付未入库、工作树可回收；审计已逐字节冻结进 evidence_snapshot",
  "F6 low: 文档⑤ R=10 0.64% 无工件但审计复现 0.6434%（事实成立、工件缺失）",
  "F7 low: ④ 表写 'Step 2000'，工件末点实为 step 1999",
  "F8 info: 判据自校准（κ=0.75σξ ⇒ σ_model=σ_input），非外部标杆；独立毛细能量路线 2.90% 互洽；OSS 仓无 LICENSE（reference-only），报告未记许可提示"
 ],
 "not_run": [
  "CAC2D 静滴对标（交付自承 not comparable，无判据）",
  "free_energy_step 相场漂移主张复跑（无工件无脚本）",
  "OSS 侧同算例数值对照（原码 380×380×1000 步通道算例，无 ref 值）",
  "GPU/CuPy 路径、网格/时间步收敛阶研究"
 ]
}$json$::jsonb, 'board-audit');

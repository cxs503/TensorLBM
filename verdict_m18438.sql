-- m18438 交差核验 verdict：对象 m18371 XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT (TensorLBM)
-- 判级 L3：真代码算子 + 多生产调用方集成 + 单步逐位验证；但完整收敛跑未出结果（队列超时）、
-- tensor_benchmark_cases 无该锚点注册行 → L4 不足。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18371 XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT', 18438, 3, $json${
 "verdict_basis": "L3：src/tensorlbm/multiphase.py:940-1060 collide_sc_single_component 是真代码算子（D2Q9 SCMP BGK 碰撞，支持 velocity_shift/edm/guo 三种 forcing scheme，含完整 docstring 与公式注释），被 5 个生产模块调用（contact_line/capillary_driven.py:318/386, contact_line/engine.py:359, contact_line/microtexture.py:851, contact_line/three_phase.py:416/560, dam_break.py:376），且从 tensorlbm.__init__ 导出（line 518, 1790, 2577）。单步验证（L=64, R=12, 近平衡初值）确认三 scheme 在 t=0 逐位一致（ρ_min=0.3700, ρ_max=4.4860, mass=3554.34）。不到 L4：完整 Laplace 定律收敛跑（4 radii × 3 schemes × 2 arms = 24 runs）提交集群 cj1910 但队列等待超预算窗口，仅完成单步验证；cross-arm error 0% 因参考实现与 TensorLBM 使用相同公式（零手抄），非独立基准；tensor_benchmark_cases 无 XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT 注册行。",
 "claimed_level_implicit": "XVAL 交叉对标通过（cross-arm σ error 0%，过 15% 与 3% 门）",
 "delivered_level": 3,
 "gap_to_claim": "完整收敛跑未完成（集群队列超时），仅单步验证；cross-arm 0% 误差源于参考实现与 TensorLBM 同源公式，非独立外部基准；tensor_benchmark_cases 缺注册行。",
 "impl_files": [
  "src/tensorlbm/multiphase.py:940-1060 (collide_sc_single_component：velocity_shift/edm/guo 三 forcing scheme 完整实现)",
  "src/tensorlbm/multiphase.py:625-713 (guo_force_term：Guo 2002 二阶力源项)",
  "src/tensorlbm/__init__.py:518,1790,2577 (导出 collide_sc_single_component)",
  "xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/run_xval.py (双臂验证 harness)",
  "commit at /root/tp-agent/work/m18371 (5 files added to xval/)"
 ],
 "entry": "python3 xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/run_xval.py",
 "run_artifacts": [
  "xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/results.json (σ ref/tl 对比：velocity_shift 0.0892/0.0892, edm 0.0895/0.0895, guo 0.0894/0.0894; error_pct 全 0.0%; pass_15/pass_3 全 true)",
  "xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/predeclared.json (预声明阈值与实验设计)",
  "xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/REPORT.md (六段报告)",
  "xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/DIFF_OSS_VS_TENSORLBM.md (差异清单)",
  "xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/run_xval.py (验证脚本)"
 ],
 "tests": "无专用单元测试（本 mission 为 XVAL 交叉对标，非单元验证）；单步验证在 run_xval.py 内完成（t=0 逐位一致检查）",
 "integration_callers": [
  "src/tensorlbm/contact_line/capillary_driven.py:318,386 (forcing='edm')",
  "src/tensorlbm/contact_line/engine.py:359",
  "src/tensorlbm/contact_line/microtexture.py:851",
  "src/tensorlbm/contact_line/three_phase.py:416,560 (forcing='edm')",
  "src/tensorlbm/dam_break.py:376",
  "src/tensorlbm/__init__.py:518,1790,2577 (导出)"
 ],
 "benchmark": {
  "status": "partial",
  "note": "完整 Laplace 收敛跑（4 radii × 3 schemes × 2 arms）提交集群 cj1910 但队列等待超预算；仅完成单步验证（L=64, R=12）",
  "single_step": {"rho_min": 0.37, "rho_max": 4.486, "mass": 3554.34, "all_schemes_bit_identical": true},
  "cross_arm_sigma": {"velocity_shift": {"ref": 0.0892, "tl": 0.0892, "error_pct": 0.0}, "edm": {"ref": 0.0895, "tl": 0.0895, "error_pct": 0.0}, "guo": {"ref": 0.0894, "tl": 0.0894, "error_pct": 0.0}},
  "tensor_benchmark_cases_registered": false
 },
 "repro_cmd": "cd /root/tp-agent/work/m18371 && python3 xval/XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT/run_xval.py",
 "external_anchor": "WITHHELD：参考实现为从 Shan-Chen 1993 / Kupershtokh 2004 / Guo 2002 论文公式转录的 clean numpy 实现，无独立外部手册原表原页"
}$json$, 'board-audit');

-- m18313 board audit verdict: 对象 m18195 XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT (TensorLBM)
-- 判级 L2：真实独立功能实现（614 行移植内核 + 12 单测 + 可复现 XVAL 扫描）；
-- L3 不成立（零集成调用方）、L4 不成立（锚点基准 0 行 + 平台 L4 门 CLAIMED_NOT_PROVEN rows=0）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18195 XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT', 18313, 2, $json${
 "verdict_basis": "L2：真实独立功能实现。src/tensorlbm/phasefield/pressure_tensor_equilibrium.py（614 行，D2Q9 自由能二元流体平衡态，压强张量进平衡态，OSS sailfish 语义移植）输入→计算→输出完整；入口 run_xval.py 审计方重跑 5.0 s 产出全套工件，verdict PASS，worst_resolved_sigma_error=0.06089506891217726 与提交工件逐位一致（仅 wall_seconds 3.8→5.0 变化）；12 个单元测试审计方重跑全过（6.29 s）；sympy 符号推导残差 0（离散二阶矩=连续自由能压强张量，机器精度 1.7e-16）；4 个 vendored OSS 文件 sha256 审计方独立复算全部匹配。但 (a) 零集成调用方（模块 docstring 明言 no solver in the repository is modified；全仓 grep 无主求解器 import）⇒ L3 不成立；(b) tensor_benchmark_cases 该 anchor 0 行（psql 直查 postgres 角色）+ 平台 /tensor/bench/l4-verdict/18195 = CLAIMED_NOT_PROVEN（rows=0，failed_checks 全 5 项）+ /tensor/bench/results?mission_id=18195 total=0 ⇒ 基准裁判无可裁行，L4 不成立。",
 "claimed_level_implicit": "XVAL 交叉对标完成（mission result.headline：verdict PASS，最差已分辨 σ 误差 6.09%；报告 id 17063 published）",
 "delivered_level": 2,
 "gap_to_claim": "XVAL 对标语义实质兑现（OSS 源码 sha256 钉死 + 符号重推 + 解析闭式 ref vs 分布函数实测 got，已分辨子集 ξ≥2 全部 <15%），但形式要件缺件：锚点未在 tensor_benchmark_cases 登记（0 行），无平台重算 ref/got/error 行（tensorbench_results 0 行），且移植内核未接入任何主求解器。按 L 级定义（级别=功能实质）判 L2，不虚标 L3/L4。",
 "impl_files": [
  "src/tensorlbm/phasefield/pressure_tensor_equilibrium.py:1-614（work/m18195@caa90595 实读；docstring 含 OSS file:line 溯源与移植取舍）"
 ],
 "impl_files_detail": [
  "src/tensorlbm/phasefield/pressure_tensor_equilibrium.py:1-60 (模块 docstring：移植动机/公式/约定/溯源)",
  "src/tensorlbm/phasefield/pressure_tensor_equilibrium.py 内 9 点各向同性 grad/lap 算子 + 平衡态构造 + 压强/σ 测量诊断（审计方读头部 60 行 + 测试断言交叉印证）"
 ],
 "entry": [
  "xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/run_xval.py"
 ],
 "entry_detail": "run_xval.py main()：审计方重跑 exit=0，5.0 s，verdict PASS，worst_resolved_sigma_error=0.06089506891217726（与提交 results.json 逐位一致）；derive_oss_pressure_tensor.py 符号推导残差 0（results.json provenance.derivation_stdout_tail 记录 Mxx-Pxx-rho ux^2=0 等）",
 "run_artifacts": [
  "artifacts/xval_pressure_tensor_equilibrium/results.json（437 行：A_sweep/oss_flat/oss_droplet/tau_sweep/step_series/tl_force_arm/verdicts/provenance）",
  "docs/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT.md（202 行，六段+差异清单）",
  "docs/figures/xval_pte_convergence.png（86050 B）",
  "docs/figures/xval_pte_fields.png（90786 B）",
  "xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/ref/oss_discrete_pressure_tensor.json（sympy 推导产物）",
  "xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/predeclared.json（预声明阈值/变量/敏感性计划）",
  "xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/ref/sailfish_src/（4 文件 sha256 钉死，审计方复算全匹配）"
 ],
 "tests": "tests/test_pressure_tensor_equilibrium_xval.py 12 passed（审计方重跑 pytest -q，6.29 s；含二阶矩闭合 1e-16、序参量闭合、O(h^2) 算子精度、σ vs (2/3)sqrt(2kappa A)、确定性）",
 "integration_callers": "无（零集成）：grep -rn pressure_tensor_equilibrium 全仓仅命中本模块自身/测试/xval 目录；模块 docstring 明言 no solver in the repository is modified ⇒ L3 不成立",
 "benchmark": "tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT' → 0 行（psql 直查）；平台 /tensor/bench/l4-verdict/18195 = CLAIMED_NOT_PROVEN（rows=0，failed_checks: execution_success/source_cluster_job/cluster_job_bound/ref_got_error_present/within_tolerance）；/tensor/bench/results?mission_id=18195 total=0。工件内 ref/got 数值真实存在：平界面 σ 误差 ξ=1 -19.56%（欠分辨，已文档化剔除）/ξ=1.5 -10.15%/ξ=2 -6.09%/ξ=3 -2.84%/ξ=4 -1.63%；液滴 ξ=2 -6.36%…-6.77%、ξ=3 -2.61%…-3.13%；力形式对照臂 -100.00%（结构性读不出机械压强，honest 记录）",
 "repro_cmd": "cd work/m18195 && python3 xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/run_xval.py（审计方复跑成功，5.0 s，数值与提交工件一致）；python3 -m pytest tests/test_pressure_tensor_equilibrium_xval.py -q（12 passed）",
 "external_anchor": "sailfish-team/sailfish @ b7140c580e1f4cb406dde947c8d8d5a8a6b2d79e（LGPL3）；vendored 4 文件 sha256 审计方独立复算匹配：lb_binary.py 92527092…、sym.py 77dfe766…、sym_equilibrium.py 6ef58388…、finite_difference_optimized.mako f5479378…；ref 出处 sym_equilibrium.py:16-83 / lb_binary.py:224-259 / mako:46-48",
 "audit_notes": "报告六段齐全（①问题+ref 出处 ②方法与实现 ③参数表 ④对照表 ref vs got 打勾叉 ⑤收敛性/敏感性 ⑥失败模式与工件 +⑦本仓 vs OSS 差异清单）；平台报告 id 17063 published；预声明阈值先于扫描固定（predeclared.json，无事后放宽）；ξ=1 欠分辨点如实记为失败模式而非 pass；对称双井 Δp_bulk≡0 故不宣称 Laplace 百分比（避免循环论证）——honest 纪律良好，无粉饰",
 "audited_head": "caa9059520b37caf4532d652148400a40161c02f",
 "audit_date": "2026-11-13"
}$json$, 'board-audit');

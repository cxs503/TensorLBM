-- m20302 交差核验 verdict：对象 m20245 XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY（TensorLBM）
-- 审计源工作树：/srv/tensor/agent/work/m19251/xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/
-- 判级 L3：系统集成（生产模块 added_mass_reference.py 已入 src/tensorlbm/；xval 目录与 6 个门禁测试已合入 main；
--          解析仲裁双编码、独立 FV 臂、被测臂、OSS 探针全链路可复现；
--          不授 L4：benchmark 注册 SQL 存在但未执行（status=pending），tensorbench_results 0 行）
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm20245 XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY', 20302, 3, $json${
 "verdict_basis": "L3 系统集成：生产模块 src/tensorlbm/added_mass_reference.py 提供解析仲裁注册表+求解器校验+可用性包络三件套；xval 目录与 tests/test_xval_mass_analogue.py（6 个不变量测试全过）已合入 main（commit 5e9059f91 + c0fe83e29）；全链路可复现（33 个 run JSON + artifacts/summary.json 位级一致；analyze.py 程序化重算零手抄）。不授 L4：docs/audit/m19251/register_benchmark_cases.sql 存在但 4 行 status=pending（未执行入库），tensorbench_results 无对应 mission 行，l4-verdict API 不可达。",
 "claimed": "m18997/m19251 交付 XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY 完整交叉验证：OSS 探针（3 树+台账，0 命中→WITHHELD）、解析仲裁（双编码残差 1.59e-9）、独立 FV 臂（+0.85%~+2.86%）、周期盒扫掠（R≥12/LR≥16 <3%）、LBM 非定常力臂（FAIL，H3 被否定）",
 "delivered_level": 3,
 "gap_to_claim": "L4 未达成：benchmark 注册 SQL 未执行（4 行 pending）；tensorbench_results 0 行；added_mass_reference.py 未合入 main（仅在 agent/m19251）",
 "impl_files": [
  "src/tensorlbm/added_mass_reference.py:1-320（analytic_cylinder_ca/check_cylinder_added_mass/cylinder_ca_envelope/main）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/mass_reference.py:1-242（analytic_circle_ca/fv_nonperiodic_ca/repo_periodic_ca/staircase_mask）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/run_xval.py:1-114（周期盒扫掠+分辨率扫掠+独立 FV 臂驱动）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/analyze.py:1-386（误差/拟合/判据/分解程序化重算）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/probe_oss_inventory.py:1-194（OSS 探针+台账检索）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/run_lbm_arm.py:1-84（LBM 非定常力臂）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/figures.py:1-123（4 张对照图生成）",
  "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/render_tables.py:1-170（报告表格程序化生成）"
 ],
 "entry": "xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY/run_xval.py（argparse 无参默认跑全臂）；生产入口 src/tensorlbm/added_mass_reference.py:main()",
 "run_artifacts": [
  "runs/index.json（18 臂汇总，位级可复现）",
  "runs/analytic.json（C_a=1.0, quadrature=1.0000000016, residual=1.59e-9）",
  "runs/repo_R8_LR4.json..repo_R16_LR16.json（10 个周期盒算例）",
  "runs/fv_R8_D6.json..fv_R16_D20.json（5 个独立 FV 算例）",
  "runs/lbm_res16.json..lbm_box16.json（6 个 LBM 非定常力算例，含 trace）",
  "runs/oss_survey.json（3 树+台账探针结果）",
  "artifacts/summary.json（16657 bytes，全量误差/拟合/包络/完整性）",
  "artifacts/fig1_confinement.png（31798 bytes）",
  "artifacts/fig2_resolution.png（26635 bytes）",
  "artifacts/fig3_lbm_arm.png（47031 bytes）",
  "artifacts/fig4_probe.png（14056 bytes）",
  "artifacts/TABLES.md（4213 bytes）"
 ],
 "tests": "tests/test_xval_mass_analogue.py → 6 passed in 3.51s（auditor rerun 一致）；断言不变量：解析注册表精确性、独立 FV 臂匹配、域独立性、周期盒在 15% 带内、单调性、OSS 探针否定 verdict",
 "integration_callers": [
  "src/tensorlbm/fsi_added_mass.py（引用 added_mass_reference）",
  "src/tensorlbm/oscillatory_cylinder_inertia.py（引用 added_mass_reference）",
  "src/tensorlbm/ibm_validation_cases.py（引用 added_mass_reference）",
  "src/tensorlbm/passive_scalar_transport_verification.py（引用 added_mass_reference）"
 ],
 "integration_note": "added_mass_reference.py 在 src/tensorlbm/ 中作为生产模块存在，被 4 个模块引用；但仅在 agent/m19251 分支，未合入 main。xval 目录与测试已合入 main（commit 5e9059f91 + c0fe83e29）。",
 "benchmark": {
  "registered_case": "docs/audit/m19251/register_benchmark_cases.sql 存在（4 行 INSERT），但 status=pending（未执行）",
  "platform_rows": "未查询到 tensorbench_results 行（SQL 未执行）",
  "ref_vs_got_internal": "analytic C_a=1.0 vs repo R=8/LR=16: 1.0809 (+8.09% <15% ✓); repo R=16/LR=16: 1.0234 (+2.34% <3% ✓); FV R=16/D20: 1.0085 (+0.85% <3% ✓); LBM res16: 1.8317 (+83.17% ✗)",
  "external": "OSS 臂 WITHHELD（3 树 0 命中 + 台账 0 命中）",
  "verdict_vs_gate": "解析臂 <3% ✓；周期盒 R≥12/LR≥16 <3% ✓；LBM 臂 FAIL（H3 被否定）"
 },
 "repro_cmd": "cd xval/XVAL-LBM-NO-MASS-ANALOGUE-IN-OSS-INVENTORY && python3 run_xval.py && python3 analyze.py && python3 figures.py; tests: python3 -m pytest tests/test_xval_mass_analogue.py -q (6 passed, 3.51s)",
 "external_anchor": "TensorBEM/no-mass-analogue-in-oss-inventory 做法移植；OSS 探针覆盖 pylbm@f9f5d16（235 源文件）、Maarten-vd-Sande/lbm@573a01c4（9）、hschwane/GraSPH（50）；台账 oss_inventory 33 行 + oss_solutions 283 行，实现类命中 0",
 "level_rationale": "L3 系统集成：生产模块 added_mass_reference.py 已入 src/tensorlbm/（被 4 模块引用），xval 与测试已合入 main，全链路可复现（33 JSON + summary.json 位级一致，6 测试全过）。不授 L4：benchmark 注册 SQL 未执行（4 行 pending），tensorbench_results 0 行，l4-verdict 不可达。不授 L5：added_mass_reference.py 未合入 main（仅 agent/m19251）。",
 "findings": [
  {"id": "F1", "severity": "medium", "what": "added_mass_reference.py 未合入 main", "detail": "仅在 agent/m19251 分支；xval 目录与测试已合入 main（5e9059f91 + c0fe83e29）", "impact": "生产模块不可用于 main 分支用户", "fix": "merge agent/m19251 into main 或 cherry-pick added_mass_reference.py"},
  {"id": "F2", "severity": "medium", "what": "benchmark 注册 SQL 未执行", "detail": "docs/audit/m19251/register_benchmark_cases.sql 存在但 4 行 status=pending", "impact": "L4 结构性不可达（无注册基准裁判）", "fix": "执行 SQL 将 4 行 status 改为 'active'"},
  {"id": "F3", "severity": "low", "what": "LBM 非定常力臂 FAIL（H3 被否定）", "detail": "KC≈0.1-1 区间 C_I 误差 +27%~+83%，线性拟合 R²=0.74 <0.90 不可用", "impact": "LBM 附加质量测量路线在该参数区间不适用", "fix": "报告已诚实披露；下一轮 KC≤0.2 或 β≥400 检验线性响应区"}
 ],
 "overstated_claims": ["无重大夸大；报告诚实披露 LBM 臂 FAIL 与 OSS 臂 WITHHELD"],
 "checks_run": [
  "pytest tests/test_xval_mass_analogue.py -v → 6 passed in 3.51s",
  "analyze.py 独立重算：index.json 与 summary.json 位级一致（18 臂全匹配）",
  "mass_reference.py 导入测试：analytic_circle_ca C_a=1.0, residual=1.59e-9; fv_nonperiodic_ca C_a=1.028228",
  "报告六段完整性：①-⑦ + 显式误差判定表，8 个 section 齐全",
  "工件清单：33 个 run JSON + 6 个 artifacts 文件 + 4 张 PNG 图",
  "OSS 探针验证：oss_survey.json 3 树 0 命中 + 台账 0 命中 → WITHHELD",
  "git 合入状态：xval 目录与测试在 main（5e9059f91 + c0fe83e29）；added_mass_reference.py 仅在 agent/m19251",
  "benchmark 注册 SQL 存在但未执行（status=pending）"
 ],
 "checks_not_run": [
  "LBM 非定常力臂全量重跑（6 算例 × 数千步，>10 min）：not_run by auditor",
  "数据库 benchmark 注册 SQL 执行验证：not_run（无 psql 访问）",
  "tensorbench_results 行查询：not_run（无 API 访问）"
 ],
 "disposition": "approve with notes: (1) merge added_mass_reference.py into main; (2) execute benchmark registration SQL; (3) LBM arm FAIL is honest and documented, not a defect.",
 "verdict_by": "board-audit",
 "level": 3
}$j$, 'board-audit');

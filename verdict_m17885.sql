-- m17885 交差核验 verdict：对象 m17840 XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE (TensorLBM)
-- 判级 L2：真代码算子 + 断言不变量的单测 + 逐位可复现工件；但无生产集成(L3 不足)、
-- 端到端 176.06% honest-FAIL 且无登记基准案例(L4 不足)、算子层"过 15% 门"是代数往返自洽而非精度。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17840 XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE', 17885, 2, $json${
 "verdict_basis": "L2：src/tensorlbm/wetting/geometric_two_candidate.py 是真代码算子（φ/solid/θ -> ghost 场 + 诊断字典，384 行，含候选法向选择/isotropic-9 色梯度/SolidColorDiff 三要素与 cap 几何接触角估值器），36 项单测断言不变量（镜像退化、θ=90 严格镜像、界面门限回退、clamp、非法参数抛错、解析圆帽角复现 <0.6°），交付工件可逐位复现（重跑 delta=0.0）。不到 L3：src/ 零生产调用方、wetting/__init__.py 未导出、求解器层用 benchmark 私有 make_step 循环。不到 L4：端到端 165.639° vs 60° = 176.06%（可复现 honest-FAIL，圆拟合 81.376° 与之互差 84.26°，h=62.19 达域高 97.2% ⇒ 末态非帽，角度不可测）；且 tensor_benchmark_cases 对本锚点 0 行登记，基准裁判缺席，无收敛性扫描。附加：交付默认配置与既有 cg_wetting2d 闭式逐值等价（V2-V1 保真度差 -3.35e-10，sign flips=0），故求解器能力净增≈0，净增落在测量配方+诊断+负知识。",
 "claimed_level_implicit": "XVAL 交叉对标通过（算子层 <=2.4e-14% 过 15% 门）；求解器层已诚实标注 honest-FAIL",
 "delivered_level": 2,
 "gap_to_claim": "算子层所谓 '过 15% 门' 是解析帽 -> BC 写 ghost -> 同模板反读 的代数往返自洽（M1），不表征物理精度；参考实现的区分性机制在镜像填充下严格退化（flips=0, V2 与 V1 逐值相同），故 '交叉对标 openLBMPM 做法' 只对标了其测量配方（初等几何 R=(b^2+4h^2)/8h）而非其特色算法；端到端精度门 176.06% 未过，且三条测量互差 84.26° 说明该宿主上端到端角度不可测。",
 "impl_files": [
  "/root/tp-agent/work/m17840/src/tensorlbm/wetting/geometric_two_candidate.py:160-299 (geometric_wetting_ghost_candidates：候选法向选择 + isometric-9/SolidColorDiff 切换 + 诊断字典)",
  "/root/tp-agent/work/m17840/src/tensorlbm/wetting/geometric_two_candidate.py:124-157 (isotropic_color_gradient / solid_color_mirror)",
  "/root/tp-agent/work/m17840/src/tensorlbm/wetting/geometric_two_candidate.py:301-384 (cap_geometry_contact_angle：openLBMPM 参考测量配方，status 降级不抛错)",
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/operator_study.py:118-172 (算子层 27 例/变体 harness，M1 往返 + M2 保真度)",
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/run.py:75-213 (求解器层 sessile drop：make_step 私有臂 + capgeom/circle/readback 三测量 + gate)",
  "commit adc9eea9 (18 files, +3170) 于 worktree /root/tp-agent/work/m17840"
 ],
 "entry": "python3 benchmarks/xval_openlbmpm_wetting/operator_study.py 与 python3 benchmarks/xval_openlbmpm_wetting/run.py --arms cot60 --steps 4000（均在本树可执行；审计在 /tmp/rerun 副本重跑 run.py 以免污染交付树）",
 "run_artifacts": [
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/operator_study.json (4 变体 × 27 例，M1 max 1.579e-14%/2.368e-14%，M2 0.00639/0.00713/0.02694，flips 0/0/0/3070)",
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/results_cot60.json (theta_capgeom=165.639, rel_err_pct_capgeom=176.06, h=62.19, b=15.67, theta_circle=81.376, theta_bc_readback=60.000, u_max=6.04e-3)",
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/results_ctl60.json (对照臂 theta_capgeom=128.967)",
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/equivalence_check.json (400 步两实现轨迹差 max|Δf|=3.46e-9, bitwise_identical=false)",
  "/root/tp-agent/work/m17840/benchmarks/xval_openlbmpm_wetting/fig_operator_xval.png, fig_solver_angle.png, fig_phase_fields.png (figures.json 登记)",
  "/root/tp-agent/work/m17885/artifacts/m17885_audit/audit_m17885.json (本次审计机器读数：记录 vs 重跑 delta 全 0.0)",
  "/root/tp-agent/work/m17885/artifacts/m17885_audit/fig_m17885_operator_solver.png, fig_m17885_solver_angles.png"
 ],
 "tests": "tests/test_wetting_geometric_two_candidate.py:52/66/83/93/108/122/131（test_candidate_degenerates_to_closed_form_with_mirror_fill / test_prescribed_solid_colour_switches_the_sign / test_theta_ninety_is_mirror_wall / test_interface_gate_leaves_mirror_plus_solid_color_diff / test_rejects_bad_angle_stencil_and_shape / test_cap_geometry_recovers_analytic_cap_angle / test_cap_geometry_reports_degenerate_field）；审计重跑 PYTHONPATH=src pytest tests/test_wetting_geometric_two_candidate.py tests/test_cg_geometrical_wetting.py -q -> 36 passed in 4.33s（断言不变量而非空跑）",
 "integration_callers": "无生产调用方。grep -rn 'geometric_two_candidate' src/ tests/ benchmarks/ -> 仅 tests/test_wetting_geometric_two_candidate.py、benchmarks/xval_openlbmpm_wetting/{operator_study.py,run.py}；src/ 内零命中（模块自身除外）；src/tensorlbm/wetting/__init__.py 最后一次改动在旧 commit 422c59c4，未导出该模块；求解器层评估走 benchmarks/xval_openlbmpm_wetting/run.py:75 make_step 私有循环",
 "benchmark": {
  "registered_cases_for_anchor": 0,
  "tensor_benchmark_cases_total": 65,
  "nearest_registered_wetting_anchor": "XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER (不同锚点/不同 BC 族)",
  "declared_criterion": "接触角相对误差 <=15% (mission goal)",
  "operator_layer_M1_max_pct": 2.3684757858670007e-14,
  "operator_layer_M1_is_roundtrip_selfconsistency": true,
  "operator_layer_M2_fidelity_no_threshold": {"V1": 0.006392257652678957, "V2": 0.0063922573174886215, "V3": 0.007128709020940319, "V4": 0.026940097597709992},
  "V2_minus_V1_mean_fidelity": -3.351903354284583e-10,
  "V4_fidelity_ratio_vs_V1": 4.2144886926482945,
  "solver_layer_theta_prescribed_deg": 60.0,
  "solver_layer_theta_capgeom_deg": 165.63896305189454,
  "solver_layer_rel_err_pct": 176.06493841982422,
  "solver_layer_gate_pass": false,
  "solver_layer_theta_circle_deg": 81.37604066967326,
  "measurement_disagreement_deg": 84.26292238222128,
  "apex_over_domain_height": 0.9717129890912806,
  "control_arm_theta_capgeom_deg": 128.9667867458407,
  "auditor_rerun_matches_record": true,
  "repro_delta_theta_deg": 0.0
 },
 "repro_cmd": "cd /root/tp-agent/work/m17840 && PYTHONPATH=src python3 -m pytest tests/test_wetting_geometric_two_candidate.py tests/test_cg_geometrical_wetting.py -q && PYTHONPATH=src python3 benchmarks/xval_openlbmpm_wetting/operator_study.py && PYTHONPATH=src python3 benchmarks/xval_openlbmpm_wetting/run.py --arms cot60 --steps 4000  # 审计实测：36 passed / 算子 5.9s 逐位一致 / 求解器 73s 165.639° 逐位一致",
 "external_anchor": {
  "reference": "PorousMediaSimulation/openLBMPM geometric-wetting-contact-angle",
  "resolvable_pointer": "oss_solutions.id=5 (repo=PorousMediaSimulation/openLBMPM, feature_slug=NEW:CAP-LBM-VALIDATION-SURFACE, tag=geometric-wetting-contact-angle)",
  "cited_doc": "docs/oss-inventory-openLBMPM.md",
  "cited_doc_present_in_delivery_commit": false,
  "cited_doc_present_in_origin_main": false,
  "cited_doc_elsewhere": "/root/tp-agent/work/m15182/docs/oss-inventory-openLBMPM.md (他任务工作树)",
  "source_code_line_citations_verified": false,
  "why_not_verified": "无外网；集群取源作业 cj1688 排队未执行，openLBMPM 源行号 (RKCG2D/AcceleratedRKGPU2D.py:1560/:1584-1632/:2430-2492, IniFiles/RKtwophasesetup2D.ini:42) 审计无法复核"
 },
 "overstated_claims": [
  "算子层 '27 例反读误差 <=2.4e-14% 过 15% 门' 被读作精度证据；实为解析场 -> BC 写 ghost -> 同模板反读 的代数往返自洽（任何一致错误亦为 0 误差）",
  "标题'交叉对标 openLBMPM 做法'强于事实：参考的两候选法向选择在镜像填充解析帽上 flips=0 且 V2 与 V1 逐值等价（ΔM2=-3.35e-10），未触到区分性自由度",
  "报告 ⑤ 引用的出处 docs/oss-inventory-openLBMPM.md 为悬空指针（不在 adc9eea9 也不在 origin/main）"
 ],
 "checks_run": [
  "psql tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE' -> 0 rows",
  "pytest 两测试文件 -> 36 passed (4.33s)",
  "operator_study.py 重跑 -> 与交付 JSON 逐项逐位一致 (delta 0.0)",
  "run.py --arms cot60 --steps 4000 重跑 -> theta_capgeom 165.639° / 176.06% / gate_pass=false，与交付 JSON 逐位一致",
  "grep 调用方 + git cat-file 外部锚可解析性"
 ],
 "checks_not_run": [
  "openLBMPM 源码逐行复核（无外网，取源作业 cj1688 未执行）",
  "oss60/oss90/ctl90 余臂求解器重算（仅复算失败臂 cot60 + 读 ctl60 交付工件）",
  "网格/时间步收敛扫描（交付未做，审计亦未做）"
 ]
}$json$, 'board-audit');
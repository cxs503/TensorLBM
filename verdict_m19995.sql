-- m19995 board audit: verdict for m19921 XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
 'TensorLBM',
 'm19921 XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS',
 19995,
 4,
 jsonb_build_object(
  'impl_files', jsonb_build_array(
    'src/tensorlbm/wetting/two_impl_contact_angle.py:133-171 wall_ghost_rows (dirichlet / neumann_flux / geometric_ghost 三族墙幽灵)',
    'src/tensorlbm/wetting/two_impl_contact_angle.py:282-350 measure_contact_angle (theta_cap 面积反演 + theta_circle 圆拟合，两独立估计器)',
    'src/tensorlbm/wetting/two_impl_contact_angle.py:227-262 initialise_matched_cap (体积配平初值，修掉 round-1 的 3% 气体过饱和偏差)',
    'src/tensorlbm/wetting/two_impl_contact_angle.py:107-125 psi_wall_from_angle (Young 律二分反解，唯一真值源)'
  ),
  'entry', 'xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py:288-300 CLI (one|sweep|summarize)；驱动 run_arm=质量守恒 Allen-Cahn 松弛 psi<-psi-omega(mu-<mu>)',
  'run_artifacts', jsonb_build_object(
    'committed', jsonb_build_array(
      'xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/summary.json (34 行分析级真值，含 per-row theta_cap/theta_circle/err/gas/iter/converged)',
      'xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/predeclared.json (冻结闸门 accept_pct=15.0, cross_arm_pct=15.0, estimator_agreement_deg=10.0)',
      'xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/verify_r3/verify_results.json (round-3 独立复算 6 行)',
      'artifacts/m19995/runs_smoke/{neumann_flux,geometric_ghost}__th090.0__baseline__xi3/result.json (本单审计复跑原始记录)'
    ),
    'declared_but_absent', 'REPORT.md §6 点名的 runs/<tag>/result.json、runs/<tag>/psi.npz、figures/fig{1,2,3}_*.png 不在树内：runs/ 被 .gitignore:108 忽略，figures/ 空目录 ⇒ declared-artifacts fail（本单实跑 make_report.py:82 UnboundLocalError 佐证）',
    'auditor_recompute', 'artifacts/m19995/recompute.json（自写代码从 summary.json 重算 34 行，pass/measurable/estimators_agree 与交付 mismatch=0；可测 25/过闸 25，可测行最大误差 9.35%，跨臂最大差 6.48%）'
  ),
  'tests', jsonb_build_object(
    'run_by_auditor', 'python3 -m pytest tests/test_two_impl_contact_angle.py -q → 34 passed in 36.71s @ main a80287b82',
    'content', 'tests/test_two_impl_contact_angle.py::test_young_law_from_first_principles (1-D 墙自由能数值积分独立推出 Young 律)、::test_flux_and_geometric_ghosts_encode_the_same_angle (两幽灵族场级等价 <0.03)、::test_library_circle_estimator_is_not_supplementary (修掉的补角缺陷回归)',
    'not_on_main', 'tests/test_two_impl_contact_angle_r3_estimators.py 仅存在于 agent/m19921（本单 pytest 报 file not found）⇒ round-3 回归未合入 main',
    'not_run_by_auditor', 'verify_r3/verify.py 未重跑；其解析帽标定表（theta_cap 1.2-8.9% 低偏）作为被审材料采信，状态 not_run'
  ),
  'integration_callers', jsonb_build_object(
    'only_caller', 'xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py::run_arm（同仓 Allen-Cahn 松弛驱动）',
    'main_solver', '无：grep -rn two_impl_contact_angle src/ TensorLBM/ 零命中；src/tensorlbm/wetting/__init__.py 未导出该模块 ⇒ L3 主求解器/平台集成不成立（无碰撞-迁移耦合、无流体润湿）'
  ),
  'benchmark', jsonb_build_object(
    'referee_query', 'SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug=''XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS'';',
    'registered_rows_for_anchor', 0,
    'tensor_benchmark_cases_total', 121,
    'note', '机器裁判表对本锚为空（anchor_slug ILIKE ''%CONTACT%'' 无匹配；tensorbench_results 亦无本锚）⇒ 唯一裁判是交付自带 predeclared.json 的解析锚，不是登记基准行',
    'ref', 'Young 律 cos(theta)=(3 psi_w - psi_w^3)/2（同一自由能的墙能解析推论；非实验），tests 中由 1-D 数值积分独立推出',
    'got', 'theta_circle 可测行：90°→90.02, 110°→110.04, 135°→143.67(base)/-, 70°→69.53(base)/-, 最大可测误差 9.35%(neumann_flux 70° theta_cap)，跨臂最大 6.48%(baseline 135°)',
    'gate_verdict', 'PASS on all 25 measurable rows vs predeclared 15%（审计自算，与交付一致）；2 个可测 dirichlet 90° 行违反 estimator_agreement_deg=10（11.17/12.87 deg）'
  ),
  'repro_cmd', jsonb_build_object(
    'executed_by_auditor', 'PYTHONPATH=$PWD/src python3 xval/XVAL-LBM-TWO-INDEPENDENT-CONTACT-ANGLE-IMPLS/run_xval.py one --arm {neumann_flux,geometric_ghost} --theta 90 --grid baseline → 8s/7s，逐位复现 summary.json（87.27/90.02，gap 2.75，iter 6000，gas +0.038）',
    'declared_chain_status', 'sweep 可跑；summarize 在干净树输出 "no runs"；make_report.py 崩于 make_report.py:82 UnboundLocalError (im 空分支未初始化) ⇒ 交付点名的 3 步复现链在干净树 2/3 不可执行',
    'full_sweep_recheck', '集群作业 cj2428（main@a80287b82, sweep --workers 16 + summarize + 逐行 diff）'
  ),
  'external_anchor', jsonb_build_object(
    'type', 'ANALYTIC（非实验/非 OSS 数值）',
    'law', 'cos(theta)=(3 psi_w - psi_w^3)/2，同一 Cahn-Hilliard 自由能的墙项解析推论',
    'independent_instrument', 'xval/.../verify_r3/{independent_estimators.py,verify.py} @ agent/m19921 403c4818e：独立 tanh 帽标定 + 独立几何圆拟合，6/6 行复现 <0.05 deg，跨臂量 |134.25-142.97|/135=6.46% 与 round-2 的 6.84% 同侧（被审件材料，本单未重跑）',
    'experimental_anchor', 'WITHHELD —— 无实验/文献定量对标（静态角，报告 §6 明示动态润湿不在范围）'
  ),
  'level_basis', 'L0-L2 成立：真实现（486 行，输入 theta→松弛→输出角度）+ 34 项单元/首原理测试通过 + 审计端到端 8s 复现。L4 成立（解析锚口径）：完整算例（3 网格 × 5 角 × xi 敏感 = 34 行）+ 逐行误差分析（15% 闸，可测 25/25 过）+ 可复现（one 逐位复现；全量扫描 cj2428）。L3 不成立（无主求解器调用方，wetting/__init__.py 未导出）。L5 不成立（静态角、omega=1 是松弛非时间积分、无流体耦合、无实验对标）。',
  'defects_found_for_remediation', jsonb_build_array(
    'F1 中：报告点名的 runs/、figures/ 未入库（runs/ 被 .gitignore:108 忽略），summarize→"no runs"，make_report.py:82 UnboundLocalError ⇒ 交付复现链 2/3 步在干净树不可执行',
    'F2 中：tensor_benchmark_cases 本锚 0 行 ⇒ 无登记容差可逐项对表，L4 依据是解析律自洽 + 自定 15% 闸',
    'F3 中（信息量）：theta=90 为结构退化点，neumann_flux 的 amp=0.75*sigma*cos(90)/kappa=0（two_impl_contact_angle.py:158）、geometric_ghost 的 cot(90)=0（:168）都退化为同一零法向梯度墙 ⇒ 两臂场逐位相同，跨臂 0.00% 是构造必然，不含独立性信息（7 个可测跨臂格中 2 格如此）',
    'F4 轻：predeclared estimator_agreement_deg=10 在 5 行被违（2 个可测 dirichlet 90 行 11.17/12.87 deg），报告结论行写无限定 PASS',
    'F5 轻：theta_cap 主量在解析帽上带 1.2-8.9% 系统低偏（采信 verify_r3 §B，本单 not_run），theta_circle 无偏 <0.02%',
    'F6 轻：round-3 回归 tests/test_two_impl_contact_angle_r3_estimators.py 未合入 main'
  ),
  'claims_overstated', jsonb_build_array(
    '"artefacts: runs/<tag>/result.json … figures/fig{1,2,3}_*.png" —— 树内为假（F1）',
    '结论行 "PASS" 未带 estimator_agreement 违约限定（F4）',
    '跨臂表把 theta=90 的 0.00% 当作通过证据呈现，实为退化格（F3）'
  ),
  'scope_audited', 'm19921 round-2 交付（cc8c081fb）+ round-3 复核自述（403c4818e）；未重跑 verify_r3/verify.py、未做动态润湿/LBM 耦合/实验对标（not_run）',
  'auditor_artifacts', jsonb_build_array(
    'audit/xval_m19995/AUDIT_M19995.md',
    'artifacts/m19995/recompute.py', 'artifacts/m19995/recompute.json',
    'artifacts/m19995/make_figs.py', 'artifacts/m19995/fig_m19995_two_impl_contact_angle.png',
    'artifacts/m19995/runs_smoke/'
  )
 ),
 'board-audit'
);
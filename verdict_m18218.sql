-- m18218 交差核验 verdict: 对象 m18073 XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS (TensorLBM)
-- 判级 L2：真实现（569 行 port + 458 行 harness + 130 行 fidelity driver，审计重跑 13.806%/15.961% 逐位复现）
--        但未入库（0 tracked）/ 测试红且失败原因错报 / 裁判表 0 行 / 无报告 / 两臂 15% 门全破 ⇒ 不授 L3-L5
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18073 XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS', 18218, 2,
$json${
 "verdict_basis": "L2：独立功能实现成立——oss_oracle_port.py 569 行（v/权重/CT 位标志/碰撞/流式/ABB/质量交换/转换阈值逐条对照上游，头部零手抄对照表）、run_xval.py 458 行（tensorlbm_koerner 与 oss_port 两臂 + 预声明 15% 门 + quirk 消融）、port_fidelity.py 130 行、tests 95 行；审计按脚本独立重跑 3 点，Z max err 13.806% / 15.961% 与交付件逐位一致（pass15=True/False 一致）。L3 拒：交付物全 untracked（git status 仅两行 ??: tests/test_xval_oracle_minimal_d2q9_fs.py、xval/oracle_minimal_d2q9_fs/；result.head_commit=null），仓内零调用点（grep -rl 仅命中测试自身），被测 free_surface_lbm_2d.py 为既有能力（HEAD 已 tracked），本单只消费不接入。L4 拒：tensor_benchmark_cases 对该 anchor_slug 0 行（全表 94 行/75 锚）；测试 5 例 1 红；预声明敏感性两轴未跑完；无六段报告。L5 拒：无新物理能力，OSS 臂自身不可用（照抄 124.377%，修 rho_prev 后 15.961%，u_max=0.8164966=sqrt(2/3) 限幅饱和，液量漂移 207.84%/19.08%）。",
 "declared_level": 2,
 "impl_files": [
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/oss_oracle_port.py:1-569 (上游等价 port；头部 11-38 零手抄对照表)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/run_xval.py:1-458 (DESIGN 预声明 76-118；run_tensorlbm 134-190 调用 src/tensorlbm/free_surface_lbm_2d.py:671 free_surface_step_2d；run_oss_port 200-256；evaluate 291-320；compare_arms 322-352)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/port_fidelity.py:1-130 (port vs 未改上游逐位对照)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/oss_original.py:1-159 (numba identity shim 注入后 import 未改上游 lbm/*)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/{make_figures.py:1-276, DIFF_OSS_VS_TENSORLBM.md}",
  "docs/audit_m18218/evidence_snapshot/ (被审代码+工件快照, SHA256SUMS.txt 固化指纹)"
 ],
 "entry": "python3 xval/oracle_minimal_d2q9_fs/run_xval.py --matrix（默认 --tmax=window_T[1]=4.0）；审计复现脚本 docs/audit_m18218/repro_check.py",
 "run_artifacts": [
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/artifacts/xval_results.json (8 runs + arm_vs_arm; sha256 96faf228...)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/artifacts/numbers.md (sha256 cfd61716...)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/artifacts/port_fidelity_a16.json (steps_compared=20, upstream_first_nonfinite_step=30, max|Δmass|=2.38e-06)",
  "/root/tp-agent/work/m18073/xval/oracle_minimal_d2q9_fs/artifacts/fig_{front_vs_mm,field_maps,convergence,tau_sensitivity,inventory_drift,mach_probe}.png",
  "/root/tp-agent/work/m18218/docs/audit_m18218/repro_results.json (审计独立重跑三任务；13.806/13.806/15.961 逐位一致)",
  "/root/tp-agent/work/m18218/docs/audit_m18218/fig_m18218_front_vs_mm.png, fig_m18218_diagnostics.png",
  "/root/tp-agent/work/m18218/docs/audit_m18218/repro_check.log"
 ],
 "tests": [
  "tests/test_xval_oracle_minimal_d2q9_fs.py::test_design_declares_owner_gate PASS",
  "tests/test_xval_oracle_minimal_d2q9_fs.py::test_port_matches_upstream_snapshot PASS",
  "tests/test_xval_oracle_minimal_d2q9_fs.py::test_port_ledger_conservative_static PASS",
  "tests/test_xval_oracle_minimal_d2q9_fs.py::test_evaluate_gate_logic PASS",
  "tests/test_xval_oracle_minimal_d2q9_fs.py::test_tensorlbm_arm_smoke FAIL —— 断言 assert Z[0]==pytest.approx(1.0, abs=0.1) 得 1.125（a=8 格量子 1/8=0.125 > 0.1，数学上不可满足）；整文件顺序跑时更早以 ModuleNotFoundError: tensorlbm.free_surface_lbm_2d 失败，因 test_port_matches_upstream_snapshot 永久移除 sys.path 中的 src（顺序污染）。mission result 将其错报为「front-advance 断言」失败。复现：python3 -m pytest tests/test_xval_oracle_minimal_d2q9_fs.py -q",
  "tests/test_xval_oracle_minimal_d2q9_fs.py 为 untracked（未入 HEAD，CI 不跑）"
 ],
 "integration_callers": [
  "无。grep -rln 'oracle_minimal_d2q9_fs' 全仓仅命中 tests/test_xval_oracle_minimal_d2q9_fs.py 自身；xval 目录无被 src/、Makefile、pyproject、CI 引用",
  "被测方 src/tensorlbm/free_surface_lbm_2d.py（730 行, git ls-files 已 tracked）属既有能力，非本单新增；本单为消费方 run_xval.py:137,173"
 ],
 "benchmark": {
  "registered_rows_for_anchor": 0,
  "registry_note": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS' → 0 rows（全表 94 行/75 锚）⇒ 无机器裁判登记，基准证据 WITHHELD",
  "external_gate_used": "任务自带 15% 判据 + 作者 DESIGN.thresholds_pct（src 内常量, 预声明）",
  "items": [
   {"case": "tensorlbm_koerner a=32 tau=1 g=5e-5", "got_Z_max_err_pct": 13.806, "ref": "Martin & Moyce 1952 digitised", "tol_pct": 15.0, "pass": true, "audit_rerun": 13.806},
   {"case": "tensorlbm_koerner a=16 tau=1", "got_Z_max_err_pct": 22.297, "tol_pct": 15.0, "pass": false},
   {"case": "oss_port a=32 as-shipped", "got_Z_max_err_pct": 124.377, "tol_pct": 15.0, "pass": false, "note": "u_max=0.8165 限幅饱和; inventory drift 207.84%"},
   {"case": "oss_port a=32 rho_prev_alias=False", "got_Z_max_err_pct": 15.961, "tol_pct": 15.0, "pass": false, "audit_rerun": 15.961},
   {"case": "arm_vs_arm a=16 g=5e-5 / a=16 g=0.1 / a=32", "got_max_rel_diff_pct": [63.636, 22.727, 53.797], "tol_pct": 15.0, "pass": false, "note": "互洽假设被否证"}
  ],
  "accuracy_standard_3pct": "13.806% > 3% ⇒ 不满足平台 3% 精度标准（任务判据为 15%）"
 },
 "repro_cmd": "cd /root/tp-agent/work/m18073 && python3 xval/oracle_minimal_d2q9_fs/run_xval.py --matrix --tmax 1.5（作者实际执行口径；默认 4.0 亦可）；审计复现：cd /root/tp-agent/work/m18218 && python3 docs/audit_m18218/repro_check.py → tensorlbm a=32 tmax=1.5: 13.806% pass; tmax=4.0: 4526 步末样 T=3.752, 仍 13.806%; oss_port rho_prev_fixed: 15.961% fail, drift 19.08%, u_max 0.8164966；测试：python3 -m pytest tests/test_xval_oracle_minimal_d2q9_fs.py -q → 4 passed 1 failed",
 "external_anchor": [
  "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312-324, Fig. 4, doi:10.1098/rsta.1952.0006；树内数字化 src/artifacts/martin_moyce_1952_dam_break.json sha256=6f766bfa461dc0d723eb5478f48444908ae740d5eddf6aefd4630dedcb22a66a（DESIGN 记 e6300b9a1852d7de... 不符，F8）",
  "/root/oss-cache/Maarten-vd-Sande_lbm HEAD 573a01c（lbm/streamcollide.py sha256 6d9d5315..., time_evolve.py 5f1a9dcd..., util.py 867f8370...）"
 ],
 "findings": [
  {"id": "F1", "sev": "blocking", "what": "0 tracked 改动：git status 仅 ?? tests/... ?? xval/oracle_minimal_d2q9_fs/；result.head_commit=null；fail_stage=budget_exhausted+forced_finish"},
  {"id": "F2", "sev": "blocking", "what": "tensor_reports mission_id=18073 为 0 行（report_id=null, report_owed=f）；DIFF 所引 REPORT_XVAL_ORACLE_MINIMAL_D2Q9_FS.md 树内不存在 ⇒ 六段报告缺失"},
  {"id": "F3", "sev": "blocking", "what": "测试红且错报：真实失败为 Z[0]=1.125 vs 1.0±0.1（格量子 0.125>0.1 不可满足），非所述 front-advance；整文件跑因 sys.path 污染先炸 ModuleNotFoundError"},
  {"id": "F4", "sev": "major", "what": "互洽假设否证：arm-vs-arm 22.7/63.6/53.8% 全破 15%；OSS 臂照抄 124.377%、修 quirk 仍 15.961%，u_max 精确饱和 sqrt(2/3)、液量漂移 207.84%/19.08% ⇒ oracle 不可作裁判"},
  {"id": "F5", "sev": "major", "what": "port 等价仅 20 步（upstream_first_nonfinite_step=30），且 port 加零除守卫为唯一语义偏差；headline「逐位等价」未限定范围"},
  {"id": "F6", "sev": "moderate", "what": "预声明敏感性未跑完：a=48 与 tau=0.6/1.4 缺失；fig_tau_sensitivity 对缺失点静默 skip"},
  {"id": "F7", "sev": "moderate", "what": "预声明 control 空转：artifacts.controls=[]，numbers.md Controls 段为空"},
  {"id": "F8", "sev": "minor", "what": "外部锚摘要不符：DESIGN sha256 e6300b9a... ≠ 实际 6f766bfa..."},
  {"id": "F9", "sev": "minor", "what": "工件 provenance：env 缺 numpy 键、workers=10≠默认 12 ⇒ 产自树外代码版本（数值仍复现）"},
  {"id": "F10", "sev": "minor", "what": "记录停表规则违反：落盘 runs 止于 T=1.5，而 DESIGN window_T=[0.5,4.0]、默认 --tmax=4.0；审计以 4.0 重跑结论不变（13.806%）"}
 ],
 "checks_not_run": [
  "OSS 上游 GPU/numba 真路径未验（仅 numba identity shim 的 CPU 路径）；上游 30 步后 NaN，长时行为无法双盲核对",
  "未补跑 a=48 与 tau=0.6/1.4（审计时间盒）",
  "未补跑静水柱 hydrostatic control（工件 controls 为空）",
  "未在 x86/集群复跑（本地 99 CPU, torch 2.7.1, py3.11.6）"
 ],
 "level_denials": {
  "L3": "交付物 untracked（0 commit）+ 零集成调用点；被测库为既有能力",
  "L4": "裁判表 0 行 + 测试红 + 敏感性不全 + 无六段报告 + 3% 精度不达标（13.806%）",
  "L5": "无新物理能力；OSS 参考臂自身失效（限幅饱和/质量漂移）"
 },
 "audit_artifacts": [
  "docs/audit_m18218/AUDIT_m18218_XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS.md",
  "docs/audit_m18218/repro_check.py",
  "docs/audit_m18218/repro_results.json",
  "docs/audit_m18218/repro_check.log",
  "docs/audit_m18218/make_audit_figures.py",
  "docs/audit_m18218/evidence_snapshot/SHA256SUMS.txt"
 ]
}$json$, 'board-audit');
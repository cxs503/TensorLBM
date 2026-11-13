import json, subprocess, pathlib
A = pathlib.Path('/root/tp-agent/work/m17890/artifacts/m17890')
B = '/root/tp-agent/work/m17846/benchmarks/xval_ib3d_diffuse_interface_mdf'
num = json.load(open(A/'audit_numbers.json'))
num = json.loads(json.dumps(num))
ev = {
 "impl_files": ["/root/tp-agent/work/m17846/src/tensorlbm/ibm3d_mdf.py"],
 "entry": f"{B}/run_xval.py",
 "run_artifacts": [
   f"{B}/artifacts/xval_summary.json",
   f"{B}/artifacts/operator_xval.json",
   f"{B}/artifacts/physics_mdf_iter5_peskin4pt.json",
   f"{B}/artifacts/physics_mdf_iter1_peskin4pt.json",
   f"{B}/artifacts/physics_mdf_iter8_peskin4pt.json",
   f"{B}/artifacts/physics_mdf_iter5_cosine4pt.json",
   f"{B}/artifacts/log_mdf_iter5_peskin4pt.txt",
   f"{B}/figures/fig3_cd_history.png",
   "/root/tp-agent/work/m17890/artifacts/m17890/auditor_physics_rerun_n5.json",
   "/root/tp-agent/work/m17890/artifacts/m17890/auditor_physics_rerun_n5.log",
   "/root/tp-agent/work/m17890/artifacts/m17890/auditor_operator_rerun.json",
   "/root/tp-agent/work/m17890/artifacts/m17890/fig_m17890_cd_gate.png",
   "/root/tp-agent/work/m17890/docs/board_audit/m17890/AUDIT.md"],
 "tests": ["tests/test_ibm3d_mdf.py（10 项；auditor 实跑 10 passed in 5.74s）"],
 "benchmark": {
   "anchor_slug": "XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF",
   "tensor_benchmark_cases_rows_before_audit": 0,
   "tolerance": "<15% per case (mission goal 预声明; owner 精度标准)",
   "ref": "Schiller-Naumann Cd=24/Re*(1+0.15Re^0.687): Re=100 -> 1.09174; 交付用 Re_measured 100.23 -> 1.09054",
   "got": {"mdf_iter1_peskin4pt": {"cd_drive": 1.4246, "err_drive_pct": 30.63, "cd_marker": 0.9140, "err_marker_pct": -16.19},
           "mdf_iter5_peskin4pt": {"cd_drive": 1.4098, "err_drive_pct": 29.28, "cd_marker": 0.9052, "err_marker_pct": -17.00},
           "mdf_iter8_peskin4pt": {"cd_drive": 1.4092, "err_drive_pct": 29.22, "cd_marker": 0.9048, "err_marker_pct": -17.03},
           "mdf_iter5_cosine4pt": {"cd_drive": 1.4266, "err_drive_pct": 30.81, "cd_marker": 0.9123, "err_marker_pct": -16.35}},
   "operator_parity_vs_executed_oss": {"stencil_weight_rel_diff": [0.0, 0.0, 1.9e-16], "index_match": 1.0,
       "grid_force_rel_max": 5.113336213409291e-16 if False else 5.113e-16, "reaction_rel_max": 1.25e-15, "sum_ds_rel_diff": 3.06e-16},
   "verdict": "FAIL: 4/4 算例、2/2 口径全越 15%（驱动口径 +29.2…+30.8%，标记口径 −16.2…−17.0%）；两口径互差 ~55% 且把 ref 夹在中间 ⇒ 未时间收敛",
   "auditor_rerun": num["auditor_physics_rerun"]},
 "repro_cmd": ("cd /tmp/m17890rt/benchmarks/xval_ib3d_diffuse_interface_mdf && "
               "python3 run_xval.py --stage operator && "
               "python3 run_xval.py --stage physics --tag auditor_n5 --impl mdf --n-iter 5 --kernel peskin4pt --max-steps 3000 --threads 8 "
               "# auditor 实跑：operator parity 与 physics Cd_drive=1.4098021406601322 / Cd_marker=0.90516944448701 与交付逐位一致；"
               "pytest tests/test_ibm3d_mdf.py -> 10 passed"),
 "integration_callers": None,
 "external_anchor": ("haimingz/vivsim @fb84b12f02e18f3b9dfdf738d6a1a318d81539f9 (MIT) 被逐字节 vendor 于 "
   f"{B}/vendor/vivsim/；auditor 复算 PROVENANCE.json sha256 9/9 一致；执行经 jaxshim/ NumPy 替身（jaxlib 无 loongarch64 轮子），原版 JAX 未运行"),
 "level": 2, "level_basis": ("L2：真实 MDF 实现 + 入口 + 可复现工件（复跑逐位一致）；L3 不授：主求解器/平台零调用方"
   "（grep 仅命中自身与 benchmarks/run_xval.py，后者还调用私有 _lbm_step）；L4 不授：基准 15% 门 4/4 失败。"),
 "overclaims_checked": [
   "『对标通过』不成立：物理层 honest-FAIL（+29.3%/−17.0%）——报告与 result 自认，未粉饰 ✅",
   "交付未入库：branch agent/m17846 相对 be78a87b 0 commit，git status 4 项 '??' ⇒ 平台上无可推送 commit",
   "基准未登记：tensor_benchmark_cases 该锚点 0 行（本单已补登记 status=failed）"],
 "defects": [
   f"{B}/run_xval.py: stage_physics 内 drive_force = g*float(rho.sum()*0.0)+g*float((rho*fluid).sum()) 首项恒 0（死代码）",
   f"{B}/run_xval.py: 驱动掩膜 fluid=(rr>D/2) 与漫散 IBM ±2 格壳层重叠 → 壳层双重注力，驱动/反力口径不闭合（报告已列为首要失败模式）",
   f"{B}/artifacts/log_mdf_iter5_peskin4pt.txt: Cd_marker 逐 50 步在 0.0089–3.4316 振荡（非物理，Re=100 球应定常）"],
 "recommended_action": ("开返修单：① 先 commit + 登记 benchmark case；② 驱动掩膜改 r>D/2+2 或改进出口 BC；"
   "③ IBM 力时域平滑、≥10^4 步判平台；④ 接入主求解器公开 API 后方可申 L3/L4。"),
 "audit_mission": 17890,
}
scope = 'm17846 XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF'
sql_ev = json.dumps(ev, ensure_ascii=False).replace("'", "''")
case = json.dumps({
  "case_name": "XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF: 3D 漫散界面多重直接力(MDF) 球 Re=100 阻力，4 组 n_iter/kernel × 2 口径(drive/marker)，跨对标 haimingz/vivsim 算子层 parity",
  "ref_source": "Schiller-Naumann 关联式 Cd=24/Re*(1+0.15Re^0.687)（解析关联，非实验）+ 被执行的 OSS 算子 parity（haimingz/vivsim @fb84b12f，MIT，逐字节 vendored）",
  "metric": "Cd（drive 闭合口径 2F_drive/(rho U^2 A) 与 marker 反力口径）逐例误差 %；算子层：模板权重/索引命中/网格力/反力/Σds 相对差",
  "tolerance": "<15% per case per 口径（mission 17846 goal 预声明；owner 精度标准）",
  "repo": "TensorLBM",
  "status": "failed",
}, ensure_ascii=False).replace("'", "''")
sql = f"""
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF', '{case.split('"case_name": "')[1].split('", "ref_source"')[0]}',
       '{case.split('"ref_source": "')[1].split('", "metric"')[0]}',
       '{case.split('"metric": "')[1].split('", "tolerance"')[0]}',
       '<15% per case per 口径',
       'TensorLBM', 'failed'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-IB3D-DIFFUSE-INTERFACE-MDF');
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', '{scope}', 17890, 2, '{sql_ev}'::jsonb, 'board-audit')
RETURNING id, level;
"""
pathlib.Path('/tmp/m17890_verdict.sql').write_text(sql)
p = subprocess.run(['su','postgres','-c','psql -d tensorplatform -f /tmp/m17890_verdict.sql'],capture_output=True,text=True)
print(p.stdout[-800:], p.stderr[-800:])

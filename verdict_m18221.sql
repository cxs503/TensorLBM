-- m18221 交差核验 verdict: 对象 m18128 XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL (TensorLBM)
-- 判级 L3：真驱动+生产 SCMP 路径上 3 个算例逐位复现+对独立重建的 PR-Maxwell 参考做误差分析；
-- L4 不授：预登记加密臂 L=96/128 未执行（网格非单调 9.079%→9.788%）+ 该锚 tensor_benchmark_cases 0 行 + Laplace 臂空。
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm18128 XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL', 18221, 3,
$json${
 "impl_files": [
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/xval_driver.py",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/run_one.py",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/snapshot.py",
  "docs/audit_m18221/evidence_snapshot/xval_driver.py",
  "/root/tp-agent/work/m18128/src/tensorlbm/multiphase.py"
 ],
 "entry": [
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/run_one.py",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/xval_driver.py"
 ],
 "run_artifacts": [
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/results.json",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/cells.jsonl",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/snapshot.npz",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/NUMBERS.md",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/oss_facts.json",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/figures/fig1_psi_conformance.png",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/figures/fig2_planar_profiles.png",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/figures/fig3_convergence.png",
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/figures/fig4_droplet_fields.png",
  "audit_m18221/cells_audit.jsonl",
  "audit_m18221/audit_pr_xval.json",
  "audit_m18221/fig_m18221_audit.png",
  "docs/audit_m18221/evidence_snapshot/SHA256SUMS.txt"
 ],
 "tests": [
  "tests/test_multiphase_psi_eos.py::test_corrected_psi_reconstructs_maxwell_pressure_lock",
  "tests/test_multiphase_psi_eos.py::test_corrected_sign_attractive_and_isotherm_loops",
  "tests/test_multiphase_psi_eos.py::test_forcing_velocity_shift_branch_is_bit_identical_to_legacy"
 ],
 "integration_callers": [
  "collide_sc_single_component",
  "benchmarks/eos_maxwell/probe_lib.py:157 psi_fn = make_psi_eos(eos, T) ; :390 run_probe -> collide_sc_single_component + solver.stream",
  "src/tensorlbm/multiphase.py:271 make_psi_eos -> :294 psi_eos_peng_robinson",
  "src/tensorlbm/multiphase3d.py:32 , multiphase3d_d3q27.py:35 import psi_peng_robinson"
 ],
 "benchmark": {
  "registered_rows_for_anchor": 0,
  "note": "tensor_benchmark_cases has 0 rows for XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL (94 rows global) -> referee is analytic PR-Maxwell + own prereg, audit rebuilt it",
  "e1_conformance": {
   "ref": "OSS kernel text evaluated (c0=cs^2,G=-1)",
   "got": "psi_eos_peng_robinson",
   "max_rel_dev": 2.2198792994400856e-16,
   "audit_recompute": 2.220446049250313e-16,
   "gate": 1e-06
  },
  "e2_ratio10_L48": {
   "ref_ratio": 10.0096,
   "got_ratio": 10.918366420659497,
   "err_pct": 9.079,
   "rho_l_ref": 5.8867,
   "rho_l_got": 5.8643927574157715,
   "rho_v_ref": 0.5881,
   "rho_v_got": 0.5371126532554626,
   "rho_v_err_pct": 8.67
  },
  "e2_ratio10_L64": {
   "ref_ratio": 10.0096,
   "got_ratio": 10.989307943289456,
   "err_pct": 9.788,
   "rho_v_err_pct": 9.21
  },
  "e2_ratio18p6_L64": {
   "ref_ratio": 18.6474,
   "got_ratio": 23.293563227304958,
   "err_pct": 24.92,
   "verdict": "honest FAIL (>15%)"
  },
  "maxwell_independent_vs_delivery_max_rel_diff": {
   "ratio10": 2.43e-06,
   "mid": 1.24e-06,
   "highT": 6.43e-07
  }
 },
 "repro_cmd": "cd /root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL && python3 run_one.py --tag audit_repro_planar_L48_ratio10 --geometry planar --L 48 --tr 0.8266 --steps 20000 --threads 4 --out /root/tp-agent/work/m18221/audit_m18221/cells_audit.jsonl  -> ratio 10.918366420659497 err 0.09079017058552016 ma 0.11543023569630247 (75.0 s), bit-identical to committed;  audit side: python3 audit_m18221/audit_pr_xval.py (E1 + independent Maxwell)",
 "external_anchor": [
  "/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/artifacts/oss_facts.json",
  "audit_m18221/audit_pr_xval.json"
 ],
 "verdict_basis": "L3: real driver + 3 production LBM cells rerun bit-identically through the production SCMP path (probe_lib -> collide_sc_single_component + stream) + error analysis vs an analytically rebuilt PR-Maxwell reference. L4 denied: prereg refinement arm (L=96/128) never ran -> grid series is non-monotone 9.079%(L48) -> 9.788%(L64), no demonstrated convergence; 0 registered benchmark rows for the anchor; Laplace arm empty. L5 denied: no new capability, psi PR pre-exists (last touched merge #1781/m15063), OSS side not runnable (kernel dead code, numba.cuda).",
 "findings": [
  {
   "id": "F1",
   "sev": "moderate",
   "what": "tensor_benchmark_cases has 0 rows for the anchor -> no registered referee"
  },
  {
   "id": "F2",
   "sev": "moderate",
   "what": "prereg refinement arm L=96/128 not executed; grid non-monotone 9.079->9.788 pct"
  },
  {
   "id": "F3",
   "sev": "moderate",
   "what": "XVAL is formula-level only: OSS kernel 0 call sites, not runnable; disclosed by the report (A3 branch)"
  },
  {
   "id": "F4",
   "sev": "minor",
   "what": "REPORT 4 cites N7 table absent from NUMBERS.md; droplet Ma=0.115 only in snapshot.npz meta"
  },
  {
   "id": "F5",
   "sev": "minor",
   "what": "laplace arm empty (results.json laplace.points=[]); no surface-tension benchmark"
  },
  {
   "id": "F6",
   "sev": "minor",
   "what": "no test added by the mission; XVAL driver has no regression guard"
  },
  {
   "id": "F7",
   "sev": "info",
   "what": "mission delta = evidence, not code: psi_eos_peng_robinson pre-exists at src/tensorlbm/multiphase.py:249"
  }
 ],
 "checks_not_run": [
  "OSS side not executed (numba.cuda/GPU absent, kernel unwired, no LICENSE)",
  "no L=96/128 convergence study, no cluster rerun",
  "D3Q19/D3Q27 SCMP + forcing='standard' arms not cross-checked",
  "no non-OSS literature anchor consulted for the P-R coexistence values"
 ],
 "declared_level": 3,
 "external_anchor_note": "openLBMPM HEAD 3d841891daaf; ShanChen2D/OptimizedD2Q9GPU.py sha256 cccd2b38...; calFluidPotentialGPUPR call sites = 0 (auditor recount); analytic P-R equal-area Maxwell rebuilt by auditor (rho_l=5.886705, rho_v=0.588105, p_sat=0.024820 at T_r=0.8266; <=2.4e-06 rel vs delivery N5)"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending';

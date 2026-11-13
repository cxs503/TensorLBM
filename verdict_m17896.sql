INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm17896 XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER', 18085, 3, $j${
 "impl_files": [
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:122 young_cos_theta",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:143 analytic_contact_angle",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:169 grad_laplacian (isotropic 9-point)",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:196 WallOrderWettingConfig",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:357 step (f/g populations + Guo force + bounce-back)",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:397 relax_to_equilibrium",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:538 contact_angle_from_field (cap inversion + layer diagnostics)",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/wall_order_parameter_2d.py:649 run_to_equilibrium"
 ],
 "impl_files_notes": "693 lines, real free-energy LBM (D2Q9, double-well f=(a/4)(psi^2-1)^2, isotropic 9-point gradient/Laplacian reading the wall ghost row, Korteweg force via Guo, half-way bounce-back). Audited commit 656d09ed (branch agent/m17896) — NOT contained in the main-lineage worktree m18085.",
 "entry": [
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py:72 arm_port",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py:111 arm_cac",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py:145 sweep",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py:161 summarize",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py:208 main"
 ],
 "run_artifacts": [
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/summary.json (tracked)",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/runs/port_psi+0.00_xi2.0/result.json + psi.npz (untracked)",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/runs/port_psi+0.35_xi3.0/result.json + psi.npz (untracked; psi (56,80) float64, mass_error=1.6e-16)",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/runs/port_psi+0.70_xi3.0/result.json + psi.npz (untracked; layer_fraction=0.998, cap_measurable=false)",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/figures/fig1_contact_angle_law.png (tracked)",
  "/root/tp-agent/work/m17896/xval/LBM_WALL_ORDER_PARAMETER_WETTING/figures/fig2_cap_fields.png (tracked)",
  "/root/tp-agent/work/m18085/artifacts/board_audit/m18085/audit_m17896.json",
  "/root/tp-agent/work/m18085/artifacts/board_audit/m18085/fig_audit_m17896.png",
  "/root/tp-agent/work/m18085/docs/board_audit/m18085/AUDIT_m17896.md"
 ],
 "run_artifacts_notes": "10 port run dirs, each result.json + psi.npz; only summary.json + 2 png are git-tracked (git ls-files), i.e. runs/* exist solely in the worktree /root/tp-agent/work/m17896.",
 "tests": [
  "tests/test_xval_wall_order_wetting.py::test_young_law_from_wall_excess_energy (16 passed total, re-run by this audit: pytest -q -> 16 passed in 59.87s)",
  "tests/test_xval_wall_order_wetting.py:102 test_cap_measurability_flag_separates_window",
  "tests/test_xval_wall_order_wetting.py:120 test_cac_wall_order_policy_derives_the_young_angle",
  "tests/test_xval_wall_order_wetting.py:134 test_cac_wall_order_ghost_matches_prescribed_angle_policy",
  "tests/test_xval_wall_order_wetting.py:152 test_ladder_contact_wall_order_entry_runs"
 ],
 "integration_callers": [
  "/root/tp-agent/work/m17896/src/tensorlbm/cac_lbm2d.py:311-356 wall_phi_policy='wall_order' (production CAC2D solver derives the wall angle via young_cos_theta)",
  "/root/tp-agent/work/m17896/src/tensorlbm/cac_lbm2d.py:57 WALL_POLICIES = (freeze, neumann, contact, wall_order)",
  "/root/tp-agent/work/m17896/src/tensorlbm/wetting/ladder.py:184-247 contact_wall_order -> CAC2D(wall_phi_policy='wall_order')",
  "audit probe m18085: ladder.contact_wall_order(0.35, steps=60, shape=(48,48), radius=12) executed end-to-end in 3.3 s, theta_measured=57.01 deg vs young 59.76 deg (-4.61%, 60 steps, not equilibrated -> connectivity only, no physics claim)",
  "LIMIT: the mission itself delivered n_cac=0 rows for this integrated arm (summary.json), so the integration is wired+tested but has no equilibrated physical result"
 ],
 "benchmark": {
  "case_row": "tensor_benchmark_cases.anchor_slug='XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER' (tolerance: <15% per point, accept_pct=15.0 predeclared)",
  "ref": "analytic Young law cos(theta)=(3 psi_w - psi_w^3)/2 of the SAME free energy (self-consistency, re-derived by this audit via wall-excess-energy quadrature to 1.8e-12)",
  "got_vs_ref": "in-window |psi_w|<=0.5: 9/9 pass, median |err| 1.3566%, max 3.6713%; out-of-window psi_w=+0.70: 29.5158% FAIL",
  "convergence": "0/10 rows converged (all hit the 30000-iteration cap; drift 2.07e-5 at psi_w=+0.35, 4.0e-4 at +0.70)",
  "verdict": "GATE NOT PASSED -> L4 denied",
  "platform_crosscheck": "GET /api/v1/tensor/bench/l4-verdict/17896 -> CLAIMED_NOT_PROVEN, rows=0, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]"
 },
 "repro_cmd": [
  "cd /root/tp-agent/work/m17896 && python3 xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py sweep && python3 xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py summarize",
  "/root/tp-agent/work/m18085/artifacts/board_audit/m18085/repro_m17896.py (audit re-run of psi_w=+0.35/+0.70 at xi=3 on commit 656d09ed source tree: theta 59.85617493741043 / 20.114821952526363, abs_diff 0.0 deg -> bit-identical, non-MOCK)"
 ],
 "external_anchor": {
  "target": "ebinan92/Fingering_dynamics lattice_boltzmann/validation.py (psi_wall_list ghost-row wetting)",
  "status": "WITHHELD",
  "why": "reference implementation diverges to NaN at its own default parameters; the artifact cited by the mission doc (runs/oss_psi+0.00/result.json) does NOT exist in the delivered tree and neither does oss_runner.py; cluster job cj1656 never delivered a number",
  "consequence": "no OSS-vs-port numeric comparison -> the 9/9 in-window agreement is a self-consistency check against the analytic law of the same free energy, not external validation"
 },
 "overclaims_vs_evidence": [
  "doc headline '8/10 points <3% (median 1.29%)' vs artifact: n_in_window=9, median 1.3566%, max 3.6713% (max is >3%)",
  "doc cites runs/oss_psi+0.00/result.json for the NaN finding; file absent from the delivered tree",
  "predeclared declares an independent CAC arm as the second support; delivered n_cac=0",
  "no tensorbench_results row -> any L4 claim is CLAIMED_NOT_PROVEN"
 ],
 "level_boundary": "L2 met (implementation + entry + real run artifacts, independently reproduced bit-identical). L3 met (CAC2D wall_phi_policy='wall_order' integration caller + exercised by 2 passing tests + audit probe). L4 NOT met (1/10 point at 29.52% > 15%, 0/10 converged, external anchor WITHHELD, zero platform-recomputed benchmark rows).",
 "audit_independence": "auditor m18085 did not author m17896; reproduction executed against the audited commit's own source tree; no number in this verdict was copied from the mission report"
}$j$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL, created_at = now();

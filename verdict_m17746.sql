INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorLBM', 'm17678 XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION', 17746, 2, $j${
 "impl_files": [
  "/root/tp-agent/work/m17678/src/tensorlbm/phasefield/cac_h_population.py",
  "/root/tp-agent/work/m17678/src/tensorlbm/cac_lbm2d.py"
 ],
 "impl_files_notes": {
  "new_module": "/root/tp-agent/work/m17678/src/tensorlbm/phasefield/cac_h_population.py:86-473 (Stencil/get_stencil L86-146, isotropic_gradient L173, isotropic_laplacian L188, interface_normal L204, counter_term L223, h_equilibrium L245, CACHPopulation L274, update_from_tclb L432, width_from_profile L442; D2Q9/D3Q19/D3Q27)",
  "arm_a": "/root/tp-agent/work/m17678/src/tensorlbm/cac_lbm2d.py (pre-existing integrated kernel, cross-arm reference)",
  "git": "UNTRACKED: git ls-files on all 6 deliverable paths is empty"
 },
 "entry": "/root/tp-agent/work/m17678/scripts/xval_m17678_tclb_cac_h_population.py",
 "entry_notes": {
  "driver": "921 lines; TCLB source parsed at run time (zero hand-copy)",
  "docs": [
   "/root/tp-agent/work/m17678/docs/XVAL_M17678_TCLB_CONSERVATIVE_AC_H_POPULATION.md",
   "/root/tp-agent/work/m17678/docs/XVAL_M17678_PREREG.md (pre-registered thresholds)"
  ],
  "all_untracked": true
 },
 "run_artifacts": [
  "/root/tp-agent/work/m17678/artifacts/xval_m17678/xval_tclb_cac_h_population.json",
  "/root/tp-agent/work/m17678/artifacts/xval_m17678/curves.json",
  "/root/tp-agent/work/m17678/artifacts/xval_m17678/fig1_static2d_profile_mass.png",
  "/root/tp-agent/work/m17678/artifacts/xval_m17678/fig4_static3d_sphere.png",
  "/root/tp-agent/work/m17746/artifacts/board_audit/m17746/audit_m17678.json",
  "/root/tp-agent/work/m17746/artifacts/board_audit/m17746/fig_audit_m17678.png",
  "/root/tp-agent/work/m17746/docs/board_audit/m17746/AUDIT_m17678.md"
 ],
 "run_artifacts_notes": {
  "full": "xval_tclb_cac_h_population.json 17069 B (graded P1..P3d)",
  "quick": "xval_tclb_cac_h_population_quick.json - auditor re-ran --quick in 100.0 s; quick adv centroid 5.27% / field 2.47% vs graded full 1.31% / 1.69% => quick is NOT a substitute for the graded run",
  "figures": "fig2_advection_cut.png, fig3_sensitivity.png also present; PNG pixel content not re-derived by auditor"
 },
 "tests": "/root/tp-agent/work/m17678/tests/test_xval_m17678_cac_h_population.py",
 "tests_notes": "3 tests (L14 counter-term vs documented anti-diffusion source, L29 static droplet conserves phase and stays put, L41 arm A vs arm B static limit) - auditor re-ran: 3 passed in 9.34 s",
 "integration_callers": [],
 "integration_callers_notes": "NONE: grep -rn 'cac_h_population|CACHPopulation' over src/ benchmarks/ tests/ examples/ excluding the module itself, its driver (scripts/xval_m17678_tclb_cac_h_population.py:42) and its test (tests/test_xval_m17678_cac_h_population.py:9) => 0 hits; not in src/tensorlbm/phasefield/__init__.py __all__; no registry/capability-contract entry => L3 NOT met",
 "benchmark": {
  "registry": "tensor_benchmark_cases: 0 rows for anchor_slug='XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION' (also 0 for ILIKE '%ALLEN%' / '%conservative%'), repo=TensorLBM mission=17678 -> platform owns no machine criterion",
  "ref": {
   "analytic": "tanh interface profile, exact translation, exact flux-form mass conservation",
   "oss_text": "TCLB models/multiphase/d3q27_pf_velocity (Dynamics.c.Rt L398 tmp1, L239 calcGamma, L928 collision, model.R IsotropicGrad/myLaplace, src/lib/lattice.R)"
  },
  "got": {
   "P1_counter_term_rel_err": 0.0,
   "P2_laplacian_rel_err": 2.3036533959436224e-16,
   "P2_gradient_max_rel_err": 1.4281160455735658,
   "P9a_rel_err": 0.2576796459661831,
   "P9b_rel_err": 6.247638978190394e-10,
   "P3_mass_drift_B": 7.441539155000438e-14,
   "P4_R_drift_B": 0.002041860221262723,
   "P5_profile_err_B": 0.003118629639963296,
   "P6_centroid_err_B": 0.01310277380787861,
   "P6_field_err_B": 0.01690103373676891,
   "P7_static_cross_arm_rel_l2": 1.1693504783301234e-06,
   "P7_advection_cross_arm_rel_l2": 0.006578457363489439,
   "P8_width_law_ratios": [
    4.166769959297429,
    2.084,
    1.031,
    0.686879475463341
   ],
   "P3d_mass_drift_3d": 1.5016523733769898e-14
  },
  "status": "PASS on the primary criterion (cross-arm L2 1.169e-6 static / 0.6578% advection, both <3%) and on analytic refs (0.312%-1.69%); FAIL (subject-declared honest FAIL, not overwritten) on P2 gradient 1.428, P9a 3D one-step 0.2577, P8 width law 3/4 M points",
  "tolerance": 0.15
 },
 "repro_cmd": "cd /root/tp-agent/work/m17678 && python3 scripts/xval_m17678_tclb_cac_h_population.py --quick  -> auditor executed, exit 0, 100.0 s (rewrote quick json); full mode (353 s) NOT re-run by auditor; python3 -m pytest tests/test_xval_m17678_cac_h_population.py -q -> 3 passed",
 "external_anchor": [
  "/root/tp-agent/work/m17678/artifacts/xval_m17678/xval_tclb_cac_h_population.json"
 ],
 "external_anchor_notes": [
  "/tmp/TCLB models/multiphase/d3q27_pf_velocity still present; model.R sha256 c8a220a943555013170872db99f3e04dab3feceacaadad888472154c1aa91f36f7 recomputed by auditor, identical to the artifact provenance block (GPL-3.0, reference-only)",
  "Liang, Xu, Chen, Wang, Chai & Shi, PRE 97, 033309 (2018) - Arm A lineage",
  "Fakhari & Mitchell lineage (TCLB d3q27_pf_velocity) - Arm B recipe"
 ],
 "level_rationale": "L1 met (operator identities P1 0.0 / P2 laplacian 2.3e-16 / P9b 6.25e-10, 3 unit tests green re-run by auditor). L2 met (473-line production module, real input->compute->output, D2Q9/D3Q19/D3Q27, driven end-to-end on 2D/3D cases). L3 NOT met (0 call sites outside its own driver/test; not in phasefield/__init__ __all__; no registry entry). L4 NOT met despite good numbers: 0 rows in tensor_benchmark_cases (no machine criterion), all 6 deliverable paths untracked (git ls-files empty -> lost on any fresh checkout), 3 self-declared gate failures including P2 gradient 1.428 and P9a 0.2577 which undercut the structural-identity claims, and the auditor reproduced only --quick (quick adv centroid 5.27% vs graded full 1.31%), so the graded full-run numbers are provenance-verified, not independently recomputed. Same blocker signature as board precedent m17664 (0 call sites + untracked => L2). L5 not applicable (consistency study, no experimental data).",
 "audit_limits": [
  "full run (353 s) not re-executed; all graded numbers read from the subject artifact, so they are verified-by-provenance, not auditor-reproduced",
  "quick mode is not a substitute: quick adv centroid 5.27% / field 2.47% vs graded full 1.31% / 1.69%",
  "the 4 subject PNGs were confirmed to exist but their pixel content was not re-derived",
  "TCLB was not compiled/run (licence + architecture): the 'ref' side is TCLB source text, not TCLB output",
  "no physical-accuracy claim: cross-arm agreement certifies recipe parity, not physics; no g-LBE coupling and no wall wetting in this mission",
  "upstream identity of /tmp/TCLB (beyond sha256 matching the artifact) not verified against GitHub"
 ],
 "overstated_claims": [
  "report wording 'structural layer identical' is only 2/3 true: gradient instrument 1.428 and 3D one-step 0.2577 remain unconverged; their root-cause attribution (own R->Python rebuild / reference pipeline) is plausible but unproven",
  "interface-width law W_eff=3MW holds only at M=1/3 (ratio 1.031); 3 of 4 M points contradict it (4.167 / 2.084 / 0.687) - W is an anti-diffusion coefficient, not a measured width"
 ],
 "findings": [
  {
   "id": "F1",
   "severity": "blocking",
   "claim": "6 deliverable paths untracked (git ls-files empty) - mission marked succeeded while artifacts live only in /root/tp-agent/work/m17678",
   "evidence": "git ls-files for module/script/test/2 docs/1 json -> all empty; git status -> 6 '??' entries"
  },
  {
   "id": "F2",
   "severity": "blocking",
   "claim": "no registered benchmark row -> platform's own gate has nothing to judge",
   "evidence": "psql tensor_benchmark_cases WHERE anchor_slug=... -> 0 rows"
  },
  {
   "id": "F3",
   "severity": "major",
   "claim": "two structural-identity instruments unconverged (1.428 / 0.2577) - root cause attributed but unproven",
   "evidence": "checks/P2_gradient_max_rel_err=1.428 (per-axis 1.428 vs 3.3e-16), checks/P9a_rel_err=0.2577, P9b=6.25e-10"
  },
  {
   "id": "F4",
   "severity": "major",
   "claim": "width law falsified at 3 of 4 mobility points; only the paired M=1/3 point validates",
   "evidence": "checks/P8_ratio_min=0.6869, P8_ratio_max=4.1668"
  },
  {
   "id": "F5",
   "severity": "minor",
   "claim": "quick mode silently different from graded full mode",
   "evidence": "auditor --quick run: adv centroid 5.266% / field 2.474% vs full 1.310% / 1.690%"
  },
  {
   "id": "F6",
   "severity": "minor",
   "claim": "width_from_profile atanh sign defect left in the frozen artifact (disclosed by subject)",
   "evidence": "P8 negative-width trace in xval_tclb_cac_h_population.json; report note 4"
  }
 ],
 "disposition": "request changes (reopen for L3-L4): commit the 6 deliverable paths; register 3 tensor_benchmark_cases rows; add one main-solver call site; fix the P2-gradient and P9a instruments and mark quick artifacts not_graded. Do NOT retune reference values to make P8 pass - honest FAIL stays."
}$j$::jsonb, 'board-audit') ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id;

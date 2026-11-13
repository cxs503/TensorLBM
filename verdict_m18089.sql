INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorLBM', 'm17981 XVAL-LBM-HORIZONTAL-SURVEY-EMPTY', 18089, 2, $j${
 "impl_files": [
  "/root/tp-agent/work/m17981/src/tensorlbm/oscillatory_cylinder_inertia.py",
  "/root/tp-agent/work/m17981/cases/oscillatory_cylinder_inertia/run.py",
  "/root/tp-agent/work/m17981/xval/horizontal_survey_empty/survey.py",
  "/root/tp-agent/work/m17981/xval/horizontal_survey_empty/run_arm.py",
  "/root/tp-agent/work/m17981/xval/horizontal_survey_empty/analyze.py",
  "/root/tp-agent/work/m17981/xval/horizontal_survey_empty/predeclared.json"
 ],
 "impl_files_notes": {
  "module": "src/tensorlbm/oscillatory_cylinder_inertia.py (411 lines): circle_mask L104, geometry_readout L113, _interior_nodes L141, _surface_normals L152, _traction_force L170, harmonic_decompose L194, inertia_coefficient L234, potential_reference L244, run_arm L265, main L376 - real body, no placeholder",
  "survey": "xval/horizontal_survey_empty/survey.py FAMILIES L52, main L161 (pure-CSV scan of the provenance dumps, no DB at run time)"
 },
 "entry": [
  "/root/tp-agent/work/m17981/cases/oscillatory_cylinder_inertia/run.py",
  "/root/tp-agent/work/m17981/src/tensorlbm/oscillatory_cylinder_inertia.py"
 ],
 "entry_notes": {
  "case_runner": "python cases/oscillatory_cylinder_inertia/run.py --case res16 --out runs/oscillatory_cylinder_inertia",
  "module_cli": "python -m tensorlbm.oscillatory_cylinder_inertia --case quick",
  "auditor_replay": "auditor ran the module CLI for 'quick' inside the m17981 tree: C_I=1.6758722921650897, err 57.8756% -> bit-identical to runs/oscillatory_cylinder_inertia/quick/result.json (deterministic, zero-variance)"
 },
 "run_artifacts": [
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/quick/result.json",
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/quick/trace.npz",
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/res16/result.json",
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/res16/trace.npz",
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/base_m17844/result.json",
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/base_m17844/trace.npz",
  "/root/tp-agent/work/m17981/runs/oscillatory_cylinder_inertia/benchmark_table.json",
  "/root/tp-agent/work/m17981/docs/assets/xval_lbm_horizontal_survey_empty/fig1_convergence.png",
  "/root/tp-agent/work/m17981/docs/assets/xval_lbm_horizontal_survey_empty/fig2_sensitivity.png",
  "/root/tp-agent/work/m17981/docs/assets/xval_lbm_horizontal_survey_empty/fig3_velocity_trace.png",
  "/root/tp-agent/work/m18089/artifacts/board_audit/m18089/audit_m17981.json",
  "/root/tp-agent/work/m18089/artifacts/board_audit/m18089/fig_audit_m17981.png",
  "/root/tp-agent/work/m18089/artifacts/board_audit/m18089/benchmark_case_row.json",
  "/root/tp-agent/work/m18089/artifacts/board_audit/m18089/pytest_m17981.log"
 ],
 "run_artifacts_notes": {
  "trace_schema": "new arm: t/F_mem/F_trac/U/omega/a0/A_solid; pre-module arm: t/U/F1/F2/dU/a_t (scalars from result.json numerics)",
  "auditor_refit": "independent least-squares re-fit of the first harmonic from trace.npz reproduces the reported C_I bit-for-bit: quick E1 1.675872 / E2 1.616430, res16 E1 1.831697 / E2 1.880691; the pre-module arm reproduces within 1.85% (it fits C_M and C_D jointly with the |U|U drag basis)",
  "figures": "three mission PNGs checked for existence/size only, pixels not re-derived"
 },
 "tests": [
  "tests/test_oscillatory_cylinder_inertia.py::test_geometry_readout_matches_circle",
  "tests/test_oscillatory_cylinder_inertia.py::test_harmonic_decompose_recovers_synthetic",
  "tests/test_oscillatory_cylinder_inertia.py::test_harmonic_window_orthogonality",
  "tests/test_oscillatory_cylinder_inertia.py::test_harmonic_decompose_flags_nonlinear_residual",
  "tests/test_oscillatory_cylinder_inertia.py::test_inertia_coefficient_normalisation",
  "tests/test_oscillatory_cylinder_inertia.py::test_potential_reference_bounds_and_conventions",
  "tests/test_oscillatory_cylinder_inertia.py::test_quick_arm_is_stable_and_self_consistent",
  "tests/test_oscillatory_cylinder_inertia.py::test_cli_list_runs"
 ],
 "tests_notes": "auditor re-ran the suite in the m17981 tree: 8 passed in 18.69 s (first auditor run 15.56 s). All eight live on helper level (geometry readout, synthetic harmonic recovery, window orthogonality, nonlinearity flag, normalisation, potential-reference bounds, quick-arm self-consistency, CLI) - none asserts the physics error against the reference, so green tests do not certify agreement",
 "integration_callers": [],
 "integration_callers_notes": "grep -rn oscillatory_cylinder_inertia over the whole m17981 tree hits only the module itself, its own test, cases/oscillatory_cylinder_inertia/run.py and docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md => 0 non-self callers; not exported in src/tensorlbm/__init__.py, no capability/registry entry => L3 NOT met",
 "benchmark": {
  "registry": "tensor_benchmark_cases row for anchor_slug='XVAL-LBM-HORIZONTAL-SURVEY-EMPTY' exists (dumped to artifacts/board_audit/m18089/benchmark_case_row.json); row text: 'gate NOT passed (status left pending on purpose)', registered by audit m17888 because the row was absent at delivery time",
  "tolerance": "<15% (m17844 predeclared, owner standard); ecosystem line <3%",
  "ref": {
   "registry_convention": {"C_M_ideal_fluid": 2.0, "source": "Morison, O'Brien, Johnson & Schaaf (1950); Sarpkaya (1986): 1 Froude-Krylov + C_a=1"},
   "delivery_convention": {"c_a_area_D12": 1.061515, "c_a_area_D16": 1.059697, "c_a_area_D24": 1.044081, "source": "tensorlbm.fsi_added_mass.potential_added_mass on the identical mask, area-normalised (internal code, +6.0% above the analytic unbounded C_a=1)"},
   "external_anchor": {"analytic_unbounded_C_a": 1.0, "note": "no experimental oscillatory-flow added-mass data used"}
  },
  "got": {
   "registry_base_E1_halfway_C_M": 1.5742086608150845,
   "registry_base_err_pct": -21.29,
   "quick_E1_area_C_I": 1.6758722921650897,
   "quick_err_vs_same_mask_pct": 57.87558714635476,
   "quick_err_vs_analytic_pct": 67.58722921650897,
   "res16_E1_area_C_I": 1.8316966383748203,
   "res16_err_vs_same_mask_pct": 72.85094365219824,
   "res16_err_vs_analytic_pct": 83.16966383748203,
   "base_E1_area_C_I": 1.6937150807149621,
   "base_err_vs_same_mask_pct": 62.22,
   "estimator_spread_pct": {"quick": 0.05944197, "res16": 0.04899403, "base_m17844": 4.86214613},
   "error_trend_over_D12_16_24_pct": [57.88, 72.85, 62.22]
  },
  "status": "FAILED - no arm within 15%, let alone the 3% ecosystem line; smallest error 57.9% is 3.9x the gate; predeclared arms res32 and kc20 were never run; honest FAIL, gate NOT passed",
  "honest_fail_respected": "auditor reproduced the FAIL rather than overwriting it; no frozen mission artifact was modified"
 },
 "repro_cmd": "cd /root/tp-agent/work/m17981 && python -m pytest tests/test_oscillatory_cylinder_inertia.py -q  (8 passed) ; PYTHONPATH=src python -m tensorlbm.oscillatory_cylinder_inertia --case quick --out /tmp/repro_quick  (C_I=1.6758722921650897 bit-identical) ; PYTHONPATH=src python xval/horizontal_survey_empty/survey.py --out <existing-dir>  (8/8 family verdicts identical; the directory must pre-exist, survey.py does not mkdir) ; full auditor tool: python tools/audit_m17981_verify.py --work /root/tp-agent/work/m17981 --out artifacts/board_audit/m18089 --bench-row artifacts/board_audit/m18089/benchmark_case_row.json",
 "external_anchor": "literature only: Morison 1950 / Sarpkaya 1986 ideal-fluid circular cylinder C_a=1; plus the local OSS survey verdict 'empty' over the dumps (oss_inventory 33 rows, oss_solutions 261 rows, 45 repos) - that emptiness is a statement about this ledger, not about the open-source world (lettuce/lbmpy ship cylinder cases; family keywords are narrow)",
 "verdict_notes": {
  "level_reason": "L2 as built: real code input->compute->output (full D2Q9 periodic box, Guo forcing, halfway bounce-back, two independent force estimators, Morison least squares), real artifacts, bit-identical replay, 8 unit tests green. NOT L3: 0 non-self callers / no registration. NOT L4: benchmark gate FAILED (19-83% error) and 2 predeclared arms never ran. NOT L1: it is a complete executable physics case, not a unit-level check.",
  "capability_state": "WITHHELD - oscillatory-flow inertia / added-mass coefficient is implemented and measurable, but not verified within 15% of any reference convention",
  "overstated_or_unsupported_claims": [
   "registry row supports only the pre-module (m17844) Morison-convention number (-21.29%); it does not contain the rework's same-mask results (+57.9%/+72.9%) that the delivery headlines",
   "docs section 5 window-sensitivity number C_I=1.541 has no landed artifact and cannot be re-derived",
   "docs section 3 states transient 1.5T / window 3T while the quick case actually runs 1.0T / 2.0T",
   "convergence claim rests on 2 resolutions (D=12,16) although 3 are predeclared (res32 never ran)",
   "platform report tensor_reports.id=16915 carries figures=[] although three PNG artifacts exist, and neither report nor docs answers the '<3%' question explicitly"
  ],
  "defects_found": [
   "same-mask reference is the repository's own fsi_added_mass output (internal), +6.0% above the analytic C_a=1 anchor; convention switched after the -21% was observed (both conventions fail, so the verdict is unchanged)",
   "KPI normalisation uses nominal a0 while KC/beta use measured U (a0_measured/a0_nominal=1.014 -> ~1.4% on C_I, does not change the verdict)",
   "survey.py --out does not mkdir, so the documented repro command fails on a fresh directory",
   "registry registration duty drifted to the auditor (row absent at delivery)"
  ],
  "not_run": [
   "physics re-run of res16/base (capacity discipline: only bit-identical replay of quick plus trace re-fit)",
   "res32 / kc20 (do not exist)",
   "pixel re-derivation of the three mission figures",
   "comparison against experimental oscillatory-flow added-mass data",
   "consistency check of the local dumps against the production oss tables"
  ]
 },
 "audit_report": "docs/board_audit/m18089/AUDIT_m17981.md",
 "auditor_tool": "tools/audit_m17981_verify.py"
}$j$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id = EXCLUDED.mission_id,
 level = EXCLUDED.level, evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL;
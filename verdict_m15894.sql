-- Board audit verdict — m15894 auditing mission m15470 (CASE-MCMP-CONTACT-ANGLE, TensorLBM)
-- Disposition: level 2 (independent functional implementation, honest-FAIL case), verdict_by='board-audit'.
--   L3 REFUTED: no main-solver / platform caller (grep over src,tests,benchmarks -> harness only);
--               the only library-side change is an 11-line docstring WARNING (semantics unchanged).
--   L4 REFUTED: case FAILS 0/6 (stage A, max |delta| = 89.9999 deg) and 0/3 (stage B hold-out,
--               max 19.12 deg) against the pre-registered criterion -> no capability validated;
--               ref theta_young is produced by the very inversion under test (weak-by-design);
--               0 referee rows in tensor_benchmark_cases, 0 tensorbench_results rows for m15470;
--               run artifacts never reached repo main (merge #1825 = ee578b8f carried 2 files).
-- Independently verified by m15894: artifact presence (41 files), ref-vs-got recomputation from the
-- JSON (not from report prose), bit-exact spot re-execution (delta 0.0 deg), pytest claim (8 passed),
-- six-section report present, repro_cmd executable.
-- Idempotent: re-running updates the same verdict row.
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15470 CASE-MCMP-CONTACT-ANGLE',
  15894,
  2,
  $json${
 "impl_files": [
  "/root/tp-agent/work/m15470/tests/benchmarks/wetting_contact_angle/wetting_bench.py",
  "/root/tp-agent/work/m15470/tests/benchmarks/wetting_contact_angle/contact_angle.py",
  "/root/tp-agent/work/m15470/tests/benchmarks/wetting_contact_angle/run_case_cli.py",
  "/root/tp-agent/work/m15470/tests/benchmarks/wetting_contact_angle/assemble.py",
  "/root/tp-agent/work/m15470/tests/benchmarks/wetting_contact_angle/make_report.py",
  "/root/tp-agent/work/m15470/TensorLBM/src/tensorlbm/wetting/__init__.py"
 ],
 "impl_notes": "wetting_bench.py = SCMP sessile-drop driver (run_case / calibration_curve / invert_calibration); contact_angle.py:74 = three estimators + theta_lib = 180 - theta_phys convention; assemble.py = verdicts + estimator_control + grid sensitivity + B2 hold-out plan; make_report.py renders tables from the JSON. Library file change in commit 4256e939 (git show confirmed): +11 lines docstring WARNING on wall_psi_from_contact_angle, code semantics unchanged.",
 "entry": "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/repro.sh",
 "entry_note": "repro.sh (mode 755, 1826 B) drives run_all.sh a|b1|b2 + assemble.py + make_report.py; single case = python run_case_cli.py --theta 70 --grid coarse --law young --out <json>",
 "run_artifacts": [
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/contact_angle_sweep.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/GENERATED_REPORT.md",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/repro.sh",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/figures/prescribed_vs_measured.png",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/figures/calibration_curve.png",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/figures/contours_young.png",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/a_coarse_70.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/a_coarse_90.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/a_coarse_110.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/a_fine_70.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/a_fine_90.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/a_fine_110.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/b2_fine_70.json",
  "/root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/cases/c_coarse_init70.json",
  "/root/tp-agent/work/m15894/audit/audit_m15894/audit_spot_a_coarse_70.json",
  "/root/tp-agent/work/m15894/audit/audit_m15894/evidence.json"
 ],
 "run_artifacts_note": [
  "41 committed files on branch agent/m15470 @ 4256e939 (worktree clean): 34 per-case JSON + contact_angle_sweep.json (91819 B, sha256 c906ede96eeb821e663debb1239ae9291a3e76ca4402552cf1dc57ce19ed400d, meta.generated_utc 2026-11-12T00:42:19Z, machine loongarch64, git_rev 3508357d) + 3 PNG + repro.sh + GENERATED_REPORT.md",
  "each case record carries theta_measured, estimator trio (cap/circle/band), mass_start/end/drift, converged, drift_deg, spurious_max, runtime_s, 8-point trace[] and the extracted interface points -> real field-derived data, not hardcoded",
  "DURABILITY GAP (finding F4): merge ee578b8f 'Merge #1825 (m15470)' into main carried only TensorLBM/src/tensorlbm/wetting/__init__.py (+11) and docs/CASE-MCMP-CONTACT-ANGLE.md (+153); git ls-tree origin/main -- artifacts/CASE-MCMP-CONTACT-ANGLE -> 0 files, and 'contact_angle_sweep.json' has 0 hits in the main tree. The numbers are auditable only while the mission work tree survives.",
  "auditor-side reproduction file audit_spot_a_coarse_70.json is NOT the mission artifact: it is m15894's independent re-execution (same loongarch64 box, 48 s, threads=4) used to prove the entry point is executable and deterministic"
 ],
 "tests": [
  "pytest tests/benchmarks/wetting_contact_angle -k 'artifact or falsified or grid_sensitivity or report_tables or calibrated or young_relation or inversion' -> 8 passed, 6 deselected in 6.22 s (re-run by the auditor in the mission work tree; the mission's claim of 8 passed is accurate)",
  "tests/benchmarks/wetting_contact_angle/test_wetting_contact_angle.py asserts invariants: stored verdicts vs recomputed deltas, honest-FAIL root-cause block presence, zero-handcopy table rendering, grid-sensitivity monotonicity"
 ],
 "integration_callers": "NONE — grep -rl over tests/, src/, benchmarks/ locates the harness only under tests/benchmarks/wetting_contact_angle/; no main-solver or platform module imports it. Library impact is docstring-only -> L3 refuted",
 "benchmark": {
  "referee_row_in_tensor_benchmark_cases": "ABSENT — SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-MCMP-CONTACT-ANGLE' returned 0 rows (whole table 25 rows). Nearest row GAP-LBM-MCMP-POISEUILLE (id 7, status pending) is 'MCMP layered Poiseuille analytic / interface position + flux + spurious current / <5%/<10%/<0.005U' — a different quantity, cannot referee a contact angle",
  "tensorbench_results_rows_mission_15470": 0,
  "platform_l4_verdict": "GET /api/v1/tensor/bench/l4-verdict/15470 -> CLAIMED_NOT_PROVEN, rows=0, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]",
  "criterion_declared": "|theta_measured - theta_young| < 10 deg per case (docs/CASE-MCMP-CONTACT-ANGLE.md sec.1 + commit message 4256e939)",
  "criterion_applied_in_code": "meta.accept_deg = 5.0 and per-case tol_deg = 5.0 -> 2x STRICTER than declared (finding F3, non-blocking: FAIL at both thresholds)",
  "stage_a_ref_vs_got": "coarse/70: ref 70.0 got 111.39459 delta +41.39 FAIL | coarse/90: ref 90.0 got 179.99994 delta +90.00 FAIL | coarse/110: ref 110.0 got OUT_OF_RANGE (needs psi_w<0; theta_max(psi_w>=0)=101.97) FAIL | fine/70: ref 70.0 got 117.70241 delta +47.70 FAIL | fine/90: ref 90.0 got 179.99994 delta +90.00 FAIL | fine/110: OUT_OF_RANGE FAIL",
  "stage_a_pass_count": "0/6 at the 5 deg code tolerance and 0/6 at the declared 10 deg; max |delta| = 89.99994270422047 deg; 2 of 6 cases retained as explicit OUT_OF_RANGE records (no case dropped, no tolerance relaxed)",
  "stage_b_holdout_ref_vs_got": "70 deg (f_w 0.580446) got 89.11933 delta +19.12 FAIL | 90 deg (f_w 0.524112) got 106.01026 delta +16.01 FAIL | 110 deg (f_w 0.45973) got 121.44394 delta +11.44 FAIL -> 0/3, 'coarse calibration transfers to fine grid' also falsified",
  "stored_verdicts_block": {"young": {"n_cases": 6, "n_pass": 0, "max_abs_delta": 89.99994270422047, "status": "FAIL"}, "calibrated": {"n_cases": 3, "n_pass": 0, "max_abs_delta": 19.119325139807316, "status": "FAIL"}},
  "error_analysis_present_in_case": "grid sensitivity 6.31 deg at f_w=0.4551 (25.0 deg at the wet end); metastability split 11.11 deg at f_w=0.52 (exceeds the 10 deg tolerance itself); analytic circular-cap estimator control bias <= 1.106 deg over 2 grids x 3 angles -> the 40-90 deg drift is model error, not estimator artefact",
  "verdict": "honest FAIL independently recomputed from the JSON artifacts (0/6 and 0/3); the mission neither dropped a case nor relaxed the criterion"
 },
 "repro_cmd": "bash /root/tp-agent/work/m15470/artifacts/CASE-MCMP-CONTACT-ANGLE/repro.sh  (executable sha256 68e872b0678b18fd...; all 9 referenced harness paths exist). Auditor executed the single-case entry point on this box: returncode 0, ~48 s, theta_measured = 111.39459003071705 -> abs diff 0.0 deg vs the mission record (bit-exact deterministic).",
 "external_anchor": "WEAK-BY-DESIGN / WITHHELD — ref theta_young is produced by the same cos(theta)=(psi_w-psi_v)/(psi_l-psi_v) inversion that is under test, so the comparison is internally consistent but not externally anchored; only the Young-equation FORM (Young, Phil. Trans. R. Soc. Lond. 95, 65-87, 1805) is external, and the mission explicitly declares page-level source verification WITHHELD (offline, no network). No experimental or literature numeric contact-angle anchor was used.",
 "findings": {
  "F1": "run_artifacts REAL and complete (41 files, sweep 91819 B parsed OK); nothing hardcoded: every case carries trace[], interface points, mass drift, runtime",
  "F2": "honest FAIL independently recomputed: stage A 0/6 (max 89.9999 deg), stage B 0/3 (max 19.12 deg), no case dropped, no tolerance relaxed -> the negative result stands",
  "F3": "non-blocking evidence inconsistency: declared criterion 10 deg vs applied 5 deg (code stricter); conclusion identical at both",
  "F4": "durability gap: main got the doc + docstring but ZERO run artifacts, so the case is not reproducible from the repo trunk (repro.sh only lives on the mission branch)",
  "F5": "no machine referee: 0 rows for this anchor in tensor_benchmark_cases, 0 tensorbench_results rows for m15470, platform l4-verdict CLAIMED_NOT_PROVEN",
  "F6": "no integration caller -> L3 refuted; library change is docstring-only (+11 lines on wall_psi_from_contact_angle)",
  "F7": "the measurement chain itself is sound (estimator control <= 1.106 deg), so this is a credible capability falsification (wall_psi_from_contact_angle is not an angle setter: measured wall law slope 5.03 vs library 1.21, neutral wall f_w* 0.512 vs 0.172, grid drift 6.31 deg, metastability 11.11 deg), not a broken harness"
 },
 "level_rationale": {
  "auditor_level": 2,
  "why": "L2 = real input->compute->output path: 34 case records from a working SCMP sessile-drop chain, deterministic (auditor bit-exact re-run) with analytic estimator control <= 1.106 deg. L3 refuted (no main-solver/platform caller; library change docstring-only). L4 refuted (benchmark FAILS 0/6 and 0/3 so no capability is validated; ref self-referential to the code under test; 0 referee rows; artifacts absent from main). L0/L1 too low: real multiphase LBM runs, not a document.",
  "value": "the falsification itself is a legitimate deliverable: the library's Young-law angle preset is now documented as unusable without a same-resolution calibration curve"
 },
 "evidence_bundle": "/root/tp-agent/work/m15894/audit/audit_m15894/evidence.json (machine-generated by /root/tp-agent/work/m15894/audit/verify_m15470.py) + /root/tp-agent/work/m15894/docs/audit-m15894-case-mcmp-contact-angle.md + figure /root/tp-agent/work/m15894/audit/audit_m15894/fig_m15894_contact_angle_audit.png"
}$json$,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      mission_id = EXCLUDED.mission_id,
      evidence = EXCLUDED.evidence,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL;
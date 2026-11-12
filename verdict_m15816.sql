-- Board audit verdict — m15816 auditing mission m15466 (CASE-RAYLEIGH-BENARD-RA1E5, TensorLBM)
-- Declared level 2 = independent functional implementation (real code, input -> compute -> output),
-- proven executable by auditor reproduction. L3 (no integration caller) and L4 (no converged run,
-- no run artifacts, no tensorbench_results row, l4-verdict=CLAIMED_NOT_PROVEN) are REFUTED.
-- The mission's own run_artifacts are objectively ABSENT (3 cluster jobs never started), recorded
-- honestly below so the machine gate can downgrade on its artifact-presence scheme.
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15466 CASE-RAYLEIGH-BENARD-RA1E5',
  15816,
  2,
  $json${
 "impl_files": [
  "/root/tp-agent/work/m15466/bench_rayleigh_benard_ra1e5.py",
  "/root/tp-agent/work/m15466/benchmarks/verified/thermal_cavity/driver.py",
  "/root/tp-agent/work/m15466/gen_figures_ra1e5.py"
 ],
 "entry": "/root/tp-agent/work/m15466/bench_rayleigh_benard_ra1e5.py",
 "run_artifacts": [
  "ABSENT — no artifacts/result_ra1e5*.json and no artifacts/figures_ra1e5/ anywhere in /root/tp-agent/work/m15466 (find -name 'result_ra1e5*' -> empty)",
  "ABSENT — the only numbers anywhere are the unconverged N=64/10k-value trio transcribed into docs/REPORT_CASE_RAYLEIGH_BENARD_RA1E5.md sec.4 (Nu=3.987, u_max=35.90, v_max=64.97); no JSON backing them (zero-handcopy rule violated)",
  "cluster_jobs 1133/1134/1137 (the advertised producers) all state=queued, started_at=NULL, attempts=0, 2h+ after submit -> no artifact could exist",
  "auditor-side reproduction (NOT the mission's artifact, kept separate): /root/tp-agent/work/m15816/artifacts/audit_m15816/evidence.json -> audit_reproduction_extra['audit_ra1e5_n32_3k.json'] sha256=1ac889bb41cec041c3cc3a406dae50b9148206a04c81bdac2ba64b5c73544d4c (N=32/3000 steps, 25.2 s, Nu=3.38107/-25.18%, u_max=41.0697/+18.29%, v_max=61.8577/-9.83%)"
 ],
 "tests": [],
 "integration_callers": "NONE — the script is an external consumer of the tensorlbm.thermal library API via benchmarks/verified/thermal_cavity/driver.py; no main-solver or platform-side caller references it (grep -rn 'rayleigh_benard_ra1e5' over tests/ and benchmarks/verified/thermal_cavity/ -> no hits; the only ra1e5 hits are the unrelated horizontal-layer RB case in tests/test_p97_n05_validation.py)",
 "benchmark": {
  "anchor_row_in_tensor_benchmark_cases": "ABSENT — 0 rows for anchor_slug='CASE-RAYLEIGH-BENARD-RA1E5' (table has 23 rows, 0 matching RAYLEIGH by slug or case_name) -> the anchor has no machine referee at all (platform ledger gap)",
  "tensorbench_results_rows_for_mission_15466": 0,
  "platform_l4_verdict": "/api/v1/tensor/bench/l4-verdict/15466 -> CLAIMED_NOT_PROVEN, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance], reason='no tensorbench_results row for this mission: the L4 benchmark claim is self-declared, not measured'",
  "ref_vs_got_only_existing_numbers": [
   {
    "q": "Nu_mean",
    "ref": 4.519,
    "got": 3.987,
    "err_pct": -11.82,
    "within_8pct": false,
    "within_3pct": false,
    "steps": 10000,
    "nx": 64,
    "converged": false
   },
   {
    "q": "u_max",
    "ref": 34.72,
    "got": 35.9,
    "err_pct": 3.4,
    "within_8pct": true,
    "within_3pct": false,
    "steps": 10000,
    "nx": 64,
    "converged": false
   },
   {
    "q": "v_max",
    "ref": 68.6,
    "got": 64.97,
    "err_pct": -5.29,
    "within_8pct": true,
    "within_3pct": false,
    "steps": 10000,
    "nx": 64,
    "converged": false
   }
  ],
  "verdict": "NO quantity <3%; no quantity converged; Nu also exceeds the mission's own 8% tolerance -> benchmark NOT passed"
 },
 "repro_cmd": "cd /root/tp-agent/work/m15816 && python3 artifacts/audit_m15816/bench_rayleigh_benard_ra1e5.py --ra 1e5 --nx 32 --max-steps 3000 --out /tmp/audit_ra1e5_n32_3k.json  # EXECUTES end-to-end, 25.2 s, emits grids/obs/err_pct/pass (exit 0); the advertised cluster form 'python3 bench_rayleigh_benard_ra1e5.py --ra 1e5 --nx 64 128 --max-steps 300000' never ran (cj1133 queued, attempts=0) and is infeasible in its 3600 s timeout (~34 ms/step at N=64 -> ~2.8 h for 300k steps)",
 "external_anchor": "de Vahl Davis, G. (1983) 'Natural convection of air in a square cavity: a bench mark numerical solution', Int. J. Numer. Methods Fluids 3(3):249-264, Table 1 p.257 — WITHHELD for per-value transcription (paper not machine-available offline). Nu=4.519 is cross-confirmed by two independent in-repo tables (conjugate_thermal/cases.py:41, heat_convection_benchmarks.py:63-70); u_max=34.72 / v_max=68.60 have no in-repo second source (differ from the literature 34.73/68.59 by 0.03%/0.015% -> consistent but unverified).",
 "level_rationale": "L2, not L1: unlike m15543 the delivered object is real executable code (real library kernel behind it), and the auditor reproduced input->compute->output with the delivered entry (JSON with err_pct/pass, sha256 recorded). Not L3: no main-solver/platform caller. Not L4: no converged case, no run artifact, no tensorbench_results row, platform referee says CLAIMED_NOT_PROVEN, and every existing number fails both 3% and (Nu) the mission's own 8%. Not L1/L0 functionally, but note the mission has literally zero tests and zero artifacts, so the maturity_gate SCHEME (L2 needs impl_files+entry+run_artifacts files that exist; L1 needs tests) will downgrade this to L0/L1 and open a rework round — consistent with the audit's judgement: the implementation is real, the evidence chain is broken.",
 "findings": [
  "F1 HIGH: phantom merge + auto-approve — mission succeeded/accepted=true, pr_gate={pr:1819, state:'merged', sha:null}, head_commit=null, yet commit 7dafdcea exists ONLY on agent/m15466 and main has no such file (propagates fake 'landed' evidence downstream)",
  "F2 HIGH: three-piece set missing run_artifacts — no result JSON, no figures dir; items ④⑤⑥ of the report are self-labelled 'pending' and the report header says '进行中'; the only numbers are hand-transcribed from a terminal (zero-handcopy rule)",
  "F3 HIGH: anchor has no machine referee — 0 rows in tensor_benchmark_cases for CASE-RAYLEIGH-BENARD-RA1E5 (23-row table, 0 RAYLEIGH), 0 rows in tensorbench_results for m15466 -> L4 unprovable by construction (platform ledger gap, propose registering the case row)",
  "F4 MEDIUM: the only measured errors fail — Nu -11.82% (>8% mission tolerance, >3% ecosystem standard); u_max 3.40% and v_max 5.29% also >3%; all at 10k unconverged steps",
  "F5 MEDIUM: runtime claim not reproducible — report says '120 s for ~10k steps at N=64'; audit measured 8.4 ms/step at N=32 (25.2 s/3000) and N=64/10k did not finish in 890 s at load 405 -> the 300k/800k-step plan does not fit its own 3600/7200 s cluster timeouts",
  "F6 MEDIUM: 'submitted to cluster' used as delivery justification — cj1133/1134/1137 queued with attempts=0 for 2h+; nodes n315/n330 healthy (load 35/128) and queue snapshot queued=135/killed=29 in 6h -> scheduler-side stall (ecosystem friction, not charged to m15466, but it produced zero evidence)",
  "F7 LOW: naming/semantics debt — anchor 'RAYLEIGH-BENARD' vs de Vahl Davis side-heated cavity (disclosed honestly in report ① and the docstring), and the repo already carries an unrelated horizontal-layer 'rb_ra1e5' in tests/test_p97_n05_validation.py -> retrieval confusion risk",
  "F8 LOW: dangling provenance pointer — driver.py docstring cites NOTES.md (pre-registered conventions) but benchmarks/verified/thermal_cavity/ contains only README.md/driver.py/result.json/run.py",
  "F9 LOW: gen_figures_ra1e5.py:plot_temperature_field re-runs the solver to obtain fields (result JSON has no field dump) -> figures cannot be regenerated from stored artifacts",
  "F10 INFO: in-repo prior art shows the same driver verified at Ra=1e3/1e4 (benchmarks/verified/thermal_cavity/result.json: n128/n256 all quantities <=3%, monotone) — so Ra=1e5 is an extension, not a new machine; the missing piece is compute + artifacts, not method development"
 ],
 "not_run": [
  "Did not run the advertised converged sweep (N=64/128/256 x 300k-800k steps): verification ticket, not a redo; cluster queue had 135 jobs queued and 0 successes in 6h",
  "Did not obtain the de Vahl Davis (1983) PDF -> u_max/v_max reference values stay WITHHELD (only Nu has an independent in-repo cross-check)",
  "N=64/10k auditor reproduction did not complete (killed at 890 s under load ~405) -> the report's preliminary trio could not be numerically re-derived; executability of the entry was established at N=32/3000 instead",
  "Did not modify the audited worktree (read-only: reproduction ran a copy inside work/m15816)"
 ],
 "artifacts": [
  "/root/tp-agent/work/m15816/docs/audit_m15466_ra1e5.md (six-section audit report)",
  "/root/tp-agent/work/m15816/artifacts/audit_m15816/evidence.json (all DB/API/reproduction numbers, machine-collected)",
  "/root/tp-agent/work/m15816/artifacts/audit_m15816/collect_evidence.py",
  "/root/tp-agent/work/m15816/artifacts/audit_m15816/make_figures.py + fig1_ra1e5_ref_vs_got.png + fig2_evidence_matrix.png"
 ],
 "scope_note": "Audited mission m15466 (succeeded, accepted=true via [auto-approve], head_commit=null), worktree /root/tp-agent/work/m15466 on branch agent/m15466, commit 7dafdcea (3 files, 621 insertions). Cross-checked: tensor_dev_missions, tensor_benchmark_cases (23 rows), tensorbench_results, cluster_jobs 1133/1134/1137 + 6h queue snapshot, run_cluster nodes, /tensor/bench/l4-verdict/15466, git branch/contains/ls-tree on main.",
 "impl_details": [
  "/root/tp-agent/work/m15466/bench_rayleigh_benard_ra1e5.py:1-210 (real entry: argparse -> driver loop -> observables -> err_pct/pass -> JSON; commit 7dafdcea on branch agent/m15466 only)",
  "/root/tp-agent/work/m15466/bench_rayleigh_benard_ra1e5.py:34-42 REF_DVD_1E5 (Nu=4.519 cross-confirmed by src/tensorlbm/conjugate_thermal/cases.py:41 and src/tensorlbm/heat_convection_benchmarks.py:63-70; u_max=34.72/v_max=68.60 WITHHELD, no in-repo second source)",
  "/root/tp-agent/work/m15466/benchmarks/verified/thermal_cavity/driver.py:1-180 (pure glue over tensorlbm.thermal: collide_bgk_force/buoyancy_force/nusselt_number/apply_temperature_boundaries -> physical kernel is the library, not the mission)",
  "/root/tp-agent/work/m15466/gen_figures_ra1e5.py:1-252 (figure script; needs artifacts/result_ra1e5.json which does not exist, and re-runs the solver for field plots -> cannot regenerate figures from stored artifacts)"
 ],
 "gate_expectation": "maturity_gate SCHEME: L2 needs impl_files+entry+run_artifacts files that exist; run_artifacts is objectively ABSENT and tests=[] -> machine gate will report L0 with accurate reasons ('L2 缺 run_artifacts', 'L1 缺 tests') and open a rework round. That is the correct remedy: the implementation exists, the evidence chain does not. Declared level 2 is a substance judgement, not an artifact claim."
}$json$::jsonb,
  'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      mission_id = EXCLUDED.mission_id,
      gate_state = 'pending',
      gate_level = NULL,
      gate_note = NULL
RETURNING id, repo, scope, level, gate_state;
-- m16205 board audit of m16183 / CASE-LBM-SQUARE-CYL-NORBERG
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM', 'm16183 CASE-LBM-SQUARE-CYL-NORBERG', 16205, 3, $json${
 "level": 3,
 "level_rationale": "L2 (real input->compute->output) and L3 (integrated: run.py drives library collide_mrt/stream/far_field_bc_2d/compute_obstacle_forces) are satisfied by the in-repo case; L4 is NOT granted despite complete-case+error-analysis+repro_cmd shape, because (a) no tensor_benchmark_cases referee row exists for the anchor, (b) the in-tree verifier verify.py cannot run on the shipped artifacts, (c) the full D32/D48 rerun was not reproduced in this audit, (d) the Cd claim depends on the force-measurement column (wet vs mem = 8.53%). Level is attributed to commit d9a45c85 (pre-existing); mission m16183 delta = 0.",
 "mission_delta": {
  "mission_id": 16183,
  "db_status": "succeeded",
  "fail_stage": "no_progress+forced_finish",
  "self_reported": "honest-FAIL (budget exhausted in reconnaissance; no implementation, no run)",
  "changed_files": [
   "/root/tp-agent/work/m16183/.scratch/m16183_checks.py"
  ],
  "report_id": null,
  "tensor_reports_rows_for_mission": 0,
  "own_run_artifacts": 0,
  "own_benchmark_numbers": 0,
  "verdict": "m16183 delivered nothing; the anchor's substance is inherited from d9a45c85 (2026-09-29)"
 },
 "impl_files": [
  "benchmarks/verified/square_cylinder/run.py:132 def run_case (geometry, sponge, seeding, forcing, FFT St)",
  "benchmarks/verified/square_cylinder/run.py:111 def square_mask (half-way BB, exact side=D)",
  "benchmarks/verified/square_cylinder/run.py:163 _step_plain / :166 _step_seeded (per-step kernel)",
  "benchmarks/verified/square_cylinder/run.py:205 cd_series = fx_np/q_dyn / :206 cd_mean over steady window",
  "benchmarks/verified/square_cylinder/run.py:214 St via np.fft.rfft on Hann-windowed Cl",
  "benchmarks/verified/square_cylinder/verify.py:1-135 in-tree gate replay (BROKEN on shipped schema)",
  "benchmarks/verified/square_cylinder/verify_traces.py (recomputes stats from raw per-step traces)",
  "runs/m16205_audit/audit_recompute.py (this audit's independent gate replay, JSON-only)"
 ],
 "entry": "python3 benchmarks/verified/square_cylinder/run.py --D 32 48 [--steps 60000 90000] --device <sdaa:0|cuda:0|cpu> [--compile-mode default|eager] [--out DIR] (argparse at run.py:main; per-grid driver loop over D/steps)",
 "run_artifacts": [
  "/root/tp-agent/work/m16183/benchmarks/verified/square_cylinder/result.json (D32/D48 summary + reference convention + sdaa_compile_limitation)",
  "/root/tp-agent/work/m16183/benchmarks/verified/square_cylinder/case_D32.json (D=32, 1280x1280, 60000 steps, mem column, data_source=SDAA run 2026-08-20)",
  "/root/tp-agent/work/m16183/benchmarks/verified/square_cylinder/case_D48.json (D=48, 1920x1920, 90000 steps, mem column)",
  "/root/tp-agent/work/m16183/benchmarks/verified/square_cylinder/case_D64.json (D=64, 2560x2560, 120000 steps, wet/pair/mem A-B columns, prereg sha256 b9928c15…)",
  "/root/tp-agent/work/m16205/runs/m16205_audit/evidence_audited/ (bit-copies of the above + run.py/verify.py/README.md/REFERENCE_AUDIT.md, sha256 recorded)",
  "/root/tp-agent/work/m16205/runs/m16205_audit/audit_recompute_result.json (gate replay output)",
  "/root/tp-agent/work/m16205/runs/m16205_audit/fig_ref_vs_got.png, fig_force_columns.png (figures derived from the shipped JSON)",
  "/root/tp-agent/work/m16205/runs/m16205_audit/smoke_D8.log (auditor smoke run D=8/4000 steps CPU — no result.json within the audit window)"
 ],
 "tests": {
  "repo_tests_for_this_case": "NONE — grep -rl square_cylinder tests/ = 0 files",
  "in_tree_verifier": "benchmarks/verified/square_cylinder/verify.py FAILS on the shipped artifact set: '[FAIL] case files found — D=[]' then IndexError at verify.py:100 (cases[ds[-1]]['pair']) because case_D32.json/case_D48.json are summary-schema, not the wet/pair/mem triple the verifier demands",
  "audit_side": "runs/m16205_audit/audit_recompute.py — 23 OK / 2 not-checkable / 3 FAIL over gates G1..G6",
  "not_run": "no repo pytest suite executed for this anchor; verify_traces.py not run (raw per-step traces not shipped: case_D64.json raw_traces_note)"
 },
 "integration_callers": [
  "benchmarks/verified/square_cylinder/run.py:88-90 from tensorlbm.boundaries import compute_obstacle_forces, far_field_bc_2d, make_sponge_strength",
  "run.py:160 compute_obstacle_forces(f, mask) — Ladd MEM, post-stream/pre-bounce-back",
  "run.py:164 far_field_bc_2d(f, u_in, mask) — inlet/side Dirichlet + outlet zero-gradient + obstacle BB in one call",
  "src/tensorlbm/solver.py:276 collide_mrt (tau_field per-cell relaxation = sponge) + stream",
  "src/tensorlbm/boundary/boundaries.py:164 compute_obstacle_forces / :201 far_field_bc_2d / :390 make_sponge_strength",
  "benchmarks/compile_route.py route_step (compile/eager routing, used at run.py:177-178)"
 ],
 "benchmark": {
  "machine_referee": "tensor_benchmark_cases WHERE anchor_slug='CASE-LBM-SQUARE-CYL-NORBERG' -> 0 rows (NO referee: metric/tolerance absent; anchor's got values cannot be machine-adjudicated)",
  "declared_reference_run_py": {
   "REF_CD": 1.48,
   "REF_ST": 0.147,
   "band_cd": [
    1.44,
    1.52
   ],
   "band_st": [
    0.145,
    0.149
   ]
  },
  "declared_reference_verify_py": {
   "REF_CD": 1.4785,
   "REF_ST": 0.1456,
   "defect": "two reference conventions coexist in the same directory -> same got yields two error sets"
  },
  "got_vs_ref": [
   {
    "D": 32,
    "force_column": "mem(compute_obstacle_forces)",
    "cd": 1.4816,
    "cd_err_pct": 0.11,
    "st": 0.1482,
    "st_err_pct": 0.82,
    "cd_pass_3pct": true,
    "st_pass_3pct": true
   },
   {
    "D": 48,
    "force_column": "mem(compute_obstacle_forces)",
    "cd": 1.4844,
    "cd_err_pct": 0.3,
    "st": 0.148,
    "st_err_pct": 0.68,
    "cd_pass_3pct": true,
    "st_pass_3pct": true
   },
   {
    "D": 64,
    "force_column": "wet",
    "cd": 1.6139,
    "cd_err_pct": 9.16,
    "st": 0.1479,
    "st_err_pct": 1.58,
    "cd_pass_3pct": false,
    "st_pass_3pct": true
   },
   {
    "D": 64,
    "force_column": "pair",
    "cd": 1.487,
    "cd_err_pct": 0.57,
    "st": 0.1479,
    "st_err_pct": 1.58,
    "cd_pass_3pct": true,
    "st_pass_3pct": true
   },
   {
    "D": 64,
    "force_column": "mem",
    "cd": 1.487,
    "cd_err_pct": 0.57,
    "st": 0.1479,
    "st_err_pct": 1.58,
    "cd_pass_3pct": true,
    "st_pass_3pct": true
   }
  ],
  "cross_grid_span": {
   "mem_cd": [
    1.4816,
    1.4844,
    1.487
   ],
   "span_cd_pct": 0.364,
   "mem_st": [
    0.1482,
    0.148,
    0.1479
   ],
   "span_st_pct": 0.203
  },
  "force_column_ab": {
   "wet_cd": 1.6139,
   "mem_cd": 1.487,
   "pair_cd": 1.487,
   "wet_minus_mem_pct": 8.53,
   "pair_minus_mem_pct": 0.0,
   "wet_cd_err_pct_vs_run_py_ref": 9.05
  },
  "gate_replay": {
   "ok": 23,
   "not_checkable": 2,
   "fail": 3,
   "failed": [
    "D64/wet |Cd err| <= 3% — +9.16%",
    "wet - mem Cd < 3% (claim must not hinge on the column) — +8.53% (wet column would give +9.05% error -> would FAIL the 3% bar)",
    "tensor_benchmark_cases row for anchor — rows=0 -> NO machine referee: got values cannot be machine-adjudicated"
   ]
  },
  "external_anchor_consistency": "anchor is named ...-NORBERG (experimental St reference) but the delivered case deliberately scores against the 2D free-stream low-blockage NUMERICAL cluster and explicitly rejects experiment (Okajima 1982 St 0.143-0.145, no Cd). Scientifically defensible, but the anchor name/reference convention mismatch must be adjudicated at anchor level."
 },
 "repro_cmd": "cd /root/tp-agent/work/m16183 && PYTHONPATH=src:benchmarks python3 benchmarks/verified/square_cylinder/run.py --D 32 48 --steps 60000 90000 --compile-mode eager --device sdaa:0   (author estimate ~19 h total; reduced smoke that the auditor actually executed: --D 8 --steps 4000 --device cpu --compile-mode eager)",
 "external_anchor": [
  "Sohankar et al. 1998 IJNMF 26:39 (Cd 1.46 / St 0.147, 2D free-stream) as cited in arXiv 2411.03124v4 Tab.1",
  "Yoon et al. 2010 (Cd 1.44 / St 0.145) — arXiv 2411.03124v4 Tab.1",
  "arXiv 2309.09197 Tab.2: Sohankar 1.477/0.146, Sahu 1.488/0.149, Sen 1.530/0.145",
  "arXiv 2308.08085v3 Tab.7-8: Fakhari&Lee 1.51/0.149, Gonzalez 1.50/0.145, uniform 1.500/0.1472",
  "arXiv 2404.12123 (Present 1.476); arXiv 2405.12834 (1.47-1.55 / 0.145-0.148)",
  "Rejected by the case itself: Okajima 1982 JFM 116,379 (experiment St 0.143-0.145, no Cd; 3D experiment vs 2D solver)",
  "Anchor name 'NORBERG' does not match the adopted (numerical) reference family — naming issue, flagged"
 ],
 "not_run": [
  "full D=32 (60000 steps) + D=48 (90000 steps) rerun — ~19 h on SDAA per author record; not reproduced in this audit",
  "D=64 (120000 steps) rerun — only its shipped case_D64.json was replayed",
  "verify_traces.py (needs raw per-step traces; not shipped)",
  "repo pytest suite for this anchor (no square_cylinder test exists)",
  "SDAA device rerun (torch_sdaa importable on the audit host, but result.json records teco-inductor fallback-to-eager defect)"
 ],
 "disposition": "reopened (request changes): register the tensor_benchmark_cases row (metric+tolerance, force-column locked), fix verify.py schema compatibility, unify the reference constants, have a mission actually deliver (m16183 delta=0) and publish the tensor_reports entry, add a tests/ regression for this case; then re-audit for L4.",
 "artifacts": [
  "/root/tp-agent/work/m16205/runs/m16205_audit/AUDIT_m16183.md (six-section audit report)",
  "/root/tp-agent/work/m16205/runs/m16205_audit/audit_recompute.py + audit_recompute_result.json",
  "/root/tp-agent/work/m16205/runs/m16205_audit/gen_figures.py + fig_ref_vs_got.png + fig_force_columns.png",
  "/root/tp-agent/work/m16205/runs/m16205_audit/gen_report.py + /tmp/report.json"
 ]
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence;

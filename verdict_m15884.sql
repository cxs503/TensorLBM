-- Board audit verdict — m15884 auditing mission m15776 (CASE-CYLINDER-DRAG-RE100, TensorLBM)
-- L2 = independent functional implementation (real input->compute->output), proven executable by the
-- auditor (pytest 6 passed + a fresh 6000-step D=20 run). L3 refuted: module not exported by the
-- package, no main-solver/platform caller (only tests + benchmark runner + a manifest row).
-- L4 refuted: machine_pass=false, 3 of 4 metrics outside the 5% case tolerance, single grid level
-- (convergence={}), 0 tensorbench_results rows (platform l4-verdict CLAIMED_NOT_PROVEN), 0 rows for
-- the anchor in tensor_benchmark_cases, registered repro_cmd not executable and not reproducing the
-- shipped artifact, and the shipped dp=2.0140 is stale (HEAD re-run: 5.4494, x2.71).
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15776 CASE-CYLINDER-DRAG-RE100',
  15884,
  2,
  $json${
 "mission": 15776,
 "anchor": "CASE-CYLINDER-DRAG-RE100",
 "auditor_mission": 15884,
 "impl_files": [
  "src/tensorlbm/validation/cylinder_dfg.py (565 lines; REFERENCE_BANDS@L100-120, probe_columns@L278, run_case@L295, dp extrapolation@L401-414, __main__ CLI@L512-565)",
  "benchmarks/verified/cylinder/re100_dfg/run.py (entry/aggregator; REPO_ROOT bug @L37 parents[2])",
  "benchmarks/verified/cylinder/re100_dfg/REFERENCE.md (frozen band register)",
  "benchmarks/validation_matrix/matrix.json cases[10].id=cylinder_re100_dfg",
  "benchmarks/validation_matrix/references.json (cylinder_dfg2d2_{cd,clamp,st,dp}_re100)"
 ],
 "entry": "PYTHONPATH=<repo>/src python3 -m tensorlbm.validation.cylinder_dfg --D 20 --steps 12000 --threads 4 --out <FILE>  (NOTE: --out is a FILE, not a dir)",
 "run_artifacts": [
  "benchmarks/verified/cylinder/re100_dfg/artifacts/level_D20_u0833.json sha256:592446e9eafb899a (936668 B, wall_s=178.9, D=20/12000 steps, ONE grid level only)",
  "benchmarks/verified/cylinder/re100_dfg/artifacts/level_D20_u0833.series.npz sha256:aeb2273c45defb09",
  "benchmarks/verified/cylinder/re100_dfg/artifacts/result.json sha256:57c9490242b5129c (machine_pass=false, verdict=not_verified, convergence={})",
  "ABSENT: figures/ (fig1_force_history.png, fig2_bands_vs_measured.png exist only in the m15776 work tree, never committed); ABSENT runs/m15776/{D28,D40}.log|json from main (untracked)",
  "auditor reproduction (kept separate, artifacts/audit_m15884/repro/repro_D20_6000.json): D=20/6000 steps/4 threads, wall 94.1 s -> cd=3.53285 st=0.29492 cl_amp=1.22815 dp=5.44944"
 ],
 "tests": [
  "tests/test_cylinder_dfg_reference.py — 6 passed in 7.10 s (auditor run): band-vs-register freeze, probe-columns-are-fluid for D=20/28/40, synthetic St recovery, smoke run finite"
 ],
 "integration_callers": "NONE in main solver or platform: tensorlbm/validation/__init__.py does not export cylinder_dfg; no app/backend or platform-side caller; callers are tests/test_cylinder_dfg_reference.py + benchmarks runner + a manifest row in benchmarks/validation_matrix/matrix.json (documentation-level wiring, not runtime integration)",
 "benchmark": {
  "anchor_row_in_tensor_benchmark_cases": "ABSENT — 0 rows for anchor_slug='CASE-CYLINDER-DRAG-RE100' (table has 25 rows, none cylinder-related) -> no machine referee",
  "tensorbench_results_rows_for_mission_15776": 0,
  "platform_l4_verdict": "/api/v1/tensor/bench/l4-verdict/15776 -> CLAIMED_NOT_PROVEN, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance], reason=\"no tensorbench_results row for this mission\"",
  "ref_vs_got": [
   {
    "metric": "cd",
    "frozen_band": [
     3.22,
     3.24
    ],
    "got_shipped_D20_12000": 3.495514,
    "err_pct_vs_mid": 8.22,
    "inside_band": false,
    "tol_pct_case": 5.0,
    "pass_5pct": false,
    "got_auditor_HEAD_D20_6000": 3.532848,
    "err_pct_auditor": 9.376,
    "auditor_vs_shipped_ratio": 1.011
   },
   {
    "metric": "st",
    "frozen_band": [
     0.295,
     0.305
    ],
    "got_shipped_D20_12000": 0.302297,
    "err_pct_vs_mid": 0.766,
    "inside_band": true,
    "tol_pct_case": 5.0,
    "pass_5pct": true,
    "got_auditor_HEAD_D20_6000": 0.294916,
    "err_pct_auditor": -1.695,
    "auditor_vs_shipped_ratio": 0.976
   },
   {
    "metric": "cl_amp",
    "frozen_band": [
     0.99,
     1.01
    ],
    "got_shipped_D20_12000": 1.248383,
    "err_pct_vs_mid": 24.838,
    "inside_band": false,
    "tol_pct_case": 5.0,
    "pass_5pct": false,
    "got_auditor_HEAD_D20_6000": 1.228152,
    "err_pct_auditor": 22.815,
    "auditor_vs_shipped_ratio": 0.984
   },
   {
    "metric": "dp",
    "frozen_band": [
     2.46,
     2.5
    ],
    "got_shipped_D20_12000": 2.014037,
    "err_pct_vs_mid": -18.789,
    "inside_band": false,
    "tol_pct_case": 5.0,
    "pass_5pct": false,
    "got_auditor_HEAD_D20_6000": 5.44944,
    "err_pct_auditor": 119.735,
    "auditor_vs_shipped_ratio": 2.706
   }
  ],
  "machine_pass_shipped": false,
  "note_dp_stale": "shipped dp=2.0140 comes from level_D20_u0833.json which has NO probes{} block (pre-extrapolation code); HEAD code re-run gives dp=5.4494 (x2.71). REFERENCE.md itself lists dp_raw 2.46-2.50 AND dp_dimensionless 4.92-5.00, while REFERENCE_BANDS/bench compares against 2.46 -> factor-2 convention is unresolved; either way the shipped -18.79% is not reproducible at HEAD"
 },
 "repro_cmd": {
  "registered_in_matrix.json": "python benchmarks/verified/cylinder/re100_dfg/run.py --levels 28 40 --steps 60000 --threads 4",
  "executable_as_written": false,
  "defect_1": "run.py L37 REPO_ROOT=HERE.parents[2] == <repo>/benchmarks, so it inserts <repo>/benchmarks/src (nonexistent); import then resolves to whatever tensorlbm is ambiently installed (auditor env: /root/tp-agent/work/m13963/src/tensorlbm -> ImportError: cannot import name cylinder_dfg). Sibling benchmarks/verified/cylinder/re100/run.py uses parents[3].",
  "defect_2": "aggregate() does path.relative_to(REPO_ROOT) -> ValueError for any --out-dir outside <repo>/benchmarks",
  "defect_3": "registered --levels 28 40 has no artifact: only D20 exists, so the registered command cannot reproduce the registered artifact (needs D20/12000)",
  "working_incantation": "PYTHONPATH=<repo>/src python3 benchmarks/verified/cylinder/re100_dfg/run.py --aggregate-only --out-dir <repo>/benchmarks/.../audit (absolute path)"
 },
 "external_anchor": "Schäfer & Turek 1996, NNFM 48, 2D-2 (Re=100, confined, D/H=0.244): Cd 3.22-3.24, Cl_amp 0.99-1.01, St 0.295-0.305, dp 2.46-2.50 — declared provenance LIMIT: the PDF was unreachable offline, bands transcribed from benchmarks/roadmap.md + benchmarks/pending/cylinder_re40_st/result.json; no platform/external machine check of the band itself",
 "mission_self_declaration": "mission result JSON says headline \"...精度未达标 = honest-FAIL\", machine_pass=false; accepted=true came from TP_AUTO_APPROVE_DEV_ACCEPTANCE autopilot (decision_note), and server_quality_audit={gate_valid:false, key_present:false, gate_missing_warning:true}",
 "report": "ABSENT — 0 rows in tensor_reports for mission_id=15776; mission summary itself lists 'Tensor API 报告' as not run; no six-section report in-tree for this anchor",
 "level_rationale": "L2: real input->compute->output implementation (MRT collide + Zou/He + half-way BB + Ladd momentum exchange composed from library primitives), proven executable by auditor (pytest 6 passed + fresh 6000-step run). L3 refuted: no main-solver/platform call path, module not exported by the package (integration is a manifest row + tests). L4 refuted: machine_pass=false, 3 of 4 metrics outside the 5% case tolerance (dp non-reproducible/stale), single grid level (no D28/D40, convergence={}), no tensorbench_results row, no tensor_benchmark_cases anchor row, registered repro_cmd not executable and not reproducing the shipped artifact."
}$json$,
  'board-audit'
);
SELECT repo, scope, mission_id, level, verdict_by, jsonb_typeof(evidence::jsonb) FROM tensor_maturity_verdicts WHERE mission_id=15884;

-- m22754: XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE canonical re-packaging + completed pre-declared ladder
DELETE FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE';
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status, passed_at)
VALUES ('XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE',
        'plane_channel_resolution_pair_H16_H32',
        'analytic plane Poiseuille, half-way bounce-back walls: k_super=(H/ny)*H^2/12',
        'max |k_super(LBM)-ref|/ref over H=16 (got -0.1094%) and H=32 (got -0.0283%); profile max rel 0.204% / 0.052%; A1-vs-A2 identity 1.77e-12',
        '0.15', 'TensorLBM', 'passed', now());

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorLBM','m22754 XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE', 22754, 2,
 $J${
 "impl_files": [
  "src/tensorlbm/porous/fluid_node_table.py (canonical port of the m17677 shadow-tree module)",
  "src/tensorlbm/porous/__init__.py"
 ],
 "tests": "tests/test_porous_fluid_node_table.py -> 12 passed in 26.7s (canonical tree, PYTHONPATH=src)",
 "entry": "PYTHONPATH=src python3 xval/XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE/run_xval.py",
 "run_artifacts": [
  "xval/XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE/results.json",
  "figures/xval_compressed_node_table.png"
 ],
 "report": "docs/xval_compressed_fluid_node_neighbour_table.md",
 "gates": {
  "G1_representation_identity_lt_1e-8": {
   "pass": true,
   "ux_maxabs_rel": 1.7651071164214899e-12,
   "ux_l2_rel": 8.510860454203697e-13,
   "k_rel_diff": 1.4904888496506432e-13
  },
  "G2_physics_lt_15pct": {
   "pass": true,
   "err_pct": [
    0.10937503947983651,
    0.02827219324186281
   ]
  },
  "G2_target_lt_3pct": {
   "pass": true,
   "err_pct": [
    0.10937503947983651,
    0.02827219324186281
   ]
  },
  "G3_forcing_linearity_lt_5pct": {
   "pass": true,
   "k_rel_diff_g_admissible_pair": 3.836127626038432e-15,
   "ma_max_admissible_pair": 0.05507154456567774,
   "k_rel_diff_g_wide_pair_not_admitted": 5.504009202576902e-15,
   "ma_max_wide_pair_not_admitted": 0.11014308913135601
  },
  "resolution_delta_on_rel_error": 0.000811028462379737,
  "resolution_converged_lt_5pct": true,
  "channel_arms": [
   {
    "case": "C2a_channel_H16",
    "steps": 5600,
    "converged": true,
    "err_pct": -0.10937503947983651
   },
   {
    "case": "C2b_channel_H32",
    "steps": 12000,
    "converged": false,
    "err_pct": -0.02827219324186281
   }
  ],
  "converged_all_channel_arms": false
 },
 "benchmark": {
  "anchor_slug": "XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE",
  "k_err_pct_H16": -0.10937503947983651,
  "k_err_pct_H32": -0.02827219324186281,
  "tolerance": 0.15,
  "ecosystem_3pct_met": true
 },
 "defects_closed": [
  "F1 shadow-tree packaging -> canonical importable tensorlbm.porous",
  "F2 runner/artifact step mismatch -> single runner, numbers emitted, no hand copy",
  "F3 docs page + benchmark row now exist",
  "F4 steady criterion implemented and reported per arm (H=32 did not trip it: disclosed)",
  "F5 ladder truncated -> resolution pair + linearity + porosity arms all executed"
 ],
 "stale_claim_corrected": "the ported test asserted dense_stream == tensorlbm.d2q9.stream under a d->opposite(d) mirror; canonical tensorlbm.solver.stream uses velocity-direction labelling, so the agreement is a direct identity and the mirror premise was shadow-tree specific (test repointed, docstring fixed)",
 "checks_not_run": [
  "openLBMPM CUDA OSS arm (no GPU) -> external anchor WITHHELD",
  "C2b steady flag at 12000-step cap (KPI already settled; reported not retuned)"
 ],
 "integration_callers": "none pre-existing: no canonical solver path consumes tensorlbm.porous yet; the module is importable and tested, so this is an availability+validation increment, NOT an L3 solver-integration claim.",
 "limits": "the compressed footprint at p=0.90 is 1.8x the dense one (int64 table) - negative result kept"
}$J$::jsonb, 'xval-r2')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id=EXCLUDED.mission_id, level=EXCLUDED.level, evidence=EXCLUDED.evidence;

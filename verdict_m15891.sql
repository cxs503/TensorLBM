INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
  'TensorLBM',
  'm15467 CASE-MCMP-LAPLACE',
  15891,
  4,
  $json$
{
  "impl_files": [
    "benchmarks/case_pool/mcmp_laplace/run_case.py:1-614 (executor: PROTOCOL/THRESHOLDS at :72-110, Pool tasks :148-200, per-G fit + tolerance family :390-445, summary write :500-560)",
    "benchmarks/case_pool/mcmp_laplace/make_report.py:1-426 (report+figure renderer from summary.json, zero hand-copy; _hero_case() added by m15891)",
    "src/tensorlbm/mcmp_verification.py:458 measure_laplace(dims=2 default, sigma=dp*R/(dims-1))",
    "src/tensorlbm/mcmp_verification.py:839 _fit_laplace_level (one-parameter through-origin fit + free-line diagnostic)",
    "src/tensorlbm/scmc_pressure_tensor.py (Route B Kirkwood-Buff mechanical tensor)"
  ],
  "entry": "CLI: python benchmarks/case_pool/mcmp_laplace/run_case.py --jobs 8 [--quick] [--out DIR]; smoke verified: --quick --jobs 8 -> artifacts/m15891_mcmp_audit/quick_smoke_summary.json (37.3 s, tol_family_pass=true, max_rel_err_law=2.50%)",
  "run_artifacts": [
    "benchmarks/case_pool/mcmp_laplace/artifacts/summary.json (4151 lines: 12 droplet + 3 planar/Route-B + 6 resolution cases, protocol+thresholds+verdict)",
    "benchmarks/case_pool/mcmp_laplace/artifacts/hero_fields.npz (rho1/rho2/ux/uy, G_12=-2.75, R_init=20, n=96)",
    "benchmarks/case_pool/mcmp_laplace/artifacts/hero_profile.json",
    "benchmarks/case_pool/mcmp_laplace/figures/{laplace_law,sigma_vs_radius,droplet_field,routeB_flatness}.png (restored by m15891; git blobs byte-identical to e547c37c)",
    "docs/CASE-MCMP-LAPLACE.md (= benchmarks/case_pool/mcmp_laplace/README.md, six sections)",
    "artifacts/m15891_mcmp_audit/evidence.json (auditor recompute of every headline number + raw-field spot check)",
    "artifacts/m15891_mcmp_audit/quick_smoke_summary.json (auditor quick re-run)",
    "figures/audit_m15891_mcmp_laplace.png"
  ],
  "tests": [
    "tests/benchmarks/test_mcmp_laplace_dims.py::test_measure_laplace_2d_uses_cylinder_factor (PASSED; asserts sigma == dp*R, not merely executes)",
    "tests/benchmarks/test_mcmp_laplace_dims.py::test_measure_laplace_3d_uses_sphere_factor (PASSED; 3D = 0.5*dp*R)",
    "tests/benchmarks/test_mcmp_laplace_dims.py::test_measure_laplace_rejects_unknown_dimension (PASSED; dims=4 -> ValueError)",
    "tests/benchmarks/test_mcmp_laplace_dims.py::test_young_laplace_calibration_fit_honours_dims (PASSED; fit factor consistent with measurement)",
    "pytest tests/benchmarks/test_mcmp_laplace_dims.py -q -> 4 passed in 4.34 s",
    "auditor invariant checks (script): through-origin sigma0 reproduces stored values to <1e-15 for all 3 levels; measure_laplace(hero_fields.npz) reproduces summary dp to 2.0e-6 rel, R exactly"
  ],
  "integration_callers": [
    "benchmarks/case_pool/mcmp_laplace/run_case.py:58 import tensorlbm.mcmp_verification as mv (simulate_static_droplet / measure_laplace / estimate_coexistence / _fit_laplace_level)",
    "benchmarks/case_pool/mcmp_laplace/run_case.py:60 from tensorlbm.scmc_pressure_tensor import SCMCPressureConfig (Route B)",
    "benchmarks/mcmp_verify_bench.py:43 import tensorlbm.mcmp_verification as mv (library-level MCMP benchmark runner)",
    "platform registry: tensor_case_pool id=16 case_code=MCMP-LAPLACE repo=TensorLBM tol_policy='<5%' status=dispatched mission_id=15467",
    "platform report: tensor_reports id=14544 scenario_key=case-pool/mcmp-laplace",
    "no platform router / main-solver registration (case harness over existing library API) -> L3 integration only partial"
  ],
  "benchmark": {
    "platform_referee": "ABSENT: SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-MCMP-LAPLACE' = 0 (the table's only MCMP row is GAP-LBM-MCMP-POISEUILLE, tol '<5%/<10%/<0.005U'); referee therefore falls back to tensor_case_pool id=16 (tol_policy '<5%') + the case's pre-registered THRESHOLDS {tolerance:0.05, r2_min:0.99, max_offset_fraction:0.05, flat_flatness_max:0.05} (run_case.py:72-78, declared fixed before the run)",
    "metric": "max_i |sigma_i - sigma_0|/(sigma_0), sigma_i = dp_i*R_i (2D cylinder Laplace law, one unknown per G level), sigma_0 = through-origin least squares",
    "got": {"G-2.5": {"sigma_0": 0.187765, "max_rel_err_pct": 2.539, "R": [11.493, 15.652, 19.662, 23.606], "dp": [0.0160588, 0.0121051, 0.0097215, 0.0081562]},
            "G-2.75": {"sigma_0": 0.204347, "max_rel_err_pct": 1.752, "R": [11.600, 15.684, 19.669, 23.608], "dp": [0.0174422, 0.0130671, 0.0105096, 0.0088075]},
            "G-3.0": {"sigma_0": 0.211879, "max_rel_err_pct": 1.760, "R": [11.654, 15.696, 19.668, 23.604], "dp": [0.0180112, 0.0135314, 0.0108850, 0.0091345]}},
    "ref": {"source": "analytic Delta p = sigma/R (2D circular droplet / cylindrical interface), sigma fitted through the origin per G level; audited independently from summary.json", "tolerance_pct": 5.0, "auditor_sigma0_match": "<1e-15 vs stored"},
    "errors_pct_vs_ref": {"G-2.5": [1.705, 0.906, 1.799, 2.539], "G-2.75": [0.986, 0.291, 1.160, 1.752], "G-3.0": [0.930, 0.244, 1.044, 1.760], "max": 2.539, "within_5pct": true, "within_3pct": true},
    "diagnostic_not_tolerance": {"free_line_sigma_vs_through_origin_pct": {"G-2.5": -5.84, "G-2.75": -3.71, "G-3.0": -3.59}, "max_offset_fraction_pct": {"G-2.5": 9.61, "G-2.75": 5.94, "G-3.0": 5.72}, "r2": [0.99964, 0.99996, 0.99998], "self_set_guardrail_le_5pct": false, "note": "curvature (Tolman-type) trend 3.6-5.8%; the headline '2.54%<5%' holds only under the pre-registered one-unknown through-origin criterion"},
    "routeB_absolute_sigma": {"valid": false, "flatness": {"G-2.5": 0.456, "G-2.75": 0.563, "G-3.0": 0.641}, "threshold": 0.05, "sigma_mech": [-0.00944, -0.01509, -0.02066], "status": "VOID by its own validity gate (P_NN conservation fails); absolute sigma has NO independent anchor -> L5 not reachable"},
    "resolution_study": {"n96_to_n128_sigma_change_pct": {"G-2.5": -1.01, "G-2.75": -0.74, "G-3.0": -0.67}},
    "roster_completeness": "12/12 droplet cases + 3/3 planar + 6/6 resolution runs present and converged (cases[].converged=true), all three G levels within the library's coexistence window (-2.3..-3.5)",
    "verdict": "PASS under the pre-registered <5% tolerance item (max 2.54%); self-set diagnostic guardrails (intercept fraction <=5%) FAIL and are reported honestly (summary.verdict.guard_family_pass=false, doc section 5)"
  },
  "repro_cmd": "cd /root/tp-agent/work/m15891 && python benchmarks/case_pool/mcmp_laplace/run_case.py --jobs 8 --out artifacts/repro_mcmp && python benchmarks/case_pool/mcmp_laplace/make_report.py  # full protocol 369 s (exceeds the 120 s control-plane timeout; auditor submitted it as cluster job cj1196, still queued at close); quick protocol verified locally: --quick --jobs 8 -> 37.3 s, tol_family_pass=true, max_rel_err=2.50%; report/figure re-render is deterministic (4 PNG git blobs byte-identical to e547c37c)",
  "external_anchor": "Laplace/Wikipedia-level analytic law Delta p = sigma/R for a 2D circular interface (cylinder), i.e. Young-Laplace 1805; the 2D-vs-3D factor disagreement inside the repo (mcmp_verification vs mcmp_verification_3d docstring, fixed by m15467 with a dims parameter) is what this case pins. NO numeric literature anchor for the absolute sigma of this linear-psi Shan-Chen coupling (ref_source in the case is the analytic form only) -> proportionality/radius-independence is validated, the absolute value is not.",
  "auditor_recompute": {
    "method": "scripts/audit_m15891_mcmp_laplace.py reads only committed artifacts; refits through-origin sigma0 and the free line from the 12 (R, dp) pairs; re-measures the hero droplet from hero_fields.npz through mcmp_verification.measure_laplace(dims=2)",
    "sigma0_agreement": "all 3 levels reproduce stored sigma_origin and max_rel_err to <1e-15 / <1e-15",
    "raw_field_spot_check": {"hero": "G_12=-2.75, R_init=20, n=96", "dp_auditor": 0.010509619, "dp_stored": 0.010509640, "dp_rel_diff": -1.96e-6, "radius_auditor": 19.669375467, "radius_stored": 19.669375438, "sigma_auditor": 0.206717650, "sigma_stored": 0.206718054},
    "reproduction_of_conclusion": "quick protocol independent re-run reproduces the same tolerance-family PASS (max 2.50% < 5%) -> the conclusion is not a one-off artifact of the committed json"
  },
  "three_piece_verification": {
    "run_artifacts": "PASS with one defect found+fixed: summary.json / hero_fields.npz / hero_profile.json committed and internally consistent, but the 4 figures referenced by README/docs were dropped by merge cf959e0d (present in e547c37c, absent from HEAD) -> dead links on main; m15891 re-rendered them from summary.json and the 4 git blobs are byte-identical to e547c37c",
    "benchmark_ref_vs_got": "PASS: all 12 points within the pre-registered 5% door (max 2.54% < 3%); auditor recomputation matches to machine precision; platform ledger tensor_benchmark_cases has 0 rows for this anchor (F1), fallback referee = tensor_case_pool id=16 '<5%'",
    "repro_cmd": "PASS (quick protocol executed by the auditor, 37.3 s; make_report.py re-render deterministic byte-for-byte); full 369 s protocol cannot run inside the 120 s control-plane shell timeout - submitted as cluster job cj1196 (queued ~380 jobs ahead at audit close, no result -> no conclusion drawn from it)"
  },
  "level_rationale": "L4 = complete benchmark case + error analysis + reproducible. Positive: 3 couplings x 4 radii + resolution study (n=96/128) + an honest negative (Route B void by its own P_NN conservation gate, 46/56/64% flatness); pre-registered <5% tolerance item passes at 2.54% max; raw-field spot check proves the numbers come from the simulated fields (dp rel diff 2e-6, float32 storage); report+figures are deterministically rendered from the committed json (regenerated PNGs byte-identical to the merged-away originals); 4 invariant pytest guards pass. Not L5: the absolute surface tension has NO independent anchor (Route B void), so what is validated is the radius-independence/proportionality of Delta p ~ 1/R plus the 2D factor - not the absolute sigma; also the free-line diagnostic exposes a 3.6-5.8% curvature trend that a stricter reading of the law would have to absorb. Not L3-in-full: the case is a harness over existing library modules (mcmp_verification, scmc_pressure_tensor) with no platform-router or main-solver registration.",
  "findings": [
    "F1 medium: tensor_benchmark_cases has 0 rows for anchor_slug='CASE-MCMP-LAPLACE' -> the 'sole referee' table does not exist for this anchor; recommend registering (metric='max_i |sigma_i-sigma_0|/sigma_0', tolerance='0.05') plus the resolution-convergence item",
    "F2 medium (fixed): the 4 figures referenced by README/docs/CASE-MCMP-LAPLACE.md were dropped by merge cf959e0d (present in e547c37c) -> dead links on main; restored in this branch, regenerated bytes identical to the historical blobs (no data loss, pure merge-filter loss)",
    "F3 medium: headline 'PASS 2.54%<5%' holds only under the pre-registered one-unknown through-origin criterion; the free-line diagnostic shows sigma drifting 3.6-5.8% over R=11.5..23.6 and an intercept fraction of 5.7-9.6% (all three levels exceed the case's own 5% intercept guardrail). Honest in the artifacts (guard_family_pass=false) but a title-only reader underestimates the curvature trend; recommend reporting both numbers at headline level",
    "F4 medium-low (fixed): make_report.py:348 labelled the hero field artifact with cases[0].G_12 (=-2.5) although hero_fields.npz is G_12=-2.75, R_init=20 -> README/docs mislabel; now read from the npz, re-render changes exactly one line",
    "F5 medium-low (open): docs/MCMP_VERIFY_BENCH.md still tabulates absolute sigma with the OLD 2D factor (sigma=dp*R/2) -> all absolute sigmas there are 2x too small, and the doc is still cited by docs/MCMP_STRATIFIED_POISEUILLE_SPEC.md; the case README flags it as legacy and gives the regeneration command (benchmarks/mcmp_verify_bench.py --report, needs a simulation re-run) - not executed here (out of audit scope, would contend for compute)",
    "F6 low: tensor_case_pool id=16 status remains 'dispatched' after the deliverable; bookkeeping lag",
    "F7 info: absolute sigma has no independent external anchor (Route B void) -> the case validates proportionality/radius-independence, not the absolute value; this is why L5 is not claimed",
    "F8 info: full protocol 369 s exceeds the 120 s control-plane hard timeout; auditor re-run submitted as cluster job cj1196 (queued, no result at close) -> no reproduction claim is made from it",
    "F9 info: the guard tests assert invariants (2D vs 3D factor, ValueError, fit/measure factor consistency) rather than merely executing code, but they use a synthetic tanh field (no solver in the loop) - unit-level lock only, no end-to-end test",
    "F10 info: tests/benchmarks/test_mcmp_verify_bench.py could not be run locally (killed at the 120 s control-plane timeout) -> status unknown/not_run, not passed"
  ],
  "not_run": [
    "Full 12+3+6 protocol re-execution (m15467's 369 s run) - cluster job cj1196 still queued at audit close, no result",
    "tests/benchmarks/test_mcmp_verify_bench.py (library-level MCMP benchmark) - killed by the 120 s control-plane timeout, status unknown",
    "docs/MCMP_VERIFY_BENCH.md regeneration (needs a simulation re-run; F5 left open on purpose)",
    "GPU/SDAA path (delivery is CPU float32)",
    "No modification of the audited case logic (verification-only); only the doc/figure defects F2/F4 were repaired"
  ],
  "scope_note": "Audited mission m15467 (succeeded), delivery commit e547c37c (PR #1822, merged cf959e0d). Audit branch agent/m15891 @ 53046bcc + 87a5176c. Audit report docs/audits/m15891_CASE-MCMP-LAPLACE_audit.md, evidence artifacts/m15891_mcmp_audit/evidence.json, figure figures/audit_m15891_mcmp_laplace.png. Verdict: L4 (complete benchmark case, pre-registered criterion PASS at 2.54%, reproducible), not L5 (no independent absolute sigma anchor)."
}
$json$::jsonb,
  'board-audit'
) ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', gate_level=NULL, gate_note=NULL
RETURNING id, repo, scope, level, gate_state;
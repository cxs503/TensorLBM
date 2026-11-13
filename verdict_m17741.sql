-- m17741 board-audit verdict: XVAL-LBM-CHAPMAN-ENSKOG-AUTO (audited mission m17672)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES ('TensorLBM', 'm17672 XVAL-LBM-CHAPMAN-ENSKOG-AUTO', 17741, 2, $json${
 "entry": [
  "/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/run_xval.py",
  "/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/make_report.py"
 ],
 "tests": [
  "tests/test_chapman_enskog.py: 14 passed in 45.01s (pytest -q at audited HEAD 15a9b393)",
  "includes D2Q5 anisotropic-fourth-moment refusal and the zeta-convention tests"
 ],
 "benchmark": {
  "ref_vs_got": {
   "sweep": {
    "n_checks": 35,
    "n_passed": 35,
    "all_below_3pct": true,
    "median_err_pct": 0.1001,
    "n_below_0p4pct": 31,
    "distinct_got_values": 31
   },
   "D_sound2d": {
    "pass": true,
    "caveat": "the 3 D4-zeta rows form got = nu_L_measured - nu_measured, i.e. one measurement is reused; ref=0 at s_e=2 is evaluated on a declared physical scale, not a relative error",
    "checks": 10,
    "quantity": "c_s and nu_L = 2(D-1)/D nu + zeta",
    "worst_err_pct": 0.19
   },
   "E_sound3d": {
    "note": "n32 -> n64 error GROWS (0.010% -> 1.155%): the 3D sound series is not mesh-converged, the worst number in the sweep sits on an unconverged arm",
    "pass": true,
    "checks": 4,
    "worst_err_pct": 1.155
   },
   "A_shear2d_bgk": {
    "got": "measured shear-wave log-decay with tensorlbm.solver.collide_bgk+stream, n32/n64, r2=1.0",
    "ref": "nu = cs2(tau-1/2) from derive_equivalent_equation(D2Q9,BGK)",
    "pass": true,
    "checks": 6,
    "worst_err_pct": 0.313
   },
   "B_shear2d_mrt": {
    "note": "the 3 s_e values per tau give identical nu (nu independent of s_e) -> 9 checks, 3 distinct got values",
    "pass": true,
    "checks": 9,
    "worst_err_pct": 0.804
   },
   "C_shear3d_bgk": {
    "got": "tensorlbm.collision_viscosity_audit.run_collision_viscosity_audit (pre-existing D3Q19 audit harness, not new code)",
    "pass": true,
    "checks": 6,
    "worst_err_pct": 0.313
   }
  },
  "not_hardcoded": "probe P2: perturbing one D2Q9 weight (east x1.10, renormalised) makes the derivation raise 'isolation of zeta failed: diagonal coefficients differ' -> nu/zeta come from the moment algebra, not a re-typed formula; symbolic values vary with the collision model (D2Q9 BGK zeta=1/30 vs D2Q9 MRT zeta=3/82). Limitation: anisotropic weight sets are refused rather than handled.",
  "machine_referee": "psql: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CHAPMAN-ENSKOG-AUTO' -> 0 rows; also 0 rows for anchor_slug ILIKE '%CHAPMAN%'. The machine referee has NO registered case/tolerance for this anchor, so the gate cannot referee this claim; the numbers below are the harness's own declared gate (gate_pct=15) re-executed by me.",
  "post_hoc_gate_amendment": "commit 15a9b393 (the shipped HEAD) changed the gate itself: 'Previously 34/35 with the only failure being nan/nan on the zero-reference row'. The 35/35 headline depends on an error definition amended AFTER the first failure. The amendment is defensible (relative error to an exactly-zero reference is undefined, and the measured -3.58e-05 vs scale 0.1 is 0.036%), but it must be read as a post-hoc criterion change affecting exactly the failing row (1 of 35).",
  "gate_has_detection_power": "docs/audit_m17741/probe_m17741.py P1: same measurement (tau=0.8, nu_meas=0.100206, r2=1.0) vs matched prediction 0.1 -> 0.206% PASS; vs mismatched prediction (tau=1.2, 0.2333) -> 57.05% FAIL. The 15% gate discriminates."
 },
 "repro_cmd": "cd /root/tp-agent/work/m17672 && PYTHONPATH=src python3 benchmarks/xval_chapman_enskog_auto/run_xval.py --outdir /tmp/xval_fresh  # executed 2026-11-13 at HEAD 15a9b393: 35/35 PASS, worst -1.155%, 0 numeric drift vs shipped result.json; plus PYTHONPATH=src python3 -m pytest tests/test_chapman_enskog.py -q -> 14 passed",
 "impl_files": [
  "/root/tp-agent/work/m17672/src/tensorlbm/chapman_enskog.py"
 ],
 "disposition": "approve_with_notes -> L2 confirmed, promotions to L3/L4 rejected; ledger/KPI must not book this as an integrated solver capability",
 "remediation": [
  "register the referee row(s): INSERT INTO tensor_benchmark_cases(anchor_slug, case_name, ref_source, metric, tolerance, repo) for the 5 arm families (or one row per check) so the gate can referee without trusting the harness's own gate_pct",
  "stamp provenance.git_head from the tree actually run and re-generate result.json at the commit that ships it",
  "declare the zero-reference error rule in the artefact (field 'zero_ref_scale') instead of only in a commit message, so the 34/35 -> 35/35 change is visible in the data",
  "extend the 3D sound arm with a third grid (n48/n96) — the n32->n64 error growth means the worst check is on an unconverged series",
  "to reach L3: call derive_equivalent_equation from the solver/audit path (e.g. collision_viscosity_audit asserting nu == predicted at config time) and export it from src/tensorlbm/__init__.py"
 ],
 "report_review": "tensor_reports id 16658 (mission 17672): six sections present in docs/XVAL-LBM-CHAPMAN-ENSKOG-AUTO.md (1 problem/reference sources, 2 method, 3 parameters, 4 derivation+comparison table, 5 convergence/sensitivity, 6 failure modes+artifacts, 7 explicit error verdict 'every key quantity < 15%: yes'); section 7 answers the <15% question explicitly but does NOT answer the owner's <3% line with the zero-reference caveat (all 35 are in fact <3%, worst 1.155%).",
 "run_artifacts": [
  "/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/result.json",
  "/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/table.csv",
  "/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/convergence.csv",
  "/root/tp-agent/work/m17672/benchmarks/xval_chapman_enskog_auto/chapman_enskog_xval.png",
  "docs/audit_m17741/fresh_rerun_result.json",
  "docs/audit_m17741/audit_m17741.png",
  "docs/audit_m17741/audit_stats.json",
  "docs/audit_m17741/probe_m17741.json"
 ],
 "external_anchor": [
  "canonical Chapman-Enskog / lattice-Boltzmann transport result nu = cs2 (tau - 1/2), zeta = 0 for athermal BGK under the full-gradient convention (Succi 2001; Wolf-Gladrow 2000; Kruger et al. 2017 ch. 6) and the stress-trace / sound-attenuation relation Re(omega) = -k^2 nu_L/2 with nu_L = 2(D-1)/D nu + zeta",
  "the module DERIVES these rather than assuming them (exact symbolic cs2 from the weights; MRT zeta=3/82, D3Q19 zeta=1/15) and the 'got' side is an independent solver measurement -> genuine two-path cross-validation, unlike a closed form re-typed twice",
  "no experimental or literature DATASET is compared: no external/field/benchmark data -> L5 not claimed"
 ],
 "level_rationale": "L1 met: 14 unit tests + exact symbolic/analytic identities, fresh re-run bit-identical. L2 met: genuinely new functional code (input stencil+weights+collision rates -> output exact symbolic nu/zeta/nu_L, 3 stencils x BGK/TRT/MRT, 573 lines, consistency assertions that actually fire on a perturbed stencil). L3 NOT met: 0 production callers, module not exported in __init__, nothing in the solver or any pipeline imports it -> no integration. L4 NOT met (as a certified level): (a) the machine referee has 0 rows for anchor_slug='XVAL-LBM-CHAPMAN-ENSKOG-AUTO', so no registered case/tolerance exists to referee the claim; (b) the 'benchmark' is the deliverable's own harness — ref and got are distinct code paths (good) but the case set is periodic shear/sound decay on the repo's own solvers, with no external reference; (c) the 35/35 headline rests on a post-hoc gate amendment (was 34/35) and the worst number sits on a non-monotone n32->n64 series. L5 NOT met: no experimental anchor. Sub-stance note: in evidence quality this case clearly exceeds the L2 bar (35 reproducible checks, worst 1.155%, all <3%), but the capability is a standalone verification tool with no integration and no referee registration, so certifying L3/L4 would be an overstatement.",
 "provenance_defect": "result.json stamps provenance.git_head=e53d994d but the artefact ships in 15a9b393 (one commit later); chapman_enskog_sha256=b9a42507... does match the file at HEAD and my independent re-derivation reproduced all 5 spot-checked derivation records with nu_rel_diff=0.0, so the drift is traceability cosmetics, not a numeric discrepancy — but a read-only auditor cannot tell from the artefact which commit produced it.",
 "check_count_caveat": "35 checks, 31 distinct got values, 5 arm families; the 9 B-arm checks carry only 3 distinct measurements (nu independent of s_e) and the 6 D-arm zeta/nu_L checks reuse the same 3 sound + 3 shear runs. '35/35' is a check count, not a capability count.",
 "integration_callers": [
  "NONE — grep -rn 'chapman_enskog' over src/ finds 0 imports of tensorlbm.chapman_enskog outside the module itself. src/tensorlbm/two_level_amr2d.py:107,111,239 use the *string* neq_scale='chapman_enskog' (an unrelated scale-factor mode); tests/test_two_level_amr2d.py:54 and tests/test_lbm_von_neumann_stability.py:102 are pre-existing, independent of this module. The module is also not re-exported by src/tensorlbm/__init__.py -> no production caller, no solver/pipeline wiring (L3 NOT met)"
 ],
 "path_note": "impl_files/entry/run_artifacts are ABSOLUTE paths into the audited worktree /root/tp-agent/work/m17672 (mission 17672, HEAD 15a9b393) because those commits are not yet in /root/tensor-ecosystem/TensorLBM (ECO at c05d7af3 predates m17672). Repo-relative equivalents: src/tensorlbm/chapman_enskog.py, benchmarks/xval_chapman_enskog_auto/* (valid after push-merge). docs/audit_m17741/* are shipped by this audit branch (m17741).",
 "impl_notes": "chapman_enskog.py:423 derive_equivalent_equation — sympy order-eps CE: f1 = -B diag(sigma) B^-1 (d_t1 + c.grad1) f_eq -> stress moments -> nu (:519), zeta (:537); conservation/isolation assertions (:483-500, :527-536) fire on a perturbed stencil (probe P2). :91 Stencil (exact Fraction weights, isotropy_checks :136), :273 bgk, :285 trt, :342 mrt, :353 d2q9_mrt, :552 longitudinal_viscosity; 573 lines",
 "entry_notes": "run_xval.py:330 main() CLI with --outdir; make_report.py:245 renders docs + png from result.json"
}$json$, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, evidence = EXCLUDED.evidence,
      gate_state = 'pending', gate_level = NULL, gate_note = NULL;

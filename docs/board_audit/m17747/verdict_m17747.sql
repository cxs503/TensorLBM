-- m17747 board audit (交差核验) of mission 17679
-- XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS (TensorLBM vs haimingz/vivsim @ fb84b12)
-- Idempotent: re-running updates nothing twice (guarded by NOT EXISTS / scope+repo lookup).
\set ON_ERROR_STOP on

BEGIN;

-- 1) referee row for the anchor: the mission never registered one (audit finding A5 = 0 rows),
--    so from now on the <15% bar is predeclared in the machine-judgeable table, not only in prose.
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS',
       'vivsim Re=200 free-stream cylinder: continuous error print vs DOI refs (30D x 20D, cyl at 10D, 150 D/U0)',
       'haimingz/vivsim @ fb84b12f02e18f3b9dfdf738d6a1a318d81539f9 examples/2d/flow_pass_cylinder.py:78-85 (CD_REF=1.389, CL_REF=0.718; DOIs 10.1016/j.ijmecsci.2024.108961, 10.1016/j.camwa.2014.05.013, 10.1017/S0022112086003014)',
       'Cd_mean vs 1.389 AND Cl_peak same-caliber vs 0.718 (also report harmonic amplitude); caliber = mean/max over the second half of the time record, as in the OSS source line 207-208',
       '<15% (owner bar, predeclared in the port as TOL_PCT=15.0; |err| strictly < 15). Round-2 status: got WITHHELD (0/3 fresh runs produced a number, NaN at 2000/2500/500 steps)',
       'TensorLBM',
       'pending'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases
                  WHERE anchor_slug = 'XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS');

-- 2) the maturity verdict (required deliverable of mission 17747)
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (
'TensorLBM',
'm17679 XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS',
17747,
2,
$json$
{
 "audit_artifact": "docs/board_audit/m17747/AUDIT_m17747.md + audit.json (A1-A8 recomputed) + fig_m17747.png + evidence/ (18 files, MANIFEST.sha256)",
 "audited_commit": "agent/m17679 @ 3161a1fc (worktree clean)",
 "impl_files": [
  "benchmarks/xval/xval_vivsim_cylinder_re200.py:195 run_case() - 30D x 20D vivsim set-up on TensorLBM primitives (MRT + solid freeze + half-way bounce-back + Ladd MEM + far_field_bc_2d + 5D sponge, seed 4 cols @ 0.10*u_in)",
  "benchmarks/xval/xval_vivsim_cylinder_re200.py:168 verdict() - predeclared |err| < 15.0 gate on both vivsim calibers (strict inequality; 15.0 itself FAILs, pinned by test)",
  "benchmarks/xval/xval_vivsim_cylinder_re200.py:128 vivsim_calibers() - last-half mean(cd) / max|cl| port of the OSS caliber at upstream lines 207-208",
  "benchmarks/xval/xval_vivsim_cylinder_re200.py:77 harmonic_amplitude() + :108 per_period_half_pp() - second caliber (least-squares amplitude, per-period median half peak-to-peak), added because max|Cl| is single-sample sensitive",
  "benchmarks/xval/xval_vivsim_re200_report.py:47/:89/:120 fig_history/fig_errors/fig_grid - programmatic summary + 3 figures"
 ],
 "entry": "benchmarks/xval/xval_vivsim_cylinder_re200.py:361 main() (argparse --D/--u/--re/--steps/--alpha/--st-seed/--out/--series/--strict); continuous err% print at :298-299 and :399; NaN detector at :292-293",
 "run_artifacts": [
  "/root/tp-agent/work/m17679/runs/xval_D10_u010.json - nan_at=2000, steps_done=1999, wall_s=244.4, mlups=0.49 (honest FAIL, caliber-matched)",
  "/root/tp-agent/work/m17679/runs/xval_D12_u010.json - nan_at=2500, wall_s=316.7 (honest FAIL)",
  "/root/tp-agent/work/m17679/runs/xval_D12_u030.json - nan_at=500, wall_s=177.1 (falsifies the low-tau hypothesis the mission itself recorded)",
  "/root/tp-agent/work/m17679/artifacts/xval_vivsim_re200/summary.json - runs_frozen + runs_fresh + fresh_caliber_matched_verdict='WITHHELD' (tracked in branch)",
  "/root/tp-agent/work/m17679/docs/figures/xval_vivsim_re200_frozen_ladder.png (tracked)",
  "audit-side replay: docs/board_audit/m17747/evidence/m17747_repro_D10.{json,log,npz} - auditor re-ran D=10/u=0.1/2000 steps in 32.5 s: nan_at=2000 identical, verdict dict identical key-by-key, pre-divergence cd_half1 rel diff 5.45e-06"
 ],
 "tests": [
  "tests/test_xval_vivsim_cylinder_re200.py - 11 tests, auditor re-ran: '11 passed, 1 warning in 4.35s' (pytest exit 0)",
  "test_gate_fails_outside_tolerance_and_at_the_boundary (:140) - asserts 15.0% itself FAILs (strict |err| < TOL)",
  "test_vivsim_cl_peak_is_single_sample_sensitive (:71) - pins the max|Cl| spike defect (one spike inflates >1.25x while the harmonic amplitude moves <0.15)",
  "test_vivsim_calibers_selects_last_half_only (:49) + test_harmonic_amplitude_recovers_amplitude_and_frequency (:89) - calibers pinned against semi-analytic series with known answers",
  "test_reference_values_are_the_oss_ones (:39) - pins CD_REF=1.389 / CL_REF=0.718 / 3 DOIs / tol=15"
 ],
 "integration_callers": [
  "NONE: 'grep -rn xval_vivsim TensorLBM/ src/' returns 0 hits; the module is reachable only from tests/test_xval_vivsim_cylinder_re200.py:20 and the two docs - therefore L3 (system integration) NOT met",
  "not registered in any solver boundary/capability registry; the anchor had 0 rows in tensor_benchmark_cases before this audit"
 ],
 "benchmark": {
  "registered_referee_rows_for_anchor_before_audit": 0,
  "registered_now_by_audit": "1 row, status=pending, tolerance predeclared '<15%' (XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS)",
  "graded_arm_fresh_caliber_matched": "0/3 valid numbers - NaN @2000 (D10/u0.1), @2500 (D12/u0.1), @500 (D12/u0.3); Cd/Cl errors are NaN, so the <15% bar was never actually applied to a caliber-matched number",
  "frozen_arm_recomputed_by_auditor": {
   "source": "benchmarks/verified/cylinder/re200/result.json (verified_date 2026-08-19; PRE-EXISTING, not produced by m17679)",
   "D48": {"Cd": 1.3386, "err_cd_pct": -3.63, "cl_peak_sqrt2_x_std": 0.6859, "err_cl_pct": -4.47, "within_15pc": true, "within_3pc": false},
   "D64": {"Cd": 1.3177, "err_cd_pct": -5.13, "cl_peak_sqrt2_x_std": 0.6647, "err_cl_pct": -7.43, "within_15pc": true, "within_3pc": false},
   "caliber_mismatch": "40D x 40D domain, cylinder at 12D, u=0.08, sponge 10D, Braza-1986 anchor; Cl is sqrt(2)*Cl_std conversion, not a measured max|Cl| -> indirect evidence only"
  },
  "convergence_order": "NOT AVAILABLE - 2 grid levels only, and they belong to the mismatched set-up"
 },
 "repro_cmd": {
  "status": "EXECUTED by the auditor, works",
  "as_documented": "cd /root/tp-agent/work/m17679 && PYTHONPATH=src python3 benchmarks/xval/xval_vivsim_cylinder_re200.py --D 16 --u 0.1 --out out.json --series series.npz",
  "auditor_run": "cd /root/tp-agent/work/m17679 && python3 benchmarks/xval/xval_vivsim_cylinder_re200.py --D 10 --u 0.1 --steps 2000 --threads 16 --out /tmp/m17747_repro_D10.json --series /tmp/m17747_repro_D10.npz  -> 32.5 s, '[xval] *** NaN at step 2000 ***', gate FAIL",
  "broken_pointer": "the mission doc's round-2 pointer 'command in benchmarks/xval/README summary line' does not exist (benchmarks/xval/ holds only 2 .py files)",
  "audit_repro": "python3 docs/board_audit/m17747/audit_m17747.py && python3 docs/board_audit/m17747/gen_figure.py (rebuilds every graded number and the figure)"
 },
 "external_anchor": [
  "haimingz/vivsim @ fb84b12f02e18f3b9dfdf738d6a1a318d81539f9 examples/2d/flow_pass_cylinder.py:78-85 - CD_REF=1.389 / CL_REF=0.718 + 3 DOIs, PARSED by the auditor from the source slice (evidence/oss_flow_pass_cylinder.py, sha256 e9693607...) and matched against the port constants: cd_match/cl_match/doi_match = true",
  "same file :37-62 - D=60, NX=30D, NY=20D, CYL (10D,10D), U0=0.05, RE=200, TM=180000; :195-208 - Cd=2h0/(D U0^2), cd_mean=mean(cd[TM//2:]), cl_max=max|cl[TM//2:]| - the caliber the port had to reproduce",
  "DOI lineage inherited from the OSS source: 10.1016/j.ijmecsci.2024.108961, 10.1016/j.camwa.2014.05.013, 10.1017/S0022112086003014 (not opened by the auditor; the OSS comment gives no averaging method, so CL_REF=0.718 carries unquantified uncertainty)"
 ],
 "level_rationale": "L1 met: 11 invariant-asserting tests green on auditor re-run. L2 met: real end-to-end code (input->compute->output) that the auditor replayed in 32.5 s and that emits JSON/CSV/NPZ plus a machine verdict; reference provenance verified line-by-line from the OSS source. L3 NOT met: 0 call sites outside benchmarks/xval + tests. L4 NOT met: 0 referee rows before this audit AND the deciding caliber-matched arm produced 0/3 valid numbers (NaN @2000/2500/500, replayed identically), while the FROZEN -3.63%/-5.13%/-4.47%/-7.43% rows come from a pre-existing verified case with a different domain (40D x 40D), u=0.08 and a sqrt(2)*Cl_std conversion - indirect evidence, not this anchor's verification; no convergence order. L5 impossible (no experimental or independent anchor). Level = 2.",
 "findings": [
  "F1 blocking-for-anchor: the anchor's own deliverable (caliber-matched TensorLBM vs 1.389/0.718) is WITHHELD - 3/3 runs NaN, so the 15% bar was never applied to a real number",
  "F2 medium: St hallucination - all three NaN runs still report a FINITE st_vs_lit_err_pct (-49.24% / -51.27% / -18.78%) because harmonic_amplitude() degenerates to the frequency floor 1/T -> St=0.1; a reader can read this as a measured error (reproduced in the auditor replay)",
  "F3 medium: the max|Cl| caliber is single-sample sensitive (pinned at 2.6x by the mission's own test) and CL_REF=0.718 has no traceable averaging method upstream, so even a fixed arm can fail 15% for caliber, not physics reasons",
  "F4 low-medium: broken round-2 pointer - docs cite benchmarks/xval/README which does not exist",
  "F5 low: the three honest-FAIL run records live under runs/ which .gitignore:108 excludes -> only summary.json's runs_fresh survives in the pushed branch (this audit snapshots them under docs/board_audit/m17747/evidence/)",
  "F6 low: 'all tables programmatically injected' is slightly overstated - the doc table hard-codes -7.42% where summary.json/recomputation give -7.43%",
  "F7 info: ST_LIT=0.197 cited to Williamson 1996/1998 without a DOI or page (labelled non-criterion)"
 ],
 "unfound": "no fabricated run (wall_s/mlups/step counts self-consistent and replayable), no overwritten FAIL, no FROZEN-as-FRESH substitution (evidence grade labelled in doc, report and summary.json)",
 "disposition": "request_changes (reopened): keep the L2 credit for the harness; round 2 must (1) localise the NaN with a (D, seed_amp) 2-D scan incl. seed_amp=0, (2) deliver a caliber-matched number at D=16/24 with u=0.05, reporting both max|Cl| and harmonic amplitude, (3) fill the referee row registered by this audit, (4) fix F2-F7",
 "audit_limits": [
  "only the D=10/u=0.1 record was replayed (250 s each for the D=12 pair)",
  "root cause of the divergence NOT localised by the auditor",
  "vivsim never executed (no JAX; 3.9e11 cell-updates upstream) - all OSS-side claims are source-level",
  "upstream checkout has no resolvable git HEAD object; fb84b12 identity rests on the working-tree content matching the cited lines",
  "platform frictions reported by the mission (cluster queued 40+ min, knowledge engine offline) not re-verified - out of this audit's domain"
 ],
 "report": "tensor_reports id=16662 - six sections 1-6 present, 1 figure, explicitly answers the '<3%?' question with 'cannot claim'"
}
$json$::jsonb,
'board-audit'
);

COMMIT;

-- expected: level 2, gate_state pending -> maturity_gate decides confirmed / reopened
SELECT id, repo, scope, mission_id, level, verdict_by, gate_state
FROM tensor_maturity_verdicts
WHERE repo = 'TensorLBM' AND scope = 'm17679 XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS';
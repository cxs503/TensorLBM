-- Addendum to verdict id 620 (m17596 board audit): F0 committed-artifact staleness + D5 run instability.
-- Applied 2026-11-13 by m17665; report tensor_reports id=16663; audit doc docs/audit_m17596/AUDIT_m17596.md sec.9
UPDATE tensor_maturity_verdicts
SET evidence = evidence
  || jsonb_build_object(
       'report', 'tensor_reports id=16663 (board audit report, figures embedded)',
       'F0_committed_artifact_stale', jsonb_build_object(
          'fact', 'run_xval.py at HEAD 10d568a6 deterministically emits exact_cut=4544.0 / ours_cut=4736.0 / rel_err=+4.23% for the E1 instance "refined 2x3x3 (checker)"; the committed result.json (and the committed report table line 117) records 1280/1280/+0.00%. Artifacts were generated 07:26:48-07:30:01Z, code changed before the 07:31:50Z commit, never re-run.',
          'consequence', 'committed artifacts cannot be regenerated from committed code; the stale numbers are the more favourable ones; D1a is +4.23% from HEAD code (inside the 15% cap, outside the 3% band)',
          'unaffected', 'D3 max|df|=0.0, E2/E3 cut numbers, E3 imbalance and D4 +5.144e-4 do reproduce bit-identically from HEAD code',
          'self_disclosure', 'the audit re-execution of the driver overwrote the working-tree artifacts (git status M on result.json/tables.md/fig1..fig3), so committed-artifact and fresh-run numbers are reported separately'),
       'D5_run_instability', jsonb_build_object(
          'committed', 'min hidden fraction 92.6% -> pass_D5 true',
          'fresh_rerun_same_node', 'min hidden fraction 68.6% -> pass_D5 FALSE; 8 of 15 shipped rows carry hidden>100% or negative exposed time',
          'reading', 'the D5 verdict flips between runs of the same code -> the overlap gate is not a stable pass; defensible claim is only "exposed exchange time indistinguishable from zero for alpha<=0.5"'))
  || jsonb_set(evidence, '{benchmark,ref_vs_got,D1a_cut_vs_exhaustive_optimum,got_rel_err_max}', '"0.00% as committed / +4.23% from HEAD code (F0)"')
  || jsonb_set(evidence, '{benchmark,ref_vs_got,D5_hidden_fraction,pass}', '"committed PASS at 92.6% but fresh re-run gives 68.6% -> pass_D5 FALSE (run-unstable)"')
WHERE repo='TensorLBM' AND scope='m17596 XVAL-LBM-BLOCKFOREST-METIS-MULTISTREAM-OVERLAP' AND verdict_by='board-audit';

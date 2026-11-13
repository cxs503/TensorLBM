-- Machine referee row for anchor XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN.
--
-- Why this file exists: the anchor stayed open after the first delivery (mission
-- 17839, commit eee0988a) because `SELECT ... FROM tensor_benchmark_cases WHERE
-- anchor_slug='XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN'` returned 0 rows -- "passes the
-- benchmark" was not machine-adjudicable.  This row names the reference basis, the
-- metric, the locked constants (duplicated in verify.py LOCKED) and the auto-check
-- command.  Idempotent: re-running replaces the row instead of duplicating it.
--
-- Applied 2026-11-13 (mission m17902); evidence: benchmarks/verified/
-- fused_push_pull_aa_pattern/{result.json,verify.py} + docs/
-- XVAL_m17902_FUSED_PUSH_PULL_AA_PATTERN_INTEGRATION.md

BEGIN;

DELETE FROM tensor_benchmark_cases WHERE anchor_slug = 'XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN';

INSERT INTO tensor_benchmark_cases
  (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
VALUES (
  'XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN',
  '2d_taylor_green_decay_d2q9_bgk_fused_push_pull_vs_ab',
  'Closed-form 2-D Taylor-Green solution as evaluated by the upstream reference itself '
  '(lbm-principles-practice/code chapter13/gpu/LBM.cu taylor_green_eval, cell-centred '
  'X=x+1/2, Y=y+1/2, rho=rho0+3P); nu=1/6 (tau=1.0), u_max=0.02 (Ma=0.0346), periodic, '
  'float64, stiffness ladder n=32/48/64 with steps 120/270/480 (steps ∝ n², fixed '
  'physical time). Referee implemented at benchmarks/verified/fused_push_pull_aa_pattern/.',
  'u_x and u_y relative L2 error vs closed-form Taylor-Green at the finest grid, for BOTH '
  'kernels (repo AB reference step_stream_collide(kernel="ab") and the upstream fused '
  'push-pull kernel); plus scheme equivalence max|f_fused - collide_bgk(f_ab)| and the '
  'cross-arm velocity relative L2; plus relative mass drift over the run; plus two '
  'negative-control arms (viscosity +25%, mirror-initialised field) that must exceed the gate.',
  '<3% per-kernel u_x/u_y rel.L2 (owner 15% gate; 3% self target, measured ~0.09%); '
  'equivalence state < 1e-12 and cross-arm velocity < 1e-10 (measured 3.3e-16 / 8.3e-15); '
  'mass drift < 1e-12 (measured ~1e-15); observed convergence order >= 1.0; both negative '
  'controls >= 15% so the gate is not toothless. '
  'auto-check: python benchmarks/verified/fused_push_pull_aa_pattern/verify.py (exit 0 = pass); '
  'regenerate: python benchmarks/verified/fused_push_pull_aa_pattern/run.py',
  'TensorLBM',
  'pending'
);

COMMIT;

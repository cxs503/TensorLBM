-- CASE-LBM-SQUARE-CYL-NORBERG — machine referee row (audit m16183 remediation item 1).
--
-- Audit m16183 found `SELECT ... WHERE anchor_slug='CASE-LBM-SQUARE-CYL-NORBERG'` returned
-- 0 rows, so "passes the benchmark" could not be machine-adjudicated by anyone.
-- The row below is the referee: it names the reference basis, the metric, and the
-- locked force column, and both constants come from refspec.py (single source).
--
-- Idempotent: re-running replaces the row for this anchor instead of duplicating it.
-- Applied 2026-11-12 (mission m16553); evidence: docs/CASE-LBM-SQUARE-CYL-NORBERG_remediation.md

BEGIN;

DELETE FROM tensor_benchmark_cases WHERE anchor_slug = 'CASE-LBM-SQUARE-CYL-NORBERG';

INSERT INTO tensor_benchmark_cases
  (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
VALUES (
  'CASE-LBM-SQUARE-CYL-NORBERG',
  'square_cylinder_re100_free_stream_vortex_shedding',
  '2D free-stream low-blockage (blockage 2.5%, 40D domain) numerical cluster, alpha=0 deg: '
  'Cd_ref = 1.48 in [1.44, 1.52] (median 1.488); St_ref = 0.147 in [0.145, 0.149]. '
  'Sources: arXiv 2411.03124v4 Tab.1, 2309.09197 Tab.2, 2308.08085v3 Tab.7/8, '
  '2404.12123, 2405.12834. NOT applicable: Okajima 1982 experiment (3D wake, St 0.143-0.145, '
  'no Cd), Sohankar 1997 channel (high blockage, Cd~2.05), Sharma&Eswaran 2004 (small domain). '
  'Cross-check: benchmarks/verified/square_cylinder/REFERENCE_AUDIT.md; locked constants: refspec.py',
  'Cd error vs Cd_ref = 1.48 and St error vs St_ref = 0.147, finest grid, scored on the locked '
  'primary force column pair (Ladd paired two-population MEM); wet column is diagnostic only',
  '|Cd err| <= 3% AND |St err| <= 3% (refspec.TOL_PCT = 3.0) on the finest grid; all grids inside the '
  'locked reference bands; monotone error with D or resolution floor <= 1%; '
  'auto-check: python benchmarks/verified/square_cylinder/verify.py [--repro-dir DIR] (exit 0 = pass), '
  'regression: pytest tests/benchmarks/test_square_cylinder_case.py',
  'TensorLBM',
  'pending'
);

COMMIT;

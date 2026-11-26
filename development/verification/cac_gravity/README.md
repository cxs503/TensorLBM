# Historical cAC gravity verification missions

These are internal development records, not published user manuals.
The original files were relocated byte-for-byte; no solver or runner was
changed or executed for this migration.

| Original location | Preserved records |
| --- | --- |
| `docs/m23012/` | [Preregistration](m23012/PREREG.json), [report](m23012/REPORT.md), [original results](m23012/results.json) |
| `docs/m23170/` | [Phase A derivation](m23170/PHASE_A_DERIVATION.md), [Phase A report](m23170/PHASE_A_REPORT.md) |

The m23012 result records `overall_pass: false`, approximately 788.65%
hydrostatic-pressure error and nonfinite phase-integral drift. The m23170
report asserts a later Phase A BGK pass; its mission directory contains no
separate machine-readable result. Neither record is a current acceptance.

The original inline references to `src/tensorlbm/cac_lbm3d.py` and
`scripts/verify_cac_hydrostatic_mass.py` point to historical names. Available
frozen snapshots are [source](../../legacy/TensorLBM/src/tensorlbm/cac_lbm3d.py)
and [runner](../../legacy/TensorLBM/scripts/verify_cac_hydrostatic_mass.py);
these are not the maintained source entry or certified exact run inputs.

- [Case registry declaration](../../../benchmarks/cross_validation/cac_gravity_historical/benchmark_case.json)
- [Original Git objects and SHA-256 migration ledger](../../../docs/architecture/migrations/cac-gravity-mission-archives-2026-11-27.json)

Current scientific acceptance: not evaluated. Original result JSON contains
nonfinite values and is preserved in its original representation.

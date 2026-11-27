# Historical NEBB steady-channel comparison (early rounds)

The independent m17963 audit's [original report](../../../docs/board_audit/m17963/AUDIT_m17759.md),
[original delivered diff](../../../docs/board_audit/m17963/rerun/DIFF_OSS_VS_TENSORLBM.md),
[original runner](../../../docs/board_audit/m17963/rerun/xval_nebb_bodyforce_driver.py) and
[original measured payload](../../../docs/board_audit/m17963/rerun/xval_nebb_bodyforce_driver.json)
remain in their pre-existing frozen audit snapshot. The three loose originals
and additional reference filename were proven byte-identical before consolidation.

The [subsequent repair report](REPORT_XVAL_NEBB_BODYFORCE.md) is a distinct
historical record and is preserved verbatim. The [original encoded figure](xval_figure_b64.json)
and its [exact decoded PNG](archived_nebb_profile.png) preserve the earlier
figure bytes; decoding is not scientific reproduction. The historical
[renderer](make_xval_figure.py) reads the same exact audit payload at its retained
path. Execute it only in an isolated checkout to avoid overwriting historical
outputs. The [current R3 tool](../../../tools/nebb_bodyforce_xval.py) reads the
same original payload and does not claim native JAX execution.

The historical m17963 audit recorded failure by the predeclared max-interior
relative metric; the repair report is not a substitute for fresh independent
acceptance. The registered transient case and remaining later XVAL rounds
are separate; neither their assets nor their verdicts change here.

[Archive-only registry entry](../../../benchmarks/cross_validation/nebb_steady_early_rounds/README.md) ·
[Migration ledger](../../../docs/architecture/migrations/nebb-early-rounds-and-sparse-audits-2026-11-27.json).

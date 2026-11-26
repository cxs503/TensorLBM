# Historical regularised boundary verification

This archive preserves the generic wall-closure cross-validation campaign (R1–R5)
and its historical conclusions. It is not a current scientific acceptance claim.
The production implementation remains in [src/tensorlbm/regularised_bc.py](../../../src/tensorlbm/regularised_bc.py).
The [2D reference and runner](../regularised_bc_2d_reference/) remains a live
test dependency; the [2D solver](../../../src/tensorlbm/regularised_bc_2d.py)
is not duplicated here.

- [Published 2D study](../../../docs/reports/XVAL_LBM_REGULARISED_BC_NONEQ_STRESS.md)
- [Round 2 audit](../../../docs/reports/XVAL_LBM_REGULARISED_BC_ROUND2_AUDIT.md)
- [Round 5 report](../../../docs/XVAL-LBM-REGULARISED-BC-NONEQ-STRESS-R5.md)
- [Archived R4 report](REPORT_R4.md), [R5 report](REPORT_R5.md),
  [Round 2 measured summary](artifacts/round2_summary.json)

To reproduce the Round 2 figures from the archived measurements, run from the repository
root: `python development/verification/regularised_bc_noneq_stress/make_round2.py`.
It writes the summary and figures in this archive; use a disposable checkout.
The historical linked figures were not tracked in the migration base. Full R4/R5
numerical campaigns and external report publishing are not part of this move.

The original checksums and path mapping are in the
[migration ledger](../../../docs/architecture/migrations/regularised-boundary-archives-2026-11-26.json).

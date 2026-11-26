# Historical contact-angle verification series

The archived R2–R5 rounds and orientation study are grouped by topic.
All historical verdicts and data remain as recorded, not current scientific acceptance.
The [registered reference case](../../../benchmarks/cross_validation/contact_angle_geodesic/benchmark_case.json)
links the original calibration, measured fields and independent audits.

- [Round 2](round-2/REPORT.md) and [summary](round-2/summary.json).
- [Round 3](round-3/REPORT_R3.md) and [measured fields](round-3/runs_r3/).
- [Round 4](round-4/REPORT_R4.md) and [results](round-4/results_r4.json).
- [Round 5](round-5/REPORT_R5.md) and [results](round-5/results_r5.json).
- [Orientation study](../../../docs/WALL-ORIENTATION-CONTACT-ANGLE-INVARIANCE.md) and
  [summary](wall-orientation/summary.json).
- [Published independent R5 audit](../../../docs/AUDIT_XVAL_TWO_IMPL_CONTACT_ANGLE_R5.md).
- [Migration ledger](../../../docs/architecture/migrations/contact-angle-early-rounds-2026-11-26.json).

Run `python development/verification/contact-angle/round-5/run_r5.py measure`
for the existing offline remeasurement of committed fields. It writes to the
ignored outputs directory; the frozen original R5 result remains untouched.
Earlier runners write beside their own archives: use a disposable checkout.
The full solver campaigns and missing historical plot assets were not
re-evaluated as part of the directory move.

# Static immersed markers for a circular cylinder

This case indexes the archived Re=20 verification and its independent board
audit. The audit found a mean drag coefficient of 0.24318 against the
preregistered reference of 2.05 (88.14% relative error, with a 15% tolerance).
The trailing drag history also failed the convergence criterion. The original
mission's raw cluster result is absent. The repository retains the audit
result, figures and audited solver snapshot.

- [Original report](../../../development/verification/static-immersed-markers/REPORT.md)
- [Preregistered case](../../../development/verification/static-immersed-markers/predeclared.json)
- [Board audit](../../../docs/board_audit/m18779/audit_report.md)
- [Machine-readable evidence](../../../docs/board_audit/m18779/verify_output.json)
- [Case manifest](benchmark_case.json)

The [archived LIFE runner](../../../development/verification/static-immersed-markers/run_xval.py)
uses Re=100 and a separate three-arm comparison. It cannot replay the Re=20
result. Its LIFE reference input (`A1_life_binary/TotalForces.out`) is absent
from the tracked tree.

The [original paths](https://github.com/cxs503/TensorLBM/tree/32bf22f8074b9232d70fa5675089e7c1756dd9cd/xval/XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES)
are available at the pinned source revision. The [migration ledger](../../../docs/architecture/migrations/static-immersed-markers-xval-2026-11-25.json)
records the checksums for the three preserved files.

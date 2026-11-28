# m15891 — historical MCMP Laplace board audit

The current script re-scores existing committed inputs without a solver run. The 2026-11-28 disposable-checkout reanalysis produced the same JSON and PNG before and after the repository-root path correction. These are newly derived artifacts, not recovered historical original files or fresh scientific acceptance.

- [Current post-processing entry](audit_m15891_mcmp_laplace.py)
- [Stable published historical report](../../../docs/audits/m15891_CASE-MCMP-LAPLACE_audit.md)
- [Frozen historical verdict SQL (not executed)](verdict_m15891.sql)
- [Case input and derived-output index](../../../benchmarks/case_pool/mcmp_laplace/evidence/m15891/README.md)
- [Migration ledger](../../../docs/architecture/migrations/board-audit-and-design-script-handoff-2026-11-28.json)

From the repository root, in a disposable checkout, run `python development/audit/m15891/audit_m15891_mcmp_laplace.py`. The script retains the original ignored/local output locations; do not overwrite the tracked evidence.

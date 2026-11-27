# Historical mission audit m20976

Historical L2 / reopened R2 audit of mission m20893 (578410df6), honest FAIL of grid-independent acceptance; audit-specific cluster job receipts not all tracked.

Original audit evidence is preserved verbatim; this directory migration is not scientific replay. Frozen scripts and SQL retain original paths, worktree assumptions and publishing destinations. They are not active case entry points; historical rendering/replay may require the audited revision or absent run outputs, and historical publishing scripts are not executed. The [migration ledger](../../../docs/architecture/migrations/oracle-and-square-cylinder-audit-docs-2026-11-27.json) resolves old tracked paths.

Case: [current runner](../../../benchmarks/xval_oracle_minimal_d2q9_fs/run.py); [published report](../../../docs/XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS_R2_RESOLUTION.md).

Original and previously archived records:

- [AUDIT_m20976_XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS_R2.md](AUDIT_m20976_XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS_R2.md)
- [build_verdict.py](build_verdict.py)
- [evidence/fit_and_richardson.json](evidence/fit_and_richardson.json)
- [evidence/job_log_recovery.json](evidence/job_log_recovery.json)
- [evidence/snapshot_a16_T0.5.json](evidence/snapshot_a16_T0.5.json)
- [evidence/verdict_evidence.json](evidence/verdict_evidence.json)
- [fit_from_artifact.py](fit_from_artifact.py)
- [parse_job_logs.py](parse_job_logs.py)
- [publish_report.py](publish_report.py)
- [repro/R2_RESOLUTION.md](repro/R2_RESOLUTION.md)
- [repro/collect_job_stdout.py](repro/collect_job_stdout.py)
- [repro/r2_cj3032_maxima.json](repro/r2_cj3032_maxima.json)
- [repro/resolution_study.py](repro/resolution_study.py)
- [verdict_m20976.sql](verdict_m20976.sql)
- [verify_r2.py](verify_r2.py)

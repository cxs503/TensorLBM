# Audit m17876 of the historical m17835 convolution-streaming delivery

[Original audit report](AUDIT_m17835.md) · [Original evidence figure](fig_m17835_evidence.png).
The [previously archived verdict](../m17835/verdict_m17835.sql) remains
at its existing path; the original historical tools in this archive are
preserved unchanged: [evidence collector](collect_evidence.py),
[verdict builder](make_verdict_sql.py), and [report publisher](publish_report.py).

The original report records L2 and a failed XVAL claim, absent machine referee
rows and historical untracked outputs. The audit did not run the upstream
TensorFlow binary. The original audit-specific machine JSON and complete mission
worktree are not available in the tracked tree. Do not run the historic DB or
publishing scripts as current acceptance.

The [related current benchmark directory](../../../benchmarks/xval_conv_as_streaming/README.md)
indexes the existing later 3-D result separately from this earlier audit.
[Migration record](../../../docs/architecture/migrations/peskin-and-conv-audit-2026-11-27.json).

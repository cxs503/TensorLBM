# Historical single-component pluggable forcing comparison

Preserve the original [protocol](predeclared.json), [measurement results](results.json),
[report](REPORT.md), and [independent audit](../../audit/m18438/AUDIT_m18438.md).
The runner explicitly selects the frozen historical checkout
`development/legacy/TensorLBM/src`, not the installed current package.
It is isolated from the active source roots; the frozen tree and original
results are unchanged. Do not use this historical runner as a current
production solver or scientific acceptance claim. Full historical execution
is deferred.

[Migration record](../../../docs/architecture/migrations/eos-pinned-forcing-2026-11-26.json).

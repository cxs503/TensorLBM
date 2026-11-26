# Historical moving-wall halfway bounce-back float64 check

The [published report](../../../docs/XVAL-LBM-MOVING-WALL-HALFWAY-BB-FLOAT64.md)
and [predeclared protocol](predeclared.json) describe the original verification.
Frozen measured outputs remain at [summary.json](summary.json) and [table.csv](table.csv).
The [external reference driver](oss_reference.py) requires the original pinned
pylbm checkout (see its documentation); the existing shim is preserved as historical
code, not new numerical evidence. No external reference run was conducted
for this relocation.

See the [migration ledger](../../../docs/architecture/migrations/regularised-boundary-archives-2026-11-26.json).

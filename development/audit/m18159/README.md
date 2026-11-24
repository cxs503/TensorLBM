# m18159 board audit tools

This directory holds the mission-local audit collector, report and figure
generators, verifier, and supporting clients. The
[published audit](../../../docs/board_audit/m18159/M18159_BOARD_AUDIT.md) and its
machine-readable evidence retain their published paths.

To regenerate the report and figures from the tracked evidence in a working
checkout, run:

```bash
python development/audit/m18159/m18159_render_report.py
python development/audit/m18159/m18159_figures.py
```

The remaining scripts depend on the original mission worktree, TensorPlatform
database, or authenticated local API. The
[original script paths](https://github.com/cxs503/TensorLBM/tree/32bf22f8074b9232d70fa5675089e7c1756dd9cd/scripts)
remain available at the source revision. The
[migration ledger](../../../docs/architecture/migrations/m18159-audit-tools-2026-11-25.json)
records the exact mappings and checksums.

# m17839 fused push–pull cross-validation evidence

This is the original CPU cross-validation delivery and the accompanying upstream
source snapshot. The [published report](../../../../../docs/XVAL_m17839_FUSED_PUSH_PULL_AA_PATTERN.md)
and [board audit](../../../../../docs/board_audit/m17884/AUDIT_m17884.md) describe its historical scope.
The parent benchmark runs its own referee and maintains its own result.

- [Original result](result.json)
- [Upstream provenance](ref/PROVENANCE.json)
- [Archived figure index](figures/index.json)

From the repository root, replay the portable path on a CPU:

```sh
OMP_NUM_THREADS=1 python benchmarks/verified/fused_push_pull_aa_pattern/evidence/m17839/run.py --quick
```

Omitting `--out` writes to `outputs/historical_xval/fused_push_pull_aa_pattern/`.
The original PNGs are absent from the tracked snapshot. The renderer can regenerate
them from the archived result. Run it in a separate worktree since it also updates the
published Markdown and figure metadata.

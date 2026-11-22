# AUDIT-RUNTEST-6 · m14496

The [published report](../../../docs/AUDIT_RUNTEST_6_TENSORLBM.md) records
the sample tests, benchmark comparisons, convergence checks and limitations of
the original audit. The [figure manifest](figures.json) retains the source
workspace's paths.

[The audit tools](tools/) retain their original filenames. Repository-relative
Python tools and the local benchmark batch use the current checkout root.
Five original shell drivers keep the old `/srv/tensor/agent/work/m14496`
workspace path for historical replay. The audit's raw logs and JSON are absent
from this checkout. The scripts look for inputs under `artifacts/audit_runtest_6/`
or `runs/audit_runtest_r6/` as they did in the original workspace.

`patch_conv_arms.py` is the historical in-place solver patch. The publication
script is a historical delivery tool. Both are retained to document the audit
and are outside the maintained solver and CI workflows.

The [migration ledger](../../../docs/architecture/migrations/runtest-r6-audit-tools-2026-11-23.json)
records each previous path and the original Git object and SHA-256.

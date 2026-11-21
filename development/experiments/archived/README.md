# Archived experiment scripts

One-off drivers kept for provenance: the `*_worker.py` / `*_launcher.py`
campaign scripts, `verify_*.py` / `diagnose_*.py` probes and the
`run_*.sh` / `launch_*.sh` shells that produced the numbers quoted in
`docs/` and `docs/reports/`.

**These are not part of the package and are not maintained.**

- Historical workspaces can contain their own package trees and test fixtures. The installable TensorLBM package lives in [src/tensorlbm](../../src/tensorlbm/).
- Not linted, formatted or type-checked: CI runs ruff and mypy over
  `src tests examples benchmarks` only.
- Pinned to the repository state and hardware of the day they ran (many
  expect the 8×5090 / SDAA dev boxes, absolute paths, or `results_*/` output
  directories that are no longer tracked). Expect them to need edits before
  they run again.

## Where things belong now

| Kind of code | Home |
|---|---|
| Library code | `src/tensorlbm/` |
| Reusable, maintained tooling | `scripts/` (linted, some covered by `tests/`) |
| Benchmarks with verified reference data | `benchmarks/` |
| Documented, runnable examples | `examples/` |
| One-off experiment / debug driver | **here** |

The [workspaces](workspaces/) retain mission-specific files under their original
relative hierarchy. [Staging snapshots](staging/) contain partial source trees
used to launch individual campaigns. To find a former root script or a mission
directory, search the [old-to-new path manifest](../../docs/architecture/migration_manifest.csv).
The release command remains at the root. The old cleanup utility is under
[scripts/maintenance](../../scripts/maintenance/).

## Reviving one

Copy it out, don't edit in place:

```bash
cp experiments/archived/<script>.py /tmp/ && cd /tmp
PYTHONPATH=/path/to/TensorLBM/src python <script>.py --help
```

If it turns into something worth keeping, promote the logic into
`src/tensorlbm/` with a test, and leave the archived copy alone as the
historical record.

# Benchmark definitions and acceptance evidence

Start with [STATUS.md](STATUS.md) and the individual case's report. Lifecycle folders
already exist; their names alone are not independent proof of physical validity.

- `pending/`: incomplete validation or unresolved acceptance blockers.
- `verified/`: a recorded acceptance claim whose scope, source revision, references
  and metrics must be checked before reuse.
- `archived/`: historical evidence and superseded campaigns; retain the reason.

Keep the runner, versioned configuration, references, thresholds, curated results
and report traceable by a stable benchmark ID. Distinguish analytic checks,
internal consistency, external experimental validation and performance measurements.
Never upgrade status solely because a runner exits zero or a directory was moved.

New raw runs go to ignored output locations. Large data should have versioned
external storage, SHA-256, a retrieval command and generating provenance. Do not
externalize existing fixtures until consumers can retrieve and validate them.

A common benchmark manifest/schema and automatic lifecycle validation are planned,
not yet implemented by the repository layout gate. See the
[test/benchmark map](../docs/architecture/test_and_benchmark_map.md),
[layout policy](../docs/architecture/repository_layout.md) and
[migration candidates](../docs/architecture/migration_manifest.csv).

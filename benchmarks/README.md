# Benchmark definitions and acceptance evidence

Find the [published case manuals and downloads](../docs/benchmarks/tutorials/README.md).
Use `python tools/benchmark_registry.py --find-asset couette` to search tracked
paths in benchmarks, docs, experiments, xval, audit and audits, including files
outside the case registry.

Start with [STATUS.md](STATUS.md) and the individual case's report. Lifecycle folders
already exist; their names alone are not independent proof of physical validity.

- `validation_matrix/`: the nine-case validation matrix (registry, criteria,
  reference provenance gate, CI-tier cases, generated three-level table).
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

Use `python tools/benchmark_registry.py --case couette_2d` from the repository
root to look up tracked runners, local references, results and tutorial downloads.
Use `--case matrix:ship_resistance_suboff` to retrieve the matrix definition,
reference provenance, reproduction command and archived comparison for that case.
A missing comparison is returned as null.
Use `--json` to export all records from the published tutorial catalogue, the validation matrix and
the immediate lifecycle/case-pool directories. Nested campaign assets and
standalone campaign folders still require manual registration. The registry
lists available evidence and leaves current acceptance unevaluated.
See the
[test/benchmark map](../docs/architecture/test_and_benchmark_map.md),
[layout policy](../docs/architecture/repository_layout.md) and
[migration candidates](../docs/architecture/migration_manifest.csv).

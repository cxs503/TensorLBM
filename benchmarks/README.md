# Benchmark definitions and acceptance evidence

Find the [published case manuals and downloads](../docs/benchmarks/tutorials/README.md).
Use `python tools/benchmark_registry.py --find-asset couette` to search tracked
paths in benchmarks, docs, experiments, xval, audit and audits, including files
outside the case registry.

Start with [STATUS.md](STATUS.md) and the individual case's report. Lifecycle folders
already exist; their names alone are not independent proof of physical validity.

- `validation_matrix/`: the nine-case validation matrix (registry, criteria,
  reference provenance gate, CI-tier cases, generated three-level table).
- [model_units/](model_units/README.md): analytic regression cases used by the
  generated model-unit CI matrix.
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
Each case variant also lists tracked files under its own `evidence/` directory,
including nested run logs and figures. The registry records these paths in
`evidence` and keeps the existing `results` field for case result inputs.
Use `--case matrix:ship_resistance_suboff` to retrieve the matrix definition,
reference provenance, reproduction command and archived comparison for that case.
A missing comparison is returned as null.
Use `--case xval:ludwig_fe_stress` to find the analytic criteria, both CLI paths, the archived result and the dated CPU replay. To register a cross-validation, add a tracked `xval/<case>/benchmark_case.json` using that case as the example.
Use `--asset-index` to export all Git-tracked benchmark files with their object IDs.
Each entry distinguishes explicit case links from membership in the nearest
registered case directory. Files with neither association remain listed and
counted by directory. Standalone campaigns use `benchmarks/<name>/benchmark_case.json`
and can be queried as `--case campaign:<name>`, for example `campaign:dam_break_ritter`.
Use `--json` to export all records from the published tutorial catalogue, the validation matrix and
the immediate lifecycle/case-pool directories, and explicitly registered campaigns and cross-validations. The registry
lists available evidence and leaves current acceptance unevaluated.
See the
[test/benchmark map](../docs/architecture/test_and_benchmark_map.md),
[layout policy](../docs/architecture/repository_layout.md) and
[migration candidates](../docs/architecture/migration_manifest.csv).

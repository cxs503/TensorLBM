# TensorLBM structure and benchmark asset preservation audit (2026-11-21)

## Scope and baseline

Read-only inspection of the `main` recursive tree at SHA `244139491515a78a9e971b83eafceba7c70d6a19` (16,509 tree entries, including directories; not a file count). No numerical code or existing asset moved. This is a planning baseline, **not** scientific revalidation.

| Root subtree | Recursive entries |
| --- | ---: |
| benchmarks | 4,138 |
| docs | 3,379 |
| xval | 1,606 |
| src | 1,575 |
| tests | 1,274 |
| experiments | 868 |
| TensorLBM | 760 |
| scripts | 738 |
| audit | 527 |
| runs | 409 |

## Confirmed findings

1. The authoritative installable library is `src/tensorlbm/` per `docs/architecture/repository_layout.md`. `TensorLBM/src/` is a historical nested copy, with its own `TensorLBM/pyproject.toml`. **Do not delete** without import/consumer and hash review.
2. `docs/benchmarks/tutorials/README.md` explicitly indexes **33 tutorial cases** with Markdown, PDF/Word where available, and `benchmarks/verified/` case links. The tree contains 33 PDF files in `docs/benchmarks/tutorials/files/`. Some indexed cases have no PDF. Preserve these assets and their links.
3. `benchmarks/STATUS.md` reports 91 verified and 60 pending **directory inventory counts dated 2026-10-05**, explicitly not independent proof of scientifically validated capabilities. Never translate those counts into current acceptance passes.
4. `benchmarks/` mixes lifecycle (`verified/`, `pending/`, `archived/`), subject (`water_entry/`, `wall_model_les/`), and mission-numbered campaign paths. `docs/` mixes public tutorials and many mission audit folders. This makes discoverability and ownership ambiguous.
5. Existing governance is **not absent**: `REPOSITORY_REFACTORING_PLAN.md`, `repository_layout.md`, `repository_layout_policy.json`, `migration_manifest.csv`, `STRUCTURE_MAP_2026-10-26.md` and `test_and_benchmark_map.md` already exist. The plan forbids bulk moves. The manifest contains completed, proposed and blocked items; do not mark the entire refactor complete.
6. The 2026-10-26 structure map reported 11 import cycles and zero static reverse imports from core to app **at that snapshot**. These are not fresh runtime findings.

## Ownership contract (proposed; non-breaking)

- `src/tensorlbm/`: single supported numerical package.
- `tests/`: code/unit/integration/regression tests; separate benchmark acceptance from ordinary green CI.
- `benchmarks/`: canonical benchmark definitions, reference provenance, acceptance criteria and curated evidence.
- `docs/benchmarks/tutorials/`: **canonical published case manuals** (33-case baseline). Do not copy them into runner folders.
- `examples/`: maintained runnable user entrypoints; `experiments/`: unaccepted research.
- `audit/` and existing historical paths: preserve pending verified archival migration; new mission artifacts must have an explicit owner and retention policy.

Prefer stable benchmark IDs and a registry mapping `id -> runner, tutorial, reference, evidence, revision, status`. Status is a *versioned verdict*, not a reason to move case identity. Existing `verified/` and `pending/` paths remain compatible until each consumer is inventoried.

## Ordered migration ledger (not yet executed)

| Priority | Asset / path | Proposed action | Blocking verification |
| --- | --- | --- | --- |
| P0 | `docs/benchmarks/tutorials/README.md` and `files/` | Preserve 33-case baseline; generate case-to-manual inventory | Check all indexed links, byte hashes, tracked PDF/DOCX availability |
| P0 | `TensorLBM/src/` and nested `pyproject.toml` | Compare with root package; classify historical vs live | Static imports, editable install, CI, external consumers, exact hashes |
| P0 | `benchmarks/verified/` and `pending/` | Inventory case IDs and source/evidence/threshold completeness | Compare validation matrix and `validation_status.json`; no status promotion |
| P1 | `benchmarks/case_pool/` and mission-named directories | Map to stable IDs; add registry aliases first | Search CI, docs, scripts and result consumers |
| P1 | `docs/board_audit/`, `docs/audit_m*/`, `audit/`, `audits/` | Separate published docs from agent mission provenance | Retention, citation, links and external report consumers |
| P1 | `xval/`, `runs/`, `artifacts/`, `experiments/` | Define source / reproducible evidence / temporary output policy | Git tracked status, checksums, retrieval and execution references |
| P2 | Flat modules in `src/tensorlbm/` | Pilot one compatible subpackage, no math changes | Import graph, compatibility shims, tests, scientific regression |

## Mandatory acceptance gates for each future move

1. Freeze base SHA and old/new path manifest with blob SHA and size.
2. Search Python imports, `pyproject.toml`, CI, tests, scripts, docs, notebook links and external CLI usage; record unresolved consumers.
3. Preserve documented import/CLI contracts or add compatibility shims.
4. Verify README/Markdown relative links and tutorial PDF/DOCX availability.
5. Run layout gate, affected unit/integration tests, benchmark-specific scientific acceptance and compare numerical outputs. **No tests were executed in this audit.**
6. Update `docs/architecture/migration_manifest.csv` with status and evidence; do not delete historical provenance or overwrite accepted results.
7. Merge only after independent review of CI and scientific evidence. A passing layout check is not scientific validation.

## Immediate implementation PRs

1. Machine-generated 33-case inventory with existence checks, checksums and links (read-only CI).
2. Canonical benchmark registry with aliases to existing paths and explicit acceptance scope.
3. Nested-tree import/packaging comparison and reference search; only then propose deletion or archival.
4. One small migration batch selected from existing manifest where external provenance and compatibility are resolved.

## Reference documents

- [Original refactoring plan](REPOSITORY_REFACTORING_PLAN.md)
- [Current layout policy](repository_layout.md)
- [Migration ledger](migration_manifest.csv)
- [2026-10-26 structure map](STRUCTURE_MAP_2026-10-26.md)
- [33-case tutorial index](../benchmarks/tutorials/README.md)
- [Benchmark status](../../benchmarks/STATUS.md)

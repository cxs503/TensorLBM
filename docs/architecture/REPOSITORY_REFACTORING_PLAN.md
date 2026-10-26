# Repository Architecture and Refactoring Plan

## Status

- Phase: initial inventory exists; low-risk Phase 2 governance is being implemented.
- Working branch: `arena/d059dc71-tensorlbm`
- Base branch: `main`
- This document does not authorize bulk file moves. Structural changes must follow an inventory of imports, public APIs, tests, CI workflows, documentation links, and benchmark provenance.

Implementation entry points:
[layout and artifact policy](repository_layout.md),
[exact-path historical exceptions](repository_layout_policy.json), and
[first migration candidates](migration_manifest.csv).
The hygiene gate does not imply the complete structural audit or migration is finished.

## Why this work is needed

The repository currently contains several distinct concerns:

1. The installable numerical solver under `src/tensorlbm/`.
2. Engineering/application services and UI under `app/` and `frontend-vue/`.
3. Tests under both `tests/` and `app/tests/`.
4. Benchmark implementations and status collections under `benchmarks/`.
5. Documentation, research experiments, run outputs, figures, and historical artifacts spread across several top-level directories and files.

These concerns can coexist in one repository, but their ownership, dependency direction, validation status, and lifecycle must be explicit.

## Non-negotiable safety rules

1. Do not change numerical algorithms as part of a file-organization-only patch.
2. Do not move or delete a file until references have been searched in Python imports, tests, CI/workflows, scripts, docs, notebooks, packaging metadata, and user-facing commands.
3. Preserve public imports and documented CLI paths. Where relocation is unavoidable, add a compatibility shim or a deprecation window.
4. Never promote an experiment to a verified benchmark solely because its script exits successfully.
5. Keep benchmark source, configuration, reference data, raw outputs, derived figures, acceptance criteria, and report provenance traceable by a stable benchmark ID.
6. Treat `pending`, `verified`, and `archived` as lifecycle states with written entry/exit criteria, not merely arbitrary storage folders.
7. Separate CPU/unit checks from GPU, long-running physics verification, and performance tests so contributors know what each green check proves.
8. Make small, reviewable pull requests. Do not combine solver refactors, numerical changes, UI changes, and large artifact moves in one PR.

## Intended architectural boundaries

### A. Numerical library

`src/tensorlbm/` owns reusable numerical algorithms, stable solver interfaces, backend abstractions, physics models, and runtime facilities needed by the library itself.

It must not import application routers, frontend code, web-server lifecycle code, or agent orchestration. Optional extensions must not silently become mandatory dependencies of the core solver.

### B. Application and orchestration

`app/` and `frontend-vue/` own API/UI, user workflows, job orchestration, platform integration, and presentation. They may depend on supported TensorLBM interfaces; the numerical library must not depend on them.

Before extracting this layer into another repository/package, inspect deployment, packaging, CI, and cross-project consumers. No repository split is proposed in Phase 0.

### C. Tests

Tests should be classified by the claim they establish:

- unit: local function/class contracts;
- integration: module and backend interactions;
- physics: conservation, analytical solutions, convergence, and known physical invariants;
- regression: a previously observed defect cannot recur;
- benchmark acceptance: explicit reference quantities and tolerances;
- performance: throughput, memory, scaling, and hardware-dependent budgets;
- platform: API, orchestration, and end-to-end user workflows.

Existing paths need not be moved immediately. First build a manifest mapping tests to these categories and the code/benchmark they protect.

### D. Benchmarks and experiments

A verified benchmark must have, at minimum:

- stable ID and problem definition;
- solver/model and parameter configuration;
- reproducible runner and environment notes;
- reference source or a justified analytical invariant;
- explicit metrics, tolerances, and acceptance decision;
- links to raw results, derived figures, and report;
- provenance for the code revision that produced the result.

Experiments may be exploratory and inconclusive. Preserve that status; do not present exploratory results as verified evidence.

### E. Generated and temporary artifacts

Root-level debug scripts, one-off comparisons, temporary logs, and generated figures must be inventoried before relocation. Classify each as maintained tool, reproducible experiment, report asset, benchmark result, or disposable temporary output. Do not delete evidence merely to make the tree look clean.

## Phased execution plan

### Phase 0 — Baseline and policy (this PR)

- Record the current architectural concerns and safety rules.
- Define the inventory required before moving files.
- Agree on dependency direction and benchmark lifecycle semantics.

### Phase 1 — Read-only structural audit

Produce machine-readable inventories for:

- top-level tree and file counts by extension/size;
- Python import graph and cycles (with dynamic imports marked as uncertain);
- package public exports and import paths used outside their defining module;
- duplicate/similar module names and likely overlapping responsibilities;
- test collection, test category, runtime/hardware requirements, and coverage targets;
- references to candidate paths in CI, scripts, docs, notebooks, packaging, and workflows;
- benchmark IDs, manifests, status, source revision, raw results, reports, and missing evidence;
- root-level debug/report/figure artifacts and their likely owners.

The audit must distinguish confirmed facts from heuristic findings.

### Phase 2 — Low-risk governance

- Add or improve directory READMEs and an index for canonical developer entry points.
- Add benchmark manifest/schema and status-transition checks without changing solver math.
- Establish root-level file admission rules and a tracked artifact policy.
- Add CI checks for architecture constraints where they can be enforced reliably.

### Phase 3 — Incremental relocation

For each small batch, include a path mapping, reference search, compatibility plan, and tests. Prefer relocating clearly temporary files first. Move core solver modules only after dependency and API review.

### Phase 4 — Boundary hardening

- Enforce one-way dependencies from application to solver.
- Stabilize public imports and optional-dependency boundaries.
- Consolidate test entry points while retaining explicit slow/GPU markers.
- Make verified benchmark evidence reproducible and traceable.

## Acceptance criteria

This refactor is successful only when:

- documented public imports continue to work;
- baseline and post-change tests can be compared;
- no unexplained numerical result changes occur;
- moved files have no broken references;
- benchmark verification status is backed by explicit evidence;
- application code does not introduce reverse dependencies into the solver core;
- contributors and coding agents can identify the correct module, test command, and benchmark evidence path from documented rules.

## First audit deliverables

1. `docs/architecture/module_inventory.md` — module responsibilities, exports, dependency findings, and confidence.
2. `docs/architecture/test_and_benchmark_map.md` — test categories and benchmark evidence/lifecycle map.
3. `docs/architecture/migration_manifest.csv` — proposed path changes, references, compatibility, owner, risk, and verification command.
4. A short review memo listing confirmed high-risk findings and recommending the smallest safe first relocation PR.

Do not invent test results. If local execution, GPU access, or a full checkout is unavailable, mark the relevant check as not run and provide an exact command for a repository runner/CI agent.

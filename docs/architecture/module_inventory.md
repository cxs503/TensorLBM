# Module Inventory — Initial Read-Only Findings

## Scope and confidence

Snapshot reviewed: `main` at commit `6ab2d9e41c49df8e14f9c0c5cd4a6c3080aa3f20`.

This is an initial source/document review, not a complete static import-graph run. Findings are separated into confirmed observations and hypotheses requiring execution or deeper source inspection.

## Confirmed observations

| Area | Evidence | Observation | Risk / next action |
|---|---|---|---|
| Installable package | `pyproject.toml` | Package discovery is rooted at `src/`; the distribution is named `tensorlbm`. | Preserve package paths and installation behavior during moves. |
| Public API surfaces | `src/tensorlbm/__init__.py`, `src/tensorlbm/api.py`, `src/tensorlbm/experimental.py` | There are multiple API surfaces: package-root lazy exports, a curated `api.py`, and an experimental surface. The package-root docstring says it exposes 786 names, while README calls the public API “small, explicit.” | Decide which surface is stable, compatibility-only, or experimental. Do not delete root exports until usage and compatibility evidence is collected. |
| Package-root export maintenance | `src/tensorlbm/__init__.py`, `CONTRIBUTING.md`, `tests/test_public_api_lazy_exports.py` (referenced in module docstring) | New exports must be synchronized across a lazy mapping, `TYPE_CHECKING` imports, and `__all__`. | Keep the existing guard; consider generating/validating the export manifest to reduce manual drift. |
| App and solver coexistence | `app/backend/main.py`, `app/tests/`, `src/tensorlbm/` | The repo contains a FastAPI platform plus the installable numerical package and separate app tests. | Keep application dependencies one-way. Audit whether any solver modules import application services. |
| Runtime path mutation | `app/backend/main.py` | The app entry point inserts the repository's `src` path into `sys.path` at runtime. | This couples app startup to repository layout. Verify deployment/package assumptions before replacing with an installed package or explicit launcher. |
| Dynamic router loading | `app/backend/main.py` | Router imports are assembled dynamically and the comment says incomplete routers may be skipped. | Review exception handling and startup diagnostics: optional routing should not hide broken required features. Do not change behavior until tests clarify intended contract. |
| Cross-layer import compatibility | `src/tensorlbm/apps/hpc.py` | HPC app code tries both `backend.services.hpc_scheduler` and `app.backend.services.hpc_scheduler` import paths. | Indicates support for multiple application import layouts. Define one canonical import path and retain compatibility only if external use is confirmed. |
| Test split | `pyproject.toml`, `tests/`, `app/tests/` | Pytest's configured `testpaths` is `tests`; platform tests live separately under `app/tests`. | Document separate commands and ensure CI explicitly collects both suites where intended. |
| Benchmark lifecycle | `benchmarks/STATUS.md`, `benchmarks/TODO.md`, `benchmarks/pending/`, `benchmarks/verified/`, `benchmarks/archived/` | Benchmark assets have lifecycle directories and status documents. | Audit whether each “verified” item has explicit criteria, provenance, and reproducible results; directory placement alone is not proof. |
| Root-level research artifacts | Root contains many `dbg_*.py`, `verify_*.py`, `compare_*.py`, `report_*.md`, and `fig_*.png` files. | Research and diagnostic assets are mixed with project entry files. | Build reference/ownership inventory before relocation; preserve evidence and scripts that are still used. |

## Initial module-boundary map

These are logical groupings observed in the current tree, not a proposal to move every module immediately.

| Existing area | Likely responsibility | Audit questions |
|---|---|---|
| `src/tensorlbm/core/` | Core abstractions and shared numerical primitives | Are core modules independent of application and high-level case modules? |
| `src/tensorlbm/backends/`, `runtime/` | Backend selection and execution support | Are backend-specific imports optional and isolated? |
| Root-level `d2q9.py`, `d3q19.py`, `d3q27.py`, collision and boundary modules | Lattice definitions and numerical kernels | Which files are canonical implementations versus compatibility or optimized variants? |
| `src/tensorlbm/phasefield/`, multiphase-related root modules | Multiphase/phase-field methods | Are algorithms, adapters, evidence gates, and benchmark drivers separated consistently? |
| `src/tensorlbm/octree_boundary/` and AMR-related modules | Octree geometry, refinement, distributed stepping | Which module owns shared geometry/data contracts, and are there cycles across the root and subpackage? |
| `src/tensorlbm/ai/`, `ml/`, `apps/` | AI models, data pipelines, runnable applications | Which features are library extensions versus user-facing application workflows? |
| `app/backend/`, `frontend-vue/` | Web platform and engineering workflow | Can this layer be installed/run independently without manipulating the solver package's import path? |

## Required next audit before any core relocation

1. Build a static import graph for all Python files under `src/tensorlbm`, `app`, `tests`, and `app/tests`.
2. Flag strongly connected components/cycles and imports from the solver package into `app`.
3. Extract package-root exports and compare them with `api.py`, `experimental.py`, README examples, and tests.
4. Search references for each candidate path across CI, packaging, scripts, docs, notebooks, and benchmarks.
5. Run baseline tests in a clean checkout and record exact commands, environment, skips, and failures before moving code.

## Confidence labels

- **Confirmed**: directly visible in the files listed above.
- **Hypothesis**: likely risk inferred from structure; must be verified by import graph or runtime tests.
- **Not assessed**: requires execution, full-tree scanning, or additional source review.


## Additional confirmed findings from CI and contributor documentation

| Area | Evidence | Confirmed observation | Required action |
|---|---|---|---|
| CI test scope | `.github/workflows/ci.yml` | The workflow's pytest command runs with the repository's configured `testpaths`; the workflow does not include an explicit `app/tests` command. The lint command covers `src tests examples benchmarks`, not `app` or `frontend-vue`. | Confirm intended platform test policy. If `app/tests` is production-critical, add a separate explicit job with app dependencies; add suitable lint/type checks deliberately rather than assuming solver CI covers it. |
| Contributor setup drift | `CONTRIBUTING.md`, `docs/development_workflow.md` | Both instruct installing `platform/requirements.txt` and starting from `platform/`, but the current tree places the implementation under `app/` and lists `app/requirements.txt`, not `platform/`. | Fix documentation to the current layout after checking deployment scripts and supported launch modes. This is a confirmed documentation/tree mismatch, not proof deployment is broken. |
| Root API ambiguity | `README.md`, `src/tensorlbm/__init__.py`, `src/tensorlbm/api.py`, `src/tensorlbm/experimental.py` | README describes a “small, explicit public API”, while package-root exports are intentionally broad (786 names per package docstring); a separate curated API and experimental API already exist. | Make the contract explicit: stable minimal API, legacy root compatibility surface, and experimental surface. Add migration guidance before changing exports. |
| Benchmark status freshness | `benchmarks/STATUS.md` | The file header says generated 2026-10-05 and the count note says refreshed 2026-10-09, while the repository tree has many more entries in lifecycle folders than the listed case table. | Treat this file as a curated status report, not a live directory inventory. Add a generated manifest/count check or state the precise scope of the report. |

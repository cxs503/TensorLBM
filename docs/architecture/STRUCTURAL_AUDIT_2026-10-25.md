# TensorLBM Structural Audit — Phase 1 Snapshot

Audit date: 2026-10-25  
Audited ref: `main` at `aef23b93835f9b7c5847e176840e8db7608f0aa9`  
Method: GitHub recursive tree metadata plus direct source/config inspection. This is a read-only structural inventory, not a full local AST/import-graph run.

## Repository size and concentration

The recursive tree contains **5,043 entries** (files and directories, not just files).

| Area | Tree entries | Interpretation |
|---|---:|---|
| `src/` | 495 | installable solver package and subpackages |
| `tests/` | 498 | solver/library tests |
| `app/` | 121 | FastAPI application and platform tests |
| `frontend-vue/` | 50 | Vue frontend source/build configuration |
| `benchmarks/` | 1942 | benchmark implementations, evidence, status collections |
| `docs/` | 749 | user/developer docs and benchmark material |
| `experiments/` | 245 | exploratory and historical research artifacts |
| `runs/` | 408 | run outputs/artifacts |

The tree has **478 Python files** under `src/tensorlbm/`, of which **353** sit directly at the package root. This is a large flat namespace; it does not by itself prove the modules should all move, but it does mean file moves need explicit ownership and import evidence.

Selected package subdirectories:

| Directory | Entries |
| `src/tensorlbm/ai/` | 39 |
| `src/tensorlbm/apps/` | 13 |
| `src/tensorlbm/backends/` | 7 |
| `src/tensorlbm/cases/` | 7 |
| `src/tensorlbm/core/` | 7 |
| `src/tensorlbm/ml/` | 11 |
| `src/tensorlbm/models/` | 3 |
| `src/tensorlbm/octree_boundary/` | 13 |
| `src/tensorlbm/phasefield/` | 10 |
| `src/tensorlbm/runtime/` | 2 |

## Confirmed structural risks

### 1. The solver package has a very wide root namespace

`src/tensorlbm/__init__.py` declares a lazy export table for **786 names**, spanning lattice primitives, collisions, boundaries, AMR, turbulence, marine cases, benchmark helpers and AI tooling. Lazy loading improves import cost but does not make the public contract small. A relocation may silently break imports even if internal tests pass.

There is also a curated `tensorlbm.api` and an experimental surface. Before moving modules, define which surface is stable and whether legacy root exports are compatibility-only.

### 2. Solver and application boundaries are porous at runtime

`app/backend/main.py` inserts the repository `src` path into `sys.path` at runtime, dynamically imports routers, and catches broad exceptions to skip unavailable routers. This makes startup tolerant but can hide broken route registration behind warnings. It also couples the application to checkout layout.

The application should depend on the installed `tensorlbm` package through a documented development/install flow. The solver library must not import app routers, frontend assets, web lifecycle code, or agent orchestration.

### 3. Job model and consumer contracts have drift signals

The platform CI run reported test calls using `Job(..., output_dir=...)`, while the current `Job.__init__` accepts `job_id, name, job_type, config` and derives `output_dir` internally. Another test double rejects `to_dict(include_result=...)` even though the production method supports that keyword. These are direct signals that test doubles and application contracts are not synchronized; each needs a main-branch comparison and an explicit contract decision.

### 4. Frontend ownership is unclear across tests and current tree

Multiple platform tests refer to `app/frontend/static/...` or `platform/frontend/static/...`, while the current tree contains `frontend-vue/src`. Tests also require `frontend-vue/dist/index.html`, which is a build output and is absent unless the workflow builds it. The fix should be a deliberate contract: source tests inspect Vue source; serving/build tests build the frontend explicitly; no test should accidentally depend on a generated artifact that CI never creates.

### 5. Numerical API names are inconsistent in test expectations

The D3Q19 module defines `equilibrium3d`, not `equilibrium`. Some platform tests import the latter. Do not add a blind alias until checking whether there is a documented compatibility expectation; otherwise update stale tests to use the canonical 3D API. The canonical symbol currently referenced by benchmark docs is `tensorlbm.d3q19.equilibrium3d`.

### 6. Repository root contains mixed-purpose files

At root, there are many one-off debug/verification/comparison scripts, generated figures and report files. They should be classified before relocation as:
- maintained developer tool;
- reproducible experiment/runner;
- benchmark evidence/report;
- generated output;
- obsolete but historically meaningful artifact.

Do not delete or mass-move these files based on filename prefixes alone.

## First failure triage queue

Observed CI run: [40173673100](https://github.com/cxs503/TensorLBM/actions/runs/40173673100).

| Priority | Cluster | Evidence | Safe next action |
|---|---|---|---|
| P0 | Baseline certainty | Python 3.11/3.12 solver jobs were still in progress at last status check | Fetch final job conclusions; do not call the baseline green |
| P1 | Frontend tests/paths | tests target missing `app/frontend` / `platform/frontend` files and unbuilt `frontend-vue/dist` | Separate Vue source-contract tests from build-and-serve integration tests |
| P1 | Job manager API drift | tests pass unsupported `output_dir` constructor arg; fake `to_dict` signature differs | Align tests and production API based on intended contract; add focused unit tests |
| P1 | Route/API contracts | live metrics 404s, expected status code mismatches, DoE tag absent | Verify route registration and endpoint contract before changing assertions |
| P1 | Test correctness | `torch` referenced without import in some test methods | Fix test import/fixture; keep physical assertions intact |
| P2 | D3Q19 naming | tests import `equilibrium`, library provides `equilibrium3d` | Decide compatibility policy, then fix the caller or provide a tested shim |
| P2 | Lint debt | Ruff reports 623 findings; 207 marked auto-fixable, 5 unsafe | Baseline by path/rule, then separate mechanical cleanup from architecture work |

The platform suite result in that run was **46 failed, 614 passed, 48 skipped, 8 errors**. The lint job failed with 623 Ruff findings. These are observed results from that run, not proof all failures originate from the architecture PR.

## Migration decision

**No source relocation is authorized by this audit snapshot.** The highest-value next code change is to repair and classify test/contract boundaries in small PRs while generating a proper static import graph. Core module moves come after public API mapping and baseline comparison.

## Commands for a full checkout / CI agent

Run from repository root:

```bash
python -m compileall -q src/tensorlbm app/backend
python -m pytest --collect-only -q tests
python -m pytest --collect-only -q app/tests
python -m ruff check src tests examples benchmarks --output-format=json > ruff-baseline.json
python -m ruff check app app/tests --output-format=json > ruff-platform-baseline.json
```

For the import graph, use an AST-based scanner that records:
- absolute and relative imports;
- imports within `tensorlbm`, `app.backend`, and frontend/build tooling;
- dynamic import sites (`__import__`, `importlib.import_module`, `find_spec`);
- unresolved imports separately from runtime-optional imports.

A static graph cannot prove runtime behavior; confirm critical edges with tests and actual import execution.

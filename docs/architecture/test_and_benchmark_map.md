# Test and Benchmark Map — Initial Inventory

## Current entry points

- Package test configuration: `pyproject.toml` sets `testpaths = ["tests"]` and adds `src` to `pythonpath`.
- Solver tests: `tests/`, including specialized folders such as `core/`, `backends/`, `runtime/`, `models/`, `ml/`, and `performance/`.
- Platform tests: `app/tests/`, including API, job manager, CAD, engineering workflow, orchestration, and UI contract tests.
- Benchmark assets: `benchmarks/pending/`, `benchmarks/verified/`, `benchmarks/archived/`, plus `docs/benchmarks/`, `runs/`, and `experiments/`.

## Important finding

The root pytest configuration names only `tests/`. The repository also has a substantial `app/tests/` suite. This does not prove the platform suite is missing from CI—workflow configuration may invoke it separately—but the CI commands must be checked and documented explicitly.

## Proposed test classification

| Category | What it proves | Typical acceptance evidence |
|---|---|---|
| Unit | Local numerical or interface behavior | Deterministic assertions on small inputs |
| Integration | Interactions across modules/backends | End-to-end module composition with controlled fixtures |
| Physics | Physical invariants or known solutions | Conservation, analytical solution, convergence trend, or trusted reference |
| Regression | A known defect stays fixed | Minimal reproducer fails before and passes after the fix |
| Benchmark acceptance | A named benchmark meets defined criteria | Versioned config, reference, metrics, tolerances, provenance, decision |
| Performance | Runtime/resource behavior on specified hardware | Warm-up policy, device, dimensions, timing and memory metrics |
| Platform | API, jobs, orchestration, UI contracts | Explicit app test command and integration fixtures |

## Benchmark lifecycle contract to implement

A status label should not be inferred from whether a runner exits with code zero.

- **pending**: definition or acceptance evidence is incomplete; no claim of verification.
- **verified**: all required metrics and tolerances pass against declared references/invariants; result is tied to a code revision and environment.
- **archived**: retained for historical traceability, with reason and superseding benchmark/version where applicable.

A benchmark manifest should contain at least:

- stable `benchmark_id` and schema version;
- problem/model definition and configuration path;
- runner path and invocation;
- reference source or analytical invariant;
- metrics, tolerances, and acceptance decision;
- code revision, environment/device, seed and relevant precision;
- raw result, derived figure, and report paths;
- status, date, and reviewer/automation provenance.

## Not yet verified

This inventory does **not** assert that the current CI omits app tests, that verified benchmarks are invalid, or that tests are duplicated. Those require workflow inspection, test collection, and benchmark-by-benchmark evidence review.

## Required follow-up

1. Inspect `.github/workflows/` and enumerate actual test commands and trigger conditions.
2. Compare test filenames/imported modules to identify ownership and likely duplication.
3. Inspect `benchmarks/STATUS.md` and representative entries in each lifecycle state.
4. Generate a manifest of benchmark IDs, source revisions, criteria, results, and missing fields.
5. Run fast solver and app test suites independently; run physics/GPU/performance suites only on suitable runners and label unavailable hardware checks as not run.


## Confirmed CI and documentation gap

The current `.github/workflows/ci.yml` installs `.[dev]` and invokes `pytest` without specifying `app/tests`; `pyproject.toml` sets `testpaths = ["tests"]`. The workflow's ruff command also omits `app` and `app/tests`. Therefore the visible standard CI workflow does not explicitly validate the platform test suite or lint the platform backend. This should be corrected only after checking whether another workflow is intentionally responsible for those checks.

Contributor-facing setup also refers to `platform/requirements.txt` and `platform/start.sh`, while the current tree contains `app/requirements.txt` and `app/start.sh`. Confirm and repair these stale paths as a small documentation fix; do not assume the deployment layout can be changed from docs alone.

## Suggested immediate low-risk PR

1. Correct contributor/developer workflow paths to match the current repository, after confirming the supported startup command.
2. Add a distinct CI job for `app/tests` using `app/requirements.txt` (or a documented minimal test extra), rather than mixing platform dependencies into the solver package extras.
3. Add a platform backend lint scope if the current codebase can pass it; otherwise create a bounded baseline and fix violations incrementally.
4. Keep numerical solver tests and physics benchmarks independent from platform test results so green checks retain clear meaning.

No tests have been run as part of this inventory. These findings are based on workflow and configuration contents only.

## First CI baseline — run 40173673100 (2026-10-25)

Observed on the architecture-governance branch; this is not a passing gate:

- `package-build`: passed (wheel build and metadata/install smoke check).
- `lint`: failed at `ruff check src tests examples benchmarks` with **623 findings**; formatter, mypy, and docs-consistency steps were skipped. This command does not lint `app/`. Because the architecture PR did not modify solver/test source in these scopes, treat this as existing lint debt unless a comparison proves otherwise; do not bulk-auto-fix it as part of a structural-only change.
- `platform-test`: **46 failed, 614 passed, 48 skipped, 8 errors**. Visible failure clusters include stale frontend/i18n paths and missing built `frontend-vue/dist/index.html`, API/route expectation mismatches, fixture/model signature drift, missing `torch` in tests, and imports expecting `tensorlbm.d3q19.equilibrium`.
- Python 3.11 and 3.12 solver jobs were still running at the last status check; final outcomes remain unconfirmed here.

### Triage policy

1. Separate failures into CI setup/fixture defects, stale tests or paths, real product regressions, and pre-existing lint debt.
2. For each platform failure, compare the same test on `main` before attributing it to a refactor.
3. Fix one coherent cluster at a time. Do not change solver numerics or relax physics/API assertions merely to obtain a green check.
4. Keep the platform suite visible while the baseline is unresolved. If a fast gate is needed, split by explicit markers and retain a separate full-suite job; do not silently exclude failing tests.
5. Enforce import-layer rules only after a reliable static import graph exists; dynamic imports must be reported as uncertain.


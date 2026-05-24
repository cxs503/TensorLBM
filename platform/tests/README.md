# TensorLBM Platform – Test Suite

End-to-end test suite for the TensorLBM B/S platform (FastAPI backend + SPA
frontend).  Built with FastAPI's `TestClient`, so no live server is required.

## Layout

| File | Scope |
|------|-------|
| `conftest.py`              | sys.path setup, `client` / `job_manager` / `waiter` fixtures |
| `test_platform_basic.py`   | `/api/health`, `/api/status`, root SPA, fallback, OpenAPI schema |
| `test_preprocess_api.py`   | polygon-mask, random-porosity-2d, voxelize-stl, unit converter |
| `test_cad_api.py`          | hull-types / preview / hull-mask / lbm-parameters / export-stl |
| `test_jobs_api.py`         | jobs list / get / logs / files / images / compare / delete / path-traversal |
| `test_job_manager.py`      | submit, cancel, diagnostics, log routing, failure handling |
| `test_postprocess_api.py`  | summary / velocity-profile / snapshot-analysis / csv (**slow**) |
| `test_solver_api.py`       | every `/api/solve/*` endpoint as a smoke run (**slow**) |
| `test_benchmarks_api.py`   | marine / multiphase / ghia / mlups / porous benchmarks (**slow**) |
| `test_websocket.py`        | `/ws` init broadcast (**opt-in**) |

## Running

Fast suite (default, ~10 s on CPU):

```bash
PYTHONPATH=src pytest platform/tests -q
```

Full suite (includes solver smoke runs and benchmark suites, ~5–20 min):

```bash
PLATFORM_SLOW_TESTS=1 PYTHONPATH=src pytest platform/tests -q
```

Enable WebSocket tests:

```bash
PLATFORM_WS_TESTS=1 PYTHONPATH=src pytest platform/tests/test_websocket.py -q
```

## Discovered & fixed defects

The test suite uncovered four real defects in the platform that were fixed
together with this change-set (see `docs/platform_test_report.md` for the
detailed analysis):

1. `routers/preprocess.py::polygon_mask` called `poly_to_mask_2d` with the
   wrong argument order and missing `device`.
2. `routers/preprocess.py::random_porosity_2d` passed `corr_length=` which
   does not exist (the library exposes `sigma=`).
3. `routers/preprocess.py::convert_units` instantiated `LBMUnitConverter`
   with obsolete keyword names (`phys_length`, `phys_velocity`, …).  The
   public API is `re, l_phys, u_phys, nu_phys, nx, u_lb`.
4. `routers/benchmarks.py::run_multiphase` passed `fast=` to
   `MultiphaseBenchmarkSuiteConfig` which does not declare it.  Replaced
   with explicit reduced sub-configs.
5. `job_manager._notify` crashed with “Event loop is closed” when the
   asyncio loop bound at startup had been torn down (e.g. between test
   runs).  Now it silently no-ops on closed loops.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Real DG-LBM solver** (`tensorlbm.dg_advection`, `tensorlbm.dg_band`): a genuine
  nodal Discontinuous-Galerkin Lattice Boltzmann hybrid.  Dimension-by-dimension
  P1-Lobatto DG advection (upwind flux, SSP-RK3, sub-cycled) with method-of-lines
  BGK collision, a packed DG-band topology, DG↔LBM P0 interface coupling
  (conservative face-trace write-back), and half-way bounce-back at solid walls.
  Validated: MMS O(Δx²)/O(Δx³) convergence, mass/momentum conservation, recovery of
  the DVBE shear viscosity, and stable end-to-end obstacle flow (2-D cylinder,
  3-D sphere, 3-D SUBOFF hull).  Enable on the SUBOFF runner with
  `DGLBMSuboffConfig(use_real_dg=True)` (the legacy gradient-correction path
  remains the default for backward compatibility).  Standalone runners:
  `examples/dg_lbm_cylinder_hybrid.py`, `examples/dg_lbm_sphere_hybrid.py`.
  Key tuned constants: 3-D RK sub-steps ≥ 16 (stiffer at low τ_dg); the band uses
  τ_dg = τ_lbm − ½ to match the exterior LBM viscosity.

### Added
- **Rate limiting**: opt-in per-IP sliding-window rate limiter in `app/backend/security.py`.
  Enabled by setting `TENSORLBM_RATE_LIMIT_REQUESTS=<N>` (requests per window, default 0 =
  disabled) and `TENSORLBM_RATE_LIMIT_WINDOW_S=<seconds>` (default 60).  Returns HTTP 429
  when the limit is exceeded.  Respects `X-Forwarded-For` for deployments behind a proxy.
  Rate limiting is enforced inside `authorize_request`, so it applies to all auth modes.
- **Job-list pagination**: `GET /api/jobs/` now accepts optional `limit`, `offset`, and
  `status` query parameters and returns a JSON envelope
  `{jobs: […], total: N, offset: K, limit: L}` instead of a bare array.  Clients that do not
  send these parameters receive the full list (backward-compatible default `limit=0`).
- **i18n completeness**: 6 new translation keys added to `en.json` / `zh.json`:
  `solve.validating`, `solve.scan_min_values`, `solve.scan_max_values`,
  `solve.scan_submitting`, `solve.scan_invalid_json`, `preprocess.no_materials`.
  Several hardcoded English strings in `app_solver.js`, `app_postprocess.js`, and
  `app_core.js` were replaced with `t()` calls so they now appear in Chinese when the
  language switcher is set to 中文.

### Added
- Marine benchmark suite now includes a **marine geometry library** case that validates
  ship hull families (Wigley/Series60/KCS) against analytical block coefficients and
  checks SUBOFF variant topology ordering (bare hull < sail < full appendage volume).
- **i18n / Chinese localisation**: The deployable web platform (`platform/`) now supports both
  **English** and **Simplified Chinese (简体中文)** with real-time in-browser switching.
  - Language switcher (`EN | 中文`) added to the global navigation bar.
  - Language preference is persisted to `localStorage` (`tensorlbm_lang`); first visit
    auto-detects from `navigator.language`.
  - Lightweight pure-frontend JSON dictionary approach:
    `platform/frontend/static/i18n/en.json`, `zh.json`, `platform/frontend/static/js/i18n.js`.
  - Terminology glossary at `platform/i18n/GLOSSARY.md`.
  - CI-friendly key-parity validator: `python platform/i18n/check_keys.py`.
  - New test module `platform/tests/test_i18n.py` covering JSON validity, key parity, and static file serving.
  - Added `README.zh-CN.md` (Simplified Chinese README) with mutual language links.

## [0.3.0] - 2026-05-24

### Added
- **Regularized BGK (RLBM)** collision operator for D2Q9 (`collide_rlbm`) and D3Q19 (`collide_rlbm3d`)
  following Latt & Chopard (2006). Projects the non-equilibrium populations onto the second-order
  Hermite polynomial subspace, filtering out ghost (non-hydrodynamic) modes and improving stability
  at low viscosity (τ → 0.5).
- **Rotating-cylinder (Magnus effect) runner** (`rotating_cylinder.py`):
  `RotatingCylinderConfig`, `run_rotating_cylinder`, plus the underlying helpers
  `rotating_wall_velocity` and `moving_wall_bounce_back` (Ladd 1994 moving-wall BC for D2Q9).
  CLI script `examples/rotating_cylinder.py` exposes a `--spin-ratio` parameter (α = ω R / u∞).
- **MLUPS benchmark extension**: `benchmarks/bench_mlups.py` now accepts `--collisions {bgk,mrt,trt,rlbm,all}`
  and compares throughput of all four collision operators on D2Q9 and D3Q19 grids.
- **Platform `/api/health` endpoint** — lightweight liveness probe independent of Torch / job state.
- **Platform `/api/jobs/compare` endpoint** — side-by-side metadata comparison for up to 10 completed
  jobs, plus a corresponding *Compare* tab in the frontend SPA.

## [0.2.0] - 2026-05-24

### Added
- Modern Python packaging via `pyproject.toml` with editable-install and dev extras support.
- Public package version metadata via `tensorlbm.__version__`.
- Structured logging helpers for long-running simulations.
- Checkpoint save/load utilities for resumable runs.
- Scientific VTK and HDF5 output helpers.
- Post-processing utilities for velocity profiles, pressure coefficient, and Q-criterion.
- Protocol-based extension interfaces for custom collision operators and boundary conditions.
- YAML/TOML configuration loading with environment-variable overrides; new `load_config_yaml` convenience function.
- Reproducibility metadata capture including git commit, Python version, and package versions.
- Mass-correction helpers for 2-D and 3-D solvers; automatic mass correction every `output_interval` steps in cylinder-flow and sphere-flow runners.
- Bouzidi interpolated bounce-back support for curved 2-D boundaries.
- Full D3Q27 lattice implementation with BGK/MRT collision and streaming helpers.
- Verification tests for Poiseuille flow and Taylor-Green vortex decay.
- Property-based tests for collision invariants and D3Q27 lattice properties.
- MLUPS benchmark scripts for 2-D and 3-D kernels.
- Progress bars (tqdm) for cylinder-flow, sphere-flow, and turbulent-channel runners.
- **Two-relaxation-time (TRT) collision** (`collide_trt`, `collide_trt3d`) for D2Q9 and D3Q19, with the Ginzburg magic-parameter default (Λ = 3/16) that eliminates Poiseuille wall-placement errors.
- **3D porous-media module** (`porous_media3d.py`): `make_random_sphere_medium`, `make_tube_array_medium_3d`, `PorousDrainageConfig3D`, `run_porous_drainage_3d`.
- **Immersed Boundary Method** (`ibm.py`): direct-forcing IBM with 2-point hat and 4-point cosine delta kernels, velocity interpolation, force spreading, and Guo body-force application.
- **Thermal LBM** (`thermal.py`): double-distribution-function model (D2Q9 momentum + D2Q5 temperature), equilibrium, BGK collision, streaming, macroscopic recovery, and Boussinesq buoyancy force.
- `CONTRIBUTING.md` with coding conventions, PR workflow, and guides for adding new solvers and benchmarks.
- Quantitative validation summary table in `README.md`.
- Ship CAD-to-flow workflow outputs for `run_ship_hull_flow`: CAD preview/statistics, optional STL export, wake-profile CSV, and post-processing summary JSON with symmetry and wake metrics.

### Changed
- Example runners now emit structured log messages instead of raw `print()` output.
- Runner metadata now includes reproducibility information.
- CI installs from the package metadata (`pip install -e ".[dev]"`), runs mypy, and reports test coverage.
- Runtime requirements now include `tqdm` for optional progress reporting.
- Ship-hull CLI and marine benchmark now exercise the full ship workflow from CAD modelling through LBM solve and quantitative post-processing checks.

## [0.1.0] - Initial release

### Added
- D2Q9 and D3Q19 lattice definitions with equilibrium and macroscopic reconstruction.
- 2-D and 3-D BGK/MRT collision operators and periodic streaming kernels.
- 2-D and 3-D boundary-condition helpers including Zou/He inlet/outlet variants.
- Example runners for cylinder, sphere, ship-hull, and water-entry simulations.
- Marine engineering extensions including Wigley hull geometry, obstacle diagnostics, and wave boundary conditions.
- Smagorinsky LES turbulence models and shared simulation utilities.
- Unit and smoke tests covering core solvers, runners, and marine features.

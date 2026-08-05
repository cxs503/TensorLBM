# SUBOFF Grid Convergence Study (R1)

## Purpose

This runner executes the production full-wet force window campaign at three or
more systematically refined grid levels for the SUBOFF bare hull (D3Q19 + MRT).
It collects the measured Ct candidate per level and computes relative-change
indicators, but **deliberately withholds any convergence or physical-validation
claim**.

The study is a **diagnostic only**: it shows how the measured Ct candidate
varies across grid resolutions, but does not assert that the solution has
converged or that the Ct values are physically validated.

## Module

`tensorlbm.suboff_grid_convergence_study`

### Public API

- `GridLevel` — one systematically refined grid level (level_id, nx, ny, nz,
  steps, capture_steps).
- `GridConvergenceStudyConfig` — study configuration (grid_levels, tau,
  inlet_velocity, hull_type, lattice, collision, cad_config).
- `run_suboff_grid_convergence_study(config, *, output_path=None)` — run the
  study and return a machine-readable convergence artifact.

## How It Works

1. For each grid level, the runner builds a SUBOFF bare-hull voxel mask using
   `tensorlbm.suboff_cad.build_suboff_mask` with the parametric DARPA
   SUBOFF-inspired geometry.
2. It creates a `GeometryAsset` and a `FullyWettedFlowConfig` with the fixed
   D3Q19 + MRT + single-phase incompressible composition, Zou-He channel
   inlet, and stationary bounce-back boundaries.
3. It runs `run_suboff_full_wet_force_window_campaign` to capture real
   post-stream/pre-bounce-back population snapshots and measure the link-wise
   momentum-exchange force on the body.
4. The Ct is normalized using the hull cross-sectional area
   `π · R²` (lattice units) as the reference area, with `ρ = 1.0` (lattice
   density) and `U = inlet_velocity` (lattice velocity).
5. Relative Ct changes between consecutive grid levels are computed as
   `|Ct_{i+1} − Ct_i| / |Ct_i|`.

## Artifact Schema

```
artifact_kind:    "suboff_grid_convergence_study"
schema:           "suboff-grid-convergence-study-r1"
status:           "diagnostic_only"
physical_validation: false
grid_levels:      [{level_id, nx, ny, nz, steps, capture_steps}, ...]
Ct_per_level:     [float, ...]
convergence_indicator:
  relative_ct_changes: [float, ...]   # N-1 entries for N levels
  ct_trend:            "decreasing" | "increasing" | "non_monotonic"
  max_relative_change: float
  convergence_claim:   "withheld"
  note:                str
per_level_results: [{level_id, grid_shape_zyx, Ct, force_time_series,
                     campaign_status, link_count, solid_cells, ...}, ...]
provenance:        {runner_api, model_identity, cad_source, ...}
provenance_hash:   sha256
```

## Constraints

- **Status is always `diagnostic_only`** — the study never claims convergence.
- **`physical_validation` is always `False`** — Ct values are lattice-unit
  diagnostics, not physical coefficients.
- **`convergence_claim` is always `"withheld"`** — relative changes are
  indicators only.
- The runner does not modify the solver hot path. It calls the existing
  `run_fully_wetted_flow` and `run_suboff_full_wet_force_window_campaign`
  production APIs.
- Only `hull_type="bare_hull"` is supported (AFF-1 equivalent).
- Only `lattice="D3Q19"` and `collision="MRT"` are supported.

## Default Grid Levels

| Level   | Grid (nx×ny×nz) | Hull Length (lu) | Hull Radius (lu) |
|---------|-----------------|-------------------|-------------------|
| coarse  | 48 × 24 × 24    | 28.8              | 1.68              |
| medium  | 64 × 32 × 32    | 38.4              | 2.24              |
| fine    | 96 × 48 × 48    | 57.6              | 3.36              |

The hull length is `0.6 × nx` and the radius is `r_over_l × length` where
`r_over_l = 1/(2 × 8.57) ≈ 0.0583` (DARPA SUBOFF L/D ≈ 8.57).

## Usage

```python
from tensorlbm.suboff_grid_convergence_study import (
    GridConvergenceStudyConfig,
    GridLevel,
    run_suboff_grid_convergence_study,
)

config = GridConvergenceStudyConfig(
    grid_levels=(
        GridLevel("coarse",  48, 24, 24, steps=4, capture_steps=(3, 4)),
        GridLevel("medium",  64, 32, 32, steps=4, capture_steps=(3, 4)),
        GridLevel("fine",    96, 48, 48, steps=4, capture_steps=(3, 4)),
    ),
)

artifact = run_suboff_grid_convergence_study(
    config,
    output_path="suboff_grid_convergence_artifact.json",
)
```

## Test Coverage

`tests/test_suboff_grid_convergence_study.py` — 14 tests covering:

- Config validation (minimum 3 levels, bare_hull only, D3Q19+MRT only,
  capture_steps validation).
- Study execution (3 grid levels, correct artifact structure).
- Per-level results (force time series, Ct, measured_candidate status,
  link_count, solid_cells).
- Ct consistency between `Ct_per_level` and `per_level_results`.
- Convergence indicator fields (relative_ct_changes, ct_trend,
  max_relative_change, convergence_claim="withheld").
- No convergence or validation claims.
- Determinism (identical results on re-run).
- Provenance (runner API, model identity, prohibition).
- JSON serializability.
- Artifact file output.

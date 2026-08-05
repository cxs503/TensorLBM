# SUBOFF Bare-Hull D3Q19+MRT Validation Runner

## Overview

This module (`tensorlbm.suboff_validation_runner`) runs a small-grid
SUBOFF bare-hull case with **D3Q19 MRT collision**, **bounce-back solid
boundary**, and **static wall** to verify the full
admission→run→force/Ct chain end-to-end.

It produces a `measured_candidate` evidence artifact — real force/Ct
observations from an actual solver loop, but with no physical validation,
convergence, or steady-state claim.

## What It Verifies

### 1. Wall-Function Admission Gate in Real Config

The runner calls the cold-path admission gate
(`wall_function_admission.require_wall_function_run`) **before any solver
execution**:

| Config | Gate Behaviour |
|--------|---------------|
| `use_wall_function=False` | **Skipped** — bounce-back only, no wall function |
| `use_wall_function=True` + D3Q19/MRT_SMAGORINSKY | **Admitted** at `IMPLEMENTATION_ONLY` |
| `use_wall_function=True` + D3Q27 | **Withheld** (`WITHHELD_UNVERIFIED_COMBINATION`) |
| `use_wall_function=True` + free_surface | **Withheld** (`WITHHELD_UNVERIFIED_COMBINATION`) |

The gate is a pure cold-path metadata check — it has no tensor arguments
and is never called per-cell or per-step.

### 2. Real Run Produces Force/Ct Time Series

The solver loop composes existing operators without modifying the hot path:

```
for each step:
    f = collide_mrt3d(f, tau)           # D3Q19 MRT collision
    f = stream3d(f)                      # D3Q19 pull-scheme streaming
    fx,fy,fz = compute_obstacle_forces_3d(f, solid)  # momentum exchange
    f = bounce_back_cells_3d(f, solid)  # static wall bounce-back
    f = zou_he_inlet_velocity_3d(f, u_in)
    f = zou_he_outlet_pressure_3d(f)
    f = bounce_back_cells_3d(f, wall_mask)
    record force/Ct
```

Per-step outputs:
- **Force time series**: `(step, fx, fy, fz)` in lattice force units
- **Ct time series**: `(step, ct, ct_fric, ct_pres)` where
  - `ct = fx / (0.5 * rho * U^2 * S)`
  - `ct_fric = (fx - pressure_drag) / dynamic_pressure`
  - `ct_pres = pressure_drag / dynamic_pressure`

### 3. measured_candidate Evidence Artifact

The evidence artifact has:

| Field | Value |
|-------|-------|
| `status` | `"measured_candidate"` |
| `physical_validation` | `false` |
| `steady_state` | `"diagnostic_withheld"` |
| `admission` | gate record (skipped/admitted/withheld) |
| `force_time_series` | per-step force samples |
| `ct_time_series` | per-step Ct samples |
| `runtime` | steps, finiteness checks, density range |
| `config` | solver configuration snapshot |

## Usage

```python
from tensorlbm.suboff_validation_runner import (
    SuboffValidationConfig,
    run_suboff_d3q19_mrt_validation,
)

# Bounce-back only (no wall function)
config = SuboffValidationConfig()  # 48×24×24, 20 steps
evidence = run_suboff_d3q19_mrt_validation(config)
evidence.write_artifact("docs/evidence/suboff-d3q19-mrt-bounceback-r1.json")

# Wall function admitted (D3Q19/MRT_SMAGORINSKY)
config = SuboffValidationConfig(use_wall_function=True)
evidence = run_suboff_d3q19_mrt_validation(config)
```

## Evidence Artifacts

- `docs/evidence/suboff-d3q19-mrt-bounceback-r1.json` — bounce-back run
- `docs/evidence/suboff-d3q19-mrt-wallfunction-admitted-r1.json` —
  wall-function-admitted run

## Design Constraints

- **No solver hot-path modification**: only existing operators are composed.
- **Cold-path admission only**: the gate is called once before the loop.
- **No physical validation claim**: `measured_candidate` is the highest
  status; `physical_validation=False` and `steady_state=diagnostic_withheld`.
- **Small grid for speed**: 48×24×24, 20 steps (~1.5s on CPU).

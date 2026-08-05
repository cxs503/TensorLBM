# SUBOFF Time Convergence Study (R1)

## Purpose

This runner executes the SUBOFF bare-hull D3Q19+MRT validation runner at four or
more different step counts on a **fixed grid** (48×24×24).  It collects the
measured Ct candidate per time level — the mean Ct over a capture window of the
final steps — and computes relative-change indicators, but **deliberately
withholds any convergence or physical-validation claim**.

The study is a **diagnostic only**: it shows how the measured Ct candidate
varies across time-step counts, but does not assert that the solution has
converged in time or that the Ct values are physically validated.

## Module

`tensorlbm.suboff_time_convergence_study`

### Public API

- `TimeLevel` — one time-step-count level (level_id, n_steps, capture_window).
- `TimeConvergenceStudyConfig` — study configuration (time_levels, nx, ny, nz,
  u_in, re, hull_length, device, lattice, collision, hull_type).
- `run_suboff_time_convergence_study(config, *, output_path=None)` — run the
  study and return a machine-readable convergence artifact.

## How It Works

1. For each time level, the runner builds a `SuboffValidationConfig` with the
   study's fixed grid (48×24×24) and physics parameters, sets `n_steps` to the
   level's step count, and calls
   `run_suboff_d3q19_mrt_validation` to run the real D3Q19+MRT+bounce-back
   solver loop.
2. The solver loop executes: MRT collision → streaming → momentum-exchange
   force measurement → bounce-back on solid → Zou-He inlet/outlet BCs →
   wall bounce-back → mass correction.
3. The measured Ct candidate for each level is the **mean Ct over the capture
   window** (the final `capture_window` steps of the run).  This represents
   the Ct at that level of time integration.
4. Relative Ct changes between consecutive time levels are computed as
   `|Ct_{i+1} − Ct_i| / |Ct_i|`.

## Artifact Schema

```
artifact_kind:    "suboff_time_convergence_study"
schema:           "suboff-time-convergence-study-r1"
status:           "diagnostic_only"
physical_validation: false
grid_shape:       {nx, ny, nz}              # fixed grid
time_levels:      [{level_id, n_steps, capture_window, capture_steps}, ...]
Ct_per_level:     [float, ...]              # mean Ct per time level
convergence_indicator:
  relative_ct_changes: [float, ...]         # N-1 entries for N levels
  ct_trend:            "decreasing" | "increasing" | "non_monotonic"
  max_relative_change: float
  convergence_claim:   "withheld"
  note:                str
per_level_results: [{level_id, n_steps, capture_window, capture_steps,
                     status, physical_validation, Ct,
                     force_time_series, ct_time_series,
                     wetted_area, dynamic_pressure, runtime, config}, ...]
provenance:        {runner_api, model_identity, grid_shape, force_method,
                    sample_phase, ct_aggregation, prohibition}
provenance_hash:   sha256
```

## Constraints

- **Status is always `diagnostic_only`** — the study never claims convergence.
- **`physical_validation` is always `False`** — Ct values are lattice-unit
  diagnostics, not physical coefficients.
- **`convergence_claim` is always `"withheld"`** — relative changes are
  indicators only.
- The runner does not modify the solver hot path. It calls the existing
  `run_suboff_d3q19_mrt_validation` production API.
- Only `hull_type="bare_hull"` is supported (AFF-1 equivalent).
- Only `lattice="D3Q19"` and `collision="MRT"` are supported.
- The grid is fixed across all time levels (default 48×24×24).

## Default Time Levels

| Level | Steps | Capture Window | Capture Steps |
|-------|-------|----------------|---------------|
| t10   | 10    | 3              | 8, 9, 10      |
| t20   | 20    | 5              | 16–20         |
| t40   | 40    | 10             | 31–40         |
| t80   | 80    | 20             | 61–80         |

## Usage

```python
from tensorlbm.suboff_time_convergence_study import (
    TimeConvergenceStudyConfig,
    TimeLevel,
    run_suboff_time_convergence_study,
)

config = TimeConvergenceStudyConfig(
    time_levels=(
        TimeLevel("t10", n_steps=10, capture_window=3),
        TimeLevel("t20", n_steps=20, capture_window=5),
        TimeLevel("t40", n_steps=40, capture_window=10),
        TimeLevel("t80", n_steps=80, capture_window=20),
    ),
)

artifact = run_suboff_time_convergence_study(
    config,
    output_path="suboff_time_convergence_artifact.json",
)
```

## Test Coverage

`tests/test_suboff_time_convergence_study.py` — 22 tests covering:

- Config validation (minimum 4 levels, bare_hull only, D3Q19+MRT only,
  n_steps/capture_window validation, duplicate level_id rejection).
- Study execution (4 time levels, correct artifact structure).
- Per-level results (force/Ct time series, measured_candidate status,
  runtime evidence, config snapshot).
- Ct consistency between `Ct_per_level` and `per_level_results`.
- Capture steps correctness.
- Convergence indicator fields (relative_ct_changes, ct_trend,
  max_relative_change, convergence_claim="withheld").
- No convergence or validation claims.
- Determinism (identical results on re-run).
- Provenance (runner API, model identity, prohibition).
- JSON serializability.
- Artifact file output.

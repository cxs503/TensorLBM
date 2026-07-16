# SUBOFF Domain Convergence Study (R1)

## Purpose

This runner fixes the hull length, grid resolution (dx = 1 lattice unit), and
step count, then varies the computational domain size across at least three
levels for the SUBOFF bare hull (D3Q19 + MRT).  Each level runs a real
D3Q19+MRT bounce-back simulation and produces a `measured_candidate` evidence
artifact with force/Ct time series.  The study collects the mean Ct per level
and computes relative-change indicators, but **deliberately withholds any
convergence or physical-validation claim**.

The study is a **diagnostic only**: it shows how the measured Ct candidate
varies across domain sizes (i.e., as blockage ratio decreases), but does not
assert that the solution has converged or that the Ct values are physically
validated.

This is distinct from the [grid convergence study](suboff_grid_convergence_study_r1.md),
which varies grid resolution (nx/ny/nz scale together, hull scales with grid).
The domain convergence study keeps the hull fixed and grows the domain around
it.

## Module

`tensorlbm.suboff_domain_convergence_study`

### Public API

- `DomainLevel` — one domain size level (level_id, nx, ny, nz).
- `DomainConvergenceStudyConfig` — study configuration (domain_levels,
  hull_length, n_steps, warmup, u_in, re, device, lattice, collision, hull_type).
- `run_suboff_domain_convergence_study(config, *, output_path=None)` — run the
  study and return a machine-readable convergence artifact.

## How It Works

1. For each domain level, the runner creates a `SuboffValidationConfig` with
   the level's domain size (nx, ny, nz) and the fixed hull_length, n_steps,
   u_in, and Re.
2. It calls `run_suboff_d3q19_mrt_validation` to execute a real
   D3Q19+MRT+bounce-back solver loop with Zou-He inlet/outlet boundaries and
   static wall bounce-back.
3. The Ct is extracted as the **mean over post-warmup steps** from the
   evidence's `ct_time_series`.  The Ct is normalized using the voxel wetted
   area as the reference area, with ρ = 1.0 (lattice density) and
   U = u_in (lattice velocity).
4. The blockage ratio is computed as `π · R² / (ny · nz)` where
   R = r_over_l · hull_length (DARPA SUBOFF L/D ≈ 8.57,
   r_over_l ≈ 0.0583).
5. Relative Ct changes between consecutive domain levels are computed as
   `|Ct_{i+1} − Ct_i| / |Ct_i|`.

## Artifact Schema

```
artifact_kind:    "suboff_domain_convergence_study"
schema:           "suboff-domain-convergence-study-r1"
status:           "diagnostic_only"
physical_validation: false
domain_levels:    [{level_id, nx, ny, nz, domain_length_lu, blockage_ratio}, ...]
Ct_per_level:     [float, ...]
convergence_indicator:
  relative_ct_changes: [float, ...]   # N-1 entries for N levels
  ct_trend:            "decreasing" | "increasing" | "non_monotonic"
  max_relative_change: float
  convergence_claim:   "withheld"
  note:                str
per_level_results: [{level_id, nx, ny, nz, Ct, ct_mean, ct_time_series,
                     force_time_series, evidence_status, physical_validation,
                     wetted_area, dynamic_pressure, blockage_ratio,
                     hull_length_lu, hull_radius_lu, domain_length_lu,
                     runtime, admission, config, ...}, ...]
provenance:        {runner_api, model_identity, cad_source, force_method,
                    ct_extraction, reference_area_mode, fixed_parameters,
                    prohibition}
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
- Hull length, step count, u_in, and Re are fixed across all domain levels.
- Domain levels must have strictly increasing nx values.

## Default Domain Levels

| Level      | Domain (nx×ny×nz) | Hull Length (lu) | Hull Radius (lu) | Blockage Ratio |
|------------|-------------------|-------------------|-------------------|----------------|
| domain_48  | 48 × 24 × 24      | 24.0              | 1.40              | 0.0107         |
| domain_64  | 64 × 32 × 32      | 24.0              | 1.40              | 0.0060         |
| domain_80  | 80 × 40 × 40      | 24.0              | 1.40              | 0.0039         |

The hull length is fixed at 24.0 lattice units across all levels.  The hull
radius is `r_over_l × hull_length` where `r_over_l = 1/(2 × 8.57) ≈ 0.0583`
(DARPA SUBOFF L/D ≈ 8.57).  The blockage ratio decreases as the domain grows,
reducing wall proximity effects on the measured Ct.

## Measured Ct Convergence Trend

The produced evidence artifact shows a **decreasing** Ct trend across domain
levels:

| Level      | Blockage Ratio | Ct      | Relative Change |
|------------|----------------|---------|-----------------|
| domain_48  | 0.0107         | 0.4900  | —               |
| domain_64  | 0.0060         | 0.4852  | 0.97%           |
| domain_80  | 0.0039         | 0.4819  | 0.69%           |

The relative Ct changes decrease between consecutive levels (0.97% → 0.69%),
suggesting the Ct is approaching an asymptotic value as the domain grows.
However, **no convergence claim is made** — this is diagnostic evidence only.

## Usage

```python
from tensorlbm.suboff_domain_convergence_study import (
    DomainConvergenceStudyConfig,
    DomainLevel,
    run_suboff_domain_convergence_study,
)

config = DomainConvergenceStudyConfig(
    domain_levels=(
        DomainLevel("domain_48", 48, 24, 24),
        DomainLevel("domain_64", 64, 32, 32),
        DomainLevel("domain_80", 80, 40, 40),
    ),
    hull_length=24.0,
    n_steps=20,
    warmup=5,
    u_in=0.06,
    re=200.0,
)

artifact = run_suboff_domain_convergence_study(
    config,
    output_path="suboff_domain_convergence_artifact.json",
)
```

## Test Coverage

`tests/test_suboff_domain_convergence_study.py` — 21 tests covering:

- Config validation (minimum 3 levels, bare_hull only, D3Q19+MRT only,
  positive hull_length/n_steps, domain level dimension validation).
- Study execution (3 domain levels, correct artifact structure).
- Fixed hull length across all levels.
- Varying domain lengths (strictly increasing).
- Decreasing blockage ratio as domain grows.
- Per-level results (force/Ct time series, measured_candidate status,
  runtime finiteness, blockage_ratio).
- Ct consistency between `Ct_per_level` and `per_level_results`.
- Convergence indicator fields (relative_ct_changes, ct_trend,
  max_relative_change, convergence_claim="withheld").
- No convergence or validation claims.
- Determinism (identical results on re-run).
- Provenance (runner API, model identity, fixed parameters, prohibition).
- JSON serializability.
- Artifact file output.

## Evidence Artifact

The produced evidence artifact is at:
`evidence/suboff_domain_convergence_artifact.json`

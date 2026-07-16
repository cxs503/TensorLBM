# Wall-function capability / compatibility / evidence contract R1

## Purpose and boundary

`tensorlbm.wall_function_admission.require_wall_function_run` is the **single
public cold-path admission boundary** for wall-function-enabled run
configurations.  It constructs the fully named request and calls
`require_wall_function` before a runner creates fields or enters its time loop.
It does not change any numerical wall operator and does not dispatch a solver.
A tuple is admitted only when every one of lattice, physics, collision,
geometry, and backend is listed.  Every other tuple raises
`WallFunctionCompatibilityError(WITHHELD_UNVERIFIED_COMBINATION)`.

The legacy helpers in `wall_model` remain directly importable solely for
reproducibility of existing internal paths; that direct import is not a public
general-feature admission API.  New callers must use the config/run boundary.

An admitted tuple is still constrained by its `ValidationLevel`.  A caller that
requires `NUMERICAL_REGRESSION` or `PHYSICAL_VALIDATION` is rejected when the
record is only `IMPLEMENTATION_ONLY`.

## Audit matrix

| Capability | Existing entry point | Admitted tuple | Evidence level | Explicitly withheld |
|---|---|---|---|---|
| FMM-like distance | `wall_model.compute_wall_distance_fmm` | `MASK_2D`/`MASK_3D`; mask-only; no collision; static voxel mask; Torch | implementation only | physical Euclidean-distance claim; lattice/physics/collision coupling; AMR |
| Log/Reichardt body-force wall function | `wall_model.wall_function_3d` | D3Q19; incompressible single phase; MRT+Smagorinsky; static voxel solid; Torch | implementation only | D3Q27; free surface/phase field; BGK/other collision; moving geometry; AMR; non-Torch backend |
| Log-law slip + moving bounce-back | `wall_model.apply_wall_model_bounce_back` | D3Q19; incompressible single phase; BGK or MRT; static/moving voxel solid; Torch | implementation only | D3Q27; multiphase/free surface; AMR; other backend; physical accuracy |
| Rough slip + moving bounce-back | `roughness.apply_rough_wall_bounce_back` | same D3Q19 mask path as above | implementation only | all unlisted combinations and physical validation |

The `wall_function_3d(..., wall_law="reichardt")` branch is an implementation
option inside the D3Q19 entry point; it has no distinct checked-in validation
and must not be reported as separately validated.

## Evidence audit

The source docstring and the SUBOFF runner repeat “AFF-8 Re=2M, Ct 0.0040 vs
0.004, <1%”.  This checkout contains no focused wall-function test and no
checked-in result/data artifact that reproduces or binds that assertion.
Existing `tests/test_d3q27_moving_wall_*` validate D3Q27 *linkwise* helper
routines, not the D3Q19 cell-mask wall-model wrapper.  Existing roughness tests
only check import/correction properties.  Consequently none of these claims is
classified as numerical regression or physical validation.

## Hot-path audit

No capability here is advertised as a hot-path-general feature:

* `compute_wall_distance_fmm` allocates `dist`, clones it every iteration, pads,
  stacks six neighbours, and calls `.item()` for convergence.
* `wall_function_3d` invokes `bool(turb.any())` plus scalar `.item()` reductions
  for drag; it also constructs masks/rolls and applies a body-force operator.
* slip/roughness paths use `any()` branch decisions and masked indexing.

These are therefore not claimed GPU-asynchronous, allocation-free, generic, or
AMR-compatible.  The contract deliberately remains separate from numerical
wall code so it cannot alter production results.

## Public runner integration

The admission boundary is exercised by the wall-enabled public configurations:

* `SuboffResistanceBenchmarkConfig(use_wall_model=True)` submits its known
  D3Q19/static-voxel/MRT+Smagorinsky/Torch wrapper tuple.  Turning on its AMR
  mode changes the geometry request to an unlisted AMR label and is rejected
  before the benchmark loop.
* `DGLBMConfig` wall flags and the legacy DG wrapper flag are rejected because
  their public config does not contain complete auditable collision/geometry
  dimensions.  Their legacy direct operators are intentionally not generalized.
* `DGLBMSuboffConfig(use_wall_function=True)` is the one explicit D3Q19
  MRT+Smagorinsky/static-voxel log-law request; it is admitted only at
  implementation level.  Its `use_wall_model` hybrid wrapper request remains
  withheld because the DG-band configuration has no audited tuple.

Free-surface and AMR are represented explicitly at the public boundary and map
to unlisted labels; D3Q27 likewise fails before dispatch.  No per-cell or
per-step admission check was added to the numerical hot paths.

## Verification

`tests/test_wall_function_contract.py` verifies an exact admitted D3Q19 tuple,
that it remains implementation-only, that D3Q27/free-surface requests fail
closed, and that a higher evidence requirement fails closed.
`tests/test_wall_function_admission_integration.py` verifies the actual public
run/config boundary, including D3Q27, free-surface, AMR, incomplete DG wrapper
requests, and the allowed D3Q19 log-law tuple.

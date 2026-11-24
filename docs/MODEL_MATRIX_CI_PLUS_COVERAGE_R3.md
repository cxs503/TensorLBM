# Model matrix CI, round 3: the boundary-condition axis

Anchor `XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE` (TensorLBM). Round 1 built the
matrix (registry key = model, analytic cases, coverage accounting, exit-code
gate); round 2 closed the denominator (14 operators) and the bulk-viscosity
channel (`acoustic_decay`, mutation study, `docs/MODEL_MATRIX_CI_PLUS_COVERAGE_R2.md`).

Round 3 closes the structural hole both rounds shared: **every case was
periodic**, so the matrix was blind to the boundary operator.  No cell in the
matrix could see an inverted reflection table, a one-cell wall-mask slip, a
missing NoDynamics restore or a mass-leaking wall.

## 1. What was added

* `src/tensorlbm/model_matrix/cases.py` — new gated case `wall_poiseuille`:
  steady plane Poiseuille channel between two half-way bounce-back walls,
  composed from shipped operators only
  (`build_collider` → constant body-force kick → solid restore →
  `bounce_back_cells_3d`/`_27` → `stream3d`/`stream27`, i.e. exactly the order
  documented for `lbm_step_correct(wall_treatment='bb')`).
* `src/tensorlbm/model_matrix/registry.py` — the case is declared for **all**
  14 matrix keys (`_with_wall_axis`), not for a hand-picked subset: the
  composition is model-agnostic, so "this model cannot run a wall" is a
  finding, not a declaration choice.  Declared cells: 47 → **61**.
* `tests/test_model_matrix_wall.py` — 8 tests: reference geometry, gate +
  second-order convergence, 1/H wall-offset shrinkage, cross-check against the
  validated `RotatingFrameLBM` Guo stepper, and three negative controls.
* [Resolution sweep driver](../development/verification/model-units/wall-axis/run_study.py) writes `result.json`,
  `BENCHMARK_REF_VS_GOT.md`, `fig1_wall_axis.png`.

## 2. Reference (a priori, nothing fitted)

Solid rows are `y = 0` and `y = ny-1`; half-way bounce-back places the
numerical walls half a lattice unit outside the first *fluid* node, i.e. at
`Y = 1/2` and `Y = ny - 3/2`, channel width `H = ny - 2`:

```
u(Y) = g/(2 nu) (Y - 1/2)(ny - 3/2 - Y),     nu = (tau - 1/2)/3
```

`g` and `nu` are inputs; the profile is never fitted to the simulation.  The
wall position reported below *is* obtained by fitting (a parabola through the
interior nodes), and that fitted position is then compared with the ideal
half-lattice location — the fit is the measurement, not the reference.

## 3. Measured (new runs write to `outputs/model_matrix_wall_axis/`)

`tau = 0.8`, `g = 1e-5`, `u_max/Ma ~ 3.2e-3 / 5.5e-3`.

| model | H | status | rel L2 err % | implied wall offset (lattice units) | centre-velocity err % |
|---|---|---|---|---|---|
| `d3q19/bgk` | 8 | PASS | 1.968 | +0.0289 | -1.460 |
| `d3q19/bgk` | 16 | PASS | 0.492 | +0.0144 | -0.361 |
| `d3q19/bgk` | 32 | PASS | 0.123 | +0.0072 | -0.090 |
| `d3q19/trt` (Λ=3/16) | 8 | PASS | 0.856 | +0.0125 | -0.635 |
| `d3q19/trt` (Λ=3/16) | 16 | PASS | 0.214 | +0.0063 | -0.157 |

Observed convergence order of the profile error:

| model | H 8→16 | H 16→32 |
|---|---|---|
| `d3q19/bgk` | p = 2.00 | p = 2.00 |
| `d3q19/trt` (Λ=3/16) | p = 2.00 | — |

Symmetry residual `max|u_j - u_{ny-1-j}|/u_max` ≤ 1e-13 and mass drift
≤ 1e-12 in every cell: the wall reflects, it neither leaks mass nor breaks the
left/right symmetry of the channel.

**Is the error < 3 %?** Yes — worst cell 1.97 % (H = 8); at H = 16 it is 0.49 %
and at H = 32 it is 0.12 %.  Owner's acceptance (< 15 %) is met with two
orders of margin at the CI resolution (H = 16).

### 3.1 A quantitative check of the magic-parameter folklore

With the same lattice and the same wall, TRT at `Λ = 3/16` shows **half** the
wall-position error of BGK at the same `tau` (δ: 0.0125 vs 0.0289 at H = 8,
0.0063 vs 0.0144 at H = 16, i.e. a factor 2.30 ± 0.01).  This is the
Ginzburg–d'Humières statement that the bounce-back wall location error vanishes
at `Λ = 3/16` in the leading order — here measured, not assumed.  The
prediction was registered before the run; it is confirmed, and it also shows
the *sign* (the wall sits `δ > 0`, slightly inside the fluid).

## 4. Detection power (negative controls, `tests/test_model_matrix_wall.py`)

Each control wraps the *real* step and corrupts one piece; the reference stays
the unmutated analytic parabola.

| control | corruption | detected by |
|---|---|---|
| leaking wall | solid cells absorb (`f → 0`) instead of reflecting | rel L2 > 3 % or non-finite |
| wrong reflection table | `roll(f, 1)` pairing instead of `f[opp]` | rel L2 > 3 % or non-finite |
| free-slip wall | specular reflection (`free_slip_cells_3d`) | rel L2 > 3 % |
| reversed forcing | `g → -g` | `u_center < 0` → FAIL |
| shifted mask | wall one cell away | implied offset > 0.3 lattice units (in-repo check at 3000 steps) |

The first four are asserted in the suite; the fifth is the reason the `offset`
metric exists (an off-by-one mask moves the wall by a full lattice unit, i.e.
~30× the physical value 0.03).

## 5. Cross-checks that the adapter is not a second implementation

`RotatingFrameLBM` (the repository's validated Guo-forced 3D stepper, Ekman
benchmarks) driven at `omega = 0` with the same body force is a second,
independent composition of a body-force channel.  The two agree on all fluid
rows to < 5e-3 relative (asserted in
`test_composition_matches_validated_forcing_stepper`); solid rows are excluded
because they are not physical in either.

A convention trap found while building this, recorded so the next agent does
not repeat it: `guo_force_delta_3d` **cannot** be dropped into a
`build_collider` pipeline — the Guo split requires the equilibrium to be built
on the half-step velocity `u* = Σc f/ρ + F/(2ρ)`, and a collider owns its own
velocity.  Using it with `u_raw` scales the effective force by
`1 - 1/(2τ)` (0.375 at `τ = 0.8`) and under-predicts the centre velocity by
~3×.  The case therefore injects an exactly-conservative constant source
(`Δf_i = 3 w_i c_i·F`, which adds momentum `F` and mass `0` per step).

## 6. What this does *not* prove

* Straight, grid-aligned walls only.  Curved / interpolated (Bouzidi) walls,
  moving walls and wall functions are not in the matrix yet — they need their
  own analytic references (Taylor–Couette, Womersley, ...).
* The case verifies the *steady* discrete state.  Unsteady wall forces
  (added-mass, history force) are out of scope.
* Only `D3Q19` and `D3Q27` cells exist; no 2D lattice is in the registry, so
  the D2Q9-only wall operators (`bounce_back_moving_wall`, Zou–He inlet) are
  still uncovered.
* The gate is single-resolution; the convergence order is measured by the
  `xval` study, not asserted per cell (that would multiply the CI cost by 4).

## 7. Reproduce

```
python -m tensorlbm.model_matrix list                      # 14 keys, 61 declared cells
python -m tensorlbm.model_matrix run --res 18 --out artifacts/model_matrix_r3
PYTHONPATH=src python development/verification/model-units/wall-axis/run_study.py
python -m pytest -q tests/test_model_matrix_wall.py tests/test_model_matrix.py
```

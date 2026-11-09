# P106W1-N04 (extension): the mixed bounce-back / specular wall on D3Q27

**Scope.** `docs/P106W1_N04_SLIP_WALL_3D.md` introduced the mixed (partial-slip)
wall on the full D3Q19 lattice and left `wall_partial_slip / D3Q27` recorded as
`NO_IMPLEMENTATION` in `tensorlbm/boundary/boundary_capability_contract.py`.
This note closes that cell: the D3Q19 implementation was generalised so the
velocity set is an *argument* (`Lattice`, `LATTICE_19`, `LATTICE_27`) instead of
a module constant, and the D3Q27 lift is validated with the same criteria that
were applied to D3Q19 — plus one criterion D3Q19 did not need (see §5, item 4).

**Status of the parent PR.** The deliverables listed for the re-opened PR
(`docs/P106W1_N04_SLIP_WALL_3D.md`, `partial_slip_wall3d.py`,
`boundary_capability_contract.py` entries, `tests/test_partial_slip_wall3d.py`)
are already on `main` (landed by merge #1482 / m7910) and pass. No second copy
of them is added here; this PR only adds the missing D3Q27 half.

---

## 1. Case and physics

Steady plane Poiseuille flow between two mixed-slip walls, driven by a constant
body force density `rho*g` along `x`, periodic in `x` and `z`.

| parameter | value | note |
|---|---|---|
| lattice | D3Q27 (and D3Q19 for cross-checks) | 27-velocity set, corner directions included |
| collision | BGK + Guo forcing | `src/tensorlbm/boundary/partial_slip_wall3d.py` |
| `tau` | 0.8 | `nu = (tau - 1/2)/3 = 0.1000000000` lu²/step |
| `g` | 1e-05 | body force density |
| `ny` | 17, 33 | fluid rows `1..ny-2`; wall planes at `y = 1/2`, `y = ny-3/2` |
| `H` | `(ny-2)/2` = 7.5, 15.5 | half height (lattice units) |
| `nx, nz` | 1, and 8×8 for the collapse run | periodic |
| `sigma` | 0, 0.25, 0.5, 0.75 | mixed weight (`0` = bounce-back, `1` = specular) |
| convergence | `max|Δu|/max|u| < 1e-10`, up to 80 000 steps | `steps`, `residual` in the JSON |

The wall itself is unchanged in form. For every direction that streams from the
wall into the fluid the post-collision population is replaced by

    f_q = (1 - sigma) f_opp(q) + sigma f_mirror(q),    mirror: (cx, cy, cz) -> (cx, -cy, cz)

The only D3Q27-specific ingredients are the tables: weights `W27`, the opposite
permutation `OPP27` and the specular mirror `MIRROR_Y_27`, all *derived* from
`tensorlbm/d3q27.py` rather than transcribed. What D3Q27 adds physically over
D3Q19 is the eight corner directions `(±1, ±1, ±1)`: they leave the `y` column
diagonally in `x` and `z` as well, so they are the directions most able to lock a
transverse pattern into the field. That is why §5 item 4 exists.

## 2. What was done

1. `partial_slip_wall3d.py`: the module-level D3Q19 tables were replaced by a
   `Lattice` descriptor (`name`, `C`, `W`, `OPP`, `MIRROR`, `Q`, broadcast-shaped
   `cx/cy/cz/w`, wall-inward masks). The kernels `_equilibrium`,
   `_macroscopic`, `_collide_bgk_guo`, the wall row and the channel solver now
   take the descriptor. `mixed_wall_row_3d` / `channel_poiseuille_3d` /
   `C19` / `W19` / `OPP19` / `MIRROR_Y` keep their signatures and values, so
   the existing D3Q19 tests are a regression pin for the refactor.
2. `mixed_wall_row_lattice` / `channel_poiseuille_lattice` are the generic entry
   points; `LATTICES = {"D3Q19", ..., "D3Q27": LATTICE_27}`.
3. `init_wave` was added to the solver: a single `x`/`z` wave (relative
   amplitude) in the initial streamwise velocity. It cannot survive an `x`/`z`-
   uniform steady state, so a run started from it proves that plane uniformity
   is *reached*, not merely preserved by a symmetric initial condition.
4. `Channel3DResult.lattice` records which velocity set produced a field.
5. Tests: `tests/test_partial_slip_wall3dq27.py` (14 cases). Evidence run:
   `artifacts/P106W1-N04/run_d3q27_slip_wall.py` →
   `artifacts/P106W1-N04/d3q27_slip_wall.json` + two figures. The contract cell
   `wall_partial_slip / D3Q27` was moved from `NO_IMPLEMENTATION` to
   `PHYSICS_VALIDATED` with these numbers as evidence.

## 3. Results

### 3.1 Slip length and the `b(sigma)` law (D3Q27 and D3Q19, two resolutions)

| lattice | ny | H | sigma | steps | u_max | b_model | b_bulk | b_quadratic |
|---|---|---|---|---|---|---|---|---|
| D3Q19 | 17 | 7.5 | 0.00 | 4010 | 0.002806 | -0.008667 | -0.008667 | -0.008667 |
| D3Q19 | 17 | 7.5 | 0.25 | 4112 | 0.002881 | 0.091333 | 0.091333 | 0.091333 |
| D3Q19 | 17 | 7.5 | 0.50 | 4319 | 0.003031 | 0.291333 | 0.291333 | 0.291333 |
| D3Q19 | 17 | 7.5 | 0.75 | 4959 | 0.003481 | 0.891333 | 0.891333 | 0.891333 |
| D3Q19 | 33 | 15.5 | 0.00 | 15743 | 0.012006 | -0.004194 | -0.004194 | -0.004193 |
| D3Q19 | 33 | 15.5 | 0.25 | 15934 | 0.012161 | 0.095806 | 0.095806 | 0.095807 |
| D3Q19 | 33 | 15.5 | 0.50 | 16320 | 0.012471 | 0.295806 | 0.295806 | 0.295807 |
| D3Q19 | 33 | 15.5 | 0.75 | 17498 | 0.013401 | 0.895806 | 0.895806 | 0.895806 |
| D3Q27 | 17 | 7.5 | 0.00 | 4010 | 0.002806 | -0.008667 | -0.008667 | -0.008667 |
| D3Q27 | 17 | 7.5 | 0.25 | 4112 | 0.002881 | 0.091333 | 0.091333 | 0.091333 |
| D3Q27 | 17 | 7.5 | 0.50 | 4319 | 0.003031 | 0.291333 | 0.291333 | 0.291333 |
| D3Q27 | 17 | 7.5 | 0.75 | 4959 | 0.003481 | 0.891333 | 0.891333 | 0.891333 |
| D3Q27 | 33 | 15.5 | 0.00 | 15743 | 0.012006 | -0.004194 | -0.004194 | -0.004193 |
| D3Q27 | 33 | 15.5 | 0.25 | 15934 | 0.012161 | 0.095806 | 0.095806 | 0.095807 |
| D3Q27 | 33 | 15.5 | 0.50 | 16320 | 0.012471 | 0.295806 | 0.295806 | 0.295807 |
| D3Q27 | 33 | 15.5 | 0.75 | 17498 | 0.013401 | 0.895806 | 0.895806 | 0.895806 |

Law fit `b = a sigma/(1-sigma) + c` on the four `sigma` points:

| fit | a | c | c*H | R^2 | monotone |
|---|---|---|---|---|---|
| D3Q19_ny17 | 0.29999999 | -0.00866670 | -0.0650002 | 1.000000000000 | True |
| D3Q19_ny33 | 0.29999996 | -0.00419384 | -0.0650045 | 1.000000000000 | True |
| D3Q27_ny17 | 0.29999999 | -0.00866670 | -0.0650002 | 1.000000000000 | True |
| D3Q27_ny33 | 0.29999996 | -0.00419384 | -0.0650045 | 1.000000000000 | True |

`c*H` is the resolution-independent lattice defect of the profile-fit route; it
is *not* a physical slip length. Both velocity sets reproduce
`a = 0.3` and `c*H = -0.065` to round-off in `a` and to ~1e-12 in `c*H`, i.e.
the mixed wall's slip law is a property of the operator, not of the velocity
set (measured difference: `|a_27 - a_19|/a = 3.54e-13`,
`|cH_27 - cH_19| = 1.30e-12` at `ny=17`).

### 3.2 Accuracy and reduction checks

| check | value |
|---|---|
| collapse_vs_1d_reduction.rel_profile_deviation | 5.79e-14 |
| collapse_vs_1d_reduction.plane_uniformity | 0.00e+00 |
| cross_lattice_profile_dev | 4.08e-14 |
| sigma0_vs_analytic_noslip_dev | 0.231% |
| sigma0_slip_length_over_H | 1.16e-03 |
| wave_start.profile_dev_vs_uniform_run | 3.03e-14 |
| wave_start.plane_spread | 3.66e-14 |
| wave_start.steps | 4319 |
| closed_loop_inversion_rel_error | 0.00e+00 |
| quadratic_fit_vs_model.max_abs_error_lu | 4.54e-07 |
| quadratic_fit_vs_model.max_rel_error | 9.73e-05 |

`collapse_vs_1d_reduction` answers the question the parent note asked of D3Q19:
does the cheap 1D column reproduce the full field? For D3Q27 it does to
5.79e-14 relative. `cross_lattice_profile_dev` compares D3Q27 with D3Q19 on
the same flow: 4.08e-14. `sigma0_vs_analytic_noslip_dev` is the negative
control, i.e. the residual of the D3Q27 result against the analytic no-slip
parabola — the same 0.231% that D3Q19 shows, which is the expected
second-order boundary error of a half-way bounce-back wall.

### 3.3 Figures

* `artifacts/P106W1-N04/d3q27_slip_wall_profiles.png` — measured D3Q27 profiles
  against the analytic slip-Poiseuille parabola (`b` taken from the same field,
  so this is a self-consistency check, not an independent benchmark) and the
  pointwise deviation on a log scale with the 3% line.
* `artifacts/P106W1-N04/d3q27_slip_wall_law.png` — `b(sigma)` for both velocity
  sets and both resolutions against the fitted law.

## 4. Notes and limitations

1. **The law is not new physics.** `a = 0.3` was already measured on D3Q19; the
   D3Q27 result is a *consistency* result (velocity-set independence), which is
   exactly what the contract needs before claiming the D3Q27 cell.
2. **Self-consistency is not validation against experiment.** Both the profile
   figure and the closed-loop inversion use `b` measured from the same field.
   The only absolute reference here is the analytic no-slip parabola at
   `sigma = 0` (0.231%) and the analytic slip-Poiseuille family, which is the
   same continuum solution the wall operator is built to reproduce. No
   experimental or high-fidelity reference is claimed in this note.
3. **`sigma = 1` has no steady state** (plug flow with secular momentum input);
   it is recorded as a signature, not as a converged result.
4. BGK only, single phase, `float64`, stationary tangential wall, one wall normal
   (`e_y`). Rotated walls, MRT/TRT, turbulence and multiphase are out of scope.
5. Cost: the D3Q27 solver is a NumPy reference implementation (27-entry Python
   streaming loop), ~1.3 s / 1000 steps at `ny=17`, `1x1` plane. It exists to
   *validate* the operator and the law, not to be the production solver.
6. The `Lattice` refactor touches a file that other work may also be editing
   (`partial_slip_wall3d.py`). All previously existing public names are kept as
   thin wrappers, and the whole D3Q19 test file is unchanged and green.

## 5. Error judgement (is it < 3%?)

| quantity | reference | measured | < 3%? |
|---|---|---|---|
| D3Q27 `sigma=0` profile vs analytic no-slip parabola | analytic | 0.231% | yes (0.231%) |
| D3Q27 vs D3Q19 profile, `sigma=0.5` | D3Q19 D3Q27 cross-check | 4.08e-14 | yes |
| D3Q27 3D field vs 1D-column reduction, `sigma=0.5` | 1D column | 5.79e-14 | yes |
| `b` law slope `a` vs the D3Q19 value 0.29999999 | D3Q19 | rel. diff 3.54e-13 | yes |
| `b` law defect `c*H` vs the D3Q19 value -0.065000 | D3Q19 | abs. diff 1.30e-12 | yes |
| quadratic-fit `b` vs curvature-free `b`, same fields | model estimator | 4.54e-07 lu (9.73e-05) | yes |
| closed-loop inversion of an analytic slip profile | the `b` used to build it | 0.00e+00 | yes (consistency only) |
| plane uniformity after a 5% `x`/`z` wave start | uniform run | 3.03e-14 | yes |

Every entry above is produced by
`artifacts/P106W1-N04/run_d3q27_slip_wall.py`; the values in the tables are
injected from `d3q27_slip_wall.json`, not transcribed by hand.

## 6. Reproduce

```bash
PYTHONPATH=src python artifacts/P106W1-N04/run_d3q27_slip_wall.py   # ~4 min, NumPy
PYTHONPATH=src python -m pytest tests/test_partial_slip_wall3dq27.py -q
PYTHONPATH=src python -m pytest tests/test_partial_slip_wall3d.py tests/test_boundary_capability_contract.py -q
```
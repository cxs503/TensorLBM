# `tensorlbm.conjugate_thermal` — DDF thermal LBM and conjugate heat transfer

Module added by mission **IN28-N03** (共轭传热验证). It provides a small,
self-contained, testable implementation of the double-distribution-function
(DDF) thermal lattice-Boltzmann model and the conjugate heat-transfer (CHT)
coupling used to validate it against literature benchmarks.

```
src/tensorlbm/conjugate_thermal/
├── ddf2d.py        D2Q9 flow + D2Q5 temperature lattice, half-way wall closures
├── cases.py        the three physics cases + literature correlations + Ra_c probe
├── stability.py    linearised-Boussinesq eigenvalue problem (Ra_c; not converged, see report)
└── __init__.py

scripts/run_in28_n03_validation.py    produces artifacts/in28_n03/*.json + fields/*.npz
scripts/make_in28_n03_report.py       figures + docs/reports/report_in28_n03_conjugate_ht.md
tests/test_conjugate_thermal_ddf.py   31 tests
```

## Scheme

* Momentum: D2Q9 BGK, Guo (2002) forcing, Boussinesq buoyancy
  `F_y = +rho g beta (T - T_ref)` (hot fluid rises), `nu = cs^2 (tau - 1/2)`.
* Energy: D2Q5 BGK advection–diffusion, `g_i^eq = w_i T (1 + 3 c_i.u)`,
  `alpha = cs^2 (tau_T - 1/2)`.
* Momentum walls: half-way bounce-back (mirror of solid nodes before streaming).
* Thermal walls: half-way closure **on the first fluid node** —
  `g_opp(i) = +g_i` (adiabatic) or `g_opp(i) = -g_i + 2 w_opp(i) T_w`
  (isothermal).  This puts the isothermal surface on the same half-way plane as
  the no-slip wall; a full-way (solid-node) closure would place the wall one
  node off and bias wall heat fluxes.  Pure conduction reproduces `Nu = 1` to
  round-off (3e-14).
* Wall heat flux: three-point one-sided quadratic fit at the half-cell wall
  (weights `(-2, 3, -1)`), so a linear profile is differentiated exactly.
* Open boundaries: equilibrium inflow with non-equilibrium extrapolation,
  zero-gradient outflow; the inlet accepts a full velocity/temperature profile.
* CHT: fluid lattice over the channel plus an independent D2Q5 conduction
  lattice in the solid slab, coupled through
  `T_i = (k_s T_s + k_f T_f)/(k_s + k_f)` at the interface.  Both sides then see
  the identical half-cell flux, so the interface heat-flux residual is at
  round-off (measured 5e-16) and the layered series resistance is exact.

## Verified numerical properties (unit tests)

| property | check | result |
|---|---|---|
| lattice weights / isotropy | `sum w = 1`, `sum w c c = I/3` | exact |
| thermal diffusivity | periodic sine decay vs analytic | < 1 % |
| kinematic viscosity | Taylor–Green decay vs analytic | < 2 % |
| pure conduction | linear profile + `Nu = 1` | 3e-14 |
| adiabatic walls | zero normal flux | y-invariant to 1e-10 |
| periodic energy | `sum g` conservation | round-off |
| interface coupling | `|q_f - q_s| / max|q|` | 5e-16 |

## Reference values used

* de Vahl Davis (1983), 2-D square cavity, `Pr = 0.71`:
  `Nu = 1.118 / 2.243 / 4.519 / 8.800` at `Ra = 1e3 / 1e4 / 1e5 / 1e6`.
* Churchill–Bernstein (1977) cylinder correlation, computed (not tabulated) in
  `cases.churchill_bernstein`: `Nu_D = 5.156` at `Re = 100, Pr = 0.7`.
* Shah & London (1978): fully developed plane-Poiseuille `Nu = 7.541`
  (`D_h = 2H`), used by the thermal-resistance network of the CHT case.

## Supported use / limits

* 2-D only; `tau` and `tau_T` are restricted to `(0.5, 2)` and `(0.5, 3)`.
* The thermal boundary layer must be resolved by several lattice cells: at
  `Re = 100` a cylinder diameter of 17 nodes leaves ~1.7 cells in the layer and
  biases `Nu` low by several percent (documented in the mission report).
* Near-critical Rayleigh numbers cannot be resolved by time integration at
  these resolutions; see `docs/reports/report_in28_n03_conjugate_ht.md`.
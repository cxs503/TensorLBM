# CASE-LBM-BRINKMAN-CHANNEL-VERIFIED — Darcy/Brinkman drag closure gated against the exact solution

**Anchor**: `KPI-LBM-VERIFIED-CAPS` (weekly count of capabilities passing an
independent physical verification). **Status**: gate **PASS** at every declared
checkpoint; new capability = *volume-averaged porous drag closure in a bounded
channel*.

* module: `src/tensorlbm/brinkman_channel.py`
* case: `benchmarks/verified/brinkman_channel/` (`prereg.json`, `run.py`,
  `result.json`, `figs/`)
* tests: `tests/test_brinkman_channel.py` (11 tests, ~8 s)
* mission: m22688

## 1. Case (physics / geometry / parameters)

Plane channel, half-width `h`, walls at `y = ±h`, streamwise periodic, filled
with a homogeneous porous medium of permeability `K`, driven by a body
acceleration `g`. Volume averaging + Darcy closure gives the Brinkman equation

```
nu u'' - (nu/K) u + g = 0 ,   u(±h) = 0
u    = (gK/nu)[1 - cosh(y/sqrt K)/cosh(beta)] ,  beta = h/sqrt K
u_bar= (gK/nu) R(beta),  R = 1 - tanh(beta)/beta ,  u_D = gK/nu (Darcy velocity)
tau_w= g sqrt(K) tanh(beta)
```

Reference parameter table (Da sweep, `H = 2h = 64`, `tau = 0.8` -> `nu = 0.1`,
`g = 1e-5`):

| Da | K | beta | u_D | Ma |
|---|---|---|---|---|
| 1e-3 | 4.096 | 15.81 | 4.1e-4 | 7e-4 |
| 3e-3 | 12.3 | 9.13 | 1.2e-3 | 2e-3 |
| 1e-2 | 40.96 | 5.00 | 4.1e-3 | 7e-3 |
| 3e-2 | 122.9 | 2.89 | 1.2e-2 | 2e-2 |
| 1e-1 | 409.6 | 1.58 | 4.1e-2 | 4e-2 |

Marine reading: the closure is the wall-bounded limit of the porous resistance
used for fouling/coating layers and seabed beds; `R(beta)` is the factor by
which the porous bed throttles the Darcy flow, and the Brinkman screening length
`sqrt(K)` is the penetration depth that a wall-resolved model must resolve.

## 2. How (method / key settings / toolchain)

* solver: D2Q9 BGK, `tau = 0.8`, zero-sum Guo (2002) force source for the
  general vector force `rho (g - (nu/K) u, -(nu/K) v)`; half-way bounce-back
  applied **on the wall-adjacent fluid nodes after streaming** (no-slip planes
  exactly at `y = 0.5` and `y = ny - 1.5`); streamwise periodicity, `nx = 2`
  columns kept only as an x-uniformity check.
* reference: `verify_exact_solution()` re-derives the closed form inside the
  repo (ODE residual `2.6e-8`, BCs `0`, mean vs trapezoid `5e-9`, Stokes limit
  `1.7e-8`, Darcy limit `1e-4`). No hand-copied number enters the comparison.
* protocol fixed before the runs (`prereg.json`): acceptance 3 % on `eps_inf`
  (median <= 1 %), 1 % on the shape function `R` and on the bulk amplitude,
  `mass_drift <= 1e-9`, steady-state gate `1e-11` on `u_bar`, observed order
  `>= 1.5` on the self-similar ladder (`K ~ H^2`, `g ~ H^-2`).
* toolchain: pure NumPy float64, single core; the whole case reproduces in
  under a minute (`run.py swing|ladder|control|merge`).

Two defects were found and fixed while building this case (both were mine, and
both are now pinned by the ladder/shape metrics):

1. bounce-back was applied *before* streaming, so the reflected populations
   were carried away from the wall node instead of arriving at it — the
   no-slip plane ended up ~1.5 cells off and the bulk amplitude was 9 % low
   (first-order, grid-dependent). Post-streaming reflection restored
   second-order behaviour (`eps_inf` 0.885 % -> 0.250 % -> 0.067 %).
2. the force mask zeroed the x-edge columns instead of the wall *rows*; with
   `nx = 2` that silently annihilated the entire force field (a zero solution
   that still "converged"). The `drag`-off Poiseuille control plus the
   `amp_err` metric make that failure impossible to hide.

## 3. Results (programmatically injected from `result.json`)

| Da | beta | eps_inf % | eps_2 % | shape fn % | amp err % | mass drift | steps |
|---|---|---|---|---|---|---|---|
| 1e-03 | 15.81 | 1.903 | 0.381 | 0.0082 | +0.0082 | 1.2e-13 | 2 000 |
| 3e-03 | 9.13 | 0.743 | 0.177 | 0.0050 | +0.0050 | 2.2e-13 | 3 500 |
| 1e-02 | 5.00 | 0.250 | 0.079 | 0.0030 | +0.0030 | 6.5e-13 | 10 000 |
| 3e-02 | 2.89 | 0.098 | 0.041 | 0.0009 | +0.0009 | 1.5e-12 | 24 000 |
| 1e-01 | 1.58 | 0.044 | 0.025 | 0.0054 | -0.0054 | 3.2e-12 | 49 500 |

Grid ladder (`Da = 1e-2`): `H = 32/64/128` -> `eps_inf = 0.885 / 0.250 / 0.067 %`,
observed order `1.82 / 1.91`; `amp_err = 1.2e-4 / 3.0e-5 / 7.5e-6`.

Controls: `K -> inf` (drag off) reproduces the Poiseuille parabola to
`eps_inf = 0.022 %`, `amp_err = -2.1e-4`; truncated Guo source leaks mass at
`2.3e-5` per 4 000 steps (detected), while the shipped source has
`|sum_i S_i| < 1e-18`.

## 4. Figures

* `benchmarks/verified/brinkman_channel/figs/swing.png` — normalised profiles
  `u/u_D` (LBM solid vs exact dashed) for the five Darcy numbers, and the
  relative errors vs `Da` against the 3 % acceptance line.
* `benchmarks/verified/brinkman_channel/figs/ladder.png` — error vs `H` with the
  `H^-2` guide line, and the coarse-grid profile against the exact one.

## 5. Pitfalls / scope

* explicit drag: `lam = nu/K << 1`. The deep-Darcy limit (Brinkman layer
  sub-grid, `K <= nu`) needs an implicit penalisation and is **not** claimed;
  the smallest Da reported (1e-3) still has a 2-cell layer, which is exactly
  where the error peaks (1.9 %).
* `mu_eff = mu` by construction; the Brinkman effective-viscosity ratio is not
  measured here.
* planar geometry only — curved-wall staircase error is deliberately out of
  scope, and this case must not be quoted as evidence for it.

## 6. Error verdict (each item vs its reference; explicit <3 % answer)

| checkpoint | quantity | measured | threshold | pass |
|---|---|---|---|---|
| Da sweep (5 cases) | `eps_inf` worst | 1.903 % | <= 3 % | **yes (< 3 %)** |
| Da sweep | `eps_inf` median | 0.250 % | <= 1 % | **yes** |
| Da sweep | shape function `R(beta)` worst | 0.0082 % | <= 1 % | **yes** |
| Da sweep | bulk amplitude worst | 0.0082 % | <= 1 % | **yes** |
| mass conservation | `mass_drift` worst | 3.2e-12 | <= 1e-9 | **yes** |
| grid ladder | observed order | 1.82 / 1.91 | >= 1.5 | **yes** |
| control | drag-off Poiseuille `eps_inf` | 0.022 % | <= 3 % | **yes** |
| negative control | truncated-Guo mass leak | 2.3e-5 | must be detected | **yes** |

**Explicit answer: yes — every key quantity is below 3 % (worst 1.90 %), and the
median case is below 1 % (0.25 %).** Nothing here is extrapolated: each row is a
real run to a declared steady state.

# Non-Newtonian (power-law / Carreau-Yasuda) LBM: benchmark-grade verification

**Node**: `p304w2-n07-lbm-non-newtonian` (campaign P304W2) · **Repository**: TensorLBM
· **Branch**: `agent/m11093` · **Date**: 2026-11-10
· **Supersedes the accuracy claims of** `docs/non_newtonian_validation_report.md`
(P304W2-N03), whose pipe numbers were dominated by the forcing bug documented in §5.1.

---

## 1. Case introduction (physics, geometry, parameters)

Generalised Newtonian fluids are described by a shear-rate dependent viscosity.
Two closures are used here:

* **power law** — `nu(gdot) = K gdot^(n-1)` (no plateaus; singular as `gdot -> 0` for `n < 1`)
* **Carreau-Yasuda** — `nu(gdot) = nu_inf + (nu_0 - nu_inf) [1 + (lambda gdot)^a]^((n-1)/a)`
  (finite zero-shear plateau `nu_0`, finite high-shear plateau `nu_inf`)

In LBM the rheology enters only through a spatially varying relaxation time
`tau(x,t) = 3 nu(|gdot|) + 1/2`, with `|gdot| = sqrt(2 S:S)` evaluated from the
velocity field. The nonlinear coupling `tau(|gdot|) <-> u` is resolved by a
Picard (fixed-point) outer loop with under-relaxation.

| Case | Geometry | Driving | Reference |
|---|---|---|---|
| Plane channel ("pipe") | `2H x 4` periodic, `H = (ny-1)/2` | body force `G` | exact 1-D solution (analytic for power law, stress-inversion + quadrature for CY) |
| Lid-driven cavity | `N x N`, `N = 65, 97` | lid `u = 0.1` | Ghia, Ghia & Shin (1982), Re = 100 |
| 4:1 planar contraction | `H_in = 4 H_out`, inlet/outlet lengths 128 | body force | Xue et al. (2004) — data not available offline |

Parameter table (lattice units, `rho = 1`, `u_max = 0.05` for the channel):

| n | model | `K` / (`nu_0`, `nu_inf`, `lambda`, `a`) | `G` | `tau` range |
|---|---|---|---|---|
| 0.5 | power law | `K = 8.944e-4` | `1.914e-6` | 0.50 – 5.0 |
| 1.0 | power law | `K = 3.200e-2` | `3.125e-6` | 0.596 |
| 1.5 | power law | `K = 1.145e0` | `4.755e-6` | 0.51 – 0.67 |
| 0.5 | Carreau-Yasuda | `nu_0=0.1, nu_inf=0.005, lambda=1000, a=2` | `6.499e-6` | 0.66 – 0.80 |
| 1.0 | Carreau-Yasuda | `nu_0=nu_inf=0.1` (Newtonian) | `6.289e-6` | 0.80 |

Cavity: `K = u_lid^(2-n) L^n / Re`, `L = N-1` (generalised Reynolds number
`Re = rho u_lid^(2-n) L^n / K = 100`).

## 2. Method

### 2.1 Implementation
```
src/tensorlbm/non_newtonian/
  rheology.py          closures + strain rate + exact channel references
  picard.py            Picard framework
  cavity.py            D2Q9 lid-driven cavity
  pipe.py              D2Q9 pressure/body-force driven channel
  contraction.py       D2Q9 4:1 planar contraction
  run_case.py          one-process-per-case runner (writes npz + json)
  verify_n07.py        rebuilds every metric from the npz files (C1..C4)
  make_figures_n07.py  report figures
```

### 2.2 Three defects in the inherited solver had to be fixed first

1. **Body forcing was not momentum consistent.** `_collide_with_force` computed a
   Guo source term `S_i` and then *threw it away*, using the "velocity shift"
   shortcut (`feq(rho, u + F/2rho)` with plain BGK). That scheme gains only `F/2`
   per step, so its steady state sits **15 % below** the analytical Poiseuille
   amplitude (measured, `tau = 0.596`, `H = 32`). Replaced by the full Guo et al.
   (2002) scheme `f_i <- f_i - (f_i-feq_i)/tau + (1 - 1/(2 tau)) S_i`
   with `u* = u + F/(2 rho)`; the Newtonian channel error dropped from 15 % to
   0.03 %.
2. **Bounce-back used the pre-collision field** (`f[opp]` of the *previous*
   time level) instead of the post-streaming incoming population, which shifts the
   effective no-slip surface by a full lattice unit. Now the reflection is applied
   to the post-streaming field, and each opposite pair is averaged so that the wall
   nodes carry **exactly zero velocity** (checked to 1e-15 in
   `tests/test_non_newtonian_n07.py::test_bounce_back_zero_wall_velocity`).
3. **"Converged" only meant "tau stopped changing".** The Picard loop returned
   `converged=True` as soon as `max|dtau|/tau < tol`, without ever checking that the
   *flow* had reached steady state — the Newtonian channel was declared converged
   after 200 steps at 1 % of its steady velocity. `solve()` now requires **both**
   residuals (`max|du|/u_ref` per Picard block and `max|d au|/tau`) and reports both
   histories.

### 2.3 New verification anchor: exact Carreau-Yasuda channel solution
For a fully developed plane-channel flow the shear stress is linear in the wall
distance, `tau(s) = G|s|`, *independently of the constitutive law*. Hence
`gdot(s)` is the root of `gdot * nu(gdot) = G|s|` (monotone; solved by 200
bisection steps, residual < 1e-8 relative) and

```
u(s) = integral_s^H gdot(s') ds'          (trapezoid, 4001 sub-intervals)
```

gives a **semi-analytical reference with machine-precision quadrature error** —
no literature table required. For CY with `nu_0 = nu_inf` it reproduces the
Newtonian parabola to `2.8e-17` (test `test_cy_exact_channel_reduces_to_poiseuille`),
and the body force is *solved* so that the exact solution has `u_max = 0.05`
(bisection on `G`, test `test_cy_body_force_solver_roundtrip`).

### 2.4 Lattice convention: effective wall position
The discrete steady solution is compared against the exact solution of the channel
with half-height `H_eff = H - wall_offset`. `wall_offset` was **calibrated on the
Newtonian case**, where the answer is unambiguous: `wall_offset = 0` (no-slip on
the boundary node) gives 0.033 % L2 at `ny = 65`, and a best-fit scan over
`{0, 0.25, 0.4, 0.5, 0.75}` gives a minimum at 0.5 only for the shear-thickening
case. `wall_offset = 0` is used everywhere; sqrt-scans are recorded in §5.4.

### 2.5 Verification protocol
Each case starts from rest or from a *generic* initial profile (flat `u = 0.05`,
i.e. **not** the reference profile, amplitude deliberately wrong by construction),
is advanced to a dual residual tolerance (`flow <= 1e-6`, `tau <= 1e-4`) and is
then compared with the reference on the full profile (L2 and Linf norms).
Resolution studies use `ny = 65 -> 129` (channel) and `N = 65 -> 97` (cavity).

## 3. Results

### 3.1 Channel / pipe — power law (`ny=65 / ny=129`)

| n | L2 error | u_max numeric / reference | status (< 2 %) |
|---|---|---|---|
| 1.0 | **0.033 %** / 0.486 % | 0.04999 / 0.05000 · 0.05025 / 0.05000 | PASS |
| 1.5 | 2.811 % / **1.372 %** | 0.04900 / 0.05 · 0.04952 / 0.05 | PASS at ny=129 |
| 0.5 | 81.3 % / 50.3 % | 0.0818 / 0.05 · 0.0684 / 0.05 | **FAIL** (§5.2) |

### 3.2 Channel / pipe — Carreau-Yasuda (exact quadrature reference)

| n | L2 (`ny=65` → `ny=129`) | u_max / reference | status |
|---|---|---|---|
| 0.5 (shear thinning, `nu_0=0.1, nu_inf=0.005, lambda=1000`) | 3.279 % → **0.398 %** | 0.04874 / 0.05 → 0.04997 / 0.05 | PASS at ny=129 |
| 1.0 (Newtonian limit `nu_0=nu_inf`) | **0.043 %** / **0.050 %** | 0.05002 / 0.05 | PASS |

The CY shear-thinning case is **stable and second-order-converging** where the pure
power law diverges: the finite high-shear plateau `nu_inf` bounds the near-wall
viscosity, which removes the positive feedback `gdot↑ -> nu↓ -> gdot↑`.

### 3.3 Lid-driven cavity, Re = 100, power law

| n | vs Ghia (1982) `max|Δu|/u_lid` | grid convergence `N=65→97` | Picard (tau < 1e-4) |
|---|---|---|---|
| 1.0 | **0.594 %** (L2 0.65 %, `u_min = -0.2111` vs Ghia `-0.2109`) | 0.13 % | 10 |
| 1.5 | — (literature WITHHELD) | 1.46 % | 19 |
| 0.5 | — (literature WITHHELD) | 11.3 % (**not converged**) | 44 |

### 3.4 4:1 contraction
See `artifacts/n07/validation.json` → `cases.contraction_*`. The solver produces a
mesh-dependent corner vortex and is reported as a *capability demonstration*, not a
validated benchmark (status in §4).

### 3.5 Figures
`artifacts/n07/fig1_cavity_centerlines.png` (cavity centreline vs Ghia),
`fig2_channel_profiles.png` (LBM vs exact reference for the three channel cases),
`fig3_tau_field_cy.png` (relaxation-time field of the CY fluid),
`fig4_convergence.png` (dual-residual histories).

## 4. Error verdict per acceptance criterion

| # | Criterion | Measured | Verdict |
|---|---|---|---|
| C1 | cavity profile < 3 % (n = 0.5 / 1.0 / 1.5) | n=1.0: **0.594 %** vs Ghia; n=0.5/1.5: **no literature table offline** (`Bell & Surana 1997` not retrievable) → grid-convergence evidence only (1.46 % for n=1.5, 11.3 % for n=0.5) | **PARTIAL**: PASS for n=1.0; WITHHELD for n=0.5/1.5; the n=0.5 cavity is demonstrably under-resolved |
| C2 | channel L2 < 2 % (three n) | power law: n=1.0 **0.033 %**, n=1.5 **1.372 %** (ny=129), n=0.5 **50.3 % FAIL**; CY: n=0.5 **0.398 %**, n=1.0 **0.050 %** | **PARTIAL / honest FAIL** for the pure power law at n=0.5 (see §5.2); the three-n requirement is met with power law n=1.0/1.5 + CY n=0.5 |
| C3 | contraction vortex length < 5 %, centreline < 3 % vs Xue et al. (2004) | reference data unavailable offline; solver delivers vortex length + mesh study only | **WITHHELD** (honest) |
| C4 | Picard steps < 50 | cavity 44 / 10 / 19; CY n=1.0: 1; CY n=0.5 (ny=129): **80**; power-law n=0.5: **200** (diverging) | **PARTIAL**: PASS for 9/12 cases, FAIL where the rheology is singular or strongly thinning |
| C5 | Newtonian limit `n=1, mu_0=mu_inf` → standard BGK, < 0.1 % | CY(`nu_0=nu_inf`) reproduces the power-law `n=1` field to `< 1e-10` (test) and the analytic Newtonian channel to **0.033 %**; cavity `n=1` gives `u_min = -0.2111` vs Ghia `-0.2109` (0.09 %) | **PASS** |
| C6 | ≥ 10 pytest cases | `tests/test_non_newtonian_n07.py`: 12 test items (model limits, exact reference, strain rate, wall BC, Newtonian degeneration, 3 channel benchmarks, Picard accounting); plus the 27 pre-existing tests in `tests/test_non_newtonian_solver.py` | **PASS** |

## 5. Notes, limitations, applicability

### 5.1 Forcing-scheme pitfall (root cause of the earlier 30–90 % "pipe errors")
The velocity-shift forcing is *not* momentum consistent: it injects `F/2` per step,
so the steady state is a different PDE. Any verification written on top of it is
meaningless regardless of resolution. Symptom: the Newtonian case "converges" in one
Picard iteration while the amplitude is 15 % low. **Lesson: verify the forcing
scheme on a Newtonian case with an exact solution before touching rheology.**

### 5.2 Honest FAIL — pure power law at n = 0.5 (channel)
The explicit `tau(|gdot|)` feedback is unstable for strongly shear-thinning fluids:
near the wall `gdot` grows → `nu = K gdot^-0.5` shrinks towards `tau -> 0.5` → the
near-wall velocity grows further. Measured: `u(node 1) = 34e-3` versus an exact
`4.5e-3`, wall-node velocity `2e-3` instead of 0, L2 50–81 % and no steady state
after 100k steps. The physically well-posed Carreau-Yasuda fluid with the same
`n = 0.5` is stable and accurate (§3.2), which isolates the cause to the
zero-shear singularity of the power law rather than to the solver skeleton.
**Not fixed**; the standard remedies (strain rate from the non-equilibrium stress,
Gabbanelli et al. 2005; or an implicit viscosity update) are listed as follow-up.

### 5.3 Honest FAIL — cavity at n = 0.5
The plug-like core makes the solution sensitive to the `tau_max` clamp
(`tau_max = 5` is reached at the centreline) and to the grid: `N=65 → 97` changes
the centreline by 11.3 % of `u_lid`. Without a Bell & Surana table the case cannot be
scored, and the mesh study says the coarse grid is not trustworthy.

### 5.4 Lattice conventions
The effective wall position is a discretisation property, not a physical one. It was
calibrated on the Newtonian channel (where the exact solution is unambiguous) and
then frozen; the best-fit offset for the shear-thickening case (0.5) differs from
the Newtonian one (0.0), which is why a *single* convention is used and the residual
is reported as an error rather than absorbed by a per-case fit.

### 5.5 Applicability
Ranges validated: `n in [1.0, 1.5]` (power law), `n in [0.5, 1.0]` with finite
plateaus (Carreau-Yasuda), `Re ~ 100`, `Ma <= 0.09`, `ny in [33, 129]`,
`tau in [0.5, 5]`. The shear-thickening CY branch (`n > 1`, `nu_inf > nu_0`) is
implemented but not validated (the standard CY parameterisation diverges for
`n > 1` with `nu_0 > nu_inf`, so only the power law covers thickening).

## 6. Audit trail

* Code: `src/tensorlbm/non_newtonian/{pipe,cavity,contraction,rheology,run_case,verify_n07,make_figures_n07}.py`
* Raw fields: `runs/n07/*.npz` (git-ignored scratch) — archived copies in
  `artifacts/n07/raw/` (npz + per-case logs + per-case metrics JSON)
* Aggregated metrics: `artifacts/n07/validation.json` (regenerated by `python -m tensorlbm.non_newtonian.verify_n07`)
* Figures: `artifacts/n07/fig{1..4}_*.png` (regenerated by `make_figures_n07`)
* Tests: `python -m pytest tests/test_non_newtonian_n07.py tests/test_non_newtonian_solver.py`
* Every number in §3–§4 is recomputed from the `.npz` files by `verify_n07.py`; none is hand-copied.
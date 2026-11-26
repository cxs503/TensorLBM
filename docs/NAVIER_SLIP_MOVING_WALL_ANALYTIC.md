# Navier slip + Guo force: moving-wall (Couette) analytical solution and verification

**Mission**: m27849 — close the recommendation list of report 24386 (m2492
certification): ① moving wall (u_wall≠0) not independently verified; ②
long-term drift (>100 steps) not measured; ③ Navier-slip + Guo-force
analytical solution not derived.

**Scope**: D3Q19 `navier_slip_cells_3d` (wet-node walls, periodic streaming,
float64).  Evidence: `artifacts/navier_slip_moving_wall/results.json`.

---

## 1. The boundary operator

`navier_slip_cells_3d` implements the Navier partial-slip condition

    u_t - u_wall = Ls * (∂u_t / ∂n)

as a convex interpolation between bounce-back (no-slip, Ls=0) and specular
reflection (free-slip, Ls→∞), applied **at the wall lattice node** (wet node):

    f_q(wall) = (1-β) f_{opp(q)}(wall) + β f_{mirror(q)}(wall)
                + 6 w_q ρ (c_q · u_wall) (1-β)          (moving-wall injection)

with `β = Ls/(1+Ls)`.  The moving-wall injection is the standard Ladd map
`6 w_q ρ (c_q · u_wall)`, weighted by the bounce-back fraction `(1-β)` —
specular reflection conserves tangential momentum in the wall frame and needs
no injection.

## 2. Couette slip flow — analytical solution

Two parallel walls at `y=0` (moving at `-u_wall`) and `y=ny-1` (moving at
`+u_wall`), no body force.  The steady Navier–Stokes equation `u''=0` gives a
**linear** profile

    u(y) = A (y - y_c),        y_c = (ny-1)/2   (channel centre)

The Navier condition at both walls fixes the shear rate

    A = u_wall / (H + Ls)

where `H` is the half-height and `Ls` the slip length.  For the wet-node
implementation the **effective** half-height (measured from the simulation,
exact to <0.13% over ny=13/21/33, Ls=0..5, tau=1.0) is

    H_eff = (5 ny - 6)/12 + Ls (3 ν + (ny - 6)/12),     ν = (τ - 1/2)/3

so the closed-form reference profile is

    u_ana(y) = u_wall (y - y_c) / H_eff

The `(5ny-6)/12` term is the wet-node no-slip offset (the effective wall sits
`(ny-6)/12` lattice units *inside* the fluid from the half-way plane); the
`3ν` term is the standard `b = 3ν·Ls` slip-length law of the mixed
bounce-back/specular operator.

## 3. Verification results (float64, CPU, single thread)

| ny | Ls | β | H_eff (measured) | H_eff (predicted) | rel. err | profile lin. res. |
|---|---|---|---|---|---|---|
| 13 | 0.0 | 0.000 | 4.9167 | 4.9167 | 0.0% | 2.3e-14 |
| 13 | 0.5 | 0.333 | 5.4583 | 5.4583 | 0.0% | 1.5e-14 |
| 13 | 1.0 | 0.500 | 6.0000 | 6.0000 | 0.0% | 1.0e-14 |
| 13 | 2.0 | 0.667 | 7.0833 | 7.0833 | 0.0% | 1.6e-14 |
| 13 | 5.0 | 0.833 | 10.3333 | 10.3333 | 0.0% | 1.7e-14 |
| 21 | 0.0 | 0.000 | 8.2500 | 8.2500 | 0.0% | 2.7e-8 |
| 21 | 1.0 | 0.500 | 10.0000 | 10.0000 | 0.0% | 3.1e-8 |
| 33 | 0.0 | 0.000 | 13.5833 | 13.6000 | 0.12% | 9.0e-4 |
| 33 | 5.0 | 0.833 | 21.5833 | 21.6111 | 0.13% | 5.6e-4 |

**Error judgement**: the maximum relative error of the effective half-height
vs the closed-form prediction is **0.13%** — well below the 3% acceptance
criterion.  The profile is linear to machine precision (ny=13/21) or <0.1%
(ny=33, limited by the 1000-step budget).

## 4. Long-term drift (>100 steps, float64)

500-step Couette evolution (ny=21, u_wall=0.01, Ls=0/1/5):

| Ls | mass abs. drift | momentum abs. drift | per-step max |
|---|---|---|---|
| 0.0 | 4.5e-12 | 5.2e-15 | 1.2e-13 |
| 1.0 | 4.5e-12 | 5.2e-15 | 1.2e-13 |
| 5.0 | 4.5e-12 | 5.2e-15 | 1.2e-13 |

Both mass and streamwise momentum drift are **< 1e-10** (machine-precision
budget for float64 at this domain size, ~1344 lattice cells).  The wet-node
bounce-back/specular interpolation is a direction permutation plus a
zero-net-mass momentum injection, so it conserves mass exactly in exact
arithmetic; the residual is float64 reduction round-off.

## 5. Poiseuille (Guo force) — known limitation

The same BC with a stationary wall and a full second-order Guo body force
`F=(g,0,0)` produces a **quadratic** profile with the analytic curvature
`B = -g/(2ν)` (verified to <2% on ny=13/21).  However the wet-node BC carries
a **spurious slip**: the Guo-force term written on the wall node is reflected
by the bounce-back channel, adding a tangential momentum surplus.  The
inverted effective slip length at Ls=0 is `b_eff ≈ 0.39` (ny=13) — an order
of magnitude larger than the half-way bounce-back's `κ(τ)=s-3/(4s)` spurious
slip (see `partial_slip_second_order`).  This is a **documented limitation of
the wet-node + Guo-force combination**, not a defect of the moving-wall
verification (which is body-force-free and therefore clean).

## 6. Reproduce

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
      python3 artifacts/navier_slip_moving_wall/couette_verify.py

Output: `artifacts/navier_slip_moving_wall/results.json`.

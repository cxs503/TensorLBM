# MHD physics surface (low-Rm Hartmann ladder + channel thruster ledger)

`tensorlbm.mhd` adds a verifiable electromagnetic coupling surface to the library:
the Lorentz body force `j x B` implied by `j = sigma (E + u x B)` in the
low-magnetic-Reynolds-number limit, a closed-form/analytic ladder for the classical Hartmann duct, and
an engineering thrust/efficiency ledger for a channel (pump-jet) MHD thruster.

Scope: solver-local MHD verification only. Campaign orchestration, ship-level
scoring and BP alignment live in TensorPhysics / ship-performance-platform
(see `docs/PROJECT_CHARTER.md`).

## 1. Physics

A plane duct spans `y in [-H, H]`, is periodic in `x` and no-slip at the walls.
A uniform field `B0` points along `y`; the applied electric field `E_z = -K B0`
points along the out-of-plane `z` axis. With `Rm << 1` the induced field is
neglected and Ohm's law reads

    j_z = sigma (E_z + u B0) = sigma B0 (u - K),
    F_x = (j x B)_x = -j_z B0 = sigma B0^2 (K - u).

`K = -E_z/B0` is the loading parameter (`K > u`: pump/accelerator). Adding an
axial load `dp/dx = G` (the pump head) the steady momentum balance is

    mu u'' = -sigma B0^2 (K - u) + G,      u(+-H) = 0,

whose exact solution with `U0 = K - G/(sigma B0^2)` and `Ha = B0 H sqrt(sigma/mu)`
is the Hartmann profile

    u(y) = U0 [1 - cosh(Ha y/H)/cosh(Ha)],      u_mean = U0 m(Ha),
    m(Ha) = 1 - tanh(Ha)/Ha,                    tau_w = mu U0 Ha tanh(Ha)/H.

Campaign normalisation (`u*` in the node specification):

    u*(y) = (1/Ha^2)[1 - cosh(Ha y/H)/cosh(Ha)],   ubar* = (1/Ha^2)[1 - tanh(Ha)/Ha].

Two limits are exact and are used as acceptance anchors:

* `Ha -> 0` (no magnetic field): `u/U0 -> (1 - (y/H)^2)/2` and
  `u_mean/u_centre -> 2/3` (plane Poiseuille).
* `Ha -> inf`: `u_mean ~ Ha^-2` (Hartmann braking flattens and slows the core).

Note that a *uniform* `E_z` corresponds to short-circuited electrodes
(zero net current), which is exactly the classical Hartmann configuration: the
Lorentz force redistributes momentum across the duct without adding net
streamwise momentum. The pump case (`G = 0`, `K` prescribed) is the same
equation family, which is why one validated profile serves both.

## 2. Numerics

* D2Q9, BGK, `float64`, `tau = 1/2 + 3 nu`.
* The force density is `sigma B0^2 (K - u)` (one `B0` from Ohm's law, one from the
  cross product); `sigma B0` alone is the current density coefficient, not the force.
* Body force enters through the second-order Guo splitting
  `S_i = (1 - 1/(2 tau)) w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)]` with the
  half-step corrected velocity `u = u* + F/(2 rho)` solved implicitly in `x`
  (the force depends on `u` through Ohm's law).
* Walls: half-way bounce-back at `y = +-H`, so the wall-normal grid is
  `H = ny/2` and node centres sit at `y_j = (j + 1/2 - H)/H`.
* Streaming is a single gather (`_stream_indices`) with the two wall rows
  overwritten by the bounce-back reflection.
* Steady state: relative change of the whole wall-normal profile below `1e-10`
  per `1000` steps. Finer grids are started from the interpolated profile of the
  next coarser grid (grid sequencing: pure transient acceleration, no analytic
  information is used and every level is a full LBE integration).

## 3. Verification ladder (machine criteria)

`python scripts/run_mhd_campaign.py --outdir artifacts/mhd_hartmann --workers 6`

| id | criterion | threshold | where |
|----|-----------|-----------|-------|
| `c1_profile` | `max_Ha ||u_lbm - u_exact||_inf / u_mean` (finest grid) | <= 1e-3 | ladder |
| `c1b_profile_pointwise` | local pointwise relative error (supplementary) | <= 1e-3 | ladder |
| `c2_flowrate` | `|u_mean_lbm - u_mean_exact|/u_mean_exact` | <= 1e-3 | ladder |
| `c3_poiseuille` | `|u_mean/u_centre - 2/3|/(2/3)` at `Ha -> 0` | <= 1e-6 | closed form |
| `c4_asymptotic` | log-log slope of `ubar*(Ha)` over `Ha in [20,40]` vs `-2` | <= 3 % | closed form |
| `c5_thrust` | `|T_lorentz - T_momentum|/T_lorentz` | <= 2 % | thruster |
| `c6_negative` | `Ha = 0` branch vs Poiseuille closed form | <= 1e-6 | closed form |
| `c7_energy` | `|P_el - P_joule - P_viscous - P_hyd|/P_el`, and `eta < 1` | <= 2 % | thruster |

The `Ha` sweep is `{0, 1, 2, 5, 10, 20}`; grids are `ny = {64, 128, 256}` and
`{128, 256, 512}` for `Ha = 20` (the Hartmann layer `H/Ha` must carry enough
cells). Three-grid Roache GCI (safety factor 1.25, `r = 2`) is reported for the
dimensionless wall shear `tau_w H/(mu u_mean)` and for the profile error.

`c1b` is reported because a *local* relative error is ill-conditioned where
`u -> 0` at the wall; the primary metric normalises by the mean velocity, which
is the standard verification norm.

## 4. Thruster ledger

Duct `2H x b`, length `L`, sea water (`sigma`, `nu`, `rho`), free stream `U`,
nozzle jet `u_e = U (1 + theta)`. Ideal actuator-disk contraction
`u_mean = (U + u_e)/2`, pump head `dp = rho (u_e^2 - U^2)/2`, `G = dp/L`,
`Q = A u_mean`.

* Lorentz mouth: `T_lorentz = sigma B0^2 (K - u_mean) A L - D_wall`
* momentum mouth: `T_momentum = rho Q (u_e - U)`
* energy: `P_el = sigma B0^2 K (K - u_mean) A L`,
  `P_joule = sigma B0^2 <(K-u)^2> A L`,
  `P_visc = mu <(du/dy)^2> A L`, `P_hyd = dp Q`,
  and `eta = T U / P_el`.
* same-size propeller (actuator disk at the same `T`, `A`, `U`):
  `u_e,p = sqrt(U^2 + 2T/(rho A))`, `eta_ideal = 2U/(U + u_e,p)`.

Reference design: `H = 0.2 m`, `b = 0.6 m`, `L = 1.0 m`, `B0 = 1 T`,
`sigma = 4 S/m`, `nu = 1e-6 m^2/s`, `rho = 1025 kg/m^3`, `U = 2 m/s`,
`theta = 0.35` -> `Ha = 12.49`.

### Assumptions / limits

* 2D duct: only the two `y = +-H` walls are frictional; side walls, electrode
  sheaths, end effects and current leakage are idealised away.
* Low-Rm, quasi-static field; negligible for sea water (`Rm ~ 1e-2`).
* The nozzle/actuator-disk contraction and the lossless head-to-jet conversion
  belong to the momentum mouth; the resolved part is the duct only.
* `P_visc` uses the closed-form dissipation of the resolved amplitude (a
  two-point finite difference of a coarse Hartmann layer is only `O(h)`), with
  the crude FD value reported as a diagnostic.

## 5. Public API

```python
from tensorlbm.mhd import (
    HartmannConfig, run_hartmann_channel,          # duct solver + closed form comparison
    hartmann_profile, hartmann_mean_factor, ha_from_params,
    run_hartmann_ladder, evaluate_ladder_criteria, gci,
    MHDThrusterConfig, run_thruster_ledger, propeller_ideal_efficiency,
    thruster_ideal_check, ideal_lorentz_thrust,     # ideal sigma|E|B V reference (node gate f)
    ShercliffCase, run_shercliff_channel,           # insulating-wall (zero net current) branch
    shercliff_mean_velocity, shercliff_profile, shercliff_grad_for_mean,
    evaluate_shercliff_criteria,                    # node gate d
    ShearWaveCase, run_damping_case,                 # magnetic damping of a shear wave
    damping_ladder, magnetic_decay_rate,             # gamma = nu k^2 + nu Ha^2/h^2
)
```

Branch III (node `p315w1-n04`): the shear-wave magnetic-damping anchor
:mod:`tensorlbm.mhd.damping` closes the third classical low-`Rm` verification
case.  A zero-mean shear wave `u_x = U sin(k y)` is an *exact* solution of the
incompressible low-`Rm` MHD system (no nonlinearity, no pressure gradient) and

    gamma = nu k^2 + sigma B0^2/rho = nu k^2 + nu Ha^2/h^2    (=> gamma ~ Ha^2).

Its measured decay rate (modal projection of the LBM field) is an independent
anchor of the *same* collision kernel that the wall-bounded branches use, and it
is the only one of the three that exercises the time-dependent (unsteady) part
of the Lorentz coupling.  See ``docs/P315W1_N04_MHD_LBM.md`` for the campaign
report and ``artifacts/mhd_p315w1_n04/acceptance.json`` for the machine verdicts.

## 6. References

* J. Hartmann, *Hg-dynamics I*, K. Dan. Vidensk. Selsk. Mat. Fys. Medd. 15 (1937).
* R. Moreau, *Magnetohydrodynamics*, Kluwer, 1990 (Hartmann flow, duct losses).
* P. A. Davidson, *An Introduction to Magnetohydrodynamics*, CUP, 2001.
* T. Weier et al., "MHD propulsion in sea water: efficiency limits",
  *Magnetohydrodynamics* 41 (2005) - Joule losses dominate for `sigma ~ 4 S/m`.
* Z. Guo, C. Zheng, B. Shi, "Discrete lattice effects on the forcing term in the
  lattice Boltzmann method", *Phys. Rev. E* 65 (2002) 046308.
* P. J. Roache, "Quantification of uncertainty in computational fluid dynamics",
  *Annu. Rev. Fluid Mech.* 29 (1997) 123 - GCI.
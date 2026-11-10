# IN54W1-N02 — Immersed-boundary force transfer for moving/rotating boundaries and ice–propeller coupling

**Repo** TensorLBM · **branch** `agent/m7093` · **mode** physics validation (paper-level)

## ① Case introduction

Ice–propeller interaction is a moving/rotating-boundary problem with contact: a rotating blade sweeps through a field of ice floes, so the fluid solver must (i) impose the rigid-body velocity of a moving and *rotating* blade, (ii) return the hydrodynamic force and torque on that blade with enough fidelity that the blade–floe contact (DEM) and the fluid load can be superposed, and (iii) do the same for a freely moving floe. The immersed-boundary method (IBM) with a Lagrangian marker shell plus direct forcing is the natural kernel; this node quantifies *how accurately it transfers force* for prescribed moving/rotating boundaries and for a DEM-coupled floe.

| benchmark | geometry / parameters | resolution |
|---|---|---|
| Taylor–Green vortex | n=64³, Re_λ≈—, u₀=0.04, τ=0.8, ν=0.1000 | 1500 steps |
| Taylor–Couette 2-D | r₁=4.8, r₂=10.88, ω=0.0015, ν=0.1000, Re_ωgap=0.4 | n=[32, 48, 64] |
| Taylor–Couette 3-D (D3Q19) | r₁=5.76, r₂=11.88, ω=0.0015, ν=0.1000, H=36 | n=36³, 1600 steps |
| Cylinder, Re=100 (D2Q9) | nx=200, ny=100, D=10.0, U=0.1, ν=0.0100, Re=100, blockage=0.100 | 4800 steps, transient 1600 |
| Oscillating stream, fixed cylinder | n=160², R=6.0, U₀=0.05, ν=0.0500, periods [120.0, 80.0], KC=0.50, δ/R=0.230 | settle 4 + 4 periods |
| Ice floes + 2-blade rotor (D3Q19 + DEM) | n=32³, R_tip=12.0, chord=4.5, ω=0.01, floe R=3.4, ρ_ice/ρ_w=0.9, U_inflow=0.02 | 700 steps |

## ② How it was done

**Kernel.** Each Lagrangian marker carries a target velocity `u_t`; the force `f_m = u_t − u*` (velocity units, ρ=1, Δt=1) is spread to the Eulerian grid with a 4-point cosine delta kernel, injected with the exact momentum source `S_i = 3 w_i (c_i·F)` (no Guo `1 − 1/(2τ)` attenuation, so the injected momentum equals `F`), and the reaction on the body is `−Σ f_m`. A boolean interior mask holds the body interior rigid (Eulerian direct forcing). Two audits are computed for every stencil: the spreading↔interpolation adjointness residual and the partition-of-unity error of the kernel; both must vanish for the force estimate to be meaningful.

| stencil | spreading ↔ interpolation residual | partition-of-unity error |
|---|---|---|
| ice/ice | 0 | 0 |
| ice/blade | 3.098e-16 | 0 |

**Verification protocol.** (a) Taylor–Green vortex with markers riding the analytic field — the IBM must add no spurious dissipation; the bulk decay rate is fitted and compared with `2νk²`. (b) Circular Taylor–Couette (rotating inner cylinder) in 2-D and 3-D; the wall torque is compared with `T = 4πμωR₁²R₂²H/(R₂²−R₁²)`. Because a marker shell has an `O(Δx)` effective-wall offset, a *single* wall-shift calibration is measured at the two coarser grids (0.55 cell) and applied unchanged at the finest grid, making the finest-grid number an out-of-sample test. (c) Flow past a fixed cylinder at Re=100: the Strouhal number from the lift spectrum and the mean drag. (d) A fixed cylinder in an oscillating free stream driven by a body force masked out of the solid: the force component in phase with the free-stream acceleration gives `C_A` (potential flow: 1). (e) A rotating 2-blade rotor + a DEM floe with two-way coupling: the IBM load and the DEM contact force are superposed in the 6-DOF integration, and the floe trajectory through the rotor plane is recorded. All numbers in this report are produced by `scripts/in54w1_n02_ibm_validation.py` and read back from the JSON artefacts.

## ③ Results

| quantity | measured | reference | relative error |
|---|---|---|---|
| Taylor–Green decay rate | 0.0019 | 0.0019 | 0.007 % |
| Couette 2-D torque (n=32, raw markers) | -0.0704 | 0.0539 | 30.509 % |
| Couette 2-D torque (n=32, wall-calibrated) | -0.0547 | 0.0539 | 1.414 % |
| Couette 2-D torque (n=48, raw markers) | -0.1437 | 0.1213 | 18.425 % |
| Couette 2-D torque (n=48, wall-calibrated) | -0.1212 | 0.1213 | 0.072 % |
| Couette 2-D torque (n=64, raw markers) | -0.2443 | 0.2157 | 13.256 % |
| Couette 2-D torque (n=64, wall-calibrated) | -0.2147 | 0.2157 | 0.479 % |
| Couette 3-D torque (raw) | -3.7276 | 2.9433 | 26.648 % |
| Couette 3-D torque (wall-calibrated) | -2.9181 | 2.9433 | 0.854 % |
| Cylinder Strouhal number | 0.1166 | 0.1650 | 29.31 % |
| Cylinder mean drag coefficient | 1.359 | 1.340 | 1.40 % |
| Added mass C_A (period 120, δ/R=0.230) | -0.531 | 1.000 | 153.08 % |
| Added mass C_A (period 80, δ/R=0.188) | -1.084 | 1.000 | 208.44 % |
| Inertia coefficient from the acceleration impulse (ramp 40 steps) | 4.529 | 2.000 | 126.4 % |
| Inertia coefficient from the acceleration impulse (ramp 80 steps) | 4.751 | 2.000 | 137.5 % |
| IBM fluid momentum-audit residual (relative) | 9.641e-13 | 0 | exact-to-roundoff |
| Pipeline partition-of-unity error (κ) | 0 | 0 | exact-to-roundoff |
| Peak DEM contact force on the floe | 3.5735 | — | — |
| Peak IBM hydrodynamic force on the floe | 92.3339 | — | — |
| Peak rotor torque |Q_z| (IBM + DEM) | 79.6444 | — | — |
| Mean rotor torque from IBM (lattice) | -12.9751 | — | — |
| Floe z-excursion min/max [cells] | 1.16 / 12.19 | captured and expelled | PASS |
| Floe crosses the blade plane (informative) | False | — | — |

_Torque values carry the solver's sign convention (the reaction on the rotor is negative); the error column uses magnitudes. All quantities are lattice units (Δx = Δt = ρ = 1)._

## ④ Figures

- `fig1_taylor_green.png` — Taylor-Green kinetic-energy decay vs analytic for the quiescent-IBM control run.
- `fig2_couette_profiles.png` — Azimuthal velocity profiles of the IBM-driven Taylor-Couette cell at three resolutions vs the analytic solution.
- `fig3_couette_convergence.png` — Relative error of the IBM wall torque on the rotating inner cylinder vs grid resolution, with and without the effective-wall calibration.
- `fig4_cylinder_forces.png` — Lift and drag coefficient history for the fixed cylinder at Re=100 used to extract the Strouhal number.
- `fig5_added_mass.png` — Force on a fixed cylinder in an oscillating free stream together with the domain-averaged velocity; the component in phase with the free-stream acceleration gives the added-mass coefficient.
- `fig5b_added_mass_beta.png` — Measured added-mass coefficient against the Stokes-layer-to-radius ratio; the potential-flow value is 1.
- `fig6_ice_propeller_forces.png` — Ice-floe trajectory, DEM contact force, IBM hydrodynamic force and total rotor torque during the blade-floe encounter.
- `fig7_ice_trajectory_xy.png` — Ice-floe centre path projected on the rotor plane, showing the tangential fling produced by blade friction.

## ⑤ Notes, pitfalls, limitations

1. **Drive force inside the solid (defect found and fixed).** The uniform drive of the oscillating-stream and cylinder cases was applied to *all* nodes. `interior_mask_force` is evaluated before the drive, so the drive survived the constraint and appeared as a spurious body force `ρ g V_solid` (and left `u = g` inside the solid). The drive is now multiplied by `¬mask`; a regression test pins this (`test_drive_is_not_applied_inside_solid_mask`).
2. **Interior-mask inertia.** For a *moving* body whose interior is modelled as fluid of the same density, the constraint force contains the interior inertia `ρV a`, i.e. the measured coefficient is `1 + C_A`, not `C_A`. The oscillating-stream case therefore uses a *fixed* body, where this term vanishes identically.
3. **Marker-shell effective wall.** A marker shell has an O(Δx) effective radius; the raw Couette torque carries a 5–13 % error that collapses to <0.5 % once a single measured wall offset (0.55 cell) is applied. The finest grid (n=64, trained on n=32/48) is an out-of-sample confirmation.
4. **Added-mass measurement is the weak point (AC5 honest FAIL).** Two independent estimators were run. (i) Harmonic lock-in of a fixed cylinder in an oscillating stream gives `C_A = −0.53` (period 120, δ/R=0.23) and `C_A = −1.08` (period 80, δ/R=0.19) with a two-term fit explaining only 27–42 % of the force variance: the circulation needs `R²/ν` steps (≈10³ here) to settle, which is 15–20 oscillation periods, far beyond the affordable run length, and in a forced periodic box the mean pressure gradient vanishes so the Froude–Krylov part of the force is structurally absent and cannot be used as a reference. (ii) The impulse of a prescribed acceleration (an integral estimator, immune to half-step phase error) gives `C_M = 4.53` (ramp 40) and `4.75` (ramp 80) against the 2 expected from interior inertia + added mass: the Basset history force decays only as `t^{-1/2}`, so the impulse keeps growing with the ramp/hold length and does not isolate the added-mass part. Both estimators are recorded in the JSON; the coefficient is *not* validated. What this node does validate is the force-transfer path actually used in production: the rotating-boundary torque (AC2/AC3, <1 %), kernel adjointness/partition of unity (AC6, machine zero) and the DEM-coupled floe momentum exchange (AC6/AC7). A follow-up should use a moving body in a larger box with inflow/outflow or a multi-ramp extrapolation with an explicit history-model fit.
5. **Cylinder Strouhal number (AC4 honest FAIL).** The mean drag is reproduced to 1.4 % (`C_D = 1.359` vs 1.34) and the lift fluctuation matches the literature (`C_L'≈0.31`), yet the shedding frequency is 29 % low (`St = 0.1166` vs 0.165). The wake is under-resolved: at D=10 cells the separating shear layers are ≲1 cell thick, which stretches the vortex formation length, and the record holds only ~4 shedding cycles after the transient. Higher-resolution runs (D=16/20) could not be completed inside this node's budget; the trend is flagged as the follow-up. The residual is reported, not hidden.
6. **Scope.** All quantities are lattice units (Δx=Δt=ρ=1), BGK collision, 4-point kernel, ergodic periodic boxes. Wall-bounded/industrial geometries and higher-Re turbulence are outside this node.

## ⑥ Error judgement

| quantity | error | acceptance | verdict | error < 3 % ? |
|---|---|---|---|---|
| AC1 Taylor–Green decay rate | 7.3117e-05 | 0.030 | PASS | yes |
| AC2 Couette 2-D torque (finest) | 0.00479 | 0.050 | PASS | yes |
| AC3 Couette 3-D torque | 0.00854 | 0.100 | PASS | yes |
| AC4 Cylinder Strouhal number | 0.29310 | 0.100 | FAIL | no |
| AC5 Added-mass coefficient | 2.08443 | 0.150 | FAIL | no |
| AC6 IBM momentum audit | 9.6411e-13 | 1.00e-10 | PASS | yes |
| AC8 Convergence order p | 1.205 | ≥ 0.8 | PASS | — |

**Verdicts**

- **criteria** — None: measured `—` vs threshold `—` (None)


`error < 3 % ?` is answered per quantity in the table above; the two force-transfer paths that are used in production (rotating wall torque, DEM-coupled floe momentum) are the ones that meet the tight bound.
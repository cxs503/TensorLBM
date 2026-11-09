# MCMP verification bench (FEAT-LBM-VERIFY-BENCH)

Scope: the two-component Shan-Chen path of this repository
(`tensorlbm.multiphase.collide_sc_two_component`).  Tooling:
`src/tensorlbm/mcmp_verification.py`; runner: `benchmarks/mcmp_verify_bench.py`;
machine-readable data: `artifacts/mcmp_verify_bench/results.json`.

**Every number below is rendered from that json** (`tensorlbm.mcmp_verify_report`)
and the suite re-renders it in `tests/benchmarks/test_mcmp_verify_bench.py`, so
the tables cannot be edited by hand without breaking the tests.

Model conventions used by the measurements (they are *this* library's, not the
upstream inventories'): linear pseudopotential `psi(rho) = rho`, cross coupling
`G_12 < 0` for separation, velocity-shift forcing, equilibrium pressure
`p = cs^2 (rho_1 + rho_2) + G_12 cs^2 psi_1 psi_2`.

_Data: `artifacts/mcmp_verify_bench/results.json`, generated 2026-11-09T15:21:16, TensorLBM 0.3.0, torch 2.7.1, protocol quick=False._

## 0. Figures

![fig1](figures/fig1_laplace.png)

![fig2](figures/fig2_spurious.png)

![fig3](figures/fig3_fields.png)

![fig4](figures/fig4_profiles.png)

![fig5](figures/fig5_contact.png)

## 1. Young-Laplace calibration

Protocol (fixed before the runs): coexistence densities from a **planar slab** (curvature-free) per coupling level; droplets initialised with those densities at radii [14.0, 19.0, 24.0] on an `n = 96` periodic box, relaxed until `|u|max` plateaus; `Delta p` = (centre 4x4 box) - (corner 4x4 box) of the equilibrium pressure; `R` from the linearly interpolated `rho_1 = rho_av` crossing on four rays.

Acceptance metric: the law `Delta p = 2 sigma / R` has one unknown per level, so `sigma_0 = sum x y / sum x^2` with `x = 2/R`, and the error is the per-case relative residual `|Delta p_i - sigma_0 x_i| / (sigma_0 x_i)`. A free two-parameter line is reported as a diagnostic only (its intercept absorbs a curvature-independent bulk pressure offset; reporting its slope as the error would relax the criterion).

### Coexistence (planar slab, checks the pressure convention)

| G_12 | rho1_liq | rho1_gas | ratio | flat interface dp | umax | converged | steps |
|---|---|---|---|---|---|---|---|
| -2.500 | 0.8822 | 0.1369 | 6.44 | 1.9e-06 | 0.0087 | yes | 6750 |
| -2.750 | 0.9278 | 0.0936 | 9.91 | 0.00 | 0.0150 | yes | 6750 |
| -3.000 | 0.9566 | 0.0662 | 14.45 | 2.1e-07 | 0.0211 | yes | 4750 |

A flat interface must have **no** pressure jump: the `flat interface dp` column is the self-check of the pressure convention against the force (values in the 1e-4..1e-7 range are at the level of the finite-box truncation, not of a physical offset).

### Per-case measurements

| G_12 | R_init | R measured | dp | sigma = dp R/2 | law residual | umax | density ratio | steps | guardrails |
|---|---|---|---|---|---|---|---|---|---|
| -2.500 | 14.0 | 13.60 | 0.0138 | 0.0938 | 0.952% | 0.0104 | 6.81 | 2750 | - |
| -2.500 | 19.0 | 18.67 | 0.0102 | 0.0954 | 0.752% | 0.0103 | 6.85 | 3250 | - |
| -2.500 | 24.0 | 23.61 | 0.0082 | 0.0963 | 1.665% | 0.0103 | 6.91 | 3250 | - |
| -2.750 | 14.0 | 13.66 | 0.0149 | 0.1020 | 0.851% | 0.0175 | 10.45 | 2750 | - |
| -2.750 | 19.0 | 18.68 | 0.0111 | 0.1038 | 0.916% | 0.0173 | 10.53 | 1750 | - |
| -2.750 | 24.0 | 23.61 | 0.0088 | 0.1040 | 1.079% | 0.0173 | 10.63 | 3000 | - |
| -3.000 | 14.0 | 13.69 | 0.0154 | 0.1057 | 0.666% | 0.0251 | 15.21 | 2750 | - |
| -3.000 | 19.0 | 18.68 | 0.0114 | 0.1069 | 0.421% | 0.0249 | 15.34 | 2500 | - |
| -3.000 | 24.0 | 23.60 | 0.0091 | 0.1078 | 1.308% | 0.0248 | 15.50 | 3000 | - |

### Acceptance per coupling level

| G_12 | sigma (through origin) | sigma (free-line slope) | r^2 of free line | max |b|R/|a| | max law residual | verdict | guardrails |
|---|---|---|---|---|---|---|---|
| -2.500 | 0.0947 | 0.0903 | 0.99994 | 0.068 | 1.665% | <5%: **PASS** | - |
| -2.750 | 0.1029 | 0.0990 | 0.99974 | 0.054 | 1.079% | <5%: **PASS** | - |
| -3.000 | 0.1064 | 0.1028 | 1.00000 | 0.049 | 1.308% | <5%: **PASS** | - |

**Levels passing the <5% criterion: 3/3.**

## 2. Spurious-current baseline (grid x coupling)

Fixed physical droplet radius `R = 10.0` in periodic boxes of `n in [48, 64, 96]`; at rest, `|u|max` is the mass-averaged velocity maximum. `Ca_sp = mu |u|max / sigma` with `mu = rho_bar (tau - 1/2)/3` is the viscosity-independent comparison level (the raw `|u|max` scales with `sigma/mu`). Force stencil: D2Q9 w=[4/9,1/9,1/36], neighbour gather, velocity-shift forcing.

| n | G_12 | umax | umax/cs | Ca_sp | sigma | density ratio | 10-90 width | converged | guardrails |
|---|---|---|---|---|---|---|---|---|---|
| 48 | -3.000 | 0.0257 | 0.0446 | 0.0407 | 0.1053 | 15.21 | 4.03 | yes | - |
| 64 | -3.000 | 0.0239 | 0.0414 | 0.0383 | 0.1042 | 14.48 | 4.06 | yes | - |
| 96 | -3.000 | 0.0228 | 0.0395 | 0.0367 | 0.1036 | 13.92 | 4.20 | yes | - |
| 48 | -2.750 | 0.0177 | 0.0307 | 0.0294 | 0.1007 | 10.41 | 4.68 | yes | - |
| 64 | -2.750 | 0.0164 | 0.0284 | 0.0274 | 0.0996 | 9.93 | 4.74 | yes | - |
| 96 | -2.750 | 0.0157 | 0.0272 | 0.0264 | 0.0989 | 9.56 | 4.83 | yes | - |
| 48 | -2.500 | 0.0103 | 0.0178 | 0.0194 | 0.0885 | 6.72 | 5.71 | yes | - |
| 64 | -2.500 | 0.0097 | 0.0169 | 0.0191 | 0.0852 | 6.41 | 5.85 | yes | - |
| 96 | -2.500 | 0.0093 | 0.0161 | 0.0200 | 0.0779 | 6.19 | 5.94 | yes | - |

Range over the table: `|u|max = 9.3231e-03 .. 2.5725e-02` lattice units (`0.0161 .. 0.0446` cs). The spurious level **increases with |G_12|** (stronger cohesion -> larger density ratio and larger interfacial force imbalance) and **decreases with resolution** at fixed physical radius, which is the expected pseudopotential behaviour; the platform criterion `spurious < 0.005 U_max` must therefore be evaluated at the resolution and coupling of the target case, not from a single entry of this table.

## 3. Contact angle / interface thickness quick check

Both checks are calibrated against **analytic fields with known answers** (no solver in the loop): a circular cap on a flat wall (R = 25, n = 96, interface width 1.5) and tanh density profiles of prescribed width. The contact angle is measured three ways, independently: the near-wall gradient projection of the geometric method (solid rows extrapolated from the fluid, gradient taken on the 2nd fluid layer), an algebraic circle fit to the mid-density contour, and the upstream base-length/height cap reconstruction.

| theta true | gradient | err | circle fit | err | cap geometry | err | node spread | guardrails |
|---|---|---|---|---|---|---|---|---|
| 30.0 | 25.48 | -4.52 | 30.07 | 0.07 | 28.07 | -1.93 | 24.3 | angle_scatter |
| 45.0 | 41.23 | -3.77 | 44.98 | -0.02 | 44.76 | -0.24 | 19.7 | - |
| 60.0 | 56.03 | -3.97 | 60.00 | -4.0e-05 | 59.49 | -0.51 | 11.7 | - |
| 90.0 | 86.84 | -3.16 | 90.00 | -0.00 | 90.00 | 0.00 | 1.5 | - |
| 120.0 | 117.82 | -2.18 | 119.99 | -0.01 | 118.53 | -1.47 | 7.4 | - |
| 135.0 | 131.80 | -3.20 | 135.00 | 9.4e-04 | 134.57 | -0.43 | 11.3 | - |

Measured accuracy: gradient method max error **4.52 deg** (mean 3.47 deg), circle fit max error **0.070 deg**, cap geometry max error **1.93 deg**. The gradient route carries a systematic positive bias (towards 90 deg) from the wall extrapolation; that is why the circle fit is reported next to it and why the angle must always travel with its `fill_order`/`first_layer` setting.

### Interface thickness (analytic tanh profiles)

| w true | w recovered | 10-90 width | 10-90 true | error | r^2 | points |
|---|---|---|---|---|---|---|
| 1.00 | 0.999 | 2.196 | 2.197 | -0.072% | 0.99994 | 4 |
| 1.50 | 1.508 | 3.314 | 3.296 | 0.558% | 0.99977 | 6 |
| 2.00 | 2.022 | 4.443 | 4.394 | 1.100% | 0.99979 | 8 |
| 3.00 | 3.029 | 6.654 | 6.592 | 0.952% | 0.99990 | 11 |

The 10-90 convention is `2 atanh(0.8) w = 2.1972 w`; the number is meaningless without it, which is why it is stated per row. The same measurement runs on the real MCMP droplets in the spurious-current table above (`10-90 width` column).

## 4. Validity domain (what the tool refuses to report)

The guardrails are only useful if they fire. This table runs couplings *outside* the usable window and records what the tooling does with them (all runs n=64, R_init=14, 6000 steps).

| G_12 | R measured | dp | umax | converged | guardrails |
|---|---|---|---|---|---|
| -2.200 | n/a | 5.901e-06 | 1.227e-06 | no | centre_not_liquid, no_crossing, not_converged |
| -2.300 | n/a | -0.0107 | 0.0049 | yes | centre_not_liquid, no_crossing |
| -3.500 | n/a | n/a | nan | no | non_finite_field, insufficient_profile |

## 5. Consumption chain (OSS inventory entries -> code)

| entry | repo | tag | symbol | where it lands | status |
|---|---|---|---|---|---|
| `21` | lbm-principles-practice/code | scmp-young-laplace-calibrator-with-guardrails | `sc_mcmp_pressure + droplet_radius` | src/tensorlbm/mcmp_verification.py :: sc_mcmp_pressure (pressure), droplet_radius (4-ray crossing with interpolation), measure_laplace (4x4 centre/corner boxes) | re-derived |
| `21` | lbm-principles-practice/code | scmp-young-laplace-calibrator-with-guardrails | `Guardrail + droplet_radius` | src/tensorlbm/mcmp_verification.py :: Guardrail, droplet_radius (returns radius=None + no_crossing / centre_not_liquid / periodic_image_risk / anisotropic), measure_laplace (non_finite_field), _fit_laplace_level (hard-guardrail veto) | defect-fix |
| `20` | lbm-principles-practice/code | scmp-two-component-planar-interface | `sc_mcmp_pressure, init_planar_slab, simulate_planar_slab, estimate_coexistence` | src/tensorlbm/mcmp_verification.py :: sc_mcmp_pressure (pair sum), init_planar_slab / simulate_planar_slab (flat slab), estimate_coexistence(geometry='planar') | re-derived |
| `5` | PorousMediaSimulation/openLBMPM | geometric-wetting-contact-angle | `geometric_contact_angle + _fill_solid` | src/tensorlbm/mcmp_verification.py :: _fill_solid (order=1 mirror / order=2 linear extrapolation), _near_wall_fluid, geometric_contact_angle (cos theta = -(grad rho . n_wall)/|grad rho|) | re-derived |
| `5` | PorousMediaSimulation/openLBMPM | geometric-wetting-contact-angle | `cap_geometry_contact_angle` | src/tensorlbm/mcmp_verification.py :: cap_geometry_contact_angle (b, h -> R = (4h^2+b^2)/(8h), theta = atan2(b, 2(R-h))) | adapted |
| `5` | PorousMediaSimulation/openLBMPM | geometric-wetting-contact-angle | `(not consumed as code)` | design reference for the future wetting BC; recorded in src/tensorlbm/mcmp_verification.py :: cap_geometry_contact_angle docstring context and report section 6 limitation | design-reference |
| `281` | lssfau/walberla | two-independent-contact-angle-impls | `interface_thickness + interface_width_10_90` | src/tensorlbm/mcmp_verification.py :: interface_thickness (10-90 width and tanh-fit w reported together), interface_width_10_90 (documented 2 atanh(0.8) conversion) | inventory-only |
| `1` | PorousMediaSimulation/openLBMPM | porter-explicit-forcing-4th-order | `FORCE_STENCIL` | src/tensorlbm/mcmp_verification.py :: FORCE_STENCIL (recorded in every spurious-current row) and spurious_current_benchmark docstring | inventory-only |
| `44` | sorush-khajepor/listLBM | shan-chen-mcmp-velocity-mixing | `barycentric_velocity` | src/tensorlbm/mcmp_verification.py :: barycentric_velocity (explicit mass averaging, clamped denominator), spurious_current_benchmark (Ca_sp reported with mu based on rho_bar) | inventory-only |

Every row is checked twice: the inventory text is read from the platform `oss_solutions` table (repo/tag/method bullet), and the upstream source quoted in that row is read from the local OSS cache before the formula is re-derived for this library's sign and pseudopotential conventions. `status=re-derived` means the upstream formula was **not** copied: the library convention differs and the adaptation is documented in the module docstring.

## 6. Known limits of these measurements

* The pressure relation is only valid for the **linear** pseudopotential that this library's MCMP force kernel uses; a non-linear psi (as in the single-component path) needs its own coefficient.
* The contact-angle tool supports flat walls only (a single solid strip); curved or rough walls return a guardrail instead of an angle.
* The contact angle of a *real* MCMP wall droplet is not measured here: the library's two-component path has no wetting boundary condition yet (dry wall only), so the tool is calibrated on analytic caps and is ready for the wetting rework, not proof of one.
* Spurious currents are reported at `tau = 1` only; they are known to depend on viscosity, so a cross-viscosity study is a separate run.

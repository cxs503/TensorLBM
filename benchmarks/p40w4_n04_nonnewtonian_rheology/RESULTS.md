# P40W4-N04 results (production mode)

generated 2026-11-10T12:09:30 · wall time 691 s · torch 2.7.1 on loongarch64

## 1. Solver control case (Newtonian)

| quantity | value |
|---|---|
| grid gap h | 32 |
| u_max relative error vs parabola | -0.008 % |
| profile L2 relative error | +0.039 % |
| steps / converged | 1500 / True |

## 2. Power law (closed-form Ostwald-de Waele anchor)

| n | h | K | L2 vs closed form | u_max err | flow-rate err | steps | converged |
|---|---|---|---|---|---|---|---|
| 0.7 | 32 | 0.017566 | +0.164 % | +0.235 % | -0.014 % | 1500 | True |
| 1.3 | 32 | 0.62602 | +0.017 % | -0.024 % | +0.028 % | 4500 | True |

## 3. Carreau-Yasuda and Cross (constitutive-inversion anchor)

| model | h | profile L2 | u_max err | flow-rate err | viscosity-curve max dev | closure round-trip | steps |
|---|---|---|---|---|---|---|---|
| Carreau-Yasuda | 48 | +0.146 % | +0.113 % | +0.182 % | +0.169 % | +0.000 % | 1000 |
| Cross | 48 | +0.095 % | +0.083 % | +0.039 % | +1.040 % | +0.000 % | 1500 |

## 4. Ice-slurry Herschel-Bulkley plug

| quantity | value |
|---|---|
| h / tau_y / K / n / m | 48 / 0.0003569 / 0.02 / 0.6 / 10000 |
| ideal plug half-width tau_y/(rho a) | 8.4000 |
| LBM fit detector | 8.0986 (-3.588 %) |
| LBM stress detector | 4.7905 (-42.971 %) |
| LBM velocity-threshold detector | 3.0479 (-63.716 %) |
| anchor fit-detector bias (regularisation) | -0.378 % |
| anchor velocity-threshold bias | -25.806 % |
| LBM vs anchor, fit detector (discretisation) | -3.222 % |
| HB profile L2 vs anchor | +6.793 % |

## 5. Grid convergence (power law n = 0.7)

drive calibrated on u_max = 0.005 (Ma^2 = 7.50e-05).

| grid gap h | L2 relative error |
|---|---|
| 16 | +0.6122 % |
| 32 | +0.1642 % |
| 64 | +0.0315 % |
| 128 | +0.0256 % |

| pair | two-grid Richardson order |
|---|---|
| 16->32 | 1.899 |
| 32->64 | 2.380 |
| 64->128 | 0.299 |

drive-floor diagnosis (h = 64): L2 = +0.0315 % at u_max = 0.005 vs +0.0510 % at u_max = 0.002; compressibility scale 2·Ma² = 1.50e-04

asymptotic range = [16, 32, 64, 128] · **fitted order = 1.611**

## 6. Gates

| id | worst | threshold | passed |
|---|---|---|---|
| G1_power_law_profile | 0.001635 | 0.03 | True |
| G2_carreau_yasuda | 0.00169 | 0.05 | True |
| G3_cross | 0.0104 | 0.05 | True |
| G4_ice_slurry_plug | 0.03588 | 0.05 | True |
| G5_grid_convergence_order | 1.611 | 2.0 | False |
| G6_pytest | 20 | 10 | True |

**all_passed = False**


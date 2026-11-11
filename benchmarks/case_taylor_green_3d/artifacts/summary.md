# CASE-TAYLOR-GREEN-3D — generated evidence tables

## 0. Reference exactness (machine-checked, spectral derivatives)

| case | curl(C)/\|C\| | convective term is a gradient | \|lap(u)+k2u\|/\|k2u\| | NS residual (rel) | exact solution |
|---|---|---|---|---|---|
| tg | 2.707e-15 | yes | 1.593e-14 | 1.593e-14 | u(t) = u0 * exp(-nu*0.0771063*t) |
| tg3d | 0.227 | **no** | n/a | n/a | **none (no closed form)** |
| abc | 6.530e-15 | yes | 2.342e-14 | 2.342e-14 | u(t) = u0 * exp(-nu*0.0385531*t) |

ABC is Beltrami: `||curl(u)-k u||/||curl(u)||` = 2.14e-15 (machine-checked, N=32).

## 1. Grid convergence (M4)

| N | k=2π/N | \|k\|² | L2 rel. error | observed order vs previous | wall [s] |
|---|---|---|---|---|---|
| 24 | 0.26180 | 0.137078 | 1.1450e-02 |  | 22.0 |
| 32 | 0.19635 | 0.077106 | 6.3281e-03 | 0.855 | 46.5 |
| 48 | 0.13090 | 0.034269 | 2.7741e-03 | 1.190 | 119.6 |
| 64 | 0.09817 | 0.019277 | 1.5539e-03 | 0.836 | 252.1 |

M4 observed order (24→48) = **2.045** [OK] (target ≥ 1.8); 24→32 = 0.855, 48→64 = 0.836.

## 3. τ sensitivity (S1)

| τ | ν=(τ-0.5)/3 | Re=A/(ν√\|k\|²) | λ_exact=2ν\|k\|² | λ_fit | rel. error | |
|---|---|---|---|---|---|---|
| 0.55 | 0.01667 | 0.43 | 2.570209e-03 | 2.587296e-03 | +6.648e-01% | [OK] |
| 0.60 | 0.03333 | 0.22 | 5.140419e-03 | 5.157556e-03 | +3.334e-01% | [OK] |
| 0.70 | 0.06667 | 0.11 | 1.028084e-02 | 1.030054e-02 | +1.916e-01% | [OK] |
| 0.80 | 0.10000 | 0.07 | 1.542126e-02 | 1.545012e-02 | +1.872e-01% | [OK] |

## 4. Amplitude / Mach sensitivity (S2)

| A | Ma=A√3 | M1 rel. error | M2 rel. error | M3 L2 rel. error |
|---|---|---|---|---|
| 1.0e-03 | 0.0017 | -1.255e+00% | +3.334e-01% | 6.298e-01% |
| 2.0e-03 | 0.0035 | -1.255e+00% | +3.334e-01% | 6.303e-01% |
| 1.0e-02 | 0.0173 | -1.257e+00% | +3.327e-01% | 6.463e-01% |
| 2.0e-02 | 0.0346 | -1.261e+00% | +3.305e-01% | 6.939e-01% |

## 5. z-dependent exact companion: Beltrami/ABC flow

| N | |k|² | M1 energy rel. error | M2 exponent rel. error | M3 L2 rel. error |
|---|---|---|---|---|
| 32 | 0.038553 | -1.403e+00% | +3.725e-01% | 7.070e-01% |
| 48 | 0.017135 | -4.304e-01% | +2.015e-01% | 2.233e-01% |

## Figures

- `runs/case_taylor_green_3d/figures/grid_convergence.png`
- `runs/case_taylor_green_3d/figures/tau_sensitivity.png`
- `runs/case_taylor_green_3d/figures/amp_sensitivity.png`
- `runs/case_taylor_green_3d/figures/abc_decay.png`


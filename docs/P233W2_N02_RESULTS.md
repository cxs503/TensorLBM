# P233W2-N02 generated results

> Auto-generated from `artifacts/ibm_ice_propeller/verdicts.json` (sha256 `5eff9919b4996c58270964dff465c757193a71e6788e5b19e8c7023d68531188`) by `benchmarks/ibm_ice_propeller/make_report.py`. No number in this file is typed by hand; re-run the driver to regenerate.

## Results

### 3.1 Cylinder loads versus literature (AC2, AC5)

| case | D [cells] | Cd | Cl (rms or mean) | Cd error | Cl error | St |
|---|---|---|---|---|---|---|
| fixed Re=100 | 8 | 1.3103 | 0.0002 | 2.22 % | 99.99 % | 0.1143 |
| fixed Re=100 | 12 | 1.2777 | 0.0000 | 4.65 % | 100.00 % | 1.4400 |
| fixed Re=100 | 16 | 1.2589 | 0.0000 | 6.05 % | 100.00 % | 0.1280 |
| rotating Re=100 | 8 | 1.4965 | 1.6963 | 37.29 % | 303.88 % | 0.1600 |
| rotating Re=100 | 12 | 1.4745 | 1.4361 | 35.27 % | 241.92 % | 0.1500 |
| rotating Re=100 | 16 | 1.4717 | 1.3159 | 35.02 % | 213.32 % | 0.1600 |

Reference values: fixed cylinder Cd = 1.34, Cl' = 0.325, St = 0.1645 (Cd = 1.34 (Tritton 1959; Roscoe 1950 benchmark value quoted in Uhlmann 2005); Cd = 1.325, Cl' = 0.325, St = 0.1645 from Ding, Shu, Yeo & Xu (2007) Phys. Fluids 19:016701, see also Braza, Chassaing & Ha Minh (1986) JFM 165:79); rotating cylinder Cd = 1.09, |Cl| = 0.42 (Stojkovic, Breuer & Durst (2002) Phys. Fluids 14:3160, Table 2 (Re = 100, spin ratio 0.5): Cd = 1.09, |Cl| = 0.42; secondary Kang, Choi & Lee (1999) Phys. Fluids 11:3312 (Cd = 1.13, |Cl| = 0.35). The lift sign follows the rotation-direction convention, so magnitudes are compared.).

| acceptance check | Cd error | Cl error | verdict |
|---|---|---|---|
| fixed cylinder Re=100 (finest grid) | 6.05 % | 100.00 % | PASS |
| rotating cylinder Re=100 alpha=0.5 (finest grid) | 35.02 % | 213.32 % | FAIL |

### 3.2 Momentum conservation (AC4)

| audit | measured | threshold |
|---|---|---|
| prescribed moving body in a periodic box (400 steps) | 5.675e-15 | 1e-6 |
| free body, fluid + solid momentum drift | 7.162e-15 | 1e-6 |

### 3.3 Grid convergence (AC5)

| quantity | grids | errors | monotone decreasing |
|---|---|---|---|
| fixed_cd | [8, 12, 16] | 0.0222, 0.0465, 0.0605 | no |
| fixed_cl | [8, 12, 16] | 0.9999, 1.0000, 1.0000 | no |
| rotating_cd | [8, 12, 16] | 0.3729, 0.3527, 0.3502 | yes |
| rotating_cl | [8, 12, 16] | 3.0388, 2.4192, 2.1332 | yes |

### 3.4 Ice load time history (AC3)

| quantity | value |
|---|---|
| peak contact load | 5.1091 |
| peak hydrodynamic load | 4.2456 |
| peak total load | 9.1102 |
| hydrodynamic Cd measured (pre-contact) | 1.5930 |
| literature Cd at the same Re | 1.4080 |
| relative error | 13.14 % |
| impulse-momentum residual | 3.357e-15 |
| contact chord at peak load [cells] | 23.734 |

Literature series status: Liu et al. (2019) propeller-ice milling load series NOT available offline (no network); substituted by (i) hydrodynamic baseline vs literature Cd at the same Re, (ii) exact momentum/impulse audit (residual reported), (iii) Korzhavin crushing-bound geometry.

### 3.5 Drag coefficient versus Reynolds number (AC6)

| Re | Cd (IBM) | Cd (literature) | error |
|---|---|---|---|
| 50 | 1.6698 | 1.5 | 11.32 % |


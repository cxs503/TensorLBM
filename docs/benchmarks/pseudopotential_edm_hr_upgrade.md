# Pseudopotential (SCMP) force-format upgrade: EDM forcing + high-order isotropic gradients, and the Hadamard-Rybczynski settling test

**Campaign node** `p97w3-n04` (TensorLBM).  Code: `src/tensorlbm/pseudopotential_edm3d.py`, benchmark: `benchmarks/pseudopotential_edm_hr_upgrade/`, tests: `tests/test_pseudopotential_edm3d.py` (38 cases).

## 1. Case introduction

A single-component Shan-Chen (SCMP) lattice-Boltzmann engine with an exponential pseudopotential `psi(rho) = 1 - exp(-rho)` and coupling `G_eff = -5` is used in two configurations:

* **static droplet** -- Young-Laplace law `dp = 2 sigma / R` and the spurious-current floor, radii `R = 6 / 8 / 10`, box `5.33 R` across;
* **buoyant droplet** -- terminal settling velocity against the Hadamard-Rybczynski (1911) law

  `U_HR = 2 (rho_l - rho_v) g R^2 (1 + lambda) / (3 mu_v (2 + 3 lambda))`, `lambda = mu_l/mu_v`, with the viscosity ratio imposed per cell through `tau(x) = 1/2 + mu(phi(x))/(rho cs^2)` so that `lambda` is independent of the density ratio.

| parameter | value | note |
|---|---|---|
| lattice | D3Q19, `cs^2 = 1/3` | float64 throughout |
| coupling | `G_eff = -5` (`G_lib = +5`) | coexistence `rho_l = 1.957`, `rho_v = 0.1596` (repo-verified) |
| interface | tanh init, half-width 2 | equilibrium width set by the EOS |
| body force | `(rho - <rho>) g zhat` | mean removed: no net momentum in the periodic box |
| vapour relaxation | `tau_v = 2.0` -> `mu_v = rho_v cs^2 (tau_v - 1/2)` | sets the viscosity scale |
| `lambda` ladder | 0.1 / 1 / 10 | `tau_l` from 0.53 to 6.3 |
| `R` ladder | 6 / 8 / 10 | box 4.4 R (settling), 5.33 R (Laplace) |

## 2. What was done

Three force-format axes were separated and measured instead of asserted:

1. **forcing scheme** — `velocity_shift` (production: `feq(rho, u + tau F/rho)`) vs the Kupershtokh exact-difference method (`feq(rho, u + F/rho) - feq(rho, u)`), which is momentum exact (`sum_i c_i df_i = F`, asserted in the tests);
2. **gradient isotropy** — the D3Q19 weight set (rank-2/4 isotropic, rank-6 residual `sum W c_x^6 - 5 sum W c_x^4 c_y^2 = -2/3`) vs a 32-neighbour set (`|c|^2 = 1,2,3,4`, weights 2/15, 1/15, 1/60, 1/120) solved *exactly* from the four rank-6 isotropy conditions over the rationals;
3. **spatial form** — `product` vs `difference`: tested, and found to be algebraically *identical* on every symmetric stencil because `sum_i W_i c_i = 0`.  This is reported as a negative result rather than claimed as an upgrade.

Toolchain: PyTorch float64, exact-rational lattice constants (the library's `W` constant is float32 and casting it to float64 leaves a 1e-8 weight error which breaks mass conservation at the 1e-8 per-collision level), `pytest` for the algebra, and this benchmark for the physics.  Every case writes its own JSON, so the campaign is resumable.

## 3. Results

### 3.1 Young-Laplace law and spurious currents

| format | stencil / forcing | `sigma_fit` | `R^2` of `dp` vs `1/R` | max `sigma_i` deviation | `max|u|` interface | `max|u|` core |
|---|---|---|---|---|---|---|
| F0_classic | d3q19 / velocity_shift | 0.0113 | 0.755206 | 366.405% | 0.134 | 0.003 |
| F3_edm_iso6 | iso6_32 / kupershtokh | -0.0040 | 0.201629 | 1441.291% | 0.124 | 0.004 |

Per-radius detail (upgraded format F3):

| `R0` | `R_eq` | `dp` | `sigma_i` | dev vs fit |
|---|---|---|---|---|
| 6.000 | 5.5601 | 0.0106 | 0.0296 | -839.418% |
| 8.000 | 7.6888 | 0.0120 | 0.0461 | -1252.692% |
| 10.000 | 9.7144 | 0.0110 | 0.0536 | -1441.291% |

### 3.2 Why the Laplace fit is noise-limited (closed-box density offset)

| format | `rho_l` at R = 6/8/10 | scatter | `dp/drho_l` | `dp` uncertainty | `2 sigma/R` range | signal/noise |
|---|---|---|---|---|---|---|
| F0_classic | 2.0852 / 2.0531 / 2.0670 | 0.032 | 0.149 | 0.005 | 0.006 .. 0.010 | 1.68 |
| F3_edm_iso6 | 2.0863 / 2.0563 / 2.0721 | 0.030 | 0.150 | 0.004 | 0.006 .. 0.011 | 1.84 |

In a closed periodic box the vapour density, and through the EOS the liquid density, depend on the drop size.  A scatter of a few 1e-3 in `rho_l` maps to a bulk-pressure uncertainty comparable to the whole curvature term `2 sigma/R`, which is why `R^2` of the `dp` vs `1/R` fit cannot reach the 0.999 target at these radii.  Fixing the vapour density (open reservoir) or measuring the pressure tensor directly would be required.

### 3.3 Periodic-image (hindered settling) study

| box `n` | `phi^{1/3}` | `U_rel` |
|---|---|---|
| 26 | 0.3423 | 0.0508 |
| 36 | 0.7937 | -2.174e-07 |
| 44 | 0.7936 | -4.062e-07 |

Fit: `U = U_inf (1 - k phi^(1/3))` with `U_inf = 0.0893`, `k = 0.1125`, `R^2 = 1.00000`.  The fit is applied as the confinement correction below; `k` is *measured*, not taken from the literature.

### 3.4 Settling vs Hadamard-Rybczynski

**F3_edm_iso6**

| `lambda` | `R` | `R_eq` | `U_rel` | `U_last` | `U_HR` | err (raw) | err (box-corrected) | `Re` | `max|u|/U_HR` | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 6.000 | 5.5153 | 0.0504 | 0.0798 | 0.0066 | +660.85% | +691.28% | 0.556 | 18.550 | **no** |
| 0.100 | 8.000 | 7.5269 | 0.0193 | 0.0353 | 0.0072 | +167.06% | +177.90% | 0.291 | 17.442 | **no** |
| 0.100 | 10.000 | 9.5007 | 0.0034 | 0.0143 | 0.0076 | -55.69% | -53.88% | 0.064 | 16.772 | **no** |
| 1.000 | 6.000 | 5.5216 | 0.0508 | 0.0793 | 0.0066 | +666.05% | +696.72% | 0.561 | 18.521 | **no** |
| 1.000 | 8.000 | 7.5292 | 0.0206 | 0.0360 | 0.0072 | +184.89% | +196.45% | 0.311 | 17.427 | **no** |
| 1.000 | 10.000 | 9.4954 | 0.0049 | 0.0156 | 0.0076 | -35.93% | -33.32% | 0.093 | 16.777 | **no** |
| 10.000 | 6.000 | 5.5738 | 0.0246 | 0.0384 | 0.0067 | +266.11% | +280.91% | 0.274 | 18.523 | **no** |
| 10.000 | 8.000 | 7.5513 | 0.0087 | 0.0183 | 0.0073 | +19.81% | +24.69% | 0.132 | 17.662 | **no** |
| 10.000 | 10.000 | 9.5262 | 0.0003895 | 0.0057 | 0.0077 | -94.91% | -94.70% | 0.007 | 17.114 | **no** |

**F0_classic**

| `lambda` | `R` | `R_eq` | `U_rel` | `U_last` | `U_HR` | err (raw) | err (box-corrected) | `Re` | `max|u|/U_HR` | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.100 | 8.000 | 8.0300 | 0.0134 | 0.0276 | 0.0083 | +61.65% | +68.66% | 0.215 | 18.840 | **no** |
| 1.000 | 8.000 | 7.7861 | 0.0181 | 0.0330 | 0.0078 | +131.04% | +140.75% | 0.282 | 19.476 | **no** |
| 10.000 | 8.000 | 17.2328 | -1.557e-09 | -1.484e-14 | 5.262e-11 | +2858.09% | +3148.00% | 5.36e-08 | 69.297 | yes |

### 3.5 Resolution extrapolation `R -> inf` (Richardson)

| `lambda` | fitted order `p` | three-point `p` | `U(R->inf)` | `U_HR` at the ladder mean | err at `R->inf` | err at `R=16` (from fit) |
|---|---|---|---|---|---|---|
| 0.1 | 1.590 | n/a | -0.0355 | 0.0072 | -595.480% | -337.515% |
| 1 | 1.531 | n/a | -0.0352 | 0.0072 | -590.953% | -317.308% |
| 10 | 1.510 | n/a | -0.0213 | 0.0072 | -394.551% | -246.947% |

### 3.6 `Re -> 0` ladder (body-force extrapolation, `lambda = 1`, `R = 6`)

| `g/g0` | `U_rel` | `U` (box-corrected) | `U_HR` | `Re` | err |
|---|---|---|---|---|---|
| 0.400 | 0.0117 | 0.0122 | 0.0027 | 0.129 | +358.84% |
| 1.000 | 0.0508 | 0.0528 | 0.0066 | 0.561 | +696.72% |
| 2.500 | -0.0201 | 0.0209 | 0.0166 | 0.221 | +25.95% |

Linear fit `U(Re) = a + b Re` gives `a = -2.542e-06` (error vs HR -100.030%) and `b = 0.0942`.

## 4. Figures

![fig1_laplace.png](artifacts/pseudopotential_edm_hr_upgrade/figures/fig1_laplace.png)

*Young-Laplace law for the three force formats: `dp` vs `1/R_eq` with the linear fit `dp = 2 sigma/R`.*

![fig2_spurious.png](artifacts/pseudopotential_edm_hr_upgrade/figures/fig2_spurious.png)

*Spurious-current floor of the static droplet (interface band, mean over R = 6/8/10); annotations are ratios to the production format.*

![fig3_settling.png](artifacts/pseudopotential_edm_hr_upgrade/figures/fig3_settling.png)

*Left: terminal settling velocity vs radius, dotted = the analytic HR law.  Right: relative error vs HR, with the 15% criterion band and the `R=16` extrapolated point.*

![fig4_box.png](artifacts/pseudopotential_edm_hr_upgrade/figures/fig4_box.png)

*Periodic-image study: terminal velocity vs `phi^(1/3)` and the fitted `U_inf (1 - k phi^(1/3))`.*

![fig5_fields.png](artifacts/pseudopotential_edm_hr_upgrade/figures/fig5_fields.png)

*Relaxed droplet: density slice and `log10|u|` slice showing that the spurious recirculation is confined to the interface.*

## 5. Pitfalls, scope and limitations

1. **The spatial "exact difference" form is a no-op.**  `sum_i W_i c_i = 0` holds identically on every symmetric stencil, so `sum W c [psi(x+c) - psi(x)]` equals `sum W c psi(x+c)` exactly.  Only the *forcing* scheme and the *isotropy* of the weight set can change anything.  The tests assert this rather than hiding it.
2. **The mechanical pressure must be sampled away from the interface.**  `p = cs^2 rho + (G_eff/2) cs^2 psi^2` dips strongly negative where `psi^2` peaks (inside the diffuse interface), so an "outside" average that includes interface cells yields a *negative* Laplace pressure.  The reported `dp` is core (`r < 0.6 R_eq`) minus far field (`r > 1.5 R_eq`).
3. **The production spurious current is large.**  `max|u|` in the interface band reaches ~1e-1 lattice units at density ratio ~12, while the core stays at ~1e-3.  The settling velocity is therefore measured as the drop's *rigid-body* drift over the core shell, and the interface band is excluded from the average.
4. **Periodic images.**  In a periodic box the droplet's images hinder its settling; the effect is linear in `phi^(1/3)` and is measured here (`k` above) rather than assumed.  The raw (`U_rel`) and corrected (`U_corr`) errors are both reported.
5. **Finite Reynolds number.**  `Re` at the reported runs is 0.561, so the measured velocity contains an Oseen-type inertial correction; the body-force ladder extrapolates to `Re -> 0`.
6. **The measured drift is not a settling velocity.**  In the settling runs the drop-core drift does not saturate: it decays for ~100 steps and then grows monotonically (the `U_last` column is systematically larger than the window mean `U_rel`, and the `converged` column is mostly `no`).  A dedicated diagnostic on the same engine shows the core drift reversing sign at step ~150 and then growing at ~4e-5/step while the total momentum of the box stays at the 1e-13 level -- the signal is buried under the interface micro-current, it is not a viscous settling response.  These numbers are reported as measured; they are **not** a measurement of the HR law.
7. **The box-size ladder is degenerate.**  Measured `U_rel` at `phi^(1/3)` = 0.342: 0.051, 0.794: -2.17e-07, 0.794: -4.06e-07 -- the drift collapses towards zero in the two larger boxes, so the fitted hindrance coefficient `k = 0.1125` (R^2 = 1.000 on 3 points) cannot be used as a physical correction.  The box-corrected errors are quoted for completeness only; the raw errors carry the same verdict.  The `Re -> 0` ladder is degenerate for the same reason (it extrapolates to `-2.542e-06`, i.e. -100.030%).
8. **Scope.**  One EOS (`psi = 1 - exp(-rho)` at `G_eff = -5`), one lattice (D3Q19), `tau_v = 2`.  Larger density ratios (lower `tau_v`) raise the spurious currents further and make `lambda = 0.1` numerically unattainable (`tau_l -> 1/2`).

## 6. Error judgement (key quantities vs baseline)

| quantity | reference | this work | error | < 3%? |
|---|---|---|---|---|
| `sigma_i` consistency (F0_classic) | the format's own fit | max dev 366.405% | +366.40% | **no** |
| `sigma_i` consistency (F3_edm_iso6) | the format's own fit | max dev 1441.291% | +1441.29% | **no** |
| Laplace linearity (F3) | `R^2 = 1` | `R^2 = 0.201629` | `1 - R^2 = 0.798` | **no** |
| settling `U` vs HR (F3_edm_iso6, box-corrected, worst case) | HR analytic | +696.72% | +696.72% | **no** |
| settling `U` vs HR (F0_classic, box-corrected, worst case) | HR analytic | +3148.00% | +3148.00% | **no** |

**Answer to "is it below 3%?"**: the Laplace-law self-consistency of the upgraded format and the spurious-current *reduction factor* are the only quantities quoted against a machine-checkable reference; the settling velocity is **not** within 3% (nor 15%) of the Hadamard-Rybczynski law at the resolutions affordable here, and this is reported as an honest FAIL of the 15% criterion rather than smoothed over.

## 7. Criteria verdicts (machine-checked)

| criterion | pass | values |
|---|---|---|
| `C1_hr_err_lt_15pct_at_R16` | FAIL | `{"values_pct": {"lam0.1": -337.5149885205343, "lam1": -317.30838846800344, "lam10": -246.94677513384696}}` |
| `C2_laplace_r2_gt_0p999` | FAIL | `{"values": {"F0_classic": 0.7552056092669074, "F3_edm_iso6": 0.20162904719536012}}` |
| `C3_spurious_lt_1e-3` | FAIL | `{"u_max_iface": {"F0_classic": 0.13359257622185827, "F3_edm_iso6": 0.12369833016115506}, "U_char_median_hr": 0.007266097278928029}` |
| `C4_richardson_order_ge_1` | PASS | `{"orders": {"lam0.1": 1.5897500000000002, "lam1": 1.5307500000000003, "lam10": 1.5101000000000002}}` |
| `C5_pytest` | PASS | `{"cases": 38}` |

**Reading the verdicts.**  `C4` is satisfied only in the narrow numerical sense that the fitted exponent of the `1/R` sequence is >= 1: the sequence itself is not a convergent settling-velocity series (the extrapolated limit is off by ~590%), so the order is a fit to a non-physical trend and carries no predictive value.  `C1`, `C2` and `C3` fail with the numbers above; the mechanism is quantified in sections 3.2 and 5.

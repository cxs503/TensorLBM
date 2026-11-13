### N1 — E1 formulation conformance (ψ from OSS source text vs TensorLBM)

| $T_r$ | T | compared ρ-range | n | max rel. dev (c₀=cs²) | c₀=3 map: ψ ratio | pass 1e-6 |
|---|---|---|---|---|---|---|
| 0.5500 | 0.0401 | 0.001–9.5 | 800 | 2.22e-16 | 3.0000 ± 2.66e-16 | ✔ |
| 0.7500 | 0.0547 | 0.001–9.4 | 795 | 2.20e-16 | 3.0000 ± 2.74e-16 | ✔ |
| 0.8266 | 0.0603 | 0.001–9.29 | 790 | 2.22e-16 | 3.0000 ± 2.83e-16 | ✔ |
| 0.9500 | 0.0693 | 0.001–9.1 | 781 | 2.21e-16 | 3.0000 ± 2.78e-16 | ✔ |

Worst deviation over all temperatures: **2.22e-16** (float64 eps = 2.22e-16); prereg gate 1e-6 → **PASS**.

### N2 — OSS snapshot facts (parsed, zero hand-copy)

| item | value |
|---|---|
| OSS HEAD | `3d841891daaf` (3d841891daafe7b6e9f65e51dd9bf20f2b0e1792 2018-10-15 Merge branch 'master' of https://github.com/VadarSkyWalker/openLBMPM) |
| P-R kernel | `calFluidPotentialGPUPR` in `ShanChen2D/OptimizedD2Q9GPU.py` |
| p_EOS (verbatim, de-continued) | `(fluidRho[i, indices] * constR *  temperatureT) / (1. -                     coeffB * fluidRho[i, indices]) - (coeffA * c…` |
| ψ (verbatim) | `sqrt(2. / (constC0 * constG) * (tmpP -                           1./3. * fluidRho[i, indices]))` |
| call sites | `{'calFluidPotentialGPUPR': 0, 'calEffectiveMassPR': 0}` |
| wiring verdict | kernel-only (dead code): zero call sites, no parameter plumbing |
| active potential | `['fluidPotential[i, indices] = fluidRho[i, indices]']` (potentialType = 'Simple') |
| OSS macro-pressure term | `[i, j] * fluidRho[i, indices] *                             fluidRho[j, indices]` |

### N3 — interaction-stencil isotropy moments (weights eval-ed from OSS literals)

| stencil | points | Σw | Σw·c_x² | A4 | A6 | A8 |
|---|---|---|---|---|---|---|
| OSS_iso4_8pt_E_N_W_S_NE_NW_SW_SE | 8 | 1.6667 | 1.0000 | 1.0000 | 0.6000 | 0.4286 |
| TensorLBM_D2Q9 | 9 | 1.0000 | 0.3333 | 1.0000 | 0.6000 | 0.4286 |

(isotropic value of A4/A6/A8 is 1; both stencils are 4th-order isotropic and identical beyond it, i.e. the OSS `iso4` shell buys nothing over D2Q9.)

### N4 — E2 thermodynamic accuracy vs analytic PR-Maxwell (production cells)

| run | geometry | L | steps | stable | ρ_l | ρ_v | ratio | ratio err | ρ_l err | ρ_v err | worst | <15 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local_planar_L48_ratio10 | planar | 48 | 20000 | ✔ | 5.8644 | 0.5371 | 10.9184 | 0.0908 | 0.0038 | 0.0867 | 0.0908 | ✔ |
| planar_L64_ratio10 | planar | 64 | 12000 | ✔ | 5.8676 | 0.5339 | 10.9893 | 0.0979 | 0.0032 | 0.0921 | 0.0979 | ✔ |
| planar_L64_mid_tr075 | planar | 64 | 12000 | ✔ | 6.5703 | 0.2821 | 23.2936 | 0.2492 | 0.0027 | 0.2016 | 0.2492 | ✘ |

### N5 — E2 reference values (PR equal-area Maxwell construction)

| tier | T_r | T | ρ_l | ρ_v | ratio | p_sat | ρ_cross | |area residual| |
|---|---|---|---|---|---|---|---|---|
| ratio10 | 0.8266 | 0.0603 | 5.8867 | 0.5881 | 10.0096 | 0.0248 | 9.2928 | -9.99e-16 |
| mid | 0.7500 | 0.0547 | 6.5879 | 0.3533 | 18.6474 | 0.0152 | 9.4091 | 3.08e-15 |
| highT | 0.9500 | 0.0693 | 4.3029 | 1.3636 | 3.1556 | 0.0475 | 9.1033 | -2.13e-15 |

### N6 — E2b pressure read-out from the measured plateaus

| run | p(ρ_l) | p(ρ_v) | p_sat (Maxwell) | p_l err | p_v err | plate mismatch |
|---|---|---|---|---|---|---|
| local_planar_L48_ratio10 | 0.0230 | 0.0234 | 0.0248 | 0.0732 | 0.0567 | 0.0164 |
| planar_L64_ratio10 | 0.0233 | 0.0233 | 0.0248 | 0.0626 | 0.0604 | 0.0022 |
| planar_L64_mid_tr075 | 0.0128 | 0.0128 | 0.0152 | 0.1618 | 0.1609 | 9.41e-04 |

### N8 — verdict under the preregistered rules

- **D1** = A0-conformant
- **D2** = A0-accurate
- **primary_cell** = planar_L64_ratio10
- **primary_worst_err** = 0.09787752345405071
- **laplace_sigma** = None
- **laplace_residual** = None
- **note** = D2 is applied to the finest planar grid at the ratio-10 tier (prereg §7).

### N9 — figures

- `fig1_psi_conformance.png` — fig1 ψ conformance
- `fig2_planar_profiles.png` — fig2 planar profiles
- `fig3_convergence.png` — fig3 convergence & sensitivity
- `fig4_droplet_fields.png` — fig4 droplet fields

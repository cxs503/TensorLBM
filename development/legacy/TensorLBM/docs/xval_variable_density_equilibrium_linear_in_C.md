# XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C

Cross-validation of the **variable-density equilibrium distribution** where density
depends linearly on concentration:

    rho(C) = rho_L + C * (rho_H - rho_L)
    f_i^eq = w_i * rho(C) * [1 + 3(e_i.u) + 9/2(e_i.u)^2 - 3/2(u.u)]

## 1. Problem and Reference Provenance

**External anchor**: Walberla PhaseField Allen-Cahn multiphase (lssfau/walberla,
`apps/showcases/PhaseFieldAllenCahn/CPU/multiphase_codegen.py`); Krüger et al.
"The Lattice Boltzmann Method" (2017), Ch. 9 on multiphase/multicomponent LBM.

**Physics**: For variable-density Poiseuille flow with constant kinematic viscosity
ν, the steady-state momentum equation is:

    d²u/dy² = -g/ν

which is **independent of density**. The velocity profile is the standard parabolic
Poiseuille profile regardless of density stratification (when ν is constant).

**Test**: Does the LBM with variable-density equilibrium recover this profile?

## 2. Method

Two density modes tested:
- **Fixed**: equilibrium uses prescribed ρ_field from concentration C
- **Evolving**: equilibrium uses actual ρ from distributions (sum of f_i)

D2Q9 lattice, BGK collision (τ=0.8), Guo forcing (second-order), float64,
body force acceleration g=1e-5, ν=0.1.

## 3. Parameters

| symbol | value |
|---|---|
| lattice | D2Q9 |
| relaxation time | τ = 0.8 |
| kinematic viscosity | ν = 0.1 |
| body force | g = 1e-5 |
| speed of sound | c_s = 1/√3 ≈ 0.577 |
| channel height | H = 64 fluid rows |
| max analytic velocity | u_max = g·H²/(8ν) = 0.0512 |

## 4. Results

| case | ρ_H/ρ_L | mode | L2 err | Max err | Ma | ρ drift | steps | verdict |
|---|---|---|---|---|---|---|---|---|
| C1_const_density | 1.00 | N/A | 0.065% | 0.087% | 0.0886 | 0.00e+00 | 30000 | **PASS** |
| C2_var_density_ratio1.1_fixed | 1.10 | fixed | 55.124% | 79.203% | 0.0213 | 4.23e-16 | 13900 | FAIL |
| C3_var_density_ratio1.1_evolving | 1.10 | evolving | 3.540% | 4.845% | 0.0844 | 1.98e-12 | 30000 | **PASS** |
| C4_var_density_ratio2_fixed | 2.00 | fixed | DIVERGED | DIVERGED | - | - | 146 | DIVERGED |
| C5_var_density_ratio2_evolving | 2.00 | evolving | 24.392% | 33.392% | 0.0591 | 2.08e-12 | 30000 | FAIL |

## 5. Convergence / Sensitivity

- **Constant density (C1)**: L2=0.065%, max=0.087% — excellent agreement with analytic
- **Variable density, evolving, ρ_H/ρ_L=1.1**: L2=3.54%, max=4.85% — PASS (<15%)
- **Variable density, evolving, ρ_H/ρ_L=2.0**: L2=24.4%, max=33.4% — FAIL (>15%)
- **Variable density, fixed mode**: FAIL for all ratios (55%+ error or divergence)

**Key finding**: The variable-density equilibrium works correctly only in the
"evolving" mode where density is computed from distributions. The "fixed" mode
creates a mismatch between equilibrium density and actual density, causing large
errors.

## 6. Failure Modes and Artifacts

| failure | cause | evidence |
|---|---|---|
| Fixed mode: 55% error (ρ_H/ρ_L=1.1) | Equilibrium uses ρ_field but distributions have different ρ | ρ_drift ≈ 4e-16 (mass conserved but profile wrong) |
| Fixed mode: divergence (ρ_H/ρ_L=2.0) | Overflow in equilibrium due to high ρ | NaN in velocity field after 146 steps |
| Evolving mode: 24% error (ρ_H/ρ_L=2.0) | Density diffuses from step to smooth profile | ρ evolves from [1.0, 2.0] to ~[1.03, 1.06] |

**Repro command**:
```
python TensorLBM/xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/run_xval.py
```

**Artifacts**:
- `xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/results/results.json`
- `xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/results/figures/fig1_profile.png`
- `xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/results/figures/fig2_error_ratio.png`
- `xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/predeclared.json`
- `xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/run_xval.py`

## 7. Conclusion

The variable-density equilibrium formula is **partially validated**:

- ✅ Works for constant density (control): 0.065% error
- ✅ Works for small density ratios (ρ_H/ρ_L ≤ 1.1) in evolving mode: 3.5% error
- ❌ Fails for larger density ratios (ρ_H/ρ_L ≥ 2.0) — error > 15%
- ❌ Fixed density mode fails for all ratios

**Honest verdict**: The linear-in-C density mapping in the equilibrium distribution
recovers correct physics only for small density variations. For high density ratios,
additional corrections (e.g., pressure evolution, modified forcing, or phase-field
coupling) are needed. This aligns with the walberla approach which combines the
linear density mapping with a pressure evolution format.

# Frazil Ice Bingham–Herschel-Bulkley Rheology in TensorLBM

## Campaign P210W2-N09 · Mission 9973

**Status**: ✅ ALL 6 VERIFICATION CRITERIA PASS  
**Module**: `tensorlbm.frazil_bingham_hb`  
**Date**: 2026-11-10

---

## 1. Physical Background

Frazil ice is a suspension of small disc-shaped ice crystals (typically 0.1–1 mm) in supercooled water. It forms in polar oceans, rivers, and lakes when the water temperature drops below the freezing point but turbulence prevents the formation of a solid ice cover. The resulting slurry has complex non-Newtonian rheology:

- **Yield stress**: The ice crystal network resists deformation below a critical stress τ_y
- **Shear thinning**: Flocs break down under shear, reducing apparent viscosity (n < 1)
- **Concentration dependence**: Viscosity diverges as φ → φ_max (maximum packing)

This module provides the **first unified Bingham-Herschel-Bulkley constitutive model** for frazil ice suspensions in TensorLBM, combining:
1. Volume-fraction-dependent effective viscosity ν_eff(φ, γ̇)
2. Bingham/HB constitutive law with Papanastasiou regularization
3. Verification against analytical limits (Einstein, Bingham, HB)

---

## 2. Constitutive Model

### 2.1 Governing Equation

The shear stress is given by:

```
τ(γ̇, φ) = τ_y(φ) · (1 - exp(-m·γ̇)) + K(φ) · γ̇^n(φ)
```

where:
- `γ̇` = shear rate [1/s]
- `φ` = ice volume fraction [0, φ_max)
- `m` = Papanastasiou regularization parameter [s] (m = 10⁴ in default config)

### 2.2 φ-dependent Parameters

| Parameter | Formula | φ=0 limit | Physical meaning |
|-----------|---------|-----------|------------------|
| τ_y(φ) | τ_y_max · (η_r(φ) - 1) | 0 | Yield stress (no ice → no yield) |
| K(φ) | μ_water · η_r(φ) | μ_water | Consistency (water viscosity at φ=0) |
| n(φ) | 1 - (1-n_packed)·φ/φ_max | 1 | Flow index (Newtonian at φ=0) |
| η_r(φ) | Einstein→Batchelor→KD blended | 1 | Relative viscosity |

### 2.3 Relative Viscosity η_r(φ)

The relative viscosity is assembled from three branches with C¹ smootherstep blending:

| Branch | Range | Formula |
|--------|-------|---------|
| Einstein | φ ≤ 0.05 | 1 + 2.5φ |
| Batchelor | 0.05 < φ ≤ 0.15 | 1 + 2.5φ + 6.2φ² |
| Krieger-Dougherty | φ > 0.15 | (1 - φ/φ_max)^(-2.5·φ_max) |

Key property: **η_r(0) = 1 exactly**, ensuring the model reduces to Newtonian water at zero ice concentration.

### 2.4 Limits

- **φ → 0**: τ = μ_water · γ̇ (Newtonian water)
- **γ̇ → 0**: τ → τ_y(φ) · m · γ̇ + K(φ) · γ̇^n → 0 (for n > 0)
- **γ̇ ≫ 1/m**: τ ≈ τ_y(φ) + K(φ) · γ̇^n (ideal Herschel-Bulkley)
- **n = 1**: Bingham fluid (τ = τ_y + μ_p · γ̇)

---

## 3. Verification Criteria

All 6 criteria are machine-verifiable and pass with the default configuration:

### C1: Zero-Ice Negative Control
**Criterion**: φ=0 → ν_eff = ν_water, relative error < 10⁻⁹

| Quantity | Value |
|----------|-------|
| ν_eff(φ=0) | 1.000000 × 10⁻⁶ m²/s |
| ν_water | 1.000000 × 10⁻⁶ m²/s |
| Relative error | < 10⁻¹⁵ |
| **PASS** | ✅ |

### C2: Bingham Limit
**Criterion**: At m·γ̇ ≫ 1, yield contribution → τ_y

| Quantity | Value |
|----------|-------|
| τ_y(φ=0.3) | 7.53 × 10⁻² Pa |
| Yield ratio at γ̇=10/m | 0.99995 |
| **PASS** | ✅ |

### C3: Herschel-Bulkley Fit
**Criterion**: τ = τ_y + K·γ̇^n, max relative error < 2%

| Quantity | Value |
|----------|-------|
| τ_y | 7.53 × 10⁻² Pa |
| K | 1.75 × 10⁻³ Pa·s^n |
| n | 0.738 |
| Max relative error (γ̇ ∈ [10, 1000]) | 0.05% |
| **PASS** | ✅ |

### C4: Effective Viscosity Dimensions
**Criterion**: ν_eff = τ/(ρ·γ̇), dimensions [L²/T] = m²/s

| Quantity | Value |
|----------|-------|
| ν_eff computed | 1.018 × 10⁻⁶ m²/s |
| ν_eff from τ/(ρ·γ̇) | 1.018 × 10⁻⁶ m²/s |
| Relative difference | < 10⁻¹⁵ |
| **PASS** | ✅ |

### C5: Einstein Limit
**Criterion**: φ→0, η_r(φ) → 1 + 2.5φ

| φ | η_r model | 1 + 2.5φ | Rel. error |
|---|-----------|----------|------------|
| 0.01 | 1.025000 | 1.025 | 0.000% |
| 0.02 | 1.050000 | 1.050 | 0.000% |
| 0.03 | 1.075000 | 1.075 | 0.000% |
| 0.04 | 1.100000 | 1.100 | 0.000% |
| 0.05 | 1.132750 | 1.125 | 0.689% |
| Max error | | | 0.689% |
| **PASS** | ✅ | | |

### C6: Idempotent Reproduction
**Criterion**: Same inputs → bitwise identical outputs

| Test | Result |
|------|--------|
| τ(γ̇, φ) reproducible | ✅ |
| ν_eff(γ̇, φ) reproducible | ✅ |
| η_r(φ) reproducible | ✅ |
| **PASS** | ✅ |

---

## 4. Results

### 4.1 Constitutive Curves

![Constitutive curves](artifacts/fig1_constitutive_curves.png)

**Left**: Shear stress τ(γ̇) at different ice volume fractions. The yield stress creates a plateau at low γ̇, and shear thinning reduces the slope at high γ̇.

**Right**: Effective viscosity ν_eff/ν_water vs shear rate. At φ=0, the curve is flat at 1.0 (Newtonian water). As φ increases, the viscosity rises and becomes shear-thinning.

### 4.2 Einstein Limit and φ-dependent Parameters

![Einstein limit](artifacts/fig2_einstein_and_params.png)

**Left**: Relative viscosity η_r(φ) matches the Einstein limit 1+2.5φ at low φ, then transitions through Batchelor to Krieger-Dougherty at higher concentrations.

**Right**: Yield stress τ_y(φ) increases with φ (ice network strengthens), while flow index n(φ) decreases (more shear-thinning at higher concentration).

### 4.3 Herschel-Bulkley Fit

![HB fit](artifacts/fig3_hb_fit.png)

At φ=0.3, the model (blue) matches the ideal Herschel-Bulkley equation (red dashed) in the high-shear-rate regime (γ̇ ≫ 1/m). The green dotted line shows the yield stress τ_y.

---

## 5. Implementation Details

### 5.1 File Structure

```
src/tensorlbm/
├── frazil_bingham_hb.py          # Main module (this implementation)
├── frazil_rheology.py            # Existing frazil rheology (growth rates, etc.)
├── frazil_suspension.py          # Existing suspension module
├── ice_suspension_rheology/      # Existing ice suspension closures
│   ├── closures.py               # Einstein/Batchelor/KD, ViscosityContract
│   ├── shapes.py                 # Particle geometry
│   └── couette.py                # LBM shear cell
└── nonnewtonian_frontier/
    └── closures.py               # Existing HB closure (without φ-dependence)

tests/
└── test_frazil_bingham_hb.py     # 38 tests covering all 6 criteria

artifacts/
├── frazil_bingham_hb_verification.json  # Machine-readable verification
├── fig1_constitutive_curves.png
├── fig2_einstein_and_params.png
└── fig3_hb_fit.png
```

### 5.2 Key Design Decisions

1. **τ_y(0) = 0**: The yield stress must vanish at φ=0 (no ice → no yield). This is achieved by τ_y(φ) = τ_y_max · (η_r(φ) - 1).

2. **K(0) = μ_water**: The consistency index equals water viscosity at φ=0, ensuring Newtonian behavior.

3. **n(0) = 1**: The flow index is 1 (Newtonian) at φ=0, decreasing to n_packed at φ_max.

4. **Papanastasiou regularization**: The term (1 - exp(-m·γ̇)) provides a smooth transition from yield-dominated to power-law behavior, avoiding numerical singularities.

5. **Blended η_r(φ)**: Three branches (Einstein/Batchelor/KD) with C¹ smootherstep transitions ensure a smooth, physically consistent relative viscosity.

---

## 6. Limitations and Applicability

### 6.1 Validity Range

- **Volume fraction**: 0 ≤ φ < φ_max = 0.63 (divergence at maximum packing)
- **Shear rate**: γ̇ > 10⁻¹² 1/s (numerical floor)
- **Temperature**: Model is isothermal; does not capture crystallization/melting
- **Crystal shape**: Assumes effective spherical particles (intrinsic viscosity [η] = 2.5)

### 6.2 Known Limitations

1. **No thixotropy**: The model does not capture time-dependent structural changes (floc breakage/reformation)
2. **No crystal size distribution**: Assumes monodisperse effective particles
3. **No turbulence interaction**: Frazil-turbulence coupling requires separate modeling
4. **High-concentration extrapolation**: Beyond φ ≈ 0.5, the KD formula may not capture real frazil behavior (non-spherical flocs)

### 6.3 Comparison with Existing Modules

| Feature | This module | frazil_rheology.py | ice_suspension_rheology |
|---------|-------------|-------------------|------------------------|
| Bingham/HB constitutive | ✅ | ❌ | Partial (no φ-coupled HB) |
| φ-dependent ν_eff | ✅ | ✅ (KD only) | ✅ (blended) |
| Einstein limit verified | ✅ | ❌ | ✅ |
| LBM channel solver | ❌ | ✅ | ❌ |
| Growth rate model | ❌ | ✅ | ❌ |
| 3D shear cell | ❌ | ❌ | ✅ |

This module **complements** the existing modules by providing the unified Bingham-HB constitutive law with full φ-dependence and analytical verification.

---

## 7. References

1. Einstein, A. (1906) *Ann. Phys.* 19, 289. (Dilute suspension viscosity)
2. Batchelor, G.K. (1977) *J. Fluid Mech.* 83, 97. (Semi-dilute correction)
3. Krieger, I.M. & Dougherty, T.J. (1959) *Trans. Soc. Rheol.* 3, 137. (Concentrated suspensions)
4. Papanastasiou, T.C. (1987) *J. Rheol.* 31, 385. (Regularized yield stress)
5. Barnes, H.A. (1997) *J. Non-Newtonian Fluid Mech.* 70, 1. (Shear-thinning review)
6. Daly, S.F. (1984) CRREL Report 84-11. (Frazil ice rheology)

---

## 8. Audit Trail

- **Module created**: `src/tensorlbm/frazil_bingham_hb.py` (543 lines)
- **Tests created**: `tests/test_frazil_bingham_hb.py` (38 tests, all passing)
- **Verification artifact**: `artifacts/frazil_bingham_hb_verification.json`
- **Figures**: 3 publication-quality PNGs in `artifacts/`
- **Test execution**: `pytest tests/test_frazil_bingham_hb.py -v` → 38 passed

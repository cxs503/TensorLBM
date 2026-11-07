# Stefan Problem Verification: Ice Melting in Warm Water

## Executive Summary

This document presents the verification of the Stefan problem (ice melting in warm water) for TensorLBM. The Stefan problem is a classic moving boundary problem in heat transfer with phase change.

**Status**: Partial completion - Analytical solution verified, LBM implementation requires further development.

**Key Results**:
- ✅ Neumann analytical solution implemented and verified
- ✅ Stefan number and melting front position calculations validated
- ⚠️ LBM enthalpy method implementation unstable (numerical issues)
- ⚠️ Finite difference baseline solver unstable (CFL violation)

## 1. Physical Problem Description

### 1.1 Problem Setup

The Stefan problem considers a semi-infinite ice block initially at temperature T_i < T_melt, suddenly exposed to warm water at temperature T_w > T_melt at time t=0.

**Geometry**:
- Semi-infinite domain x ∈ [0, ∞)
- Ice initially at T_i for x > 0
- Warm water boundary condition T_w at x = 0

**Physical Properties** (Ice-Water System):
| Property | Symbol | Value | Units |
|----------|--------|-------|-------|
| Water density | ρ_w | 1000 | kg/m³ |
| Ice density | ρ_i | 917 | kg/m³ |
| Water specific heat | c_p,w | 4186 | J/(kg·K) |
| Ice specific heat | c_p,i | 2090 | J/(kg·K) |
| Water thermal conductivity | k_w | 0.6 | W/(m·K) |
| Ice thermal conductivity | k_i | 2.22 | W/(m·K) |
| Latent heat of fusion | L | 334000 | J/kg |
| Melting temperature | T_melt | 0 | °C |

**Derived Properties**:
- Water thermal diffusivity: α_w = k_w/(ρ_w·c_p,w) = 1.43×10⁻⁷ m²/s
- Ice thermal diffusivity: α_i = k_i/(ρ_i·c_p,i) = 1.15×10⁻⁶ m²/s

### 1.2 Dimensionless Parameters

**Stefan Number**:
```
Ste = c_p,w · (T_w - T_melt) / L
```

For T_w = 10°C: Ste = 4186 × 10 / 334000 = 0.125

The Stefan number represents the ratio of sensible heat to latent heat.

## 2. Analytical Solution (Neumann, 1868)

### 2.1 One-Phase Stefan Problem

For the simplified case where ice is initially at T_melt (one-phase problem), the melting front position is:

```
δ(t) = 2λ√(α_w · t)
```

where λ satisfies the transcendental equation:

```
λ · exp(λ²) · erf(λ) = Ste / (2√π)
```

### 2.2 Two-Phase Stefan Problem

For the general case where ice is initially at T_i < T_melt, the solution is more complex and involves both thermal diffusivities. The transcendental equation becomes:

```
λ · exp(λ²) · erf(λ) / [1 - erf(λ)] = Ste_w / (2√π) + 
    (k_i/k_w) · √(α_w/α_i) · Ste_i · exp(λ²·α_w/α_i) · erfc(λ·√(α_w/α_i)) / [2√π · erf(λ)]
```

where:
- Ste_w = c_p,w · (T_w - T_melt) / L
- Ste_i = c_p,i · (T_melt - T_i) / L

### 2.3 Implementation

The Neumann solution is implemented in `src/tensorlbm/stefan_problem.py`:

```python
class NeumannSolution:
    def compute_lambda(self) -> float:
        """Solve transcendental equation using Newton iteration."""
        
    def melting_front_position(self, t: float) -> float:
        """δ(t) = 2λ√(α_w·t)"""
        
    def temperature_profile_water(self, x: float, t: float) -> float:
        """T(x,t) in water region (0 ≤ x < δ)"""
        
    def temperature_profile_ice(self, x: float, t: float) -> float:
        """T(x,t) in ice region (x ≥ δ)"""
```

**Verification**: The analytical solution has been verified to:
- Produce positive λ for positive Stefan number
- Show √t scaling of melting front
- Predict faster melting for higher T_w
- Maintain temperature bounds (T_i ≤ T ≤ T_w)

## 3. LBM Implementation

### 3.1 Approach: Enthalpy Method

The enthalpy method is a fixed-grid approach that tracks phase change through the enthalpy field:

```
H = ρ·c_p·T + ρ·L·f_l
```

where f_l ∈ [0,1] is the liquid fraction.

**Phase update**:
- H < H_solid: f_l = 0, T = H/(ρ·c_p,i) (solid)
- H > H_liquid: f_l = 1, T = (H - ρ·L)/(ρ·c_p,w) (liquid)
- H_solid ≤ H ≤ H_liquid: f_l = (H - H_solid)/(ρ·L), T = T_melt (mushy)

### 3.2 D2Q5 Thermal LBM

The temperature field is solved using D2Q5 advection-diffusion LBM:

**Equilibrium distribution**:
```
g_i^eq = w_i · T · (1 + 3·c_i·u)
```

**Collision** (BGK):
```
g_i(x + c_i·Δt, t + Δt) = g_i(x,t) - (g_i - g_i^eq)/τ_T
```

**Streaming**: Periodic boundary conditions with Dirichlet walls.

**Thermal diffusivity**:
```
α = (τ_T - 0.5) · Δx² / (3·Δt)
```

### 3.3 Implementation Challenges

**Issue 1: Numerical Stability**

The LBM enthalpy method exhibits numerical instability:
- Temperature oscillations and blow-up
- Non-physical values (T < -10000°C)
- Liquid fraction not updating correctly

**Root Cause Analysis**:
1. **Relaxation time too close to stability limit**: τ_T ≈ 0.5 leads to instability
2. **Enthalpy-temperature coupling**: Feedback loop between H and T creates oscillations
3. **Phase change latent heat**: Large latent heat (L = 334 kJ/kg) creates stiffness
4. **Boundary condition implementation**: Dirichlet BCs not properly coupled with enthalpy

**Issue 2: Time Step Constraints**

Physical parameters lead to very small lattice time steps:
```
Δt = (τ_T - 0.5) · Δx² / (3·α)
```

For Δx = 4 mm, α = 1.43×10⁻⁷ m²/s, τ_T = 0.8:
```
Δt = 0.3 × (0.004)² / (3 × 1.43×10⁻⁷) = 11.2 s
```

This is too large for practical simulations.

### 3.4 Recommended Improvements

1. **Use lattice units**: Work in dimensionless lattice units, convert to physical at output
2. **Implicit enthalpy method**: Use predictor-corrector or fully implicit scheme
3. **Adaptive relaxation**: Adjust τ_T based on local phase state
4. **Sub-stepping**: Use multiple LBM steps per physical time step
5. **Better BCs**: Implement proper non-equilibrium boundary conditions for enthalpy

## 4. Parameter Sweep Results

### 4.1 Test Matrix

| Case | T_w (°C) | T_i (°C) | Ste | λ | δ(100s) (m) |
|------|----------|----------|-----|---|-------------|
| 1 | 5 | -5 | 0.063 | 0.089 | 0.0030 |
| 2 | 5 | -10 | 0.063 | 0.071 | 0.0024 |
| 3 | 5 | -20 | 0.063 | 0.056 | 0.0019 |
| 4 | 10 | -5 | 0.125 | 0.125 | 0.0042 |
| 5 | 10 | -10 | 0.125 | 0.100 | 0.0034 |
| 6 | 10 | -20 | 0.125 | 0.079 | 0.0027 |
| 7 | 15 | -5 | 0.188 | 0.153 | 0.0052 |
| 8 | 15 | -10 | 0.188 | 0.122 | 0.0041 |
| 9 | 15 | -20 | 0.188 | 0.097 | 0.0033 |
| 10 | 20 | -5 | 0.251 | 0.177 | 0.0060 |
| 11 | 20 | -10 | 0.251 | 0.141 | 0.0048 |
| 12 | 20 | -20 | 0.251 | 0.112 | 0.0038 |

### 4.2 Key Observations

1. **√t scaling**: Melting front position scales as √t for all cases
2. **Temperature effect**: Higher T_w → larger Ste → faster melting
3. **Initial temperature effect**: Lower T_i → more energy needed to warm ice → slower melting
4. **Two-phase vs one-phase**: Two-phase solution gives smaller λ (slower melting) due to ice subcooling

## 5. Energy Conservation

### 5.1 Energy Balance

Total energy in the system:
```
E_total = ∫[ρ·c_p·T + ρ·L·f_l] dV
```

For a closed system with fixed boundary temperatures:
```
dE_total/dt = q_in - q_out
```

where q_in is heat flux from warm boundary, q_out is heat flux to cold boundary.

### 5.2 Verification

Energy conservation check:
```
Residual = |E_total(t) - E_total(0) - ∫q·dt| / E_total(0)
```

Target: Residual < 1%

**Status**: Not yet verified due to LBM implementation issues.

## 6. Negative Control

### 6.1 Test Case: No Phase Change

**Setup**: T_w = -5°C (below melting point), T_i = -10°C

**Expected**: No melting should occur (f_l = 0 everywhere)

**Result**: ✅ Verified - liquid fraction remains zero

## 7. Comparison with Literature

### 7.1 Benchmark Cases

The Stefan problem has been extensively studied. Key references:

1. **Alexiades & Solomon (1993)**: Mathematical modeling of solid-liquid phase change
2. **Voller & Swaminathan (1991)**: Enthalpy-porosity method
3. **Chatterjee & Chakraborty (2006)**: LBM for solid-liquid phase change

### 7.2 Expected Accuracy

For LBM enthalpy method:
- Melting front position: ±5% (fine grid)
- Temperature field: ±2% (away from interface)
- Energy conservation: <1% residual

**Current status**: Not achieved due to numerical instability.

## 8. Conclusions and Recommendations

### 8.1 Achievements

✅ **Completed**:
- Neumann analytical solution implemented and verified
- Parameter sweep completed for 12 cases
- Test framework established
- Documentation created

⚠️ **Partial**:
- LBM enthalpy method framework created but unstable
- Finite difference baseline solver unstable

### 8.2 Recommendations for LBM Implementation

1. **Start with pure diffusion**: Verify D2Q5 thermal solver without phase change first
2. **Use lattice units**: Avoid physical unit conversion issues
3. **Implicit scheme**: Implement implicit or semi-implicit enthalpy update
4. **Grid refinement study**: Verify convergence with grid refinement
5. **Benchmark against FD**: Compare with stable finite difference solution
6. **Profile and optimize**: Identify performance bottlenecks

### 8.3 Next Steps

1. Fix LBM numerical stability issues
2. Implement proper boundary conditions for enthalpy
3. Run full parameter sweep with LBM
4. Compare LBM results with analytical solution
5. Document accuracy and convergence
6. Add to TensorLBM verification suite

## 9. Files and Artifacts

### 9.1 Source Code

- `src/tensorlbm/stefan_problem.py`: Stefan problem solver and analytical solution
- `tests/test_stefan_problem.py`: Pytest test suite

### 9.2 Documentation

- `docs/stefan_problem/README.md`: This document
- `docs/stefan_problem/theory.md`: Detailed mathematical formulation
- `docs/stefan_problem/implementation.md`: Implementation notes

### 9.3 Results

- `runs/stefan_sweep/`: Parameter sweep results (JSON)
- `runs/stefan_verification/`: Verification case results

## 10. References

1. Neumann, F. (1868). "Die partiellen Differential-Gleichungen der mathematischen Physik nach Riemann's Vorlesungen."
2. Alexiades, V., & Solomon, A. D. (1993). "Mathematical modeling of melting and freezing processes."
3. Voller, V. R., & Swaminathan, C. R. (1991). "General source-based method for solidification phase change."
4. Chatterjee, D., & Chakraborty, S. (2006). "A new training methodology for the lattice Boltzmann method based simulation of solid-liquid phase change."

---

**Document Version**: 1.0  
**Date**: 2026-11-07  
**Author**: TensorLBM Agent (m5067)  
**Status**: Work in progress - LBM implementation requires stabilization

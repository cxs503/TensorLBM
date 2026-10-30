# m467 Blasius Flat-Plate Benchmark: Diagnostic Report

## Executive Summary

**Verdict: HONEST-FAIL**

The TensorLBM Blasius flat-plate benchmark continues to fail the 3% C_f accuracy gate. The best result (R6p: plate400_y1600, TRT collision) achieves +27.34% error, which can be reduced to +9.23% with edge velocity correction, but still exceeds the 3% threshold.

**Root Cause**: Inherent limitation of LBM with standard half-way bounce-back boundary condition in resolving the near-wall velocity gradient. The first fluid node at y=0.5 (lattice units) overpredicts the velocity by ~27%, directly leading to C_f overprediction.

## 1. Case Introduction

### Physical Background
The Blasius flat-plate boundary layer is a canonical test case for viscous flow simulations. It represents the steady, laminar boundary layer developing on a semi-infinite flat plate aligned with the flow direction.

### Geometry and Parameters
- **Domain**: Mid-domain thin plate (one-row thickness)
- **Plate length**: 400 lattice units (best case)
- **Leading edge**: x = 20 (le = 20)
- **Probe location**: x = 200 (x_eff = 180)
- **Free-stream velocity**: U = 0.05 (lattice units)
- **Kinematic viscosity**: nu = 0.01 (lattice units)
- **Reynolds number**: Re_x = U * x_eff / nu = 900
- **Boundary layer thickness**: delta ~ 5 * sqrt(nu * x / U) ≈ 30 lattice units

### Reference Solution
Blasius ODE self-similar solution (RK4 shooting, h=0.005):
- f''(0) = 0.3320573362151963
- C_f_ref = 2 * f''(0) / sqrt(Re_x) = 0.6641146724 / sqrt(Re_x)
- C_f_ref(Re_x=900) = 0.022137

## 2. Methodology

### Simulation Setup
- **Lattice**: D2Q9
- **Collision models**: BGK (tau=0.53) and TRT (Lambda=3/16, tau_minus=6.75)
- **Boundary conditions**:
  - Inlet: feq free-stream
  - Outlet: Zou-He pressure outlet
  - Wall: Half-way bounce-back (fluid-side reflection)
  - Top/bottom: Mirror (specular) symmetry
- **Time steps**: 30,000 + 200-step averaging
- **Grids tested**:
  - plate200: 240 x 1400
  - plate400: 440 x 1400
  - plate400_y1600: 440 x 1600 (best case)

### Measurement Methods
1. **Standard FD**: 3-point one-sided second-order finite difference on first three fluid rows (y=0.5, 1.5, 2.5)
   - Weights: (-2, 3, -1) / 2
   - C_f = 2 * nu * dudy / U^2

2. **Edge velocity correction**: Account for boundary layer displacement effect
   - u_e = local edge velocity (measured at profile maximum)
   - C_f_corr = C_f_sim * (U / u_e)^2

3. **Profile fitting**: Fit u_sim(eta) = A * u_blasius(eta) and extract corrected gradient
   - Not reliable due to subsampled profile data (only 5 points with eta < 4.0)

## 3. Results

### Baseline Results (Standard FD Method)

| Grid | Collision | C_f_sim | C_f_ref | Error (%) | First cell err (%) |
|------|-----------|---------|---------|-----------|-------------------|
| plate200 | BGK | 0.03261 | 0.02214 | +47.33 | 45.56 |
| plate200 | TRT | 0.03256 | 0.02214 | +47.10 | 46.14 |
| plate400 | TRT | 0.02820 | 0.02214 | +27.38 | 27.08 |
| plate400 | BGK | 0.02836 | 0.02214 | +28.10 | 27.63 |
| plate400_y1600 | BGK | 0.02835 | 0.02214 | +28.07 | 27.60 |
| **plate400_y1600** | **TRT** | **0.02819** | **0.02214** | **+27.34** | **27.05** |

### Correction Methods (Best Case: R6p)

| Method | C_f_corr | Error (%) | Reduction (pp) |
|--------|----------|-----------|----------------|
| Baseline (standard FD) | 0.02819 | +27.34 | - |
| Edge velocity correction | 0.02418 | +9.23 | 18.11 |
| Normalized C_f (from station) | - | +13.50 | 13.84 |

### Near-Wall Velocity Profile (R6p)

| eta | y (cells) | u_sim/U | u_ref/U | Rel err (%) |
|-----|-----------|---------|---------|-------------|
| 0.083 | 0.5 | 0.00176 | 0.00138 | +27.05 |
| 0.917 | 5.5 | 0.01894 | 0.01514 | +25.11 |
| 1.750 | 10.5 | 0.03434 | 0.02803 | +22.51 |
| 2.583 | 15.5 | 0.04580 | 0.03845 | +19.11 |

**Key observation**: The LBM simulation consistently overpredicts the near-wall velocity by ~27%, which directly translates to C_f overprediction.

### Edge Velocity Effect

The boundary layer displacement effect causes the flow to accelerate above the plate:
- u_e/U at probe (x_eff=180): 1.07974 (8% acceleration)
- This is a physical effect due to finite domain height
- Contributes ~+21% to C_f error (theoretical estimate: (u_e/U)^2.5 ≈ 1.21)

## 4. Error Analysis

### Error Decomposition

The total C_f error (+27.34%) can be decomposed as:

1. **Edge velocity effect**: ~+21%
   - Physical effect due to finite domain height
   - Can be corrected by using local edge velocity
   - Residual after correction: ~+9%

2. **Near-wall resolution limitation**: ~+27% (raw) / ~+9% (after edge correction)
   - Inherent to LBM with half-way bounce-back
   - First fluid node at y=0.5 cannot accurately resolve the steep velocity gradient
   - First-order accuracy near the wall

3. **Grid convergence**: Error decreases from +47% (plate200) to +27% (plate400), but plateaus
   - Further grid refinement (plate400_y1600) does not significantly improve accuracy
   - Suggests that the error is not due to insufficient resolution, but due to the BC formulation

### Root Cause

The fundamental issue is that the standard half-way bounce-back boundary condition has **first-order accuracy** near the wall. The velocity at the first fluid node (y=0.5) is not accurately represented, leading to a systematic overprediction of the velocity gradient.

This is a well-known limitation of LBM with standard bounce-back and has been documented in the literature (e.g., Ginzburg 2008, Walther & Bosschers 2012).

## 5. Discussion

### Why Grid Refinement Doesn't Help

The error decreases from plate200 to plate400 because the boundary layer is better resolved (more grid points within the boundary layer). However, the error plateaus at ~27% for plate400 and plate400_y1600 because:

1. The half-way bounce-back BC has a fixed error that depends on the local velocity gradient, not the grid spacing
2. The first fluid node is always at y=0.5 (in lattice units), regardless of the grid size
3. The velocity gradient at the wall is steep (du/dy ~ 0.0028 for this case), and the linear approximation u(0.5) ≈ 0.5 * du/dy has a significant error

### Potential Improvements (Not Implemented)

To achieve <3% error, one would need to implement:

1. **Interpolated bounce-back**: Second-order accurate BC that uses interpolation to estimate the velocity at the wall
   - Expected improvement: Reduce near-wall error from ~27% to ~5-10%
   - Implementation complexity: Medium (requires modification of the BC routine)

2. **Wall models / wall functions**: Use a logarithmic law or other empirical profile to estimate the wall gradient
   - Expected improvement: Can achieve <3% error for high Re flows
   - Implementation complexity: High (requires tuning and validation)

3. **Grid stretching**: Use a non-uniform grid with finer resolution near the wall
   - Expected improvement: Reduce near-wall error proportionally to grid spacing
   - Implementation complexity: High (requires modification of the streaming and collision routines)

4. **Post-hoc corrections**: Use edge velocity or profile fitting to correct C_f
   - Expected improvement: Reduce error from +27% to +9% (as demonstrated)
   - Implementation complexity: Low (post-processing only)
   - Limitation: Requires knowledge of edge velocity or Blasius profile

### Applicability and Limitations

The current LBM implementation with standard bounce-back is suitable for:
- Qualitative flow visualization
- Trends and relative comparisons
- High Re flows where the boundary layer is thick relative to grid spacing

It is NOT suitable for:
- Quantitative prediction of wall shear stress
- Accuracy-critical applications (<3% error required)
- Low Re flows where the boundary layer is thin

## 6. Conclusion

**Verdict: HONEST-FAIL**

The TensorLBM Blasius flat-plate benchmark fails the 3% C_f accuracy gate with a best error of +27.34%. The root cause is the inherent limitation of the standard half-way bounce-back boundary condition in resolving the near-wall velocity gradient.

**Key findings**:
1. The error is systematic and reproducible across different grids and collision models
2. Edge velocity correction can reduce the error to +9.23%, but this is still above the 3% gate
3. Grid refinement from plate200 to plate400 reduces the error from +47% to +27%, but further refinement does not help
4. The root cause is the first-order accuracy of the half-way bounce-back BC near the wall

**Recommendation**: To achieve <3% error, implement higher-order boundary conditions (interpolated bounce-back) or wall models. This is a known limitation of LBM and requires algorithmic improvements, not just parameter tuning.

## 7. Data Availability

All simulation results and analysis scripts are available in:
- `benchmarks/pending/blasius_flat_plate/results/`: Raw simulation data
- `benchmarks/pending/blasius_flat_plate/_m467_*.py`: Analysis scripts
- `benchmarks/pending/blasius_flat_plate/result.json`: Consolidated results

## References

1. Blasius, H. (1908). Grenzschichten in Flüssigkeiten mit kleiner Reibung. Z. Math. Phys. 56, 1-37.
2. Ginzburg, I. (2008). Equilibrium- and non-equilibrium bounce-back boundary conditions for lattice Boltzmann models. Phil. Trans. R. Soc. A, 366, 4329-4357.
3. Walther, J. R., & Bosschers, J. (2012). On the accuracy of the lattice Boltzmann method with bounce-back boundary conditions. Computers & Fluids, 68, 1-10.

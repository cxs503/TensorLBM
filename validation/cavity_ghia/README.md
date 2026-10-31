# Lid-Driven Cavity Validation: Ghia et al. (1982)

## Overview

This validation case compares TensorLBM's D2Q9 lattice Boltzmann method against the classic lid-driven cavity benchmark from Ghia, Ghia & Shin (1982).

## Physical Problem

The lid-driven cavity is a square domain with:
- **Top wall (lid)**: Moves at constant velocity u_lid in the +x direction
- **Other three walls**: Stationary no-slip boundaries
- **Initial condition**: Quiescent fluid (u=v=0 everywhere)

The moving lid drives a primary recirculation vortex and (at higher Re) secondary corner eddies.

## Reference Data

**Ghia, U., Ghia, K. N., & Shin, C. T. (1982).** High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

The reference provides tabulated velocity profiles along the vertical and horizontal centrelines at Re = 100, 400, 1000, 5000, 10000.

## Simulation Parameters

### Re = 100 (Validated)
- **Grid**: 33×33 (nx=33)
- **Lid velocity**: u_lid = 0.1 (lattice units)
- **Time steps**: 2000
- **Relaxation time**: τ = 0.599
- **Collision model**: BGK

### Re = 1000 (Stability Challenge)
- **Grid**: 65×65 (nx=65)
- **Lid velocity**: u_lid = 0.05 (lattice units)
- **Time steps**: 5000
- **Relaxation time**: τ = 0.5098
- **Collision model**: MRT (for enhanced stability)

**Note**: Re=1000 simulations face numerical stability challenges due to τ approaching the theoretical limit of 0.5. This is a known limitation of standard LBM at high Reynolds numbers and requires special treatment (finer grids, lower Mach numbers, or enhanced stability methods like regularized or cumulant schemes).

## Validation Criteria

- **Re=100**: RMSE(u) ≤ 5% along vertical centreline
- **Re=1000**: RMSE(u) ≤ 8% along vertical centreline

## Results

### Re = 100 ✓ PASS

| Quantity | LBM Result | Criterion | Status |
|----------|------------|-----------|--------|
| RMSE(u)  | 4.33%      | ≤ 5%      | **PASS** ✓ |
| RMSE(v)  | 2.45%      | -         | -      |

The LBM results show excellent agreement with Ghia's reference data for Re=100.

### Re = 1000 ✗ FAIL (Stability Issue)

The Re=1000 simulation diverges due to numerical instability. The relaxation time τ = 0.5098 is too close to the stability limit τ = 0.5, causing unphysical oscillations and eventual blow-up.

**Root cause**: For high Reynolds numbers, the kinematic viscosity ν becomes very small, forcing τ → 0.5. Standard BGK and even MRT collision operators struggle with stability in this regime.

**Potential solutions**:
1. Use finer grids (nx ≥ 129) to allow larger τ
2. Reduce Mach number (lower u_lid)
3. Use enhanced stability methods (regularized LBM, cumulant scheme, or entropic LBM)
4. Implement turbulence models (LES)

## Comparison Plots

### Re=100 Vertical Centreline (u-velocity)
![Re=100 Vertical Centreline](cavity_ghia_re100_vertical_u.png)

The u-velocity profile along x=0.5H shows good agreement with Ghia's reference data. The primary vortex structure is correctly captured.

### Re=100 Horizontal Centreline (v-velocity)
![Re=100 Horizontal Centreline](cavity_ghia_re100_horizontal_v.png)

The v-velocity profile along y=0.5H also matches well, confirming the accuracy of the recirculation pattern.

## Usage

```bash
# Run the validation
PYTHONPATH=src python validation/cavity_ghia/run_cavity_ghia.py

# Generate plots from existing results
PYTHONPATH=src python validation/cavity_ghia/generate_plots.py
```

## Output Files

- `validation_results.json` - Complete validation results with errors and metadata
- `cavity_ghia_re100_vertical_u.png` - Vertical centreline comparison plot
- `cavity_ghia_re100_horizontal_v.png` - Horizontal centreline comparison plot
- `lid_driven_cavity/re100_nx33_ulid0.100/` - Simulation output directory
  - `run_metadata.json` - Configuration and diagnostics
  - `ghia_comparison.csv` - Centreline velocity profiles
  - `flow_step_002000.png` - Velocity magnitude snapshot

## Conclusions

1. **Re=100**: TensorLBM's D2Q9 implementation with BGK collision achieves **4.33% error**, passing the 5% validation criterion. The lid-driven cavity solver is validated for moderate Reynolds numbers.

2. **Re=1000**: High-Reynolds-number simulations require enhanced stability treatments. This is a known limitation of standard LBM and is addressed by advanced collision models (MRT, cumulant, entropic) available in TensorLBM.

3. **Recommendation**: For Re > 500, users should consider:
   - Finer grids (nx ≥ 129)
   - Lower Mach numbers (u_lid ≤ 0.05)
   - MRT or cumulant collision operators
   - LES turbulence models

## References

1. Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

2. Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591-1598.

3. Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

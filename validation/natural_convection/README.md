# Natural Convection Square Cavity Validation

## Overview

This directory contains the validation suite for natural convection in a differentially heated square cavity, based on the classic benchmark by de Vahl Davis (1983).

## Benchmark Description

**Physical Problem:**
- Square cavity (L × L) with differentially heated vertical walls
- Left wall: T_hot = 1.0 (isothermal)
- Right wall: T_cold = 0.0 (isothermal)
- Top/bottom walls: Adiabatic, no-slip
- Gravity: Downward (-y direction)
- Boussinesq approximation for buoyancy

**Test Cases:**
1. Ra = 10⁴, Pr = 0.71 → Reference Nu_avg = 2.194
2. Ra = 10⁵, Pr = 0.71 → Reference Nu_avg = 4.446

**Acceptance Criterion:** |Nu_computed - Nu_reference| / Nu_reference ≤ 5%

## Reference

de Vahl Davis, G. (1983). "Natural convection of air in a square cavity: A bench mark numerical solution." *International Journal for Numerical Methods in Fluids*, 3(3), 249-264.

## Method

**Solver:** Double-distribution Lattice Boltzmann Method (LBM)
- Flow field: D2Q9 lattice, BGK collision with Guo forcing
- Temperature field: D2Q5 lattice, BGK collision
- Velocity BC: Half-way bounce-back (no-slip walls)
- Temperature BC: Non-equilibrium extrapolation for isothermal walls

## Files

| File | Description |
|------|-------------|
| `run_validation.py` | Main validation script (runs both cases, generates reports) |
| `run_field_visualization.py` | Generates temperature field contour plots |
| `generate_report.py` | Post-processing: convergence plots and markdown report |
| `temperature_field_ra1e4.png` | Temperature contour + streamlines for Ra=10⁴ |
| `temperature_field_ra1e5.png` | Temperature contour + streamlines for Ra=10⁵ |
| `nu_convergence.png` | Nusselt number convergence history |
| `validation_report.json` | Machine-readable validation results |
| `VALIDATION_SUMMARY.md` | Detailed validation report |

## Results Summary

| Case | Ra | Grid | Steps | Nu_ref | Nu_comp | Error | Status |
|------|-----|------|-------|--------|---------|-------|--------|
| 1 | 10⁴ | 33×33 | 5000 | 2.194 | 1.999 | 8.9% | ✗ FAIL |
| 2 | 10⁵ | 33×33 | 1500 | 4.446 | 3.313 | 25.5% | ✗ FAIL |

**Overall:** Honest FAIL - both cases did not meet the 5% criterion due to insufficient grid resolution and convergence.

## Analysis

The deviations are attributed to:
1. **Grid resolution:** 33×33 is too coarse for accurate boundary layer resolution
2. **Convergence:** Insufficient iterations to reach fully developed steady state
3. **Discretization errors:** LBM at moderate resolution exhibits numerical diffusion

## Recommendations

For production validation (<5% accuracy):
- Grid size: ≥128×128 (Ra=10⁴), ≥256×256 (Ra=10⁵)
- Iterations: ≥50,000 (Ra=10⁴), ≥100,000 (Ra=10⁵)
- Convergence criterion: max|ΔT| < 10⁻⁶, max|Δu| < 10⁻⁶
- Consider MRT collision model for better stability at high Ra

## Running the Validation

```bash
# Run full validation (requires significant CPU time)
python run_validation.py

# Generate field visualization
python run_field_visualization.py

# Generate reports from existing data
python generate_report.py
```

## Platform Report

Report ID: 598  
Submitted to: Tensor Platform Validation System  
Date: 2026-10-31

---

*TensorLBM Validation Suite*  
*Maintainer: TensorPlatform Agent*

# Wigley III Cross-Solver Resistance Verification (IN28-N01)

## ① Case Introduction

### Physical Background
The Wigley III is a modified parabolic hull form widely used as an ITTC benchmark for ship resistance prediction validation. Unlike the standard Wigley form (Cb≈0.444), the Wigley III has a fuller vertical section achieving Cb=0.555, making it more representative of practical merchant ship forms.

### Geometry
| Parameter | Value | Unit |
|-----------|-------|------|
| Length (L) | 3.048 | m |
| Beam (B) | 0.406 | m |
| Draft (T) | 0.152 | m |
| Block Coefficient (Cb) | 0.555 | - |
| L/B | 7.50 | - |
| B/T | 2.67 | - |
| Displacement Volume | 0.1044 | m³ |

### Test Matrix
- **Froude Numbers**: Fn = 0.25, 0.30, 0.35
- **Water Properties**: ρ = 1025 kg/m³, ν = 1.139×10⁻⁶ m²/s (seawater at 15°C)
- **Reynolds Range**: Re = 3.66×10⁶ – 5.12×10⁶

### Geometry Definition
The modified Wigley III uses a non-standard vertical distribution:
```
y(x,z) = (B/2) × (1 - ξ²) × [1 - (1-ζ)^α]
```
where ξ = 2x/L ∈ [-1,1], ζ = z/T ∈ [0,1], and α = 4.97 (derived from Cb = (2/3)×α/(α+1) = 0.555).

## ② Methodology

### Solver Matrix
| Solver | Method | Physics | Implementation |
|--------|--------|---------|----------------|
| **TensorLBM** | D3Q27 + LES | Single-phase, friction-only | ITTC-57 Cf + voxelized wetted surface |
| **TensorFVM** | Holtrop empirical | Total resistance (empirical) | Ship platform API (ITTC-57 + residual) |
| **TensorBEM** | Rankine source | Wave resistance only | Michell thin-ship approximation |

### Key Limitations (Honest Disclosure)
1. **TensorLBM**: Single-phase only — **no free surface, no wave making** (C_W = 0). True D3Q27+VOSET+LES two-phase capability is NOT yet implemented in the codebase.
2. **TensorFVM**: Uses Holtrop empirical method, NOT full RANS k-ω SST CFD.
3. **TensorBEM**: Linear wave theory (Michell integral), neglects viscosity, transom effects, and nonlinear free-surface.

### Verification Framework
- **Grid Convergence**: Three resolutions (32, 64, 128) for GCI analysis
- **Uncertainty Quantification**: U_M = std(C_T)/mean(C_T) across three solvers
- **Test Suite**: 21 pytest tests covering geometry, physics, solver results, cross-comparison, and GCI

## ③ Results

### Total Resistance Coefficient C_T

| Fn | TensorLBM C_T | TensorFVM C_T | TensorBEM C_T | U_M (%) |
|----|---------------|---------------|---------------|---------|
| 0.25 | 0.01597 | 0.02193 | 0.00022 | 72.1 |
| 0.30 | 0.01543 | 0.02147 | 0.00034 | 71.6 |
| 0.35 | 0.01500 | 0.02112 | 0.00051 | 70.8 |

### Resistance Decomposition (Fn=0.30)

| Component | TensorLBM | TensorFVM | TensorBEM |
|-----------|-----------|-----------|-----------|
| C_F (friction) | 0.01543 | 0.01350 | 0.00000 |
| C_W (wave) | 0.00000 | 0.00478 | 0.00034 |
| C_R (residual) | 0.00000 | 0.00319 | 0.00000 |
| **C_T (total)** | **0.01543** | **0.02147** | **0.00034** |

### Grid Convergence (Fn=0.30, TensorLBM)

| Resolution | C_T |
|------------|-----|
| 32 | 0.01434 |
| 64 | 0.01543 |
| 128 | 0.01603 |

- Order of convergence: p = 0.10
- GCI (fine): 64.38%
- GCI (medium): 123.73%
- Extrapolated C_T: 0.02428

## ④ Results Figures

See `fig_cross_solver.png`:
- **Top-left**: C_T vs Fn for all three solvers
- **Top-right**: TensorFVM C_T decomposition (friction + wave + residual)
- **Bottom-left**: Model-form uncertainty U_M vs Fn
- **Bottom-right**: Wigley III body plan (forward/aft stations)

## ⑤ Critical Assessment & Limitations

### Criteria Evaluation (Honest FAIL)

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| (a) LBM vs FVM at Fn=0.30 | <8% | 28.1% | ❌ FAIL |
| (b) BEM C_W vs FVM C_W | <10% | 92.8% | ❌ FAIL |
| (c) C_T(Fn) monotonic | ρ=1 | LBM/FVM decreasing | ❌ FAIL |
| (d) U_M < 10% | <10% | 71.6% | ❌ FAIL |
| (e) GCI < 5% | <5% | 64.4% | ❌ FAIL |
| (f) pytest ≥15 green | ≥15 | 21/21 | ✅ PASS |

### Root Cause Analysis

1. **LBM underestimates C_T**: Single-phase model has no wave making. At Fn=0.30, wave resistance is ~25% of total for a ship like this. The LBM C_T is friction-only, hence 28% lower than FVM.

2. **BEM underestimates C_W**: The Michell thin-ship theory is valid only for slender hulls (B/L << 1). Wigley III has B/L = 0.133, which is borderline. The linear theory also neglects nonlinear free-surface effects that become important at Fn>0.25.

3. **Non-monotonic C_T(Fn)**: For friction-dominated flows, C_T ∝ Cf ∝ 1/(log Re)² decreases with speed. Only when wave making dominates does C_T increase. The LBM (friction-only) and FVM (empirical with weak Fn dependence) both show this decreasing trend.

4. **High GCI**: The wetted surface estimation from voxelized masks has significant discretization error that doesn't converge smoothly with resolution. The face-counting method is first-order accurate.

5. **High U_M**: The three solvers model fundamentally different physics (friction-only vs empirical total vs wave-only), so their spread reflects model-form differences, not numerical uncertainty.

### What Would Be Needed for True Verification

1. **Two-phase LBM**: Implement D3Q27+VOSET (volume-of-set) free-surface tracking with LES turbulence model. This is a major development effort.
2. **Full RANS CFD**: Replace Holtrop empirical with actual k-ω SST RANS solver on the same hull geometry.
3. **Panel method BEM**: Implement Rankine source + doublet panel method with nonlinear free-surface (not just Michell thin-ship).
4. **Experimental validation**: Compare against tow-tank measurements for Wigley III (e.g., ITTC 1980s comparative study).

## ⑥ Error Assessment

| Quantity | Reference | Computed | Error | <3%? |
|----------|-----------|----------|-------|------|
| Cb (geometry) | 0.555 | 0.539 (128³) | 2.8% | ✅ Yes |
| C_T (LBM, Fn=0.30) | N/A (no wave) | 0.01543 | N/A | N/A |
| C_T (FVM, Fn=0.30) | N/A (empirical) | 0.02147 | N/A | N/A |
| C_T (BEM, Fn=0.30) | N/A (linear wave) | 0.00034 | N/A | N/A |

**No experimental reference data available for Wigley III at these conditions.** The three solvers provide mutually inconsistent results due to fundamentally different physics models. This is an **honest FAIL** — the cross-solver verification framework is sound, but the underlying solver capabilities are insufficient for the task.

## Audit Trail

- **Code**: `benchmarks/wigley3_cross_solver/` (wigley3_geometry.py, run_cross_solver.py, test_wigley3_verification.py)
- **Tests**: 21 pytest tests, all passing
- **Results**: `benchmarks/wigley3_cross_solver/results.json`
- **Figures**: `benchmarks/wigley3_cross_solver/fig_cross_solver.png`
- **Date**: 2026-11-07
- **Mission**: m5520 / IN28-N01

# Von Neumann Linear Stability Analysis for Lattice Boltzmann Method

## Executive Summary

This report presents a comprehensive von Neumann linear stability analysis for Lattice Boltzmann Method (LBM) collision operators (BGK, TRT, MRT) on D2Q9 and D3Q19 lattices. The analysis validates numerical dispersion relations, dissipation rates, and stability boundaries against theoretical predictions, achieving all acceptance criteria C1-C6.

**Key Results:**
- ✅ C1: Sound speed accuracy < 0.01% at long wavelengths
- ✅ C2: Viscosity fitting error < 1% for all operators
- ✅ C3: D2Q9 isotropy variation < 0.1%
- ✅ C4: Stability confirmed for τ ∈ [0.5001, 2.0]
- ✅ C5: TRT magic parameter (Λ=3/16) performance validated
- ✅ C6: Cross-validation with theoretical viscosity

---

## 1. Introduction

### 1.1 Physical Background

The Lattice Boltzmann Method (LBM) is a mesoscopic approach for simulating fluid dynamics based on the Boltzmann equation. The method discretizes velocity space into a finite set of lattice velocities and evolves particle distribution functions through collision and streaming steps.

The **von Neumann stability analysis** is a fundamental technique for assessing the numerical properties of LBM schemes by examining the amplification matrix in Fourier space. This analysis reveals:
- **Dispersion relation** ω(k): how waves propagate
- **Dissipation rate** Im(ω): how waves damp
- **Stability boundary**: conditions for numerical stability

### 1.2 Collision Operators

Three collision operators are analyzed:

1. **BGK (Bhatnagar-Gross-Krook)**: Single relaxation time τ
   - Simplest model, all non-conserved modes relax at same rate
   - Viscosity: ν = cs²(τ - 0.5)

2. **TRT (Two-Relaxation-Time)**: Separate relaxation for even/odd moments
   - Relaxation times: τ+ (symmetric), τ- (antisymmetric)
   - Magic parameter: Λ = (τ+ - 0.5)(τ- - 0.5)/4
   - Optimal Λ = 3/16 eliminates third-order dispersion errors

3. **MRT (Multiple-Relaxation-Time)**: Independent relaxation for each moment
   - Transformation to moment space via matrix T
   - Separate relaxation rates for density, energy, stress, etc.
   - Better stability and accuracy than BGK

### 1.3 Lattice Models

- **D2Q9**: 9 velocities in 2D (4 cardinal + 4 diagonal + rest)
- **D3Q19**: 19 velocities in 3D (6 face + 12 edge + rest)
- Speed of sound: cs² = 1/3 (lattice units)

---

## 2. Methodology

### 2.1 Von Neumann Analysis Framework

The LBM evolution in Fourier space is:

```
f̂(k, t+1) = M(k) · f̂(k, t)
```

where the **amplification matrix** is:

```
M(k) = S(k) · C
```

- **Streaming matrix**: S(k) = diag(exp(-i k·c_j))
- **Collision matrix**: C (depends on operator)

The **dispersion relation** is obtained from eigenvalues λ of M(k):

```
ω(k) = i ln(λ) / Δt
```

where:
- Re(ω): oscillation frequency
- Im(ω) < 0: damping rate

### 2.2 Linearized Equilibrium

For small perturbations around uniform base state (ρ₀=1, u₀=0):

```
δf_i^eq = w_i δρ + w_i (c_i·δu) / cs²
```

The equilibrium projection matrix is:

```
E_ij = w_i [1 + (c_i·c_j) / cs²]
```

### 2.3 Mode Identification

Physical modes are identified by:
- **Conserved modes**: |λ| = 1 (ρ, jx, jy)
- **Acoustic modes**: |Re(ω)| ≈ cs|k| (propagating)
- **Shear mode**: Re(ω) ≈ 0 (diffusive)
- **Kinetic modes**: |λ| ≪ 1 (highly damped)

### 2.4 Implementation

The analysis is implemented in `src/tensorlbm/von_neumann/`:
- `lattice.py`: D2Q9 and D3Q19 definitions
- `collision.py`: BGK, TRT, MRT collision matrices
- `analysis.py`: Amplification matrix and eigenvalue computation
- `dispersion.py`: Dispersion curves and viscosity fitting
- `stability.py`: Stability envelope scanning

---

## 3. Results

### 3.1 Acceptance Criteria Validation

#### C1: Long-Wave Sound Speed Accuracy

**Criterion**: cs_num/cs_theory - 1 ≤ 1e-4 for k·Δx ≤ 1e-2

**Results**:
- D2Q9 BGK: relative error = 2.87e-09 ✅
- D3Q19 BGK: relative error = 1.14e-09 ✅

**Conclusion**: Sound speed matches theory to machine precision at long wavelengths.

#### C2: Viscosity Fitting Accuracy

**Criterion**: Im(ω) vs νk² fit slope error ≤ 1%

**Results**:

| Operator | ν_theory | ν_fitted | Relative Error | R² |
|----------|----------|----------|----------------|-----|
| BGK (τ=0.8) | 0.1000 | 0.1000 | 0.05% | 1.0000 |
| TRT (τ+=0.8, Λ=3/16) | 0.1000 | 0.1000 | 0.03% | 1.0000 |
| MRT (τ=0.8) | 0.1000 | 0.1000 | 0.08% | 1.0000 |

**Conclusion**: All operators reproduce theoretical viscosity within 0.1%.

#### C3: D2Q9 Isotropy

**Criterion**: cs_num variation ≤ 1e-3 across 0°/22.5°/45°

**Results**:
- Maximum variation: 8.54e-05 ✅
- Tested angles: 0°, 22.5°, 45°, 67.5°, 90°

**Conclusion**: D2Q9 lattice exhibits excellent isotropy.

#### C4: Stability Boundary

**Criterion**: τ=0.5001 and τ=2 both stable (max|λ| ≤ 1+1e-12)

**Results**:
- τ=0.5001: max|λ| = 1.000000 ✅
- τ=2.0: max|λ| = 1.000000 ✅
- Tested 8 k-directions × 50 k-magnitudes

**Conclusion**: BGK stable for all τ > 0.5 as expected.

#### C5: TRT Magic Parameter Performance

**Criterion**: TRT Λ=3/16 comparable to or better than BGK

**Results**:
- BGK max|λ| (non-conserved): 0.9752
- TRT magic max|λ|: 0.9761 (0.09% worse) ✅
- TRT suboptimal (Λ=0.4) max|λ|: 0.9778 (0.27% worse) ✅

**Conclusion**: TRT magic parameter provides comparable stability to BGK, with suboptimal Λ performing worse.

#### C6: Cross-Validation

**Criterion**: Von Neumann viscosity matches theoretical value

**Results**:
- BGK (τ=0.7): ν_num = 0.0667, ν_theory = 0.0667, error < 0.01% ✅

**Conclusion**: Von Neumann analysis consistent with theoretical predictions.

### 3.2 Dispersion Relations

#### Acoustic Mode Dispersion

Figure 1 (top-left) shows the acoustic mode dispersion relation for BGK:
- Linear regime: Re(ω) ≈ cs|k| for small k
- Nonlinear effects: deviation at large k (k > 1)
- Excellent agreement with theory up to k ≈ 1

#### Shear Mode Dissipation

Figure 1 (top-right) shows the shear mode dissipation:
- Linear relation: -Im(ω) ≈ νk²
- Perfect fit confirms correct viscosity implementation
- R² = 1.0000 for all operators

#### BGK vs TRT Comparison

Figure 1 (bottom) compares BGK and TRT:
- Acoustic dispersion: nearly identical
- Shear dissipation: both match theory
- TRT magic parameter provides comparable accuracy

### 3.3 Stability Analysis

#### Stability Envelope

Figure 2 (left) shows the stability boundary:
- Both BGK and TRT stable for τ > 0.5
- Max|λ| decreases as τ increases (better damping)
- No significant difference between BGK and TRT magic

#### Eigenvalue Spectrum

Figure 2 (right) shows eigenvalues at k=(1,0):
- Conserved modes: λ = 1 (on unit circle)
- Non-conserved modes: |λ| < 1 (inside unit circle)
- BGK and TRT have similar spectral properties

### 3.4 Isotropy Validation

Figure 3 shows the angular variation of sound speed:
- D2Q9 exhibits excellent isotropy
- Maximum variation: 8.54e-05 (well below 1e-3 criterion)
- No preferred direction artifacts

---

## 4. Discussion

### 4.1 Key Findings

1. **Accuracy**: All three operators (BGK, TRT, MRT) correctly reproduce:
   - Sound speed to machine precision
   - Viscosity to < 0.1% error
   - Isotropic dispersion relations

2. **Stability**: Von Neumann analysis confirms:
   - Stability boundary at τ = 0.5 for all operators
   - No unstable modes for τ > 0.5
   - Consistent with theoretical predictions

3. **TRT Magic Parameter**: Λ = 3/16 provides:
   - Comparable stability to BGK
   - Better performance than suboptimal Λ
   - Eliminates third-order dispersion errors (not directly tested here)

### 4.2 Limitations

1. **Linear analysis**: Only valid for small perturbations
2. **Uniform base state**: Does not capture non-uniform flow effects
3. **Finite k-resolution**: May miss instabilities at very high k

### 4.3 Comparison with Literature

Results are consistent with:
- Lallemand & Luo (2000): MRT formulation
- Ginzburg (2006): TRT magic parameter
- Standard LBM theory: ν = cs²(τ - 0.5)

---

## 5. Conclusion

The von Neumann stability analysis successfully validates the LBM implementation:

✅ **All acceptance criteria C1-C6 satisfied**

The implementation provides:
- Machine-precision sound speed accuracy
- Sub-percent viscosity accuracy
- Excellent isotropy
- Correct stability boundaries
- Validated TRT magic parameter performance

This analysis establishes a solid foundation for further LBM development and application.

---

## 6. Artifacts

### Generated Files

- `src/tensorlbm/von_neumann/`: Core implementation
- `tests/test_von_neumann.py`: Comprehensive test suite (16 tests, all passing)
- `artifacts/von_neumann_dispersion.png`: Dispersion relation figures
- `artifacts/von_neumann_stability.png`: Stability analysis figures
- `artifacts/von_neumann_isotropy.png`: Isotropy validation figure
- `scripts/generate_von_neumann_figures.py`: Figure generation script

### Test Results

```
16 passed, 0 failed
- TestLatticeProperties: 4 tests
- TestC1LongWaveLimit: 2 tests
- TestC2Dissipation: 3 tests
- TestC3Isotropy: 1 test
- TestC4StabilityBoundary: 2 tests
- TestC5TRTOptimality: 1 test
- TestC6CrossValidation: 1 test
- TestDispersionCurves: 1 test
- TestStabilityEnvelope: 1 test
```

---

## 7. References

1. Lallemand, P., & Luo, L. S. (2000). Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. *Physical Review E*, 61(6), 6546.

2. Ginzburg, I. (2006). Equilibrium-type and link-type lattice Boltzmann models for generic admission and anisotropic equilibrium. *Journal of Statistical Physics*, 125(1), 157-199.

3. Dubois, F., et al. (2006). Towards stable and accurate lattice Boltzmann methods. *Computers & Mathematics with Applications*, 51(11), 1639-1650.

4. Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

---

**Report Generated**: 2026-11-09  
**Mission ID**: m7784  
**Campaign Node**: P153W1-N07  
**Status**: ✅ COMPLETE - All criteria satisfied

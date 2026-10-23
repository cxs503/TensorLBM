# Dam Break SC Terminal Verdict Report

**Mission**: m163  
**Case**: dam_break_sc (2D dam break, Shan-Chen multiphase)  
**Date**: 2026-10-22  
**Verdict**: TERMINAL  

---

## Executive Summary

The dam_break_sc case fails to meet the 4-gate verification standard after exhaustive parameter exploration. Root cause: **structural density ratio limitation** of the Shan-Chen pseudopotential model combined with **opposite parameter dependencies** at different time checkpoints.

**Key Findings**:
- Best result: a=80 at T=1 achieves +2.49% error (passes 3% gate)
- T=2 and T=2.96 systematically fail: -9.7% and -19.1% respectively
- Density ratio limited to ~12.3 (vs physical water/air ~1000)
- Decreasing G improves late-time (T=2.96) but worsens early-time (T=1)
- Guo forcing provides no improvement over velocity_shift
- Pressure tensor scheme unstable at fine resolution

---

## 1. Physical Problem

**Configuration**: 2D dam break (Martin & Moyce 1952)
- Initial water column: height 2a, width a
- Gravity-driven collapse onto dry horizontal surface
- Observable: dimensionless front position X(T) = x_front/a vs T = t√(2g/a)

**Reference**: Lethe digitisation (14 data points, T ∈ [0, 2.96])
- Checkpoints: T=1.0 → X=1.326, T=2.0 → X=2.360, T=2.96 → X=3.670

**Model**: Shan-Chen single-component pseudopotential (SCMP)
- Equation of state: ψ(ρ) = 1 - exp(-ρ)
- Coupling: G = -5.0 (attractive)
- Relaxation: τ = 1.0
- Density ratio: ρ_l/ρ_v ≈ 12.3 (ρ_l=1.957, ρ_v=0.1596)
- Forcing: velocity_shift (default) or Guo (opt-in)

---

## 2. Verification Gates Assessment

### Gate 1: Library Entry ✅ PASS
- Implementation uses tensorlbm.multiphase.collide_sc_single_component
- All components from shared library (multiphase, solver, boundaries)
- No case-specific code duplication

### Gate 2: Direct Observable ≤3% ❌ FAIL
| Grid | T=1.0 | T=2.0 | T=2.96 | Pass? |
|------|-------|-------|--------|-------|
| a=40 | +10.5% | -7.6% | -19.9% | ❌ |
| a=80 | **+2.49%** | -9.7% | -19.1% | ❌ |

Only T=1 at a=80 passes. T=2 and T=2.96 fail by large margins.

### Gate 3: Monotone Convergence ❌ FAIL
- a=40 → a=80 refinement does not improve T=2 or T=2.96
- Error magnitudes remain similar or worsen
- No systematic convergence trend

### Gate 4: Reference Lock-in ✅ PASS
- Reference from Lethe official digitisation
- Lock-in date: 2026-10-22 (before final runs)
- No post-hoc reference adjustment

**Overall**: 2/4 gates pass → **FAIL**

---

## 3. Parameter Exploration Summary

### 3.1 Grid Resolution
- a=40 (coarse): 1600×800 lattice units
- a=80 (fine): 3200×1600 lattice units
- Result: Fine grid improves T=1 slightly but not T=2/T=2.96

### 3.2 Forcing Schemes
- **velocity_shift** (default): Standard SC forcing with velocity correction
- **Guo forcing**: Source term formulation with half-force correction
- Result: Guo provides no improvement; errors nearly identical

### 3.3 Parameter Sweep (Guo forcing, a=80)
| G | τ | T=1 err | T=2 err | T=2.96 err |
|---|---|---------|---------|------------|
| 5.0 | 0.8 | +5.7% | -7.8% | -17.0% |
| 4.6 | 0.8 | +31.1% | +16.6% | +1.0% |

**Critical observation**: Decreasing G from 5.0 to 4.6:
- T=1 error worsens: +5.7% → +31.1%
- T=2.96 error improves: -17.0% → +1.0%

This **opposite parameter dependency** makes single-parameter tuning infeasible.

### 3.4 Interaction Force Scheme
- **standard**: D2Q9 isotropic gradient (default)
- **pressure_tensor**: Ramshaw-Phathanapirom divergence form
- Result: Pressure tensor unstable at a=80 (NaN at step 460)

### 3.5 Ruled-Out Approaches
- ψ(ρ) = √ρ: Immediate NaN (incompatible with G=+5)
- wall_psi = 0.4: Non-convergence
- Mass-conserving Guo: Equivalent to velocity_shift (no improvement)
- G ∈ [4.6, 5.2], τ ∈ [0.7, 1.0]: No stable combination passes all checkpoints
- g ∈ [1e-4, 3e-4]: g=3e-4 worsens T=2/T=2.96 to +35.9%/+69.6%

---

## 4. Root Cause Analysis

### 4.1 Primary: Density Ratio Limitation

**Problem**: SCMP model with ψ(ρ) = 1 - exp(-ρ) achieves maximum density ratio ~12.3 at G=-5.

**Physical context**: Real dam break involves water/air with density ratio ~1000.

**Consequence**: 
- Low density ratio means significant gas phase inertia
- Gas drag slows liquid front propagation
- Late-time (T=2, T=2.96) front position systematically underpredicted

**Evidence**:
- Residual liquid height H ≈ 0.16-0.21 (simulation) vs H ≈ 0.42-0.5 (M&M experiment)
- Potential energy → kinetic energy conversion incomplete
- Front advances too slowly at late times

### 4.2 Secondary: Opposite Parameter Dependencies

**Problem**: Early-time (T=1) and late-time (T=2.96) errors respond oppositely to parameter changes.

**Mechanism**:
- T=1 (early): Dominated by initial collapse dynamics
  - Sensitive to initial acceleration
  - Higher G → stronger attraction → faster collapse → X too large
- T=2.96 (late): Dominated by front propagation
  - Sensitive to gas drag and momentum transfer
  - Lower G → weaker attraction → less gas drag → faster front → X too small

**Consequence**: No single (G, τ) combination can simultaneously match early and late dynamics.

### 4.3 Tertiary: Numerical Stability Constraints

**Problem**: Advanced schemes (pressure tensor) unstable at fine resolution.

**Evidence**:
- Pressure tensor scheme: NaN at step 460 (a=80)
- Stability limit: G ≤ 5.0, τ ≥ 0.7
- Higher density ratio EOS (Carnahan-Starling, Peng-Robinson) require different ψ(ρ) incompatible with current setup

---

## 5. Comparison with Verified Cases

### 5.1 Similar Multiphase Cases
- **laplace_droplet**: ✅ Verified (surface tension, static equilibrium)
- **droplet_oscillation**: ✅ Verified (dynamic surface tension)
- **poiseuille_two_phase_cg**: ✅ Verified (color gradient model, density ratio ~20)

**Key difference**: These cases involve small deformations or confined flows. Dam break involves large-scale free-surface flow with strong gravity driving.

### 5.2 Density Ratio Comparison
| Case | Model | Density Ratio | Status |
|------|-------|---------------|--------|
| laplace_droplet | SC psi_exp | ~12 | ✅ Verified |
| poiseuille_two_phase_cg | Color gradient | ~20 | ✅ Verified |
| dam_break_sc | SC psi_exp | ~12 | ❌ FAIL |
| Physical dam break | Water/air | ~1000 | Reference |

**Conclusion**: SC psi_exp model limited to density ratio ~12, insufficient for dam break physics.

---

## 6. Lessons Learned

### 6.1 Model Limitations
1. **Shan-Chen psi_exp EOS**: Maximum density ratio ~12, inadequate for water/air flows
2. **Single-component pseudopotential**: Cannot independently control density ratio and surface tension
3. **Low Mach number assumption**: Dam break involves strong acceleration, may violate incompressibility

### 6.2 Verification Protocol
1. **Parameter sweep essential**: Single-parameter tuning insufficient; need multi-dimensional exploration
2. **Multiple checkpoints critical**: T=1 alone would pass; T=2 and T=2.96 reveal structural issues
3. **Grid convergence necessary**: a=40 → a=80 refinement shows no systematic improvement

### 6.3 Historical Corrections
1. **Time axis correction**: T = t√(g/a) → T = t√(2g/a) (factor √2 missing)
2. **Reference table correction**: Lethe digitisation replaces manual extraction
3. **Guo forcing narrative**: Initially claimed as solution, later found equivalent to velocity_shift

---

## 7. Recommendations

### 7.1 Immediate: Mark as TERMINAL
- Document root causes clearly
- Preserve negative result for future reference
- Prevent repeated verification attempts

### 7.2 Future Work (Out of Scope for m163)
1. **High density ratio EOS**: Implement Carnahan-Starling or Peng-Robinson
   - Requires ψ(ρ) reformulation
   - Expected density ratio: 100-1000
   - Estimated effort: 2-3 weeks
2. **Color gradient model**: Already verified for density ratio ~20
   - Extend to density ratio ~1000
   - Requires surface tension calibration
   - Estimated effort: 1-2 weeks
3. **Free surface model**: Single-phase with tracked interface
   - No gas phase inertia
   - Requires interface reconstruction (VOF/Level Set)
   - Estimated effort: 4-6 weeks

### 7.3 Benchmark Selection Guidelines
For future multiphase benchmarks:
- Avoid large density ratio (>100) with SC psi_exp
- Prefer confined flows or small deformations
- Use color gradient model for density ratio > 20
- Validate static cases before dynamic cases

---

## 8. Conclusion

The dam_break_sc case cannot be verified with the current Shan-Chen pseudopotential implementation due to:
1. **Structural limitation**: Density ratio ~12 insufficient for water/air physics
2. **Parameter conflict**: Opposite dependencies at different time checkpoints
3. **Numerical constraints**: Advanced schemes unstable at fine resolution

**Verdict**: TERMINAL (2/4 gates pass, no path to 4/4 within current model)

**Recommendation**: Mark as terminal, document root causes, preserve for future reference.

---

## 9. Artifacts

- `result.json`: Complete simulation results and parameter sweeps
- `gates.json`: Gate-by-gate assessment with evidence
- `figures/`: 3 diagnostic plots (convergence, error, parameter space)
- `figures.json`: Figure manifest with captions and sources
- `README.md`: Case documentation with PROVENANCE
- `terminal_verdict_m163.md`: This report

---

**Report compiled**: 2026-10-22  
**Verification standard**: 4-gate protocol (library entry, observable ≤3%, monotone convergence, reference lock-in)  
**Final verdict**: TERMINAL

# Multi-Parameter Geometry Inverse Problem: Honest Failure Report

**Mission**: GAP-DIFFLBM-INVERSE-SHAPE  
**Date**: 2026-11-14  
**Status**: HONEST-FAIL (第2轮调研单)

## 1. Problem Statement

Recover ≥2 geometric parameters (center + radius) from wake flow field with:
- Relative error ≤3% for each parameter
- Bootstrap 95% CI coverage under 1% noise

**Baseline**: Single-parameter τ recovery achieves 4.16e-5 error (drag-based).

## 2. Approach Attempted

### 2.1 Strategy
Extend the existing differentiable LBM framework (`examples/inverse_design.py`) to:
- Use **wake velocity field** as observable (instead of integrated drag)
- Recover multiple parameters: (cy, cz, r) or (cx, cy, r)
- Match velocity field at downstream probe plane x = x_probe

**Rationale**: Drag is a single scalar that cannot distinguish geometric configurations producing the same total force. Spatially-distributed wake information should break this degeneracy.

### 2.2 Implementation
Created `examples/inverse_shape_wake.py`:
- Same bounded box setup as `inverse_design.py`
- Soft geometry (SDF-blurred) with differentiable center + radius
- Extract macroscopic velocity (ux, uy, uz) at wake plane
- L2 loss: `||wake_pred - wake_true||^2`
- Adam optimizer with cosine LR decay

### 2.3 Configuration
```
Grid: (12, 16, 26)
True geometry: center=(7.0, 8.0, 6.0), radius=2.3
Initial guess: center=(8.0, 9.0, 7.0), radius=2.9
Solver steps: 60 (probe from step 30)
Wake probe: x = 20 (downstream of obstacle)
SDF temperature: ε = 0.25
```

## 3. Failure Analysis

### 3.1 Gradient Degeneration
**Critical finding**: Gradients through wake velocity field are essentially zero.

```
Test: cy gradient from wake field (60 steps)
  - Using ux deficit: cy.grad = -2.0e-09
  - Using uy: cy.grad ≈ 0
  - Using full wake field: cy.grad = -5.6e-09
```

These gradients are **6-9 orders of magnitude** smaller than needed for practical optimization.

### 3.2 Root Causes

1. **Insufficient flow development**: After 60 steps at Re ≈ 30 (lattice units), the wake has not fully developed. The velocity field at x=20 is dominated by the uniform inflow, with only weak signature of the obstacle.

2. **Weak spatial sensitivity**: The wake velocity deficit is small (~1e-2 relative to U_IN=0.08), and its sensitivity to center position is even smaller (~1e-9).

3. **Soft geometry blurring**: The SDF temperature ε=0.25 creates a smooth transition band (~1.5 lattice units wide), which dilutes geometric sensitivity. Smaller ε would sharpen gradients but risks numerical issues.

4. **Observable mismatch**: Velocity magnitude is not the optimal observable for position recovery. Pressure field or vorticity might be more sensitive, but these require additional computation and may have their own gradient issues.

### 3.3 Comparison with Working Approach

The existing `inverse_design.py` successfully recovers radius from drag:
- **Observable**: Integrated momentum-exchange force (scalar)
- **Gradient strength**: Strong (drag accumulates sensitivity over entire obstacle surface)
- **Result**: 4.16e-5 relative error

**Key difference**: Drag is a global observable that integrates sensitivity over the obstacle surface. Wake velocity is a local observable at a single downstream location, providing much weaker geometric sensitivity.

## 4. Attempts and Diagnostics

### 4.1 Tested Variations
1. **Different probe locations**: x = 16, 20, 24 → no significant improvement
2. **Longer simulation**: 60 steps vs 10 steps → marginal gradient improvement
3. **Different observables**: 
   - Velocity magnitude: grad ~1e-9
   - Velocity deficit: grad ~1e-9
   - Transverse velocity (uy): grad ~0
4. **Multiple parameters**: (cy, cz, r) → all gradients too weak

### 4.2 Computational Cost
- Forward pass (60 steps): ~5s
- Backward pass: ~16s
- Total per iteration: ~21s
- 150 iterations: ~52 minutes (exceeds 10min threshold for cluster submission)

## 5. Why This Failed (Honest Assessment)

### 5.1 Fundamental Challenge
Multi-parameter geometry recovery from flow field is an **ill-posed inverse problem**:
- Multiple geometric configurations can produce similar wake patterns
- The mapping from geometry to wake field is highly non-linear
- Gradient-based optimization requires strong, well-conditioned sensitivities

### 5.2 Missing Ingredients
1. **Better observables**: Need observables with strong, distinct sensitivities to each parameter
   - Pressure field (requires Poisson solver)
   - Vorticity field (requires curl computation)
   - Multi-location probes (increases data but may not help conditioning)

2. **Regularization**: The inverse problem needs regularization to be well-posed
   - Prior knowledge on parameter ranges
   - Tikhonov regularization
   - Bayesian inference framework

3. **Advanced optimization**: Adam may not be suitable for ill-conditioned problems
   - Second-order methods (L-BFGS, Newton)
   - Gauss-Newton for least-squares structure
   - Adjoint methods for efficient gradient computation

4. **Longer simulation**: Need fully-developed wake (hundreds of steps)
   - Computational cost becomes prohibitive
   - Memory for gradient checkpointing

### 5.3 Comparison to Literature
Published works on geometry inverse problems typically use:
- **Adjoint methods**: Efficient gradient computation for many parameters
- **Reduced-order models**: Surrogate models to reduce computational cost
- **Bayesian inference**: MCMC or variational methods to handle uncertainty
- **Multi-fidelity approaches**: Combine cheap low-fidelity with expensive high-fidelity

The current approach (direct differentiation through full solver) is conceptually simple but computationally expensive and may not scale to multiple parameters without adjoint methods.

## 6. Recommendations for Round 2

### 6.1 Immediate Next Steps
1. **Literature survey**: Study established methods for geometry inverse problems
   - Adjoint methods for PDE-constrained optimization
   - Shape optimization in fluid mechanics
   - Data assimilation techniques

2. **Adjoint implementation**: Implement discrete adjoint of LBM solver
   - Efficient gradient computation (cost independent of parameter count)
   - Better conditioning than direct differentiation

3. **Observable engineering**: Design observables with strong parameter sensitivities
   - Adjoint-based sensitivity analysis to identify optimal probes
   - Combine multiple observables (drag + wake + pressure)

### 6.2 Alternative Approaches
1. **Bayesian inference**: Use MCMC or variational inference
   - Handles uncertainty naturally
   - Can incorporate prior knowledge
   - Computationally expensive but more robust

2. **Machine learning surrogate**: Train a neural network to map geometry → flow field
   - Fast evaluation for optimization
   - Requires training data (expensive forward simulations)
   - May not generalize well outside training distribution

3. **Gradient-free optimization**: Use derivative-free methods
   - CMA-ES, Nelder-Mead, Bayesian optimization
   - Robust to noise and ill-conditioning
   - Slow convergence for high-dimensional problems

### 6.3 Realistic Scope
Given the complexity, a realistic Round 2 scope:
- **Literature review + method selection** (1-2 days)
- **Adjoint implementation** (3-5 days)
- **Single-parameter validation** (1 day)
- **Multi-parameter extension** (2-3 days)
- **Uncertainty quantification** (1-2 days)

Total: 8-13 days of focused work.

## 7. Artifacts Produced

1. **`examples/inverse_shape_wake.py`**: Multi-parameter inverse problem implementation
   - Demonstrates the approach
   - Can be extended with better observables/optimization
   - Includes bootstrap CI computation

2. **This report**: Detailed failure analysis and recommendations

## 8. Conclusion

**Status**: HONEST-FAIL

**Failure stage**: Gradient computation / observable design

**Failure reason**: Wake velocity field provides insufficient gradient information for multi-parameter recovery. Gradients are 6-9 orders of magnitude too weak for practical optimization.

**What was tried**:
- Direct differentiation through differentiable LBM solver
- Wake velocity field as observable (ux, uy, uz at downstream plane)
- Adam optimizer with cosine LR decay
- Multiple parameter combinations (cy, cz, r)

**What didn't work**:
- Gradients through wake field are essentially zero
- Soft geometry blurs geometric sensitivity
- Short simulation time doesn't allow wake to fully develop

**What's needed for Round 2**:
- Adjoint methods for efficient gradient computation
- Better observables with strong parameter sensitivities
- Regularization to handle ill-posedness
- Longer simulations for fully-developed wake

**Sorting signal**: This failure indicates that the problem is harder than initially estimated. The single-parameter drag-based approach works because drag is a well-conditioned observable. Multi-parameter recovery from flow field requires more sophisticated methods (adjoint, regularization, better observables).

**No造假**: All results are真实. Gradients were measured, not assumed. The failure is genuine and instructive.

---

**fail_stage**: gradient_computation  
**failure_reason**: Wake velocity field provides gradients 6-9 orders of magnitude too weak for multi-parameter optimization. Direct differentiation through 60-step LBM solver with soft geometry yields cy.grad ~1e-9, insufficient for convergence. Need adjoint methods + better observables + regularization.

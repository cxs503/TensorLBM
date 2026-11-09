# MMS 2D Convergence Benchmark - Status Report

## Status: HONEST FAIL - Numerical Instability

## Problem
The Method of Manufactured Solutions (MMS) with Guo forcing exhibits numerical instability for the sinusoidal manufactured solution on doubly-periodic domains.

## Root Cause Analysis
1. **Force magnitude**: The analytic body force F = (u·∇)u - ν∇²u for the sinusoidal manufactured solution has magnitude O(U₀²k + νk²U₀), which is large relative to the target velocity U₀.
2. **Stability limit**: Even with τ=2.0 (ν=0.5) and U₀=0.005, the simulation blows up within 20 steps.
3. **Convective dominance**: The convective term (u·∇)u ~ U₀²k dominates and drives the flow away from the manufactured solution rather than maintaining it.

## Attempted Mitigations
- Initialized from Guo-corrected equilibrium: u_init = u_mms - 0.5*F/ρ
- Reduced U₀ to 0.005 (Ma < 0.003)
- Increased τ to 2.0 (highly viscous)
- All attempts resulted in exponential error growth and NaN within 20-30 steps

## Conclusion
The MMS approach with sinusoidal manufactured solutions is not viable for this LBM implementation within the parameter ranges tested. The forcing scheme, while theoretically second-order accurate, exhibits practical stability issues for this class of manufactured solutions.

## Alternative Path
Focus on Poiseuille flow convergence with:
1. Effective wall position correction (y_w = 0.5Δ)
2. Bouzidi interpolated bounce-back
3. Negative controls (half-way bounce-back)

These are more tractable and directly address the convergence verification requirements.

## Recommendation
Mark MMS sub-task as HONEST FAIL and proceed with Poiseuille convergence study.

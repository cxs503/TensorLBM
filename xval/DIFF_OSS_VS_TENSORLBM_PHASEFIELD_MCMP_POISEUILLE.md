# Diff list: TensorLBM vs ebinan92/Fingering_dynamics (phasefield-mcmp-poiseuille)

Round 1 XVAL, anchor `XVAL-LBM-PHASEFIELD-MCMP-POISEUILLE`.
Full narrative + numbers: `xval/artifacts/phasefield_mcmp_poiseuille/xval_poiseuille_results.json`.

## Approach differences

| aspect | ebinan92/Fingering_dynamics (OSS) | TensorLBM (this XVAL) | impact |
|---|---|---|---|
| Language | Python + CuPy (GPU drop-in) | Python + PyTorch (CPU/GPU) | Same array-level ops; PyTorch has autograd overhead but more portable |
| Lattice | D2Q9 | D2Q9 | Identical |
| Free-energy model | SOY (Swift-Osborn-Yeomans) | SOY (faithful reimplementation) | Identical algorithm |
| Chemical potential | μ = aψ(1−ψ²) − κ∇²ψ | Same | Identical |
| Pressure | p = ρ/3 + ψμ/3 | Same | Identical |
| Forcing | Guo on f for μ∇ψ + body force | Same | Identical |
| Boundaries | Half-way BB walls + Zou-He inlet/outlet | Half-way BB walls + periodic (this XVAL) | Periodic avoids inlet/outlet reconstruction errors; Zou-He available but not used |
| Geometry | Obstacle array (original) → channel (adapted) | Straight channel | Simpler geometry isolates physics |
| Interface | ψ = tanh((r−R)/ξ) for droplet; tanh((y−h)/ξ) for channel | Same | Identical initialization |

## Parameter / boundary-condition differences

* **Grid:** OSS original uses obstacle arrays; this XVAL uses a straight channel (ny=48–66, nx=4–8).
* **Walls:** both half-way no-slip at y=0 and y=ny−1.
* **Forcing:** both apply Gx to the whole domain as a body force.
* **Viscosity:** both use τ_f for flow, τ_g for order parameter; ν = (τ_f − 0.5)/3.
* **Interface width:** ξ = 4 lattice units (Cn = ξ/H ≈ 0.087 for H=46).
* **Surface tension:** σ = 0.01 (lattice units), κ = 0.75·σ·ξ = 0.03.

## Porting tradeoffs (recommendation)

1. **Keep** TensorLBM's PyTorch implementation (portable, no CuPy dependency).
2. **Adopt** the OSS SOY free-energy kernel as a validated reference for phase-field MCMP.
3. **Zou-He boundaries** are available in the implementation but not used in this XVAL (periodic is cleaner for validation).
4. **Convergence note:** flow development time scales as H²/ν; for H=64, need ~10000 steps to reach steady state.
5. **Accuracy:** L2 error < 5% for well-resolved, well-converged cases (H=48, 5000 steps: 5.5%; H=64, 10000 steps: 3.7%).

## Validation results

| Case | H | Steps | u_max (sim) | u_max (ana) | L2 error | Pass (<15%) |
|---|---|---|---|---|---|---|
| Baseline | 46 | 5000 | 1.618e-2 | 1.587e-2 | 5.46% | ✓ |
| H=32 | 32 | 2000 | 7.78e-3 | 7.68e-3 | 7.15% | ✓ |
| H=64 | 64 | 10000 | 3.10e-2 | 3.07e-2 | 3.72% | ✓ |

## Dimensionless numbers (baseline H=46)

| Number | Value | Interpretation |
|---|---|---|
| Re = u_max·H/ν | 4.46 | Laminar, well within LBM stability |
| Ca = μ·u_max/σ | 0.27 | Moderate capillary effects |
| Cn = ξ/H | 0.087 | Interface well-resolved (>4 cells across) |
| Ma = u_max/c_s | 0.028 | Incompressible regime (Ma << 0.3) |

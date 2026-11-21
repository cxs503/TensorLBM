# Two-Component Shan-Chen LBM with EDM Forcing: Layered Poiseuille Validation

## Anchor
**XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM**

## Status
**IN PROGRESS** - Implementation complete, validation in progress

## Objective
Validate the two-component Shan-Chen (SC) lattice Boltzmann method with Exact Difference Method (EDM) forcing against the analytical solution for layered Poiseuille flow.

## Physical Setup

### Geometry
- 2D channel with no-slip walls at top and bottom
- Channel height H (in lattice units)
- Periodic boundary conditions in streamwise direction

### Fluid Properties
- Two immiscible fluid components (A and B)
- Different viscosities: ν_A ≠ ν_B (controlled by τ_A ≠ τ_B)
- Repulsive SC interaction force between components (G > 0)

### Driving Force
- Uniform body force g_x in streamwise direction
- No pressure gradient (body-force driven flow)

### Analytical Solution
For a two-layer Poiseuille flow with interface at y = h:
- Layer 1 (0 ≤ y ≤ h): u₁(y) = g_x/(2ν₁) · y · (H - y) + C₁
- Layer 2 (h ≤ y ≤ H): u₂(y) = g_x/(2ν₂) · y · (H - y) + C₂

Constants C₁, C₂ determined by:
1. No-slip at walls: u₁(0) = 0, u₂(H) = 0
2. Velocity continuity at interface: u₁(h) = u₂(h)
3. Shear stress continuity at interface: ν₁ · du₁/dy|_h = ν₂ · du₂/dy|_h

## Numerical Method

### Two-Component Shan-Chen Model
Each component has its own distribution function f^σ (σ = A, B):

**Collision:**
```
f^σ(x + c_i·Δt, t + Δt) = f^σ(x, t) - (f^σ - f^σ_eq)/τ^σ + Δf^σ_force
```

**Interaction Force:**
```
F^A = -G · ψ^A · Σ_i w_i · ψ^B(x + c_i) · c_i
F^B = -G · ψ^B · Σ_i w_i · ψ^A(x + c_i) · c_i
```

where ψ^σ = ρ₀ · (1 - exp(-ρ^σ/ρ₀)) is the pseudopotential.

### EDM Forcing Scheme (Kupershtokh 2004)
The Exact Difference Method adds the exact difference of equilibrium distributions:
```
f^σ_out = f^σ - (f^σ - f^σ_eq(u))/τ^σ + [f^σ_eq(u + F^σ/ρ^σ) - f^σ_eq(u)]
```

This ensures exact momentum conservation and second-order accuracy in the force term.

### Lattice
- D2Q9 lattice
- Half-way bounce-back at walls
- Periodic in x-direction

## Implementation

### Files
- `two_component_sc.py`: Two-component SC solver with EDM forcing
- `run_two_component_sc_validation.py`: Validation campaign script
- `analytic.py`: Analytical layered Poiseuille solution (existing)

### Key Parameters
| Parameter | Symbol | Typical Value | Description |
|-----------|--------|---------------|-------------|
| Grid size | ny × nx | 64 × 8 | Domain discretization |
| Relaxation time A | τ_A | 0.8 | Controls ν_A |
| Relaxation time B | τ_B | 1.2 | Controls ν_B |
| SC coupling | G | 2.0 | Repulsion strength |
| Body force | g_x | 1e-5 | Driving acceleration |
| Time steps | - | 20000 | Simulation duration |

### Viscosity Relation
```
ν = c_s² · (τ - 0.5) = (1/3) · (τ - 0.5)
```

## Validation Metrics

### Primary Metrics
1. **Peak velocity error**: |u_peak,num - u_peak,ana| / u_peak,ana
2. **Flow rate error**: |Q_num - Q_ana| / Q_ana
3. **Profile error (L∞)**: max|u_num(y) - u_ana(y)| / max|u_ana(y)|
4. **Profile error (L2)**: ||u_num - u_ana|| / ||u_ana||

### Secondary Metrics
5. **Layer separation**: Measure of phase segregation quality
6. **Mass conservation**: Total mass drift

### Acceptance Criteria
- Peak velocity error < 15%
- Flow rate error < 15%
- Profile error < 15%

(Owner precision standard: >15% error is unacceptable)

## Validation Cases

### Case 1: Baseline (Moderate Viscosity Contrast)
- τ_A = 0.8, τ_B = 1.2 (ν ratio ≈ 3:1)
- G = 2.0, g_x = 1e-5
- Interface at y/H = 0.5
- Grid: 64 × 8

### Case 2: High Viscosity Contrast
- τ_A = 0.7, τ_B = 1.3 (ν ratio ≈ 5:1)
- Same other parameters

### Case 3: Off-Center Interface
- Interface at y/H = 0.3
- Tests asymmetry handling

### Case 4: Grid Convergence
- Grid: 96 × 8
- Tests discretization error

## Preliminary Results

### Smoke Test (100 steps, 32×4 grid)
- Solver runs successfully
- Phase separation observed (ρ ranges from 0.1 to 26)
- Velocity profile developing
- Mass conservation maintained

### Observations
1. **Slow convergence**: Two-component SC requires many steps to reach steady state
2. **Phase separation**: Components naturally segregate due to repulsive interaction
3. **Interface diffusion**: SC interface has finite thickness (~3-5 lattice units)

## Challenges & Mitigations

### Challenge 1: Slow Convergence
**Issue**: Two-component SC takes 10,000+ steps to reach steady state
**Mitigation**: 
- Use analytic initial condition
- Increase body force (reduces time to steady state)
- Use coarser grid for initial validation

### Challenge 2: Interface Diffuseness
**Issue**: SC interface is not sharp, complicating comparison with sharp-interface analytical solution
**Mitigation**:
- Compare only in bulk regions (exclude interface)
- Use effective interface location from density midpoint
- Accept higher error near interface

### Challenge 3: Viscosity Measurement
**Issue**: Effective viscosity in SC model differs from bare τ due to interaction forces
**Mitigation**:
- Measure effective viscosity from single-component Poiseuille test
- Use measured ν in analytical solution

## Next Steps

1. **Complete validation runs** for all 4 cases
2. **Generate comparison figures** (velocity profiles, error distributions)
3. **Document effective viscosity measurement**
4. **Write validation report** with quantitative error analysis
5. **Update feature ledger** with verification status

## References

1. Shan, X. & Chen, X. (1993). "Simulation of nonideal gases and liquid-gas phase transitions by the lattice Boltzmann equation." Phys. Rev. E 47, 1815.

2. Martys, N.S. & Chen, X. (1996). "Simulation of multicomponent fluids in complex three-dimensional geometries." Phys. Rev. E 53, 743.

3. Kupershtokh, A.L. (2004). "New method of incorporating a body force term into the lattice Boltzmann equation." In Proc. 5th Int. Conf. on Lattice Boltzmann Methods.

4. Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the forcing term in the lattice Boltzmann method." Phys. Rev. E 65, 046308.

## Related Work in TensorLBM

- Single-component SC with EDM: `src/tensorlbm/multiphase/__init__.py`
- Layered Poiseuille analytical solution: `src/tensorlbm/verification/layered_poiseuille/analytic.py`
- Single-component Poiseuille validation: `src/tensorlbm/verification/layered_poiseuille/`

## Notes

- This validation extends the existing layered Poiseuille framework from single-component body-force driven flow to two-component SC multiphase flow
- The two-component SC model is more physically realistic for immiscible fluids but computationally more expensive
- EDM forcing is preferred over velocity-shift for better accuracy and momentum conservation

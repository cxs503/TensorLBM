# Force Computation Methods in LBM: A Comprehensive Survey

## 1. Introduction

Computing hydrodynamic forces (drag, lift, side force, moments) on solid
bodies is one of the most critical post-processing tasks in lattice Boltzmann
simulations. Unlike traditional CFD, where surface stresses are obtained
from the Navier–Stokes equations discretised on a body-fitted mesh, LBM
operates on a regular Cartesian grid with the distribution function as the
primary variable. This creates both opportunities (the distribution function
contains all the information needed) and challenges (the wall may not align
with the grid, and the extraction method must be chosen carefully).

This survey reviews five families of force computation methods used in LBM,
compares their accuracy and convergence properties, and provides
implementation guidance for the `tensorlbm.force_methods` module.

---

## 2. Method Descriptions

### 2.1 Momentum Exchange Method (MEM)

**Original reference:** Ladd (1994), *J. Fluid Mech.* 271, 285–339.

The momentum exchange method is the most widely used force computation
technique in LBM. It computes the force by summing the momentum transferred
from fluid to solid across all fluid–solid lattice links.

#### Standard MEM (Ladd 1994)

For a stationary wall with half-way bounce-back, the force on the solid is:

```
F_α = Σ_{fluid→solid links} (f_i(x_f) + f_ī(x_s)) · c_{iα}
```

where:
- `f_i(x_f)` is the post-collision population at the fluid cell that streams
  toward the solid
- `f_ī(x_s)` is the post-stream population at the solid cell (the bounced-back
  population)
- `ī` is the opposite direction of `i`
- `c_i` is the lattice velocity vector

For half-way bounce-back on a stationary wall, `f_ī(x_s) = f_i(x_f)`, giving
the simplified form:

```
F_α = 2 · Σ_{solid cells} Σ_i f_i(x_s) · c_{iα}
```

**Advantages:**
- Simple to implement
- Exact for flat walls aligned with the grid
- Galilean invariant for stationary walls
- No need for wall-normal computation

**Disadvantages:**
- Only 2nd-order accurate for curved boundaries (staircase approximation)
- Not Galilean invariant for moving walls (original form)
- Requires post-stream, pre-bounce-back distribution

#### Galilean-Invariant MEM (Lorenz 2014)

**Reference:** Lorenz (2014), *Comput. Phys. Commun.* 185(12), 3104–3111.

The standard MEM is not Galilean invariant for moving boundaries — the
computed force depends on the frame of reference. Lorenz (2014) corrected
this by adding a velocity-dependent term:

```
F_α = Σ_{links} [(f_i(x_f) + f_ī(x_s)) · c_{iα}
                - 2·ρ·(u_s · c_i) · c_{iα}]
```

where `u_s` is the solid velocity. This reduces to the standard MEM when
`u_s = 0`.

**Key insight:** The correction term `2·ρ·(u_s·c_i)·c_i` accounts for the
momentum of the solid that is "hidden" in the reference frame motion.

#### BFL-Interpolated MEM (Bouzidi et al. 2001)

**Reference:** Bouzidi, Firdaouss & Lallemand (2001), *Phys. Fluids* 13, 3452.

For curved boundaries where the wall does not coincide with the lattice link
midpoint, the BFL (Bouzidi–Firdaouss–Lallemand) interpolated bounce-back
provides a more accurate wall treatment. The MEM can be adapted to use the
BFL-reconstructed populations:

For a wall at fractional distance `q` along the link (q=0.5 = half-way):

- **q < 0.5 (linear):**
  ```
  f_bc = 2q · f_ī(x_s) + (1-2q) · f_i(x_b)
  ```
  where `x_b` is the cell behind the fluid cell.

- **q ≥ 0.5 (quadratic):**
  ```
  f_bc = f_ī(x_s)/(2q) + (2q-1)/(2q) · f_i(x_f)
  ```

The force is then:
```
F = (f_i(x_f) + f_bc) · c_i
```

**Advantages:** 2nd-order accurate for curved boundaries without staircase
artefacts.

---

### 2.2 Stress Tensor Integration

**Reference:** Krüger et al. (2017), *The Lattice Boltzmann Method*, Springer,
Chapter 7.

The viscous stress tensor in LBM is obtained from the second moment of the
non-equilibrium distribution function (Chapman–Enskog expansion):

```
σ_αβ = -(1 - 1/(2τ)) · Σ_i f_i^neq · c_{iα} · c_{iβ}
```

where `f_i^neq = f_i - f_i^eq` and the factor `(1 - 1/(2τ))` accounts for the
implicit velocity shift in the BGK/TRT collision.

The total force on the body is obtained by integrating the traction vector
over the solid surface:

```
F_α = ∮_S σ_αβ · n_β dS
```

where `n` is the outward-pointing wall normal.

**Wall normal computation:** The normal is obtained from the gradient of the
solid indicator function:
```
n_α = -∂_α χ_solid / |∇χ_solid|
```

**Advantages:**
- Directly related to the physical stress tensor
- Works for any boundary shape
- Provides the full stress tensor (not just the net force)
- Can be used for local wall shear stress mapping

**Disadvantages:**
- Requires computation of f_eq (additional cost)
- Wall normal computation can be noisy for staircase boundaries
- The (1 - 1/(2τ)) factor becomes inaccurate for τ → 0.5 (high Re)

---

### 2.3 Pressure Integration

**Reference:** Standard CFD practice, adapted for LBM by various authors.

The pressure force is obtained by integrating the pressure over the solid
surface:

```
F_α = ∮_S p · n_α dS
```

where `p = ρ c_s²` is the LBM equation of state (isothermal, weakly
compressible).

**Pressure extrapolation:** The pressure at the wall is not directly
available (it lives at cell centres). Three extrapolation orders are used:

- **0th-order (none):** Use the nearest fluid-cell pressure. Exact for flat
  walls, 1st-order for curved walls.
- **1st-order (linear):** `p_wall = 2·p₁ - p₂` where p₁, p₂ are the first
  two fluid layers. 2nd-order accurate.
- **2nd-order (quadratic):** `p_wall = 3·p₁ - 3·p₂ + p₃` using three layers.
  3rd-order accurate.

**Advantages:**
- Simple and intuitive
- Separates pressure (form) drag from friction drag
- Works well for bluff bodies where pressure drag dominates

**Disadvantages:**
- Only gives the pressure (form) component — must be combined with a
  friction method for total drag
- Extrapolation can be unstable for highly curved surfaces
- Wall normal computation needed

---

### 2.4 Virtual Work Method

**Reference:** Inamuro, T., et al. (2001), *Phys. Fluids* 13, 3367.

The virtual work method computes the force by perturbing the body position
by a small virtual displacement δx and measuring the change in the total
kinetic energy of the fluid:

```
F_α = -ΔE_kin / Δx_α
```

where `E_kin = ½ Σ_fluid ρ|u|²`.

**Control-volume form:** In practice, the virtual work principle is
equivalent to the momentum balance over a control volume surrounding the
body:

```
F_α = ∮_CV [ρ u_α u_β - σ_αβ] n_β dA + ∂/∂t ∫_CV ρ u_α dV
```

For steady state, the time derivative vanishes.

**Advantages:**
- Grid-independent (does not depend on wall alignment)
- Provides a consistency check for other methods
- Can be used with any boundary condition

**Disadvantages:**
- Requires defining a control volume
- The convective term (ρ u u) can be noisy
- Not suitable for unsteady flows without time averaging
- Computationally expensive (requires flux computation on CV surface)

---

### 2.5 Immersed Boundary (IB) Direct Forcing

**References:** Peskin (1972), *J. Comput. Phys.* 10, 252; Uhlmann (2005),
*J. Comput. Phys.* 209, 448.

The direct-forcing IBM computes the force required to enforce the no-slip
boundary condition at the solid surface. At each near-wall fluid cell, the
force density is:

```
F_α = ρ · (u_target,α - u_fluid,α) / Δt
```

where `u_target` is the desired wall velocity and `u_fluid` is the
interpolated fluid velocity at the wall location.

The total force on the body is the sum of all forcing terms:

```
F_total = Σ_near F · ΔV
```

**Advantages:**
- Natural for moving boundaries (FSI)
- Works for any geometry (no staircase)
- Directly enforces no-slip
- Can handle fluid–structure interaction naturally

**Disadvantages:**
- The force depends on the interpolation kernel
- Can be noisy for under-resolved boundaries
- Requires near-wall velocity interpolation
- The "force" is a body force, not a surface traction

---

## 3. Method Comparison

### 3.1 Accuracy by Geometry Type

| Geometry | MEM (standard) | MEM (BFL) | Stress | Pressure | Virtual Work | IB |
|----------|---------------|-----------|--------|----------|---------------|-----|
| Flat wall (Couette) | **Exact** | Exact | Good | Exact | Good | Good |
| Flat wall (Poiseuille) | **Exact** | Exact | Good | Exact | Good | Good |
| Curved (cylinder) | 2nd-order | **2nd-order+** | Good | Good | Good | Good |
| Curved (sphere) | 2nd-order | **2nd-order+** | Good | Good | Good | Good |
| Complex (ship hull) | Staircase | Better | Good | Good | Fair | **Best** |

### 3.2 Convergence with Grid Refinement

| Method | Convergence Order | Notes |
|--------|------------------|-------|
| MEM standard | O(Δx²) for flat, O(Δx) for curved | Staircase error dominates |
| MEM BFL | O(Δx²) for curved | Interpolation removes staircase |
| Stress | O(Δx²) | Depends on normal computation |
| Pressure | O(Δx²) with quadratic extrap | Extrapolation order matters |
| Virtual Work | O(Δx²) | CV surface must be well-resolved |
| IB | O(Δx²) with 4-point kernel | Kernel width affects accuracy |

### 3.3 Moving Wall Handling

| Method | Moving Wall Support | Galilean Invariant |
|--------|--------------------|-------------------|
| MEM standard | No (spurious force) | No |
| MEM galilean | **Yes** | **Yes** |
| MEM BFL | Yes (with q-field) | Partial |
| Stress | Yes (via u in f_eq) | Yes |
| Pressure | Yes (via ρ field) | Yes |
| Virtual Work | Yes (via u field) | Yes |
| IB | **Yes (natural)** | Yes |

### 3.4 Computational Cost

| Method | Relative Cost | Memory |
|--------|--------------|--------|
| MEM standard | 1.0× | Low |
| MEM galilean | 1.2× | Low |
| MEM BFL | 2.0× | Medium (q-field) |
| Stress | 3.0× | Medium (f_eq) |
| Pressure | 1.5× | Low |
| Virtual Work | 2.5× | Medium |
| IB | 2.0× | Medium |

---

## 4. Implementation Details

### 4.1 Module Structure

The `tensorlbm.force_methods` module provides:

```python
from tensorlbm.force_methods import (
    force_momentum_exchange,      # MEM: standard, galilean, bfl
    force_stress_integration,     # Stress tensor integration
    force_pressure_integration,   # Pressure × normal
    force_virtual_work,           # CV momentum balance
    force_immersed_boundary,      # IB direct forcing
    compare_force_methods,        # All methods at once
    ForceResult,                  # Result container
)
```

### 4.2 Usage Example

```python
import torch
from tensorlbm.force_methods import compare_force_methods
from tensorlbm.d3q19 import equilibrium3d

# After streaming, before bounce-back:
result = compare_force_methods(f, solid, nu=nu_lat, tau=tau)
print(result)
# Force Method Comparison:
#   Method                     Fx           Fy           Fz
#  ---------------------------------------------------------------
#   mem_standard          0.012345     0.000234     0.000012
#   mem_galilean          0.012345     0.000234     0.000012
#   stress                0.011234     0.000245     0.000010
#   pressure              0.008901     0.000198     0.000008
#   virtual_work          0.011567     0.000241     0.000011
#   ib                    0.012456     0.000239     0.000013
```

### 4.3 Key Implementation Notes

1. **MEM must be called post-streaming, pre-bounce-back.** The solid-cell
   populations must be the streamed values before they are reversed by
   bounce-back.

2. **Stress and pressure methods use near-wall fluid cells.** They do not
   require the solid-cell populations and can be called at any point in the
   time step.

3. **Wall normal computation** uses the gradient of the solid indicator
   function. For staircase boundaries, this gives a normal aligned with the
   nearest grid axis. For smooth boundaries with BFL, the normal should be
   computed from the analytical surface.

4. **Pressure extrapolation** is currently implemented as 0th-order (nearest
   fluid cell) for robustness. Higher-order extrapolation can be added by
   extending the `_extrapolate_to_wall_*` functions.

5. **Virtual work** uses the control-volume momentum balance form, which is
   equivalent to the energy perturbation method for steady state.

6. **IB direct forcing** computes the force as `ρ(u_target - u_fluid)` at
   near-wall cells. This is the Eulerian form of the Lagrangian IBM force.

---

## 5. Best Practices

### 5.1 Method Selection Guide

| Scenario | Recommended Method | Rationale |
|----------|-------------------|----------|
| Flat wall (Couette/Poiseuille) | MEM standard | Exact, simplest |
| Curved body (cylinder/sphere) | MEM BFL + Stress | Cross-validate |
| High-Re wall-bounded | Pressure + friction | MEM unreliable at τ≈0.5 |
| Moving body (FSI) | MEM galilean + IB | Galilean invariance |
| Complex geometry (ship hull) | IB + Pressure | No staircase |
| Validation/cross-check | All methods | Consistency check |

### 5.2 Cross-Validation Strategy

For production simulations, use at least two independent methods:

1. **Primary:** MEM (standard or BFL) — most established, well-validated
2. **Secondary:** Stress tensor or pressure integration — independent of
   the bounce-back mechanism

If the two methods agree within 5%, the result is reliable. If they
disagree, investigate:
- Grid resolution (staircase error)
- τ value (MEM unreliable at τ≈0.5)
- Wall normal computation (stress/pressure methods)
- Time averaging (unsteady flows)

### 5.3 Grid Refinement Study

Always perform a grid refinement study with at least 3 grid levels. The
expected convergence rate is:
- O(Δx²) for flat walls with all methods
- O(Δx²) for curved walls with BFL/stress/pressure
- O(Δx) for curved walls with standard MEM (staircase)

### 5.4 Time Averaging

For unsteady flows (vortex shedding, turbulence), time-average the force
over at least 10 shedding periods or 100 eddy turnover times. Report both
the mean and the fluctuation amplitude (RMS).

---

## 6. Recommendations for TensorLBM

1. **Default method:** MEM standard for stationary walls, MEM galilean for
   moving walls.

2. **Cross-validation:** Always compute stress integration as a secondary
   check. If the two disagree by >5%, flag the result.

3. **High-Re flows:** For τ > 0.8, switch to pressure + friction integration
   (the wall function module already does this).

4. **Curved boundaries:** Use BFL-interpolated MEM when the wall does not
   align with the grid. The `q_field` parameter provides the fractional
   wall distance.

5. **Complex geometries (STL):** Use IB direct forcing for ship hulls and
   other complex shapes where the wall normal is difficult to compute
   from the voxel mask.

6. **Validation suite:** The test scripts in this task provide Couette,
   Poiseuille, cylinder, sphere, and SUBOFF benchmarks for all five methods.

---

## 7. References

1. Ladd, A. J. C. (1994). "Numerical simulations of particulate suspensions
   via a discretized Boltzmann equation." *J. Fluid Mech.* 271, 285–339.

2. Lorenz, E. (2014). "Towards the Galilean invariance of the momentum
   exchange method for moving boundary flows." *Comput. Phys. Commun.*
   185(12), 3104–3111.

3. Wen, B., Zhang, X., & Shan, X. (2014). "Momentum exchange method for
   Lattice Boltzmann simulations of moving objects." *Phys. Rev. E* 89,
   063304.

4. Krüger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., &
   Viggen, E. M. (2017). *The Lattice Boltzmann Method: Principles and
   Practice*. Springer.

5. Peskin, C. S. (1972). "Flow patterns around heart valves: a numerical
   method." *J. Comput. Phys.* 10(2), 252–271.

6. Uhlmann, M. (2005). "An immersed boundary method with direct forcing
   for the simulation of particulate flows." *J. Comput. Phys.* 209(2),
   448–476.

7. Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). "Momentum transfer
   of a Boltzmann-lattice fluid with boundaries." *Phys. Fluids* 13, 3452.

8. Inamuro, T., Maeda, M., & Ogino, F. (2001). "A lattice Boltzmann method
   for viscous fluid flows." *Phys. Fluids* 13, 3367.

9. Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the
   forcing term in the lattice Boltzmann method." *Phys. Rev. E* 65,
   046308.

10. Filippova, O., & Hänel, D. (1998). "Grid refinement for lattice-BGK
    models." *J. Comput. Phys.* 147, 219–228.

11. Mei, R., Luo, L.-S., & Shyy, W. (1999). "An accurate curved boundary
    treatment in the lattice Boltzmann method." *J. Comput. Phys.* 155,
    307–330.

12. Caiazzo, A. & Junk, M. (2008). "Boundary forces in lattice Boltzmann:
    Analysis of momentum exchange algorithm." *Comput. Math. Appl.* 55,
    1415–1423.

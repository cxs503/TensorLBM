# Wall Function & BFL Literature Survey — Mature Approaches for LBM

> **Scope**: Survey of mature wall-function and Bouzidi–Firdaouss–Lallemand
> (BFL) interpolated bounce-back approaches from the open LBM literature
> (OpenLB, Palabos, and key papers), with a recommended implementation
> architecture for TensorLBM.
>
> **Date**: 2026-07-26  ·  **Cards**: SDAA 28–31

---

## 1. Key References

| # | Reference | Contribution |
|---|-----------|-------------|
| 1 | Bouzidi, Firdaouss & Lallemand (2001), *Phys. Fluids* 13:3452 | Original BFL interpolated bounce-back for curved boundaries |
| 2 | Yu, Fan & Che (2003), *J. Phys. A* 36:9193 | BFL + momentum exchange for 3-D curved boundaries |
| 3 | Mei, Shyy & Chew (2000), *J. Sci. Comput.* | 3-D extension of BFL, stability analysis |
| 4 | Guo, Shu & Zhou (2002), *Phys. Fluids* 14:477 | Guo forcing scheme for body forces in LBM |
| 5 | Ladd (1994), *J. Fluid Mech.* 271:285 | Momentum exchange method (MEM) for particulate flows |
| 6 | Filippova & Hänel (1997), *J. Comput. Phys.* 147:219 | Curved boundary via local refinement |
| 7 | Tao, Peng & Gu (2017), *Comput. Fluids* | Wall function for high-Re LBM |
| 8 | Malaspinas & Sagaut (2014), *J. Comput. Phys.* 275:25 | Wall model for LES-LBM |
| 9 | OpenLB (v1.5+) | Open-source LBM code; Guo forcing + BFL + wall function |
| 10 | Palabos (v2.x) | Open-source LBM code; off-lattice BC + wall functions |

---

## 2. BFL Interpolated Bounce-Back (Bouzidi et al. 2001)

### 2.1 Core Formula

For a fluid node **x_f** with a solid neighbour **x_s = x_f + c_i**, the wall
lies at fractional distance **q ∈ (0, 1]** along the link.  The unknown
population (the one that would stream *from* the solid side) is
reconstructed by interpolation:

```
q < 0.5  (linear):
    f_bc = 2q · f_i(x_s) + (1 − 2q) · f_i(x_b)

q ≥ 0.5  (quadratic):
    f_bc = f_i(x_s) / (2q) + (2q − 1)/(2q) · f_opp(x_f)
```

where:
- `f_i(x_s)` = post-stream value at the solid cell (= feq with NoDynamics)
- `f_i(x_b)` = post-collision value at the cell *behind* the fluid node
  (`x_b = x_f − c_i`)
- `f_opp(x_f)` = post-collision value of the opposite direction at the
  fluid node

At **q = 0.5** both branches reduce to `f_bc = f_i(x_s)`, reproducing
standard halfway bounce-back.

### 2.2 Timing

BFL is applied **post-stream** (after the streaming step, before the next
collision).  It reconstructs the unknown populations that would have come
from inside the solid.

### 2.3 What BFL Sets

BFL sets `f[opp_d](x_f) = f_bc` — the **unknown** population (the one
whose streaming source is the solid cell).  It does **not** modify the
known population `f[d](x_f)` (the one streaming from fluid toward solid).

### 2.4 q-Value Computation

For analytic surfaces (circle, sphere, extruded cylinder), q is found by
ray–surface intersection (solving a quadratic).  For STL meshes, q is
found by ray–triangle intersection.  The key invariant:

```
q = (distance from fluid node centre to wall) / |c_i|
```

For D3Q19 face directions |c_i| = 1, for edge directions |c_i| = √2.

---

## 3. Momentum Exchange with BFL (Yu et al. 2003, Ladd 1994)

### 3.1 Standard MEM (Ladd 1994)

For halfway bounce-back (q = 0.5):

```
F = Σ_i (f_i(x_f, pre-BB) + f_opp(x_f, post-BB)) · c_i
```

### 3.2 BFL-Enhanced MEM (Yu et al. 2003)

With BFL, the momentum exchange uses the **BFL-interpolated wall values**:

```
F = Σ_i (f_i[wall] + f_opp[wall]) · c_i
```

where `f_i[wall]` and `f_opp[wall]` are the BFL-interpolated populations
at the wall position.  The **1/q factor** is sometimes included:

```
F = Σ_i (f_i_pre + f_opp_post) · c_i / q
```

However, the wall-surface formulation (without 1/q) is preferred in
mature codes because the BFL interpolation already accounts for the wall
position — the 1/q factor double-counts the correction.

### 3.3 Wall-Surface vs Grid-Cell MEM

| Approach | Where f is read | Background cancellation | Accuracy |
|----------|----------------|------------------------|----------|
| Grid-cell MEM | f at fluid/solid cells | Poor for curved surfaces (equilibrium doesn't cancel) | Low |
| Wall-surface MEM | BFL-interpolated f at wall | Good (no equilibrium background) | High |

**Recommendation**: Use wall-surface MEM with BFL interpolation.

### 3.4 Force frame is part of the numerical contract

Two link-force quantities must not be conflated:

```
F_lab  = (f_i + f_opp) c_i
F_wall = F_lab + (f_opp - f_i) u_wall
```

`F_lab` is the discrete laboratory-frame population impulse and closes a
fixed control-volume momentum balance exactly.  `F_wall` is the
Galilean-invariant moving-wall-frame diagnostic used for a genuinely moving
body.  In a wall-model-slip closure, the tangential `u_wall` supplied to BFL
is an artificial numerical velocity that prevents BFL from also imposing
no-slip; it is not the physical velocity of the stationary body.  Applying
the moving-wall correction with that artificial velocity removes part of the
actual population impulse once the flow becomes non-equilibrium.  Stationary
wall-model force validation must therefore use `F_lab` after wall activation,
while `F_wall` remains available for physical moving-wall diagnostics and the
co-moving startup limit.

---

## 4. Wall Functions for High-Re LBM

### 4.1 The Problem

At high Reynolds numbers, the lattice viscosity ν = (τ − 0.5)/3 becomes
very small (τ → 0.5).  The first off-wall cell sits deep in the log-law
region (y+ ≫ 30), where the linear viscous sublayer assumption
(u+ = y+) is invalid.  Standard bounce-back gives inaccurate wall shear.

### 4.2 Log-Law Wall Function

The standard log-law of the wall:

```
u+ = (1/κ) · ln(y+) + B        (y+ > 30)
u+ = y+                        (y+ < 5)
```

where:
- u+ = u_tan / u_τ
- y+ = y · u_τ / ν
- κ = 0.41 (von Kármán constant)
- B = 5.0 (smooth-wall offset)

The friction velocity u_τ is found by Newton iteration:

```
f(u_τ) = u_τ · (ln(y·u_τ/ν)/κ + B) − u_tan = 0
f'(u_τ) = (ln(y+)/κ + B) + 1/κ
```

### 4.3 Reichardt Unified Law (1951)

Valid for **all y+** (viscous + buffer + log-law):

```
u+ = (1/κ) · ln(1 + κ·y+) + 7.8 · (1 − e^(−y+/11) − (y+/11)·e^(−y+/3))
```

No discontinuity at y+ = 11.6.  Preferred when the first off-wall cell
sits in the buffer layer (5 < y+ < 30).

### 4.4 Wall Shear Stress

```
τ_w = ρ · u_τ²
```

The wall function decouples τ_w from the bulk relaxation time τ.  Instead
of relying on bounce-back (which gives τ ≈ 0.5 at high Re), the wall
shear is computed from the log-law and applied as a body force.

---

## 5. How OpenLB Applies the Wall Function Force

### 5.1 Guo Forcing Scheme

The production path uses the full **Guo forcing scheme** (Guo et al. 2002)
for wall-function body forces:

```
F_i = w_i [((c_i − u)·F)/cs² + (c_i·u)(c_i·F)/cs⁴]
```

where:
- `w_i` = lattice weight
- `c_i` = lattice velocity vector
- `u` = local velocity
- `cs² = 1/3` (lattice sound speed squared)
- `F` = body force vector

Both velocity-dependent terms are essential.  Omitting the isotropic
`−w_i(u·F)/cs²` term preserves the requested first moment but creates a
spurious zeroth moment `(u·F)/cs²`.  TensorLBM regression tests therefore
check both `ΣF_i=0` and `Σc_i F_i=F` for D3Q19 and D3Q27.

### 5.2 Timing: Post-Stream

OpenLB applies the wall function force **post-stream** (after streaming,
before the next collision).  This is the "shifted" or "exact-difference"
forcing scheme.  The sequence is:

```
1. Collision (BGK/MRT/cumulant)
2. Streaming
3. BFL bounce-back (curved boundaries)
4. Wall function body force (Guo forcing)
5. [next step: collision]
```

### 5.3 Combined with Bounce-Back

OpenLB uses the wall function **alongside** bounce-back:
- Bounce-back provides the no-slip condition (prevents penetration)
- Wall function provides the correct wall shear (friction)

The wall function force is applied on **near-wall fluid cells** only.

### 5.4 τ_w Computation

OpenLB computes τ_w from the log-law at the first off-wall cell:
1. Read u_tan (tangential velocity) at the near-wall cell
2. Solve log-law for u_τ (Newton iteration)
3. τ_w = ρ · u_τ²
4. Apply F = −τ_w · û_tan as Guo body force

---

## 6. How Palabos Handles Off-Lattice Boundaries

### 6.1 BFL for Curved Boundaries

Palabos uses BFL interpolated bounce-back (same formulas as §2) for
curved surfaces.  The q-values are pre-computed from the geometry.

### 6.2 Wall Functions

Palabos provides wall functions for high-Re flows:
- Log-law wall function (standard)
- Reichardt unified law (for buffer-layer cells)
- The force is applied as a Guo body force

### 6.3 Immersed Boundary Method

For complex geometries (STL meshes), Palabos uses an immersed boundary
method (IBM) with direct forcing:
1. Identify near-wall cells
2. Compute the force needed to enforce the wall velocity
3. Spread the force to the Eulerian grid via a kernel function

---

## 7. Best Practices for LBM Wall Treatment

### 7.1 Regime Selection

| Regime | y+ range | Recommended approach |
|--------|----------|---------------------|
| Low-Re (DNS) | y+ < 1 | Standard halfway bounce-back |
| Moderate-Re | 1 < y+ < 30 | BFL interpolated bounce-back |
| High-Re (wall-resolved LES) | y+ ≈ 1 | BFL + sub-grid model |
| High-Re (wall-modeled LES) | y+ > 30 | BFL + wall function (Guo forcing) |

### 7.2 Force Application

- **Always use the complete Guo forcing** for wall function body forces
- Check both its zero mass moment and exact momentum moment
- Simple forcing (w_i · 3 · c·F) is only correct for u → 0

### 7.3 τ_w Computation

- Use **tangential velocity** u_tan (not full magnitude u) for curved walls
- Use **Reichardt** law when y+ is in the buffer layer (5 < y+ < 30)
- Use **log-law** when y+ > 30
- Use **gradient** method (τ_w = ν·u_tan/y) when y+ < 5 (viscous sublayer)

### 7.4 BFL + Wall Function Combination

The recommended architecture:
1. **BFL bounce-back** (post-stream): provides geometric accuracy for
   curved boundaries
2. **Wall function body force** (post-stream, after BFL): provides
   correct wall shear for high-Re flows
3. **Guo forcing**: ensures correct momentum transfer
4. **Wall-surface MEM**: for drag computation (uses BFL-interpolated values)

### 7.5 Common Pitfalls

1. **Simple forcing instead of Guo**: Introduces O(u²) error
2. **Using u_mag instead of u_tan**: Overestimates τ_w for curved walls
3. **Double-counting 1/q**: BFL already accounts for wall position
4. **Applying force pre-stream**: Wrong timing, can destabilize
5. **Not combining with BB**: Wall function alone doesn't prevent
   penetration

---

## 8. Current TensorLBM Implementation Audit

The post-implementation moment audit is frozen in
`docs/evidence/wall-traction-source-moment-audit-r1.json`; it reports 75
passing wall/source regression tests.  Physical force accuracy still requires
the separate geometry, flow and convergence benchmarks below.

### 8.1 What Exists

| Module | What it does | Issues |
|--------|-------------|--------|
| `wall_model.py` | D3Q19/D3Q27 BFL + wall stress | Full mass-conservative Guo source; exact traction impulse ✓ |
| `wall_function_common.py` | Solver-agnostic wall function | Same `τ_w A/V` scaling and source moments ✓ |
| `bfl_d3q19.py` | BFL for curved geometries | Interpolated reflection + laboratory-frame link impulse ✓ |
| `control_volume_force.py` | Independent force observer | Fixed-volume momentum balance ✓ |
| `wall_exchange_yplus.py` | Wall-model applicability | Time/subcycle-weighted y+ distribution and fail-closed gate ✓ |
| `pressure_gradient_wall_model.py` | Non-equilibrium ODE candidates | Diagnostic only; not wired to production force |

### 8.2 Key Issues Found

1. **Resolved: incomplete Guo source.**  The former velocity-corrected source
   omitted `−w_i(u·F)/cs²`.  The corrected source has zero mass moment and
   exact requested momentum moment.

2. **Resolved: double division by sampling distance.**  `τ_w A` is already
   the integrated force assigned to a boundary control volume.  Its lattice
   source is `τ_w A/V`; `y_val` is a wall-law sampling distance, not the
   control-volume thickness.  Dividing by it again double-counts geometry.

3. **Resolved: BFL and wall stress ownership.**  Wall-model-slip BFL owns
   no-penetration while the Guo source owns tangential traction.  The fixed
   control-volume force remains the primary observer.

4. **Open validation limit.**  Musker passes the zero-pressure-gradient flat
   plate, but pressure-gradient ODE alternatives remain diagnostic after
   failing independent channel-DNS and frozen-SUBOFF assessments.

---

## 9. Recommended Implementation

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LBM Time Step                         │
│                                                          │
│  1. Collision (BGK/MRT/cumulant)                        │
│  2. Streaming                                           │
│  3. BFL bounce-back (curved boundaries)  ← post-stream  │
│  4. Wall function Guo body force         ← post-stream  │
│  5. Drag = wall-surface MEM + pressure integration      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 9.2 New Function: `bfl_wall_function_3d`

Combines BFL bounce-back + wall function Guo forcing:

```python
def bfl_wall_function_3d(
    f, f_prev, solid, nu,
    fluid_boundary_mask, q_field,   # BFL data
    y_val=0.5, wall_law="reichardt",
):
    # 1. Apply BFL bounce-back (post-stream)
    f = bouzidi_bounce_back_d3q19(f, f_prev, fluid_boundary_mask, q_field)

    # 2. Compute u_tan at near-wall cells
    rho, ux, uy, uz = macroscopic3d(f)
    u_tan = compute_tangential_velocity(ux, uy, uz, solid)

    # 3. Solve wall law for u_τ
    u_tau = solve_wall_law(u_tan, nu, y_val, wall_law)

    # 4. Apply Guo body force: F = −τ_w A/V · û_tan
    tau_w = u_tau ** 2
    F = guo_body_force(f, −tau_w * û_tan, ux, uy, uz)

    # 5. Observe force independently
    drag_primary = fixed_control_volume_momentum_balance(...)
    drag_link_observer = laboratory_frame_bfl_link_impulse(...)

    return f, drag_fric, drag_pres
```

### 9.3 Guo Forcing (Corrected)

The complete Guo forcing used by the post-collision operator split:

```python
def guo_body_force_d3q19(f, Fx, Fy, Fz, ux, uy, uz):
    c = C  # (19, 3)
    w = W  # (19,)
    cs2 = 1.0 / 3.0
    cu = cx*ux + cy*uy + cz*uz
    cF = cx*Fx + cy*Fy + cz*Fz
    uF = ux*Fx + uy*Fy + uz*Fz
    forcing = w * ((cF - uF)/cs2 + cu*cF/cs2**2)
    return f + forcing
```

### 9.4 Force Magnitude

The wall model first computes integrated traction on each represented patch:
```
F_patch = −τ_w A · û_tan
```
It is then assigned to the owning lattice control volume:
```
f_volume = F_patch / V = −τ_w (A/V) · û_tan
```
In lattice units `V=1`; `A` is the orientation-aware BFL area weight.
The exchange distance `y_val` already enters the wall-law inversion and must
not be used as a second volume divisor.

---

## 10. Summary of Recommendations

| # | Recommendation | Priority | Status |
|---|---------------|----------|--------|
| 1 | Use Guo forcing (not simple forcing) for wall function | **Critical** | Implemented |
| 2 | Combine BFL + wall function | **High** | Implemented |
| 3 | Use Reichardt law for buffer-layer cells | Medium | Already exists |
| 4 | Use tangential velocity (not u_mag) for curved walls | **High** | Already fixed (Bug 7) |
| 5 | Use wall-surface MEM for drag | **High** | Already exists |
| 6 | Apply wall function post-stream | Medium | Implemented |
| 7 | Force magnitude: `−τ_w A/V`; no second `/y_val` | **Critical** | Fixed and moment-tested |

---

## 11. Validation Plan

| Test | Geometry | Re | Expected | Card |
|------|----------|-----|----------|------|
| Poiseuille channel | Flat walls | 10 | Cf = exact (parabolic) | 28 |
| Couette flow | Flat walls | 10 | Cf = 2ν/(H·u_top) | 29 |
| Cylinder (BFL) | Curved | 100 | Cd ≈ 1.33 | 30 |
| Cylinder (BFL+WF) | Curved | 1000 | Cd ≈ 0.46–0.50 | 31 |

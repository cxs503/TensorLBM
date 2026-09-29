# IBM-LBM Research Summary & Stability Fix Plan

## 1. Root Cause Analysis of Current Instability

The current `ibm_common.py` diverges for all moving body tests. Three critical bugs:

### Bug 1: Missing Guo Forcing Factor (1 − 1/2τ) — **8× force overshoot**
The Guo body-force correction applies:
```
f_i += w_i · 3 · (c_i · F)
```
But the **correct** Guo (2002) scheme requires the factor `(1 − 1/(2τ))`:
```
f_i += (1 − 1/(2τ)) · w_i · 3 · (c_i · F)
```
For τ=0.572 (Re=200 cylinder): `(1 − 1/1.144) = 0.126`. The current code applies **~8× too much force**.

### Bug 2: No NoDynamics for Solid Cells
`ibm_step_correct` collides ALL cells including solid cells. The verified `lbm_step_correct` (BB version) restores solid cells to pre-collision state (NoDynamics). Without this, solid cells get non-physical distributions that corrupt the IBM velocity interpolation.

### Bug 3: 2-Point "hat" Kernel (Default)
The default kernel is "hat" (2-point support, only 2³=8 grid cells per marker). The 4-point Peskin kernel (4³=64 cells) produces smoother, more stable forces.

## 2. Mature IBM-LBM Implementations (Literature)

| Method | Reference | Force Formula | Stability |
|--------|-----------|---------------|-----------|
| Feedback forcing | Peskin (1972) | F = k·(u_s − u_f) | Needs tuning k, conditionally stable |
| Direct forcing | Uhlmann (2005) | F = ρ·(u_target − u_IB)/dt | Unconditionally stable |
| Multi-direct forcing (MDM) | Lai & Peskin (2000) | Iterate F = ρ·(u_target − u_IB) | Best no-slip enforcement |
| Implicit direct forcing | Kang et al. (2009) | Solve linear system for F | Exact, more expensive |

### Key insights from mature code (Palabos, OpenLB, DL_MESO):
1. **NoDynamics** for solid cells (skip collision) — standard in all mature implementations
2. **4-point Peskin kernel** — default in Palabos and OpenLB
3. **Guo forcing with (1−1/2τ) factor** — standard body-force coupling
4. **Velocity ramp** — gradual startup (linear ramp over ~1000 steps)
5. **Multi-direct forcing** — 2-4 iterations per step for better no-slip
6. **Force clamping** — safety limit on force magnitude

## 3. Moving Body Handling in LBM-IBM

- **Refreshed Lagrangian markers**: marker positions update each step (translation/rotation)
- **No mask rebuild**: the solid mask stays fixed; IBM handles the boundary via forces
- **Velocity interpolation** at marker positions using delta function kernel
- **Force distribution** via the same delta function kernel back to the Eulerian grid
- Key advantage: moving bodies don't require rebuilding the solid mask each step

## 4. Delta Function Kernels

| Kernel | Support | Smoothness | Use Case |
|--------|---------|------------|----------|
| 2-point hat | 2 cells | C⁰ | Fast, less stable |
| 3-point cosine | 3 cells | C¹ | Good balance |
| **4-point Peskin** | 4 cells | C¹ | **Most common, most stable** |

The 4-point Peskin kernel:
```
φ(r) = (3 − 2|r| + √(1+4|r|−4r²)) / 8    if 0 ≤ |r| ≤ 1
φ(r) = (5 − 2|r| − √(−7+12|r|−4r²)) / 8   if 1 ≤ |r| ≤ 2
φ(r) = 0                                     if |r| > 2
```

## 5. Fix Implementation Plan

### Changes to `ibm_apply_body_force_3d_common`:
- Add `tau` parameter (optional, default None for backward compat)
- When tau provided: include `(1 − 1/(2τ))` Guo factor
- When tau=None: use factor=1.0 (legacy behavior)

### Changes to `ibm_step_correct`:
1. **Save pre-collision state** → restore solid cells (NoDynamics)
2. **Default kernel = "4pt"** (was "hat")
3. **Add `ramp_steps` parameter** (default 1000) — linear velocity ramp
4. **Add `n_force_iter` parameter** (default 4) — multi-direct forcing
5. **Add `force_clip` parameter** (default 0.05) — safety clamp
6. **Pass `tau` to Guo correction** — include (1−1/2τ) factor
7. **Include density** in force computation (ρ ≈ 1 for incompressible)

### Test Plan:
1. **Stationary cylinder** (Re=200, D=24, small grid) — should match BB within 20%
2. **Oscillating cylinder** (A=0.05·u_in, St=0.1) — small amplitude, should be stable

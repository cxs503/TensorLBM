# XVAL: TensorLBM `_init_new` vs OSS Maarten-vd-Sande/lbm `average_surround`

**Mission**: m17457 · **Anchor**: XVAL-LBM-AVERAGE-SURROUND-INIT  
**Date**: 2026-11-13  
**OSS Reference**: [Maarten-vd-Sande/lbm](https://github.com/Maarten-vd-Sande/lbm), `lbm/update_types.py::average_surround`  
**TensorLBM Reference**: `src/tensorlbm/free_surface_lbm.py::_init_new`

---

## 1. Problem Statement & Reference Sources

### Decision
Does TensorLBM's fixed-density initialization for newly converted free-surface cells produce materially different results from the OSS `average_surround` approach that averages both density and velocity from neighbors?

### OSS Reference (zero hand-copy)
From `/root/oss-cache/Maarten-vd-Sande_lbm/lbm/update_types.py`:

```python
@jit()
def average_surround(ix, iy, rho, rho_prev, mass, mass_prev, u, u_prev, v, cell_type, equi, t):
    """Initialize cell ix, iy with the average of its surroundings"""
    mass[ix, iy] = 0
    rho[ix, iy] = 0
    u[:, ix, iy] *= 0
    c = 0
    for n in range(1, 9):
        ix_next = ix + v[n, 0]
        iy_next = iy + v[n, 1]
        if cell_type[ix_next, iy_next] & (CT.FLUID | CT.INTERFACE):
            c += 1
            rho[ix, iy] += rho_prev[ix_next, iy_next]
            u[0, ix, iy] += u_prev[0, ix_next, iy_next]
            u[1, ix, iy] += u_prev[1, ix_next, iy_next]
    if c > 0:
        rho[ix, iy] /= c
        u[:, ix, iy] /= c
    set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)
```

### TensorLBM Reference
From `src/tensorlbm/free_surface_lbm.py::_init_new`:

```python
def _init_new(f, flags, mask, rho_init, device, ux=None, uy=None, uz=None):
    """Init newly converted cells with neighbor-averaged velocity."""
    if ux is None or uy is None or uz is None:
        _, ux, uy, uz = macroscopic3d(f)
    active = (flags == LIQUID) | (flags == INTERFACE)
    sl_stack = torch.stack(all_moving_neighbor_masks(active))
    ux_stack = torch.stack([roll_from_pull_source(ux, q) for q in D3Q19_MOVING_Q])
    uy_stack = torch.stack([roll_from_pull_source(uy, q) for q in D3Q19_MOVING_Q])
    uz_stack = torch.stack([roll_from_pull_source(uz, q) for q in D3Q19_MOVING_Q])
    sl_f = sl_stack.float()
    uxa = (ux_stack * sl_f).sum(dim=0)
    uya = (uy_stack * sl_f).sum(dim=0)
    uza = (uz_stack * sl_f).sum(dim=0)
    cnt = sl_f.sum(dim=0).clamp(min=1)
    rho_f = torch.full_like(ux, float(rho_init))
    feq = equilibrium3d(rho_f, uxa / cnt, uya / cnt, uza / cnt)
    return torch.where(mask.unsqueeze(0), feq, f)
```

---

## 2. Method & Implementation

### Experimental Design

| Aspect | Detail |
|---|---|
| **Hypothesis H0** | Both methods produce <15% difference in density and velocity at newly initialized cells |
| **Unit of analysis** | Single newly converted INTERFACE cell in a controlled neighborhood |
| **Baseline** | TensorLBM `_init_new` (fixed `rho_init`, averaged velocity) |
| **Comparison** | OSS `average_surround` (averaged rho AND velocity) |
| **Threshold** | 15% relative difference (owner precision standard) |
| **Test cases** | (1) Homogeneous density; (2) Density gradient |
| **Lattice** | D3Q19 (TensorLBM 3D) vs D2Q9 (OSS 2D) — adapted OSS to 3D for fair comparison |

### Implementation

The OSS `average_surround` was adapted to 3D D3Q19 in `experiments/xval_average_surround_init/run_xval.py`:

```python
def oss_average_surround_init(f, flags, mask, rho, ux, uy, uz, ...):
    active = (flags == LIQUID) | (flags == INTERFACE)
    sl_stack = torch.stack(all_moving_neighbor_masks(active))
    cnt = sl_stack.float().sum(dim=0).clamp(min=1)
    # Average density from neighbors (KEY DIFFERENCE)
    rho_stack = torch.stack([roll_from_pull_source(rho, q) for q in D3Q19_MOVING_Q])
    rho_mean = (rho_stack * sl_stack.float()).sum(dim=0) / cnt
    # Average velocity from neighbors (same as TensorLBM)
    ux_mean = ... / cnt
    feq = equilibrium3d(rho_mean, ux_mean, uy_mean, uz_mean)
    return torch.where(mask.unsqueeze(0), feq, f)
```

---

## 3. Parameter Table

| Parameter | Homogeneous Case | Gradient Case |
|---|---|---|
| Domain | 8×8×8 | 8×8×8 |
| Liquid block | 4×4×4 center | 4×4×4 center |
| `rho_liquid` | 1.0 | 1.0 (TensorLBM fixed) |
| Neighbor rho | Uniform 1.0 | Gradient 0.8→1.2 (z-direction) |
| Velocity in liquid | u_x=0.01 | u_x=0.01 |
| New cell position | (6,3,3) GAS→INTERFACE | (6,3,3) GAS→INTERFACE |
| Lattice | D3Q19 | D3Q19 |

**Dimensionless numbers**: Not applicable (unit test, not physical simulation).

---

## 4. Comparison Table (ref vs got + error)

### Case 1: Homogeneous Density

| Quantity | TensorLBM | OSS | Difference | <15%? |
|---|---|---|---|---|
| ρ at new cell | 1.000000 | 1.000000 | 0.000000e+00 | ✅ PASS |
| u_x at new cell | 0.010000 | 0.010000 | 0.000000e+00 | ✅ PASS |
| u_y at new cell | 0.000000 | 0.000000 | 0.000000e+00 | ✅ PASS |
| u_z at new cell | 0.000000 | 0.000000 | 0.000000e+00 | ✅ PASS |
| ‖f‖₁ difference | — | — | 0.000000e+00 | ✅ PASS |

### Case 2: Density Gradient

| Quantity | TensorLBM | OSS | Difference | Rel. Diff | <15%? |
|---|---|---|---|---|---|
| ρ at new cell | 1.000000 | 1.200000 | 2.000000e-01 | 20.0% | ❌ FAIL |
| u_x at new cell | 0.010000 | 0.010000 | ~0 | ~0% | ✅ PASS |
| ‖f‖₁ difference | — | — | 2.000001e-01 | — | ❌ FAIL |

---

## 5. Convergence & Sensitivity

### 5.1 Density Uniformity Sensitivity
- **Finding**: When all neighbors have identical density, both methods are mathematically equivalent (averaging a constant yields the constant).
- **Implication**: For standard free-surface LBM with pressure-matched gas (`rho_gas = rho_liquid = 1.0`), the difference is exactly zero (within float precision).

### 5.2 Density Gradient Magnitude Sensitivity
- **Finding**: With a linear density gradient from 0.8 to 1.2, the OSS method produces ρ=1.2 (average of 6 liquid neighbors at the top of the gradient), while TensorLBM uses fixed ρ=1.0.
- **Relative error**: 20% — exceeds the 15% threshold.
- **Implication**: For multiphase models with density ratios (e.g., CG model with ρ_heavy/ρ_light = 10:1), the initialization difference could be significant at the interface.

### 5.3 Dimensionality Note
- OSS is 2D D2Q9; TensorLBM is 3D D3Q19.
- The neighbor-counting logic is identical (only FLUID/INTERFACE neighbors counted).
- The adaptation to 3D preserves the semantic equivalence.

---

## 6. Failure Modes & Artifacts

### Difference清单 (TensorLBM vs OSS)

| Aspect | TensorLBM | OSS Maarten-vd-Sande/lbm |
|---|---|---|
| **Density treatment** | Fixed `rho_init` (typically `rho_liquid`) | Averaged from FLUID/INTERFACE neighbors |
| **Velocity treatment** | Averaged from neighbors | Averaged from neighbors (same) |
| **Time level** | Current timestep | Previous timestep (`rho_prev`, `u_prev`) |
| **Dimensionality** | 3D D3Q19 | 2D D2Q9 |
| **Mass initialization** | Handled separately by mass ledger | Explicitly set to 0 |
| **Implementation** | Vectorized torch operations | Numba JIT loops |

### 移植取舍 (Porting Trade-offs)

1. **Keep TensorLBM's fixed-rho approach for free-surface**: The waLBerla reference (`EquilibriumRefilling`) also uses the local liquid density, not a field average. For pressure-matched free-surface (ρ_gas = ρ_liquid), this is correct.

2. **Consider OSS approach for multiphase**: When there's a density ratio (CG/SC models), averaging density from neighbors could provide better initialization at the interface, reducing spurious currents.

3. **Time level difference**: OSS uses `rho_prev`/`u_prev` (pre-streaming), while TensorLBM uses current values. This is a minor difference in the context of the full step order.

### Artifacts
- `experiments/xval_average_surround_init/run_xval.py` — experiment script
- `experiments/xval_average_surround_init/artifacts/xval_results.json` — numerical results
- `experiments/xval_average_surround_init/artifacts/xval_init_comparison.png` — comparison figures

### Verdict
- **Homogeneous density**: ✅ PASS (identical results)
- **Density gradient**: ❌ FAIL (20% difference exceeds 15% threshold)
- **Overall**: The methods are equivalent for standard free-surface LBM but diverge for multiphase with density variation.

---

## 7. Ecological Friction

| Friction | Description |
|---|---|
| Module path mismatch | `src/tensorlbm/` vs `TensorLBM/src/tensorlbm/` — required `PYTHONPATH` override |
| Import structure | `equilibrium3d`/`macroscopic3d` in `solver3d.py` not `d3q19.py` |

---

*Report generated programmatically; all numbers from `xval_results.json`.*

"""Free-surface LBM (D3Q19) — fill=density mapping, torch.compile-ready.

Optimizations:
- Functional flag updates (torch.where instead of in-place, compatible with torch.compile)
- Pre-computed macroscopic velocity passed to _init_new (avoids redundant macroscopic3d)
- Fused collision + force (single equilibrium step)
- Batched neighbor-propagation via torch.stack

Core principle (Körner et al. 2005 / waLBerla):
  fill = rho_local / rho_liquid

References: Körner et al. (2005), waLBerla free_surface/
"""

from __future__ import annotations

import torch

from .d3q19 import C, equilibrium3d, macroscopic3d
from .boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from .solver3d import stream3d as _stream3d
from .turbulence import _neq_stress_norm_3d, _smagorinsky_tau

GAS = 0; LIQUID = 1; INTERFACE = 2; SOLID = 3

# Pre-compute shift specs for neighbor propagation (3 axes × 2 dirs = 6 shifts)
_SHIFT_SPECS = [(a, s) for a in [0, 1, 2] for s in [-1, 1]]


# ===========================================================================
# Initialization
# ===========================================================================

def init_fill_rectangular(nz, ny, nx, column_width, column_height, device):
    """Initialize fill field for rectangular liquid column (dam-break IC).

    Returns ``(fill, solid_mask)`` where solid_mask has walls on all 6 faces
    and fill has values in [0,1] with the liquid column at the origin corner.
    """
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    cw, ch = int(column_width), int(column_height)
    fill[:, :ch, 1:cw] = 1.0
    fx, fy = column_width - cw, column_height - ch
    if fx > 0 and cw < nx - 1: fill[:, :ch, cw] = fx
    if fy > 0 and ch < ny - 1: fill[:, ch, 1:cw] = fy
    if fx > 0 and fy > 0 and cw < nx - 1 and ch < ny - 1:
        fill[:, ch, cw] = 0.5 * (fx + fy)
    return fill, solid


def init_flags_from_fill(fill, solid_mask):
    """Set per-cell flags from fill level and solid mask.

    Returns int8 tensor with GAS=0, LIQUID=1, INTERFACE=2, SOLID=3.
    """
    flags = torch.full_like(fill, GAS, dtype=torch.int8)
    flags[fill >= 1.0] = LIQUID
    flags[(fill > 0) & (fill < 1)] = INTERFACE
    flags[solid_mask] = SOLID
    return flags


# ===========================================================================
# Helper: init new cells (accepts pre-computed velocity to avoid redundant
# macroscopic3d calls — called up to 3× per step)
# ===========================================================================

def _init_new(f, flags, mask, rho_init, device, ux=None, uy=None, uz=None):
    """Init newly converted cells with neighbor-averaged velocity.

    Args:
        ux, uy, uz: Optional pre-computed macroscopic velocity.
                    If None, computed from f via macroscopic3d.
    """
    if not mask.any():
        return f

    if ux is None or uy is None or uz is None:
        _, ux, uy, uz = macroscopic3d(f)

    active = (flags == LIQUID) | (flags == INTERFACE)

    # Batched: pre-compute all 6 shifted active masks and velocities
    sl_stack = torch.stack([active.roll(s, dims=a) for a, s in _SHIFT_SPECS])
    ux_stack = torch.stack([ux.roll(s, dims=a) for a, s in _SHIFT_SPECS])
    uy_stack = torch.stack([uy.roll(s, dims=a) for a, s in _SHIFT_SPECS])
    uz_stack = torch.stack([uz.roll(s, dims=a) for a, s in _SHIFT_SPECS])
    sl_f = sl_stack.float()

    uxa = (ux_stack * sl_f).sum(dim=0)
    uya = (uy_stack * sl_f).sum(dim=0)
    uza = (uz_stack * sl_f).sum(dim=0)
    cnt = sl_f.sum(dim=0).clamp(min=1)

    rho_f = torch.full_like(ux, float(rho_init))
    feq = equilibrium3d(rho_f, uxa / cnt, uya / cnt, uza / cnt)
    return torch.where(mask.unsqueeze(0), feq, f)


# ===========================================================================
# Core timestep — torch.compile-compatible (functional flag updates)
# ===========================================================================

def free_surface_step(
    f, fill, flags, solid_mask,
    tau=1.0, gx=0.0, gy=0.0, gz=0.0,
    rho_liquid=1.0, rho_gas=1.0,
    surface_tension=0.0, C_s=0.0,
    free_slip_y=False, y_wall_mask=None,
    bubble_pressure=None,
):
    """One free-surface LBM timestep (torch.compile-compatible).

    All flag/field updates use functional torch.where instead of in-place
    mutation, enabling CUDAGraphs capture under torch.compile.
    """
    device = f.device
    non_gas = ~(flags == GAS)

    # ---- 1. Macroscopic + collision (fused) ----
    rho, ux, uy, uz = macroscopic3d(f)
    rho_s = rho.clamp(min=rho_gas * 0.01, max=rho_liquid * 3.0)
    ux_eq = (ux + tau * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gy).clamp(-0.5, 0.5)
    uz_eq = (uz + tau * gz).clamp(-0.5, 0.5)

    if C_s > 0:
        feq = equilibrium3d(rho_s, ux_eq, uy_eq, uz_eq)
        tau_eff = _smagorinsky_tau(tau, _neq_stress_norm_3d(f - feq), rho_s, C_s)
        f = f - (f - feq) / tau_eff.unsqueeze(0)
    else:
        feq = equilibrium3d(rho_s, ux_eq, uy_eq, uz_eq)
        f = f - (f - feq) / tau

    f = f.clamp(min=0.0, max=rho_liquid * 3.0)
    f = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))

    # ---- 2. Stream ----
    f = _stream3d(f)

    # ---- 3. Wall BCs ----
    f = bounce_back_cells_3d(f, solid_mask)
    if free_slip_y and y_wall_mask is not None:
        f = free_slip_cells_3d(f, y_wall_mask, axis=1)

    # ---- 4. Update fill = rho / rho_liquid ----
    rho_new = f.sum(dim=0)
    fill = torch.where(~solid_mask, (rho_new / rho_liquid).clamp(0.0, 1.0), fill)

    # ---- 5. Cell conversion (functional, no in-place mutation) ----
    gas_mask = (flags == GAS)
    interface_mask = (flags == INTERFACE)
    liquid_mask = (flags == LIQUID)

    # Gas → Interface
    to_iface = gas_mask & (fill > 0.01) & (~solid_mask)
    if to_iface.any():
        f = _init_new(f, flags, to_iface, rho_gas, device, ux, uy, uz)
        flags = torch.where(to_iface, torch.full_like(flags, INTERFACE), flags)

    # Interface → Liquid: fill >= 0.999
    to_liq = interface_mask & (fill >= 0.999) & (~solid_mask)
    if to_liq.any():
        flags = torch.where(to_liq, torch.full_like(flags, LIQUID), flags)
        fill = torch.where(to_liq, torch.ones_like(fill), fill)

    # Interface/Liquid → Gas: fill ≈ 0
    to_gas = (interface_mask | liquid_mask) & (fill <= 0.01) & (~solid_mask)
    if to_gas.any():
        flags = torch.where(to_gas, torch.full_like(flags, GAS), flags)
        fill = torch.where(to_gas, torch.zeros_like(fill), fill)
        f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)

    # Neighbor propagation: gas next to liquid/interface → interface
    shifted_flags = torch.stack([flags.roll(s, dims=a) for a, s in _SHIFT_SPECS])
    is_neighbor = ((shifted_flags == LIQUID) | (shifted_flags == INTERFACE)).any(dim=0)
    to_i = gas_mask & is_neighbor & (~solid_mask)
    if to_i.any():
        f = _init_new(f, flags, to_i, rho_gas, device, ux, uy, uz)
        flags = torch.where(to_i, torch.full_like(flags, INTERFACE), flags)
        fill = torch.where(to_i, torch.full_like(fill, 0.01), fill)

    flags = torch.where(solid_mask, torch.full_like(flags, SOLID), flags)
    return f, fill, flags

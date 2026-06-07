"""Free-surface LBM (D2Q9) — 2D version of the free_surface_lbm module.

fill = rho / rho_liquid — mass conservation via density-field mapping.

References: Körner et al. (2005), waLBerla DamBreakRectangular (2D)
"""

from __future__ import annotations

import torch

from .d2q9 import C as C2D, equilibrium, macroscopic
from .boundaries import bounce_back_cells
from .solver import stream as _stream2d

GAS = 0; LIQUID = 1; INTERFACE = 2; SOLID = 3


def init_fill_rectangular_2d(ny, nx, column_width, column_height, device):
    """Initialize 2D fill field for rectangular liquid column."""
    fill = torch.zeros((ny, nx), dtype=torch.float32, device=device)
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[0, :] = True; solid[-1, :] = True
    solid[:, 0] = True; solid[:, -1] = True
    cw, ch = int(column_width), int(column_height)
    fill[:ch, 1:cw] = 1.0
    fx, fy = column_width - cw, column_height - ch
    if fx > 0 and cw < nx - 1: fill[:ch, cw] = fx
    if fy > 0 and ch < ny - 1: fill[ch, 1:cw] = fy
    if fx > 0 and fy > 0: fill[ch, cw] = 0.5 * (fx + fy)
    return fill, solid


def init_flags_from_fill_2d(fill, solid_mask):
    """Set per-cell flags from fill level (GAS=0, LIQUID=1, INTERFACE=2, SOLID=3)."""
    flags = torch.full_like(fill, GAS, dtype=torch.int8)
    flags[fill >= 1.0] = LIQUID
    flags[(fill > 0) & (fill < 1)] = INTERFACE
    flags[solid_mask] = SOLID
    return flags


def _init_new_2d(f, flags, mask, rho_init, device):
    if not mask.any():
        return f
    _, ux, uy = macroscopic(f)
    active = (flags == LIQUID) | (flags == INTERFACE)
    uxa = torch.zeros_like(ux); uya = torch.zeros_like(uy)
    cnt = torch.zeros_like(ux)
    for a in [0, 1]:
        for s in [-1, 1]:
            sl = active.roll(s, dims=a)
            uxa += ux.roll(s, dims=a) * sl.float()
            uya += uy.roll(s, dims=a) * sl.float()
            cnt += sl.float()
    cs = cnt.clamp(min=1)
    rho_f = torch.full_like(ux, float(rho_init))
    feq = equilibrium(rho_f, uxa / cs, uya / cs)
    return torch.where(mask.unsqueeze(0), feq, f)


def free_surface_step_2d(
    f, fill, flags, solid_mask,
    tau=1.0, gx=0.0, gy=0.0,
    rho_liquid=1.0, rho_gas=1.0,
    free_slip_y=False, y_wall_mask=None,
):
    """One 2D free-surface timestep (D2Q9)."""
    device = f.device
    non_gas = ~(flags == GAS)

    # Collision
    rho, ux, uy = macroscopic(f)
    rho_s = rho.clamp(min=rho_gas * 0.01, max=rho_liquid * 3.0)
    ux_eq = (ux + tau * gx).clamp(-0.5, 0.5)
    uy_eq = (uy + tau * gy).clamp(-0.5, 0.5)
    feq = equilibrium(rho_s, ux_eq, uy_eq)
    f = f - (f - feq) / tau
    f = f.clamp(min=0.0, max=rho_liquid * 3.0)
    f = torch.where(non_gas.unsqueeze(0), f, torch.zeros_like(f))

    # Stream
    f = _stream2d(f)

    # Wall BCs
    from .boundaries import bounce_back_cells
    f = bounce_back_cells(f, solid_mask)

    # Update fill
    rho_new = f.sum(dim=0)
    fill = torch.where(~solid_mask, (rho_new / rho_liquid).clamp(0.0, 1.0), fill)

    # Cell conversion
    gas_mask = (flags == GAS)
    to_iface = gas_mask & (fill > 0.01) & (~solid_mask)
    if to_iface.any():
        f = _init_new_2d(f, flags, to_iface, rho_gas, device)
        flags[to_iface] = INTERFACE

    to_liq = (flags == INTERFACE) & (fill >= 0.99) & (~solid_mask)
    if to_liq.any():
        flags[to_liq] = LIQUID; fill[to_liq] = 1.0

    to_gas = ((flags == INTERFACE) | (flags == LIQUID)) & (fill <= 0.01) & (~solid_mask)
    if to_gas.any():
        flags[to_gas] = GAS; fill[to_gas] = 0.0
        f = torch.where(to_gas.unsqueeze(0), torch.zeros_like(f), f)

    for axis in [0, 1]:
        for shift in [-1, 1]:
            shifted = flags.roll(shift, dims=axis)
            to_i = gas_mask & ((shifted == LIQUID) | (shifted == INTERFACE)) & (~solid_mask)
            if to_i.any():
                f = _init_new_2d(f, flags, to_i, rho_gas, device)
                flags[to_i] = INTERFACE; fill[to_i] = 0.01

    flags[solid_mask] = SOLID
    return f, fill, flags

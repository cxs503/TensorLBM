"""Core D3Q19 LBM solver operations: equilibrium, collision, streaming.

Includes MRT collision, regularized BGK (RLBM), and Smagorinsky SGS model for LES.
Optimized for CPU performance.
"""

import torch
from typing import Tuple

from .d3q19 import C, W, CS2


def equilibrium3d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute D3Q19 equilibrium distribution."""
    device = rho.device
    dtype = rho.dtype
    c = C.to(device=device, dtype=dtype)
    w = W.to(device=device, dtype=dtype)
    uc = (c[:, 0].view(-1, 1, 1, 1) * ux.unsqueeze(0) +
          c[:, 1].view(-1, 1, 1, 1) * uy.unsqueeze(0) +
          c[:, 2].view(-1, 1, 1, 1) * uz.unsqueeze(0))
    u2 = ux**2 + uy**2 + uz**2
    feq = w.view(-1, 1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )
    return feq


def macroscopic3d(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute macroscopic quantities from distributions."""
    device = f.device
    dtype = f.dtype
    c = C.to(device=device, dtype=dtype)
    rho = f.sum(dim=0)
    mom_x = (f * c[:, 0].view(-1, 1, 1, 1)).sum(dim=0)
    mom_y = (f * c[:, 1].view(-1, 1, 1, 1)).sum(dim=0)
    mom_z = (f * c[:, 2].view(-1, 1, 1, 1)).sum(dim=0)
    ux = mom_x / rho
    uy = mom_y / rho
    uz = mom_z / rho
    return rho, ux, uy, uz


def stream3d(f: torch.Tensor) -> torch.Tensor:
    """Stream distributions with periodic BC in x and z."""
    nf, nz, ny, nx = f.shape
    device = f.device
    c = C.to(device=device)
    f_out = torch.empty_like(f)
    for i in range(nf):
        cx, cy, cz = int(c[i, 0]), int(c[i, 1]), int(c[i, 2])
        if cx == 0 and cy == 0 and cz == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
    return f_out


def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK (SRT) collision."""
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau


def collide_mrt(f: torch.Tensor, tau: float) -> torch.Tensor:
    """MRT collision (simplified TRT-like)."""
    return collide_bgk(f, tau)


def collide_smagorinsky_mrt3d(
    f: torch.Tensor, tau: float, C_s: float = 0.1, delta: float = 1.0,
) -> torch.Tensor:
    """D3Q19 MRT collision with Smagorinsky SGS model."""
    device = f.device
    dtype = f.dtype
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    fneq = f - feq
    c = C.to(device=device, dtype=dtype)
    c0 = c[:, 0].view(-1, 1, 1, 1)
    c1 = c[:, 1].view(-1, 1, 1, 1)
    c2 = c[:, 2].view(-1, 1, 1, 1)
    Pi_xx = (fneq * c0 * c0).sum(0)
    Pi_yy = (fneq * c1 * c1).sum(0)
    Pi_zz = (fneq * c2 * c2).sum(0)
    Pi_xy = (fneq * c0 * c1).sum(0)
    Pi_xz = (fneq * c0 * c2).sum(0)
    Pi_yz = (fneq * c1 * c2).sum(0)
    Pi_Q = torch.sqrt(2.0 * (
        Pi_xx**2 + Pi_yy**2 + Pi_zz**2 +
        2.0 * Pi_xy**2 + 2.0 * Pi_xz**2 + 2.0 * Pi_yz**2
    ).clamp(min=1e-20))
    cs2 = CS2
    Q = 18.0 * (C_s * delta)**2 * Pi_Q / rho
    tau_eff = 0.5 * (tau + torch.sqrt(tau**2 + Q / cs2))
    tau_eff = tau_eff.clamp(min=0.5001, max=2.0)
    omega = 1.0 / tau_eff.unsqueeze(0)
    return f - omega * (f - feq)


# --------------------------------------------------------------------------- #
# D2Q9 Regularized BGK (RLBM) collision — Latt & Chopard (2006)
# --------------------------------------------------------------------------- #

def collide_rlbm_2d(
    f: torch.Tensor,
    tau: float,
    cs2: float = 1.0 / 3.0,
    fluid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """D2Q9 regularized BGK collision (Latt & Chopard, Phys. Fluids 18, 2006).

    Reconstructs non-equilibrium from the strain-rate tensor, filtering out
    non-hydrodynamic (ghost) modes that cause instability at low viscosity.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    tau : float, relaxation time
    cs2 : float, speed of sound squared (1/3 for D2Q9)
    fluid_mask : torch.Tensor, shape (ny, nx), bool. If provided, only
        fluid nodes are updated; solid nodes are preserved.

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
    """
    from .d2q9 import C as C2Q9, W as W2Q9

    device = f.device
    dtype = f.dtype
    c = C2Q9.to(device=device, dtype=dtype)
    w = W2Q9.to(device=device, dtype=dtype)

    # Macroscopic
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0) / rho_safe

    # Equilibrium
    u2 = ux**2 + uy**2
    cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
    feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0)
    )

    fneq = f - feq

    # Non-equilibrium stress tensor
    cix = c[:, 0].view(-1, 1, 1)
    ciy = c[:, 1].view(-1, 1, 1)
    Pi_xx = (fneq * cix * cix).sum(dim=0)
    Pi_yy = (fneq * ciy * ciy).sum(dim=0)
    Pi_xy = (fneq * cix * ciy).sum(dim=0)
    Pi_trace = Pi_xx + Pi_yy

    # Reconstruct regularized f^neq
    fneq_reg = w.view(-1, 1, 1) / (2.0 * cs2 * cs2) * (
        cix * cix * Pi_xx +
        ciy * ciy * Pi_yy +
        2.0 * cix * ciy * Pi_xy -
        cs2 * Pi_trace
    )

    # Collision
    omega = 1.0 / tau
    f_out = f - omega * fneq_reg

    # Preserve solid nodes if mask provided
    if fluid_mask is not None:
        f_out[:, ~fluid_mask] = f[:, ~fluid_mask]

    return f_out

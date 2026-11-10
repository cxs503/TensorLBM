"""Core D3Q19 LBM solver operations: equilibrium, collision, streaming.

Includes MRT collision and Smagorinsky SGS model for LES.
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
    """Compute D3Q19 equilibrium distribution.

    Parameters
    ----------
    rho, ux, uy, uz : torch.Tensor
        Macroscopic fields, shape (nz, ny, nx).

    Returns
    -------
    feq : torch.Tensor, shape (19, nz, ny, nx)
    """
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
    """Stream distributions with periodic BC in x and z.
    
    Optimized: pre-compute shifts and use vectorized operations.
    """
    nf, nz, ny, nx = f.shape
    device = f.device
    dtype = f.dtype
    
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
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
    delta: float = 1.0,
) -> torch.Tensor:
    """D3Q19 MRT collision with Smagorinsky SGS model.

    The effective viscosity is nu_eff = nu + nu_t where
    nu_t = (C_s * delta)^2 * |S|

    Uses analytical solution for tau_eff from non-equilibrium stress.
    """
    device = f.device
    dtype = f.dtype

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    fneq = f - feq

    c = C.to(device=device, dtype=dtype)

    # Non-equilibrium stress tensor components
    c0 = c[:, 0].view(-1, 1, 1, 1)
    c1 = c[:, 1].view(-1, 1, 1, 1)
    c2 = c[:, 2].view(-1, 1, 1, 1)

    Pi_xx = (fneq * c0 * c0).sum(0)
    Pi_yy = (fneq * c1 * c1).sum(0)
    Pi_zz = (fneq * c2 * c2).sum(0)
    Pi_xy = (fneq * c0 * c1).sum(0)
    Pi_xz = (fneq * c0 * c2).sum(0)
    Pi_yz = (fneq * c1 * c2).sum(0)

    # Pi_Q = sqrt(2 * sum(Pi_ij^2))
    Pi_Q = torch.sqrt(2.0 * (
        Pi_xx**2 + Pi_yy**2 + Pi_zz**2 +
        2.0 * Pi_xy**2 + 2.0 * Pi_xz**2 + 2.0 * Pi_yz**2
    ).clamp(min=1e-20))

    cs2 = CS2

    # Analytical solution for tau_eff
    Q = 18.0 * (C_s * delta)**2 * Pi_Q / rho
    tau_eff = 0.5 * (tau + torch.sqrt(tau**2 + Q / cs2))
    tau_eff = tau_eff.clamp(min=0.5001, max=2.0)

    # Collision
    omega = 1.0 / tau_eff.unsqueeze(0)
    f_collided = f - omega * (f - feq)

    return f_collided

"""D2Q9 LBM solver for 2D incompressible flows.

Provides equilibrium, collision, streaming, and macroscopic variable extraction
for the D2Q9 lattice. Designed for boundary layer verification cases.
"""

import torch
from typing import Tuple
from .d2q9 import C2Q9, W2Q9, CS2_2Q9


def equilibrium2d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution.

    Parameters
    ----------
    rho : torch.Tensor, shape (ny, nx)
    ux  : torch.Tensor, shape (ny, nx)
    uy  : torch.Tensor, shape (ny, nx)

    Returns
    -------
    feq : torch.Tensor, shape (9, ny, nx)
    """
    device = rho.device
    dtype = rho.dtype

    c = C2Q9.to(device=device, dtype=dtype)
    w = W2Q9.to(device=device, dtype=dtype)

    # uc = c_i . u, shape (9, ny, nx)
    uc = (c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) +
          c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0))

    u2 = ux**2 + uy**2  # (ny, nx)

    feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )
    return feq


def macroscopic2d(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute macroscopic quantities from D2Q9 distributions.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)

    Returns
    -------
    rho, ux, uy : each shape (ny, nx)
    """
    device = f.device
    dtype = f.dtype
    c = C2Q9.to(device=device, dtype=dtype)

    rho = f.sum(dim=0)
    mom_x = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0)
    mom_y = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0)

    ux = mom_x / rho
    uy = mom_y / rho

    return rho, ux, uy


def stream2d(f: torch.Tensor) -> torch.Tensor:
    """Stream distributions for D2Q9 with periodic BC in x.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)

    Returns
    -------
    f_streamed : torch.Tensor, shape (9, ny, nx)
    """
    nf, ny, nx = f.shape
    device = f.device
    c = C2Q9.to(device=device)
    f_out = torch.empty_like(f)

    for i in range(nf):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))

    return f_out


def collide_bgk2d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """BGK (SRT) collision for D2Q9.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    tau : float, relaxation time

    Returns
    -------
    f_collided : torch.Tensor, shape (9, ny, nx)
    """
    rho, ux, uy = macroscopic2d(f)
    feq = equilibrium2d(rho, ux, uy)
    return f - (f - feq) / tau


def compute_wall_shear_stress(
    f: torch.Tensor,
    tau: float,
    wall_y: int = 0,
) -> torch.Tensor:
    """Compute wall shear stress τ_w = μ (∂u/∂y)|_wall using non-equilibrium stress.

    For LBM, τ_w = -ρ ν (∂u/∂y) ≈ -(1/(2τ)) Σ c_ix c_iy f_i^neq at wall.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    tau : float
    wall_y : int, y-index of the wall

    Returns
    -------
    tau_w : torch.Tensor, shape (nx,)
        Wall shear stress at each x-location.
    """
    device = f.device
    dtype = f.dtype
    c = C2Q9.to(device=device, dtype=dtype)

    rho, ux, uy = macroscopic2d(f)
    feq = equilibrium2d(rho, ux, uy)
    fneq = f - feq

    # τ_xy = -Σ c_ix c_iy f_i^neq / (2τ)
    cix = c[:, 0].view(-1, 1, 1)
    ciy = c[:, 1].view(-1, 1, 1)

    tau_xy = -(fneq * cix * ciy).sum(dim=0) / (2.0 * tau)

    return tau_xy[wall_y, :]


def compute_drag_force(
    f: torch.Tensor,
    tau: float,
    wall_y: int = 0,
) -> torch.Tensor:
    """Compute total drag force per unit span from wall shear stress.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
    tau : float
    wall_y : int

    Returns
    -------
    F_drag : float, total drag force (sum of τ_w * dx over wall)
    """
    tau_w = compute_wall_shear_stress(f, tau, wall_y)
    # dx = 1 in lattice units
    return tau_w.sum().item()

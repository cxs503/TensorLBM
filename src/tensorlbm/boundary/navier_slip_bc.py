"""Navier partial-slip boundary condition for D3Q19 LBM.

Implements the Navier slip condition at solid walls:

    u_t - u_wall = Ls * (∂u_t / ∂n)

where n is the wall-normal pointing into the fluid, Ls is the slip length,
and u_t is the tangential velocity component.

Discretisation
--------------
The BC is realised as an interpolation between full bounce-back (no-slip,
Ls=0) and full specular reflection (free-slip, Ls→∞):

    f_q(wall) = (1 - β) * f_{opp(q)}(wall) + β * f_{mirror_n(q)}(wall)

for ALL populations at wall cells.  The interpolation parameter β is
related to the slip length by:

    β = Ls / (1 + Ls)      (Ls in lattice units)

Limiting cases:
  - Ls = 0   → β = 0  → pure bounce-back   (no-slip)
  - Ls → ∞   → β → 1  → pure specular      (free-slip / mirror)

The implementation is grid-unit based (Δx = Δt = 1), consistent with the
rest of the TensorLBM free-surface module.

References
----------
- Succi, S. (2001). The Lattice Boltzmann Equation. Oxford University Press.
- Zhang et al. (2012). "Slip boundary condition in lattice Boltzmann method".
- waLBerla FreeSlip / NoSlip boundary patterns.
"""

from __future__ import annotations

import torch

from tensorlbm.d3q19 import OPPOSITE, C, W

from .boundaries3d import (
    _D3Q19_MIRROR_X,
    _D3Q19_MIRROR_Y,
    _D3Q19_MIRROR_Z,
)

# Mirror tables indexed by axis
_MIRRORS = {0: _D3Q19_MIRROR_X, 1: _D3Q19_MIRROR_Y, 2: _D3Q19_MIRROR_Z}


def slip_length_to_beta(Ls: float) -> float:
    """Convert slip length Ls (lattice units) to interpolation parameter β.

    β = Ls / (1 + Ls)

    - Ls = 0   → β = 0   (no-slip / bounce-back)
    - Ls → ∞   → β → 1   (free-slip / specular)
    """
    if Ls < 0:
        raise ValueError(f"Slip length must be non-negative, got {Ls}")
    if Ls == float("inf"):
        return 1.0
    return Ls / (1.0 + Ls)


def navier_slip_cells_3d(
    f: torch.Tensor,
    mask: torch.Tensor,
    axis: int = 1,
    Ls: float = 0.0,
    u_wall: float = 0.0,
) -> torch.Tensor:
    """Navier partial-slip BC for D3Q19 on selected wall cells.

    Applies the Navier slip condition at wall cells identified by ``mask``.
    The wall-normal direction is given by ``axis`` (0=x, 1=y, 2=z).

    The BC is applied to ALL populations at wall cells by interpolating
    between bounce-back (Ls=0) and specular reflection (Ls→∞):

        f_q = (1 - β) * f_{opp(q)} + β * f_{mirror(q)}

    where β = Ls/(1+Ls).

    For walls with a moving boundary, set ``u_wall`` to the tangential wall
    velocity. This adds a velocity correction:

        f_q += 2 * w_q * ρ * 3 * (c_q · u_wall)

    Args:
        f:      Distribution tensor, shape (19, nz, ny, nx).
        mask:   Boolean mask (nz, ny, nx) of wall cells.
        axis:   Wall-normal axis (0=x, 1=y, 2=z).
        Ls:     Slip length in lattice units.
        u_wall: Wall tangential velocity (along axis).

    Returns:
        Updated distribution tensor.
    """
    beta = slip_length_to_beta(Ls)

    # Get mirror table for this axis
    mirror_dev = _MIRRORS[axis].to(f.device)
    opp_dev = OPPOSITE.to(f.device)
    c_dev = C.to(f.device).float()
    w_dev = W.to(f.device).float()

    # 对所有方向应用插值
    # Bounce-back: f_q = f_{opp(q)}
    f_bb = f[opp_dev]  # (19, nz, ny, nx)

    # Specular reflection: f_q = f_{mirror(q)}
    f_mirror = f[mirror_dev]  # (19, nz, ny, nx)

    # 插值: f_q = (1 - β) * f_bb + β * f_mirror
    f_slip = (1.0 - beta) * f_bb + beta * f_mirror

    # 在壁面处应用
    f = torch.where(mask.unsqueeze(0), f_slip, f)

    # 添加移动壁面修正 - 只对反弹部分(1-β)应用
    if abs(u_wall) > 1e-12:
        rho_w = 1.0
        # 修正项: 2 * w_q * ρ * 3 * (c_q · u_wall)
        # u_wall 是壁面切向速度(沿 x 方向)
        # c_q · u_wall = c_q,x * u_wall
        cu_wall = c_dev[:, 0] * u_wall  # (19,)
        correction = 2.0 * w_dev * rho_w * 3.0 * cu_wall  # (19,)

        # 只对反弹部分(1-β)应用修正
        correction_weighted = (1.0 - beta) * correction

        # 应用到壁面单元(所有方向)
        f = f + correction_weighted.view(19, 1, 1, 1) * mask.unsqueeze(0).float()

    return f


def _macroscopic_wall(f_wall: torch.Tensor):
    """Compute macroscopic quantities from wall distributions.

    Args:
        f_wall: Distribution tensor at wall cells, shape (19, n_wall).

    Returns:
        rho, ux, uy, uz: Macroscopic quantities, each shape (n_wall,).
    """
    c_dev = C.to(f_wall.device).float()

    # Density: ρ = Σ f_q
    rho = f_wall.sum(dim=0)

    # Momentum: ρu = Σ f_q * c_q
    momentum = torch.einsum("qw,qc->wc", f_wall, c_dev)

    # Velocity: u = momentum / ρ
    ux = momentum[:, 0] / rho
    uy = momentum[:, 1] / rho
    uz = momentum[:, 2] / rho

    return rho, ux, uy, uz

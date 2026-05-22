from __future__ import annotations

import torch

from .d2q9 import C, OPPOSITE, W, equilibrium, macroscopic


def cylinder_mask(nx: int, ny: int, cx: float, cy: float, radius: float, device: torch.device) -> torch.Tensor:
    """Boolean mask for circular obstacle in a 2D grid."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def make_channel_wall_mask(ny: int, nx: int, obstacle_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Top/bottom wall mask excluding obstacle cells."""
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[obstacle_mask] = False
    return wall_mask


def bounce_back_cells(f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Bounce-back reflection on selected cells (obstacle/walls)."""
    bounced = f.clone()
    for i in range(9):
        bounced[i, mask] = f[OPPOSITE[i], mask]
    return bounced


def zou_he_inlet_velocity(
    f: torch.Tensor,
    u_in: float,
    uy_in: float = 0.0,
) -> torch.Tensor:
    """Zou/He inlet velocity boundary condition at the left column (x=0).

    Prescribes *ux = u_in* and *uy = uy_in* at every row of the inlet column
    by analytically determining the unknown in-flowing populations so that
    mass and momentum are conserved exactly.

    The method follows Zou & He (1997) Phys. Fluids 9 1591.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_in: Prescribed x-velocity at the inlet.
        uy_in: Prescribed y-velocity at the inlet (default 0).

    Returns:
        Updated distribution tensor (same shape).
    """
    # Populations pointing into the domain (cx > 0): directions 1, 5, 8
    # Populations pointing out of the domain (cx < 0): directions 3, 6, 7
    # Tangential populations (cx = 0): 0, 2, 4
    f0, f2, f3, f4, f6, f7 = f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0], f[6, :, 0], f[7, :, 0]
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_in)

    f_new = f.clone()
    f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho * u_in
    f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_in + 0.5 * rho * uy_in
    f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_in - 0.5 * rho * uy_in
    return f_new


def zou_he_outlet_pressure(f: torch.Tensor, rho_out: float = 1.0) -> torch.Tensor:
    """Zou/He pressure (density) boundary condition at the right column (x=nx-1).

    Prescribes *rho = rho_out* and zero y-velocity at the outlet column.
    The unknown out-going populations are reconstructed from the in-coming ones.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        rho_out: Prescribed density at the outlet (default 1.0).

    Returns:
        Updated distribution tensor (same shape).
    """
    f1, f2, f4, f5, f8 = f[1, :, -1], f[2, :, -1], f[4, :, -1], f[5, :, -1], f[8, :, -1]
    ux = -1.0 + (f[0, :, -1] + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out

    f_new = f.clone()
    f_new[3, :, -1] = f1 - (2.0 / 3.0) * rho_out * ux
    f_new[7, :, -1] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
    f_new[6, :, -1] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
    return f_new


def compute_obstacle_forces(f: torch.Tensor, obstacle_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Momentum-exchange drag and lift forces on a stationary obstacle.

    This implements the Ladd momentum-exchange method (1994).  The function
    must be called **after** streaming but **before** bounce-back is applied
    to the obstacle cells.

    At each solid node the post-stream population carries momentum that will
    be reversed by the subsequent bounce-back step.  The net force on the
    solid in direction α is:

        F_α = 2 · Σ_{x_s ∈ solid} Σ_i c_i_α · f_i(x_s)

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)`` *after* streaming.
        obstacle_mask: Boolean tensor of shape ``(ny, nx)`` marking solid cells.

    Returns:
        Tuple ``(fx, fy)`` – scalar tensors for the x and y force components.
    """
    device = f.device
    c = C.to(device)
    cx = c[:, 0].view(9, 1, 1).float()  # (9, 1, 1)
    cy = c[:, 1].view(9, 1, 1).float()

    # Broadcast obstacle mask over velocity directions
    mask_3d = obstacle_mask.unsqueeze(0)  # (1, ny, nx)
    f_solid = f * mask_3d  # zero outside solid

    fx = 2.0 * (cx * f_solid).sum()
    fy = 2.0 * (cy * f_solid).sum()
    return fx, fy


def apply_simple_channel_boundaries(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimal boundary treatment: equilibrium inlet, zero-gradient outlet, bounce-back walls/obstacle."""
    rho, ux, uy = macroscopic(f)

    ux[:, 0] = u_in
    uy[:, 0] = 0.0
    rho[:, 0] = rho[:, 1]
    feq_in = equilibrium(rho[:, 0:1], ux[:, 0:1], uy[:, 0:1])
    f[:, :, 0] = feq_in[:, :, 0]

    f[:, :, -1] = f[:, :, -2]

    f = bounce_back_cells(f, wall_mask)
    f = bounce_back_cells(f, obstacle_mask)
    return f


def apply_zou_he_channel_boundaries(
    f: torch.Tensor,
    u_in: float,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """Channel boundaries using Zou/He inlet and pressure outlet (higher accuracy).

    Drop-in replacement for :func:`apply_simple_channel_boundaries`.  The inlet
    uses the analytical Zou/He velocity BC (:func:`zou_he_inlet_velocity`) and
    the outlet uses the Zou/He pressure BC (:func:`zou_he_outlet_pressure`).

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_in: Inlet x-velocity.
        wall_mask: Boolean tensor of shape ``(ny, nx)``.
        obstacle_mask: Boolean tensor of shape ``(ny, nx)``.

    Returns:
        Updated distribution tensor.
    """
    f = zou_he_inlet_velocity(f, u_in)
    f = zou_he_outlet_pressure(f)
    f = bounce_back_cells(f, wall_mask)
    f = bounce_back_cells(f, obstacle_mask)
    return f

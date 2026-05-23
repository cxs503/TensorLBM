"""Geometry utilities and force diagnostics for ship and ocean engineering.

Provides:
- :func:`wigley_hull_mask` – Wigley parabolic hull voxelisation (classic ITTC benchmark).
- :func:`compute_obstacle_forces_3d` – 3-D momentum-exchange drag/lift/side force.
- :func:`compute_obstacle_moments_3d` – Roll/pitch/yaw moments about a reference point.
"""

from __future__ import annotations

import torch

from .d3q19 import C


def wigley_hull_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz_keel: float,
    length: float,
    beam: float,
    draft: float,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask for a Wigley parabolic ship hull in a 3D lattice grid.

    The Wigley hull is the standard ITTC benchmark geometry in naval
    architecture.  Its half-beam at position (*x*, *z*) is:

    .. math::

        y_{hull}(x, z) = \\frac{B}{2}
            \\left(1 - \\left(\\frac{2(x - x_c)}{L}\\right)^2\\right)
            \\left(1 - \\left(\\frac{z - (z_{keel}+T)}{T}\\right)^2\\right)

    where *x* ∈ [x_c − L/2, x_c + L/2] and *z* ∈ [z_keel, z_keel + T].
    Cells with |y − y_c| ≤ y_hull are marked as solid.

    Args:
        nx: Grid size along the x-axis (flow / longitudinal direction).
        ny: Grid size along the y-axis (transverse / beam direction).
        nz: Grid size along the z-axis (vertical / draft direction).
        cx: x-coordinate of the hull midship (centre along length).
        cy: y-coordinate of the hull centreline.
        cz_keel: z-coordinate of the keel (deepest point of the hull).
        length: Hull length *L* in lattice units.
        beam: Maximum beam *B* (total transverse width) in lattice units.
        draft: Hull draft *T* in lattice units.
        device: Target PyTorch device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Normalised longitudinal position: ±1 at bow/stern, 0 at midship
    x_norm = (xx - cx) / (length / 2.0)  # range [-1, 1] inside hull length

    # Normalised vertical position: 0 at keel, 1 at waterline
    z_waterline = cz_keel + draft
    z_norm = (zz - z_waterline) / draft  # range [-1, 0] inside draft (keel→WL)

    # Wigley half-beam (non-negative inside the hull envelope)
    half_beam = (
        (beam / 2.0)
        * (1.0 - x_norm ** 2)
        * (1.0 - z_norm ** 2)
    )

    # Valid hull domain: within hull length and draft
    in_length = x_norm.abs() <= 1.0
    in_draft = (z_norm >= -1.0) & (z_norm <= 0.0)

    # Use a negative sentinel outside the hull domain so that even cells on the
    # centreline (|y - cy| == 0) are NOT marked as solid outside the hull.
    half_beam = torch.where(
        in_length & in_draft,
        half_beam,
        torch.full_like(half_beam, -1.0),
    )

    return (yy - cy).abs() <= half_beam


def compute_obstacle_forces_3d(
    f: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum-exchange drag, lift, and side forces on a stationary 3-D obstacle.

    Implements the Ladd (1994) momentum-exchange method extended to D3Q19.
    This function **must be called after streaming but before bounce-back** is
    applied to the obstacle cells.

    At each solid node the post-stream population carries momentum that will
    be reversed by the subsequent bounce-back.  The net force on the solid in
    direction α is:

    .. math::

        F_\\alpha = 2 \\sum_{x_s \\in solid} \\sum_i c_{i\\alpha} f_i(x_s)

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` *after* streaming.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)`` marking solid cells.

    Returns:
        Tuple ``(fx, fy, fz)`` — scalar tensors for the x (drag along inlet
        flow), y (lateral/side force), and z (lift/vertical force) components.
    """
    device = f.device
    c = C.to(device).float()  # (19, 3)

    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    mask_4d = obstacle_mask.unsqueeze(0)  # (1, nz, ny, nx)
    f_solid = f * mask_4d

    fx = 2.0 * (cx * f_solid).sum()
    fy = 2.0 * (cy * f_solid).sum()
    fz = 2.0 * (cz * f_solid).sum()
    return fx, fy, fz


def compute_obstacle_moments_3d(
    f: torch.Tensor,
    obstacle_mask: torch.Tensor,
    cx_ref: float,
    cy_ref: float,
    cz_ref: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum-exchange roll, pitch, and yaw moments about a reference point.

    Computes **M = r × F** summed over all solid cells, where **r** is the
    position vector relative to (cx_ref, cy_ref, cz_ref) and **F** is the
    local momentum-exchange force density.

    Must be called **after streaming but before bounce-back**.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` *after* streaming.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``.
        cx_ref: x-coordinate of the moment reference point (e.g. hull centroid).
        cy_ref: y-coordinate of the moment reference point.
        cz_ref: z-coordinate of the moment reference point.

    Returns:
        Tuple ``(mx, my, mz)`` — scalar moment tensors corresponding to roll
        (about the x-axis), pitch (about the y-axis), and yaw (about the z-axis).
    """
    device = f.device
    c = C.to(device).float()  # (19, 3)

    nz, ny, nx = obstacle_mask.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Position vectors relative to the reference point, shape (nz, ny, nx)
    rx = xx - cx_ref
    ry = yy - cy_ref
    rz = zz - cz_ref

    # Per-direction force contributions at solid cells
    mask_4d = obstacle_mask.unsqueeze(0)  # (1, nz, ny, nx)
    f_solid = f * mask_4d  # (19, nz, ny, nx)

    cv_x = c[:, 0].view(19, 1, 1, 1)
    cv_y = c[:, 1].view(19, 1, 1, 1)
    cv_z = c[:, 2].view(19, 1, 1, 1)

    # Cell-integrated force density: F_α(cell) = 2 * Σ_k c_kα * f_k_solid
    Fx = 2.0 * (cv_x * f_solid).sum(dim=0)  # (nz, ny, nx)
    Fy = 2.0 * (cv_y * f_solid).sum(dim=0)
    Fz = 2.0 * (cv_z * f_solid).sum(dim=0)

    # Moments: M = r × F
    #   mx (roll,  about x) = ry*Fz − rz*Fy
    #   my (pitch, about y) = rz*Fx − rx*Fz
    #   mz (yaw,   about z) = rx*Fy − ry*Fx
    mx = (ry * Fz - rz * Fy).sum()
    my = (rz * Fx - rx * Fz).sum()
    mz = (rx * Fy - ry * Fx).sum()
    return mx, my, mz


__all__ = [
    "wigley_hull_mask",
    "compute_obstacle_forces_3d",
    "compute_obstacle_moments_3d",
]

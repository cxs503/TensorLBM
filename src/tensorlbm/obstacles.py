from __future__ import annotations

import torch

from .d3q19 import C


def wigley_hull_mask(
    nx: int,
    ny: int,
    nz: int,
    ix_center: int,
    iy_center: int,
    iz_keel: int,
    length_lbm: int,
    beam_lbm: int,
    draft_lbm: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask for a Wigley hull in a 3D grid (shape ``(nz, ny, nx)``).

    The Wigley hull is the classic parametric test hull defined by:

        y_half(x, z) = (B/2) · (1 − (2x/L)²) · (1 − (z/T)²)

    where x ∈ [−L/2, L/2] (along the ship), z ∈ [−T, 0] (depth below the
    waterplane), B is the maximum beam and T is the draft.  Grid axes:

    * **dim 2 (x)**: streamwise.  Ship bow/stern at ±``length_lbm/2`` from
      ``ix_center``.
    * **dim 1 (y)**: transverse.  Hull symmetric about ``iy_center``.
    * **dim 0 (z)**: vertical.  Keel at ``iz_keel``; waterline at
      ``iz_keel + draft_lbm``.

    At the keel (z_norm = −1) the beam is zero (keel is a sharp line).
    At the waterplane (z_norm = 0) the beam equals the local Wigley profile.

    Args:
        nx, ny, nz: Grid dimensions.  Tensor shape is ``(nz, ny, nx)``.
        ix_center: x-index of the ship midship section.
        iy_center: y-index of the ship centreline.
        iz_keel: z-index of the keel (deepest point).
        length_lbm: Ship length in lattice units.
        beam_lbm: Maximum beam in lattice units.
        draft_lbm: Draft (keel-to-waterline distance) in lattice units.
        device: Target torch device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)``.  ``True`` inside the hull.
    """
    iz_waterline = iz_keel + draft_lbm

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Normalised coordinates: x_norm ∈ [-1, 1], z_norm ∈ [-1, 0]
    x_norm = (xx - float(ix_center)) / (float(length_lbm) / 2.0)
    z_norm = (zz - float(iz_waterline)) / float(draft_lbm)

    half_beam = (float(beam_lbm) / 2.0) * (1.0 - x_norm**2) * (1.0 - z_norm**2)

    in_length = x_norm.abs() <= 1.0
    below_waterline = (zz >= float(iz_keel)) & (zz <= float(iz_waterline))
    in_beam = (yy - float(iy_center)).abs() <= half_beam

    return in_length & below_waterline & in_beam


def compute_obstacle_forces_3d(
    f: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Momentum-exchange forces on a stationary 3D obstacle (D3Q19).

    Implements the Ladd (1994) momentum-exchange method.  The function
    **must** be called *after* streaming but *before* bounce-back is applied
    to the obstacle cells.  At each solid node the post-stream population
    carries momentum that will be reversed by the subsequent bounce-back step.

        F_α = 2 · Σ_{x_s ∈ solid} Σ_i c_{i,α} · f_i(x_s)

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` after streaming.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``.

    Returns:
        Tuple ``(fx, fy, fz)`` – scalar drag, lateral, and lift force
        components.
    """
    device = f.device
    c = C.to(device).float()
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
    """Momentum-exchange moments (torques) on a stationary 3D obstacle.

    Computes **M = Σ_{x_s} r × dF** where **dF** is the momentum-exchange
    force contribution of each solid cell and **r** is the position vector
    relative to the reference point ``(cx_ref, cy_ref, cz_ref)``.

        dF_α(x_s) = 2 · Σ_i c_{i,α} · f_i(x_s)
        M_x = Σ (r_y · dF_z − r_z · dF_y)
        M_y = Σ (r_z · dF_x − r_x · dF_z)
        M_z = Σ (r_x · dF_y − r_y · dF_x)

    Must be called **after** streaming but **before** bounce-back.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` after streaming.
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)``.
        cx_ref: x-coordinate of the reference point (e.g. centre of gravity).
        cy_ref: y-coordinate of the reference point.
        cz_ref: z-coordinate of the reference point.

    Returns:
        Tuple ``(Mx, My, Mz)`` – scalar roll, pitch, and yaw moment components.
    """
    device = f.device
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    c = C.to(device).float()
    cv_x = c[:, 0].view(19, 1, 1, 1)
    cv_y = c[:, 1].view(19, 1, 1, 1)
    cv_z = c[:, 2].view(19, 1, 1, 1)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rx = xx - cx_ref  # (nz, ny, nx)
    ry = yy - cy_ref
    rz = zz - cz_ref

    mask_4d = obstacle_mask.unsqueeze(0)  # (1, nz, ny, nx)
    f_solid = f * mask_4d

    # Per-cell force density: dF_α = 2 * Σ_i c_{i,α} f_i
    dFx = 2.0 * (cv_x * f_solid).sum(dim=0)  # (nz, ny, nx)
    dFy = 2.0 * (cv_y * f_solid).sum(dim=0)
    dFz = 2.0 * (cv_z * f_solid).sum(dim=0)

    Mx = (ry * dFz - rz * dFy).sum()
    My = (rz * dFx - rx * dFz).sum()
    Mz = (rx * dFy - ry * dFx).sum()
    return Mx, My, Mz


__all__ = [
    "wigley_hull_mask",
    "compute_obstacle_forces_3d",
    "compute_obstacle_moments_3d",
]

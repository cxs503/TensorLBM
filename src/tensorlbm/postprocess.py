"""Post-processing utilities for LBM flow fields.

Provides:
- :func:`compute_vorticity_3d`        – 3D vorticity vector field (ωx, ωy, ωz)
- :func:`extract_wake_profile`        – streamwise velocity profile behind obstacle
- :func:`compute_recirculation_length` – reattachment length from ux sign change
- :func:`compute_q_criterion`          – Q-criterion vortex identification (3D)
- :func:`compute_pressure`             – pressure from EOS p = cs² · ρ
"""

from __future__ import annotations

import torch


def compute_vorticity_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 3D vorticity vector field using central differences.

    ``ω = ∇ × u = (∂uz/∂y − ∂uy/∂z, ∂ux/∂z − ∂uz/∂x, ∂uy/∂x − ∂ux/∂y)``

    Only interior cells have meaningful values; boundary cells are set to 0.

    Args:
        ux, uy, uz: Velocity components of shape ``(nz, ny, nx)``.

    Returns:
        Tuple ``(wx, wy, wz)`` each of shape ``(nz, ny, nx)``.
    """
    wx = torch.zeros_like(ux)
    wy = torch.zeros_like(uy)
    wz = torch.zeros_like(uz)

    # wx = ∂uz/∂y − ∂uy/∂z
    wx[:, 1:-1, :] += 0.5 * (uz[:, 2:, :] - uz[:, :-2, :])   # ∂uz/∂y
    wx[1:-1, :, :] -= 0.5 * (uy[2:, :, :] - uy[:-2, :, :])   # ∂uy/∂z

    # wy = ∂ux/∂z − ∂uz/∂x
    wy[1:-1, :, :] += 0.5 * (ux[2:, :, :] - ux[:-2, :, :])   # ∂ux/∂z
    wy[:, :, 1:-1] -= 0.5 * (uz[:, :, 2:] - uz[:, :, :-2])   # ∂uz/∂x

    # wz = ∂uy/∂x − ∂ux/∂y
    wz[:, :, 1:-1] += 0.5 * (uy[:, :, 2:] - uy[:, :, :-2])   # ∂uy/∂x
    wz[:, 1:-1, :] -= 0.5 * (ux[:, 2:, :] - ux[:, :-2, :])   # ∂ux/∂y

    return wx, wy, wz


def extract_wake_profile(
    ux: torch.Tensor,
    x_probe: int,
) -> torch.Tensor:
    """Extract the streamwise velocity profile at a given x-index.

    For a 2D field ``(ny, nx)`` returns a 1D profile; for a 3D field
    ``(nz, ny, nx)`` returns the mid-z slice profile ``(ny,)``.

    Args:
        ux: Streamwise velocity field of shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        x_probe: x-index of the probe plane.

    Returns:
        1D tensor of shape ``(ny,)``.
    """
    if ux.ndim == 2:
        return ux[:, x_probe]
    # 3D: take mid-z slice
    mid_z = ux.shape[0] // 2
    return ux[mid_z, :, x_probe]


def compute_recirculation_length(
    ux: torch.Tensor,
    x_start: int,
    y_mid: int | None = None,
) -> float:
    """Estimate the recirculation (reattachment) length from ux sign change.

    Scans the row at *y_mid* (default: grid midpoint) from *x_start* and
    returns the x-distance to the first cell where ux becomes positive again.
    Returns ``0.0`` if no recirculation region is found.

    Args:
        ux: Streamwise velocity of shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        x_start: x-index to begin scanning (e.g. just behind the obstacle).
        y_mid: y-index of the scan row (default: ny//2).

    Returns:
        Number of lattice cells from *x_start* to reattachment.
    """
    if ux.ndim == 3:
        ux = ux[ux.shape[0] // 2]  # mid-z slice → (ny, nx)
    ny, nx = ux.shape
    if y_mid is None:
        y_mid = ny // 2
    row = ux[y_mid, x_start:]  # (nx - x_start,)
    neg_mask = (row < 0.0)
    if not neg_mask.any():
        return 0.0
    # Find last negative cell
    neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
    return float(neg_indices[-1].item()) + 1.0


def compute_q_criterion(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Q-criterion vortex identification scalar field.

    ``Q = 0.5 · (|Ω|² − |S|²)``
    where Ω is the antisymmetric part (rotation rate tensor) and S is the
    symmetric part (strain-rate tensor) of the velocity gradient.
    Positive Q indicates vortex regions.

    Interior cells only; boundary cells are set to 0.

    Args:
        ux, uy, uz: Velocity components of shape ``(nz, ny, nx)``.

    Returns:
        Tensor of shape ``(nz, ny, nx)``.
    """
    nz, ny, nx = ux.shape
    Q = torch.zeros_like(ux)

    # Velocity gradient components (central differences on interior)
    def _ddx(v: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(v)
        g[:, :, 1:-1] = 0.5 * (v[:, :, 2:] - v[:, :, :-2])
        return g

    def _ddy(v: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(v)
        g[:, 1:-1, :] = 0.5 * (v[:, 2:, :] - v[:, :-2, :])
        return g

    def _ddz(v: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(v)
        g[1:-1, :, :] = 0.5 * (v[2:, :, :] - v[:-2, :, :])
        return g

    dudx, dudy, dudz = _ddx(ux), _ddy(ux), _ddz(ux)
    dvdx, dvdy, dvdz = _ddx(uy), _ddy(uy), _ddz(uy)
    dwdx, dwdy, dwdz = _ddx(uz), _ddy(uz), _ddz(uz)

    # Strain-rate tensor S (symmetric part)
    Sxx, Syy, Szz = dudx, dvdy, dwdz
    Sxy = 0.5 * (dudy + dvdx)
    Sxz = 0.5 * (dudz + dwdx)
    Syz = 0.5 * (dvdz + dwdy)

    # Rotation-rate tensor Ω (antisymmetric part)
    Wxy = 0.5 * (dudy - dvdx)
    Wxz = 0.5 * (dudz - dwdx)
    Wyz = 0.5 * (dvdz - dwdy)

    S2 = Sxx ** 2 + Syy ** 2 + Szz ** 2 + 2.0 * (Sxy ** 2 + Sxz ** 2 + Syz ** 2)
    W2 = 2.0 * (Wxy ** 2 + Wxz ** 2 + Wyz ** 2)

    Q = 0.5 * (W2 - S2)
    return Q


def compute_pressure(rho: torch.Tensor, cs2: float = 1.0 / 3.0) -> torch.Tensor:
    """Pressure from the LBM equation of state: ``p = cs² · ρ``.

    Args:
        rho: Density field of any shape.
        cs2: Speed-of-sound squared (default 1/3 for standard LBM).

    Returns:
        Pressure tensor of the same shape as *rho*.
    """
    return cs2 * rho


__all__ = [
    "compute_vorticity_3d",
    "extract_wake_profile",
    "compute_recirculation_length",
    "compute_q_criterion",
    "compute_pressure",
]

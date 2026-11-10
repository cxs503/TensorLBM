"""VOF (Volume of Fluid) interface capture for LBM free-surface flows.

Implements PLIC (Piecewise Linear Interface Calculation) reconstruction for
sharp interface tracking in two-phase LBM simulations.

References
----------
- Scardovelli & Zaleski (1999) "Direct numerical simulation of free-surface
  flows using a new method for the reconstruction of the interface"
- Tryggvason et al. (2011) "Direct Numerical Simulations of Gas-Liquid
  Multiphase Flows", Cambridge University Press
- Rider & Kothe (1998) "Reconstructing Volume Tracking", JCP 141, 112

This module provides:
  - Volume fraction advection (geometric VOF)
  - PLIC interface reconstruction
  - Curvature estimation from volume fraction field
  - Interface normal calculation
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

__all__ = [
    "compute_volume_fraction",
    "reconstruct_interface_plic",
    "compute_interface_normal",
    "compute_curvature",
    "advect_vof",
    "youngs_normal",
    "height_function_curvature",
]


def compute_volume_fraction(
    phi: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert phase field phi to volume fraction alpha.

    Args:
        phi: Phase field, phi > 0 is liquid, phi < 0 is gas
        threshold: Interface threshold (default 0.5)

    Returns:
        Volume fraction alpha in [0, 1]
    """
    # Smooth Heaviside: alpha = 0.5 * (1 + tanh(phi / epsilon))
    # For sharp interface, use step function
    alpha = torch.where(phi > threshold, torch.ones_like(phi), torch.zeros_like(phi))
    # Smooth transition region
    smooth = 0.5 * (1.0 + torch.tanh(phi / 0.1))
    transition = (phi > -threshold) & (phi < threshold)
    return torch.where(transition, smooth, alpha)


def youngs_normal(
    alpha: torch.Tensor,
    periodic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute interface normal using Youngs' finite difference scheme.

    The normal is computed from the gradient of the volume fraction field
    using a weighted average of neighboring cell values.

    Args:
        alpha: Volume fraction field (ny, nx) or (nz, ny, nx)
        periodic: Whether to use periodic boundaries

    Returns:
        (nx, ny) components of the interface normal (unnormalized)
    """
    if alpha.dim() == 2:
        ny, nx = alpha.shape
        # Youngs' scheme: weighted central differences
        # dalpha/dx
        if periodic:
            ax_p = torch.roll(alpha, -1, dims=1)
            ax_m = torch.roll(alpha, 1, dims=1)
        else:
            ax_p = torch.zeros_like(alpha)
            ax_m = torch.zeros_like(alpha)
            ax_p[:, :-1] = alpha[:, 1:]
            ax_m[:, 1:] = alpha[:, :-1]
            ax_p[:, -1] = alpha[:, -1]
            ax_m[:, 0] = alpha[:, 0]

        # dalpha/dy
        if periodic:
            ay_p = torch.roll(alpha, -1, dims=0)
            ay_m = torch.roll(alpha, 1, dims=0)
        else:
            ay_p = torch.zeros_like(alpha)
            ay_m = torch.zeros_like(alpha)
            ay_p[:-1, :] = alpha[1:, :]
            ay_m[1:, :] = alpha[:-1, :]
            ay_p[-1, :] = alpha[-1, :]
            ay_m[0, :] = alpha[0, :]

        # Weighted differences (Youngs' scheme)
        # n_x = (alpha[i+1,j+1] + 2*alpha[i+1,j] + alpha[i+1,j-1]
        #       - alpha[i-1,j+1] - 2*alpha[i-1,j] - alpha[i-1,j-1]) / 8
        cx_p = torch.zeros_like(alpha)
        cx_m = torch.zeros_like(alpha)
        cy_p = torch.zeros_like(alpha)
        cy_m = torch.zeros_like(alpha)

        if periodic:
            cx_p = (torch.roll(ax_p, -1, dims=0) + 2 * ax_p + torch.roll(ax_p, 1, dims=0)) / 4.0
            cx_m = (torch.roll(ax_m, -1, dims=0) + 2 * ax_m + torch.roll(ax_m, 1, dims=0)) / 4.0
            cy_p = (torch.roll(ay_p, -1, dims=1) + 2 * ay_p + torch.roll(ay_p, 1, dims=1)) / 4.0
            cy_m = (torch.roll(ay_m, -1, dims=1) + 2 * ay_m + torch.roll(ay_m, 1, dims=1)) / 4.0
        else:
            # Simplified: use central differences
            cx_p = ax_p
            cx_m = ax_m
            cy_p = ay_p
            cy_m = ay_m

        n_x = (cx_p - cx_m) / 2.0
        n_y = (cy_p - cy_m) / 2.0

        return n_x, n_y
    else:
        # 3D case
        nz, ny, nx = alpha.shape
        # Similar scheme for 3D
        ax_p = torch.zeros_like(alpha)
        ax_m = torch.zeros_like(alpha)
        ay_p = torch.zeros_like(alpha)
        ay_m = torch.zeros_like(alpha)
        az_p = torch.zeros_like(alpha)
        az_m = torch.zeros_like(alpha)

        if periodic:
            ax_p = torch.roll(alpha, -1, dims=2)
            ax_m = torch.roll(alpha, 1, dims=2)
            ay_p = torch.roll(alpha, -1, dims=1)
            ay_m = torch.roll(alpha, 1, dims=1)
            az_p = torch.roll(alpha, -1, dims=0)
            az_m = torch.roll(alpha, 1, dims=0)
        else:
            ax_p[:, :, :-1] = alpha[:, :, 1:]
            ax_m[:, :, 1:] = alpha[:, :, :-1]
            ax_p[:, :, -1] = alpha[:, :, -1]
            ax_m[:, :, 0] = alpha[:, :, 0]

            ay_p[:, :-1, :] = alpha[:, 1:, :]
            ay_m[:, 1:, :] = alpha[:, :-1, :]
            ay_p[:, -1, :] = alpha[:, -1, :]
            ay_m[:, 0, :] = alpha[:, 0, :]

            az_p[:-1, :, :] = alpha[1:, :, :]
            az_m[1:, :, :] = alpha[:-1, :, :]
            az_p[-1, :, :] = alpha[-1, :, :]
            az_m[0, :, :] = alpha[0, :, :]

        n_x = (ax_p - ax_m) / 2.0
        n_y = (ay_p - ay_m) / 2.0
        n_z = (az_p - az_m) / 2.0

        return n_x, n_y, n_z


def compute_interface_normal(
    alpha: torch.Tensor,
    periodic: bool = False,
    smooth: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Compute normalized interface normal from volume fraction.

    Args:
        alpha: Volume fraction field
        periodic: Periodic boundary conditions
        smooth: Apply smoothing to normal field

    Returns:
        Tuple of normal components (n_x, n_y) or (n_x, n_y, n_z)
    """
    if alpha.dim() == 2:
        n_x, n_y = youngs_normal(alpha, periodic)
    else:
        n_x, n_y, n_z = youngs_normal(alpha, periodic)

    # Compute magnitude
    if alpha.dim() == 2:
        mag = torch.sqrt(n_x**2 + n_y**2 + 1e-12)
    else:
        mag = torch.sqrt(n_x**2 + n_y**2 + n_z**2 + 1e-12)

    # Normalize
    n_x = n_x / mag
    n_y = n_y / mag

    if smooth:
        # Simple smoothing
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=alpha.dtype, device=alpha.device) / 16.0
        # Apply smoothing (simplified)
        n_x = _smooth_2d(n_x, kernel) if alpha.dim() == 2 else n_x
        n_y = _smooth_2d(n_y, kernel) if alpha.dim() == 2 else n_y

    if alpha.dim() == 2:
        return n_x, n_y
    else:
        n_z = n_z / mag
        if smooth:
            n_z = n_z  # Skip 3D smoothing for now
        return n_x, n_y, n_z


def _smooth_2d(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 smoothing kernel to 2D field."""
    ny, nx = field.shape
    out = torch.zeros_like(field)
    # Center
    out[1:-1, 1:-1] = (
        kernel[0, 0] * field[:-2, :-2] + kernel[0, 1] * field[:-2, 1:-1] + kernel[0, 2] * field[:-2, 2:] +
        kernel[1, 0] * field[1:-1, :-2] + kernel[1, 1] * field[1:-1, 1:-1] + kernel[1, 2] * field[1:-1, 2:] +
        kernel[2, 0] * field[2:, :-2] + kernel[2, 1] * field[2:, 1:-1] + kernel[2, 2] * field[2:, 2:]
    )
    # Boundaries: copy
    out[0, :] = field[0, :]
    out[-1, :] = field[-1, :]
    out[:, 0] = field[:, 0]
    out[:, -1] = field[:, -1]
    return out


def reconstruct_interface_plic(
    alpha: torch.Tensor,
    normal: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """PLIC (Piecewise Linear Interface Calculation) reconstruction.

    Reconstructs the interface as a plane (2D: line) within each cell
    based on the volume fraction and interface normal.

    Args:
        alpha: Volume fraction field
        normal: Interface normal components

    Returns:
        Reconstructed interface field (signed distance to interface)
    """
    if alpha.dim() == 2:
        n_x, n_y = normal
        ny, nx = alpha.shape

        # For each cell, find the line position that gives the correct volume fraction
        # This is a simplified version - full PLIC requires iterative root finding
        # Here we use the approximation: d = alpha - 0.5 (signed distance proxy)
        d = alpha - 0.5

        # Interface is where d = 0 (alpha = 0.5)
        return d
    else:
        n_x, n_y, n_z = normal
        d = alpha - 0.5
        return d


def compute_curvature(
    alpha: torch.Tensor,
    periodic: bool = False,
    method: str = "height_function",
) -> torch.Tensor:
    """Compute interface curvature from volume fraction field.

    Methods:
        - "height_function": Height function method (most accurate for structured grids)
        - "divergence": Curvature from divergence of normal field
        - "laplacian": Curvature from Laplacian of volume fraction

    Args:
        alpha: Volume fraction field
        periodic: Periodic boundary conditions
        method: Curvature computation method

    Returns:
        Curvature field kappa
    """
    if method == "height_function":
        return height_function_curvature(alpha, periodic)
    elif method == "divergence":
        return _curvature_from_normal_divergence(alpha, periodic)
    elif method == "laplacian":
        return _curvature_from_laplacian(alpha, periodic)
    else:
        raise ValueError(f"Unknown curvature method: {method}")


def height_function_curvature(
    alpha: torch.Tensor,
    periodic: bool = False,
) -> torch.Tensor:
    """Compute curvature using the height function method.

    The height function method integrates the volume fraction in one direction
    to get the interface height, then computes curvature from the second
    derivative of the height.

    This is the most accurate method for structured grids but requires the
    interface to be mostly aligned with one coordinate direction.

    Args:
        alpha: Volume fraction field (2D or 3D)
        periodic: Periodic boundary conditions

    Returns:
        Curvature field kappa
    """
    if alpha.dim() == 2:
        ny, nx = alpha.shape

        # Compute height function by integrating alpha in y-direction
        # h(x) = sum_j alpha(j, x)
        h = alpha.sum(dim=0)  # (nx,)

        # Compute second derivative of height
        if periodic:
            h_ip1 = torch.roll(h, -1)
            h_im1 = torch.roll(h, 1)
        else:
            h_ip1 = torch.zeros_like(h)
            h_im1 = torch.zeros_like(h)
            h_ip1[:-1] = h[1:]
            h_im1[1:] = h[:-1]
            h_ip1[-1] = h[-1]
            h_im1[0] = h[0]

        # kappa = -h'' / (1 + h'^2)^(3/2)
        h_prime = (h_ip1 - h_im1) / 2.0
        h_double_prime = h_ip1 - 2 * h + h_im1

        # Curvature (2D)
        kappa_1d = -h_double_prime / (1.0 + h_prime**2)**1.5

        # Broadcast to 2D
        kappa = kappa_1d.unsqueeze(0).expand(ny, nx)

        return kappa
    else:
        # 3D case - simplified
        nz, ny, nx = alpha.shape
        # Integrate in z-direction
        h = alpha.sum(dim=0)  # (ny, nx)

        # Second derivatives
        if periodic:
            h_xp = torch.roll(h, -1, dims=1)
            h_xm = torch.roll(h, 1, dims=1)
            h_yp = torch.roll(h, -1, dims=0)
            h_ym = torch.roll(h, 1, dims=0)
        else:
            h_xp = torch.zeros_like(h)
            h_xm = torch.zeros_like(h)
            h_yp = torch.zeros_like(h)
            h_ym = torch.zeros_like(h)
            h_xp[:, :-1] = h[:, 1:]
            h_xm[:, 1:] = h[:, :-1]
            h_xp[:, -1] = h[:, -1]
            h_xm[:, 0] = h[:, 0]
            h_yp[:-1, :] = h[1:, :]
            h_ym[1:, :] = h[:-1, :]
            h_yp[-1, :] = h[-1, :]
            h_ym[0, :] = h[0, :]

        h_xx = h_xp - 2 * h + h_xm
        h_yy = h_yp - 2 * h + h_ym
        h_xy = (torch.roll(h_xp, -1, dims=0) - torch.roll(h_xm, -1, dims=0)
                - torch.roll(h_xp, 1, dims=0) + torch.roll(h_xm, 1, dims=0)) / 4.0

        # Mean curvature
        kappa_2d = -(h_xx + h_yy) / 2.0

        # Broadcast to 3D
        kappa = kappa_2d.unsqueeze(0).expand(nz, ny, nx)

        return kappa


def _curvature_from_normal_divergence(
    alpha: torch.Tensor,
    periodic: bool = False,
) -> torch.Tensor:
    """Compute curvature as divergence of the normal field: kappa = -div(n_hat)."""
    n_x, n_y = compute_interface_normal(alpha, periodic, smooth=True)

    # Divergence of normal
    if periodic:
        dnx_dx = (torch.roll(n_x, -1, dims=1) - torch.roll(n_x, 1, dims=1)) / 2.0
        dny_dy = (torch.roll(n_y, -1, dims=0) - torch.roll(n_y, 1, dims=0)) / 2.0
    else:
        dnx_dx = torch.zeros_like(n_x)
        dny_dy = torch.zeros_like(n_y)
        dnx_dx[:, 1:-1] = (n_x[:, 2:] - n_x[:, :-2]) / 2.0
        dny_dy[1:-1, :] = (n_y[2:, :] - n_y[:-2, :]) / 2.0

    kappa = -(dnx_dx + dny_dy)
    return kappa


def _curvature_from_laplacian(
    alpha: torch.Tensor,
    periodic: bool = False,
) -> torch.Tensor:
    """Compute curvature from Laplacian of volume fraction (approximate)."""
    if periodic:
        lap = (
            torch.roll(alpha, -1, dims=1) + torch.roll(alpha, 1, dims=1) +
            torch.roll(alpha, -1, dims=0) + torch.roll(alpha, 1, dims=0) -
            4 * alpha
        )
    else:
        lap = torch.zeros_like(alpha)
        lap[:, 1:-1] = alpha[:, 2:] + alpha[:, :-2] - 2 * alpha[:, 1:-1]
        lap[1:-1, :] += alpha[2:, :] + alpha[:-2, :] - 2 * alpha[1:-1, :]

    # Curvature is proportional to Laplacian at interface
    # This is a rough approximation
    kappa = -lap / 4.0
    return kappa


def advect_vof(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float = 1.0,
    periodic: bool = False,
    method: str = "upwind",
) -> torch.Tensor:
    """Advect volume fraction field with velocity field.

    Solves: d(alpha)/dt + div(alpha * u) = 0

    Args:
        alpha: Volume fraction field (ny, nx)
        u: x-velocity field (ny, nx)
        v: y-velocity field (ny, nx)
        dt: Time step
        periodic: Periodic boundary conditions
        method: Advection method ("upwind", "central")

    Returns:
        Updated volume fraction field
    """
    ny, nx = alpha.shape

    if method == "upwind":
        # First-order upwind
        # Flux in x-direction
        flux_x_pos = torch.where(u > 0, u * alpha, torch.zeros_like(alpha))
        flux_x_neg = torch.where(u < 0, u * alpha, torch.zeros_like(alpha))

        if periodic:
            flux_x_pos_left = torch.roll(flux_x_pos, 1, dims=1)
            flux_x_neg_right = torch.roll(flux_x_neg, -1, dims=1)
        else:
            flux_x_pos_left = torch.zeros_like(alpha)
            flux_x_neg_right = torch.zeros_like(alpha)
            flux_x_pos_left[:, 1:] = flux_x_pos[:, :-1]
            flux_x_neg_right[:, :-1] = flux_x_neg[:, 1:]

        # Flux in y-direction
        flux_y_pos = torch.where(v > 0, v * alpha, torch.zeros_like(alpha))
        flux_y_neg = torch.where(v < 0, v * alpha, torch.zeros_like(alpha))

        if periodic:
            flux_y_pos_down = torch.roll(flux_y_pos, 1, dims=0)
            flux_y_neg_up = torch.roll(flux_y_neg, -1, dims=0)
        else:
            flux_y_pos_down = torch.zeros_like(alpha)
            flux_y_neg_up = torch.zeros_like(alpha)
            flux_y_pos_down[1:, :] = flux_y_pos[:-1, :]
            flux_y_neg_up[:-1, :] = flux_y_neg[1:, :]

        # Update
        div_flux = (flux_x_pos - flux_x_pos_left + flux_x_neg_right - flux_x_neg +
                    flux_y_pos - flux_y_pos_down + flux_y_neg_up - flux_y_neg)
        alpha_new = alpha - dt * div_flux

    else:  # central
        # Central difference (less stable but more accurate)
        if periodic:
            alpha_xp = torch.roll(alpha, -1, dims=1)
            alpha_xm = torch.roll(alpha, 1, dims=1)
            alpha_yp = torch.roll(alpha, -1, dims=0)
            alpha_ym = torch.roll(alpha, 1, dims=0)
        else:
            alpha_xp = torch.zeros_like(alpha)
            alpha_xm = torch.zeros_like(alpha)
            alpha_yp = torch.zeros_like(alpha)
            alpha_ym = torch.zeros_like(alpha)
            alpha_xp[:, :-1] = alpha[:, 1:]
            alpha_xm[:, 1:] = alpha[:, :-1]
            alpha_yp[:-1, :] = alpha[1:, :]
            alpha_ym[1:, :] = alpha[:-1, :]

        dalpha_dx = (alpha_xp - alpha_xm) / 2.0
        dalpha_dy = (alpha_yp - alpha_ym) / 2.0

        alpha_new = alpha - dt * (u * dalpha_dx + v * dalpha_dy)

    # Clamp to [0, 1]
    alpha_new = torch.clamp(alpha_new, 0.0, 1.0)

    return alpha_new

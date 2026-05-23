"""Post-processing utilities for TensorLBM simulation data.

Provides:
- :func:`extract_velocity_profile`  – velocity slice at a fixed x or y position.
- :func:`compute_pressure_coefficient` – pressure coefficient Cp field.
- :func:`compute_q_criterion`       – Q-criterion for 3-D vortex identification.
"""
from __future__ import annotations

import torch


def extract_velocity_profile(
    ux: torch.Tensor,
    uy: torch.Tensor,
    axis: str = "x",
    index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a 1-D velocity profile by slicing the 2-D velocity fields.

    Args:
        ux: x-velocity field, shape ``(ny, nx)``.
        uy: y-velocity field, shape ``(ny, nx)``.
        axis: ``"x"`` to slice at a constant *x* (returns a profile along y),
              ``"y"`` to slice at a constant *y* (returns a profile along x).
        index: Grid index along the chosen axis.

    Returns:
        Tuple ``(ux_profile, uy_profile)`` — 1-D tensors of length ``ny``
        (when *axis* = ``"x"``) or ``nx`` (when *axis* = ``"y"``).
    """
    if axis == "x":
        return ux[:, index], uy[:, index]
    if axis == "y":
        return ux[index, :], uy[index, :]
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def compute_pressure_coefficient(
    rho: torch.Tensor,
    u_in: float,
    rho_ref: float = 1.0,
    cs2: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Compute the pressure coefficient field Cp.

    In LBM the equation of state is :math:`p = c_s^2 \\rho`, so the pressure
    fluctuation relative to the reference state is:

    .. math::

        C_p = \\frac{p - p_{ref}}{\\tfrac{1}{2} \\rho_{ref} U^2}
            = \\frac{c_s^2 (\\rho - \\rho_{ref})}{\\tfrac{1}{2} \\rho_{ref} U^2}

    Args:
        rho: Density field of shape ``(ny, nx)`` or ``(nz, ny, nx)``.
        u_in: Reference inlet velocity :math:`U`.
        rho_ref: Reference density (default 1.0).
        cs2: Lattice speed of sound squared (default 1/3).

    Returns:
        Cp field of the same shape as *rho*.
    """
    dyn_pressure = 0.5 * rho_ref * u_in**2
    if dyn_pressure == 0.0:
        return torch.zeros_like(rho)
    p = cs2 * rho
    p_ref = cs2 * rho_ref
    return (p - p_ref) / dyn_pressure


def compute_q_criterion(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Compute the Q-criterion for 3-D vortex identification.

    The Q-criterion is defined as:

    .. math::

        Q = \\tfrac{1}{2}\\left(\\|\\boldsymbol{\\Omega}\\|_F^2
            - \\|\\mathbf{S}\\|_F^2\\right)

    where :math:`\\boldsymbol{\\Omega}` is the antisymmetric (rotation) part
    and :math:`\\mathbf{S}` is the symmetric (strain-rate) part of the
    velocity gradient tensor. Vortex cores are regions where *Q* > 0.

    Uses second-order central differences for interior cells; boundary rows
    use forward/backward differences.

    Args:
        ux: x-velocity, shape ``(nz, ny, nx)``.
        uy: y-velocity, shape ``(nz, ny, nx)``.
        uz: z-velocity, shape ``(nz, ny, nx)``.

    Returns:
        Q-criterion field of shape ``(nz, ny, nx)``.
    """

    def _grad(field: torch.Tensor, dim: int) -> torch.Tensor:
        """Central-difference gradient along *dim* with edge padding."""
        g = torch.zeros_like(field)
        if dim == 0:
            g[1:-1] = 0.5 * (field[2:] - field[:-2])
            g[0] = field[1] - field[0]
            g[-1] = field[-1] - field[-2]
        elif dim == 1:
            g[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
            g[:, 0] = field[:, 1] - field[:, 0]
            g[:, -1] = field[:, -1] - field[:, -2]
        else:
            g[:, :, 1:-1] = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
            g[:, :, 0] = field[:, :, 1] - field[:, :, 0]
            g[:, :, -1] = field[:, :, -1] - field[:, :, -2]
        return g

    dudx, dudy, dudz = _grad(ux, 2), _grad(ux, 1), _grad(ux, 0)
    dvdx, dvdy, dvdz = _grad(uy, 2), _grad(uy, 1), _grad(uy, 0)
    dwdx, dwdy, dwdz = _grad(uz, 2), _grad(uz, 1), _grad(uz, 0)

    s_xx = dudx
    s_yy = dvdy
    s_zz = dwdz
    s_xy = 0.5 * (dudy + dvdx)
    s_xz = 0.5 * (dudz + dwdx)
    s_yz = 0.5 * (dvdz + dwdy)
    s_sq = s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)

    w_xy = 0.5 * (dudy - dvdx)
    w_xz = 0.5 * (dudz - dwdx)
    w_yz = 0.5 * (dvdz - dwdy)
    omega_sq = 2.0 * (w_xy**2 + w_xz**2 + w_yz**2)

    return 0.5 * (omega_sq - s_sq)


__all__ = [
    "extract_velocity_profile",
    "compute_pressure_coefficient",
    "compute_q_criterion",
]

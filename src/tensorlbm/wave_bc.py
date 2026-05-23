"""Airy linear-wave inlet boundary conditions for ocean engineering simulations.

Implements the Airy (linear, small-amplitude) wave theory to prescribe
time-varying velocity profiles at the inlet face of a 3-D LBM domain,
allowing simulation of regular ocean waves interacting with ship hulls and
offshore structures.

The horizontal velocity at the inlet (*x* = 0) at depth *z* and time-step *t*
is derived from Airy wave theory for finite water depth *H*:

.. math::

    u_x(z, t) = U_0 + U_w \\cos(\\omega t)
                \\frac{\\cosh(k (z - z_{bed}))}{\\cosh(k H)}

.. math::

    u_z(z, t) = -U_w \\sin(\\omega t)
                \\frac{\\sinh(k (z - z_{bed}))}{\\sinh(k H)}

where :math:`\\omega = 2\\pi / T_w`, :math:`k = 2\\pi / \\lambda`,
:math:`U_0` is the mean current, and :math:`U_w` is the horizontal velocity
amplitude at the free surface.

Exported functions
------------------
- :func:`airy_wave_velocity_3d`         – velocity profile tensor for one step
- :func:`zou_he_inlet_velocity_profile_3d` – Zou/He inlet with a 2-D velocity field
- :func:`apply_wave_inlet_3d`           – full-step wave inlet + wall BC helper
"""

from __future__ import annotations

import math

import torch

from .boundaries3d import bounce_back_cells_3d
from .d3q19 import OPPOSITE, equilibrium3d


def airy_wave_velocity_3d(
    nz: int,
    ny: int,
    step: int,
    u_mean: float,
    wave_amp: float,
    wave_period: float,
    wave_k: float,
    water_depth: float,
    z_bed: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Airy wave velocity components at the inlet plane for one time step.

    Returns the prescribed inlet velocity field (ux, uy, uz) of shape
    ``(nz, ny)`` that encodes the depth-dependent horizontal and vertical wave
    kinematics per Airy linear wave theory.

    Args:
        nz: Number of vertical lattice cells.
        ny: Number of lateral lattice cells.
        step: Current simulation time step (used as time *t* in LBM units).
        u_mean: Mean current x-velocity (always positive, in lattice units).
        wave_amp: Horizontal velocity amplitude at the free surface in lattice
                  units (i.e. *U_w = A · ω* where *A* is wave height / 2).
                  Should be small (≪ speed of sound cs ≈ 0.577).
        wave_period: Wave period in LBM time steps.
        wave_k: Wave number k = 2π / λ (in units of 1 / lattice spacing).
        water_depth: Water depth *H* in lattice units.
        z_bed: z-index of the sea bed (bottom of the water column).
        device: Target PyTorch device.

    Returns:
        Tuple ``(ux, uy, uz)`` each of shape ``(nz, ny)``.
        *uy* is zero (no lateral wave component at a normally-incident inlet).
    """
    omega = 2.0 * math.pi / wave_period
    phase = omega * step  # ωt at x=0

    z_coords = torch.arange(nz, device=device, dtype=torch.float32)
    depth_from_bed = z_coords - z_bed  # (nz,)

    kH = wave_k * water_depth
    sinh_kH = math.sinh(kH) if kH > 1e-8 else 1e-8
    cosh_kH = math.cosh(kH)

    # Horizontal velocity depth profile: cosh(k*(z-z_bed)) / cosh(k*H)
    cosh_z = torch.cosh(wave_k * depth_from_bed.clamp(min=0.0))  # (nz,)
    ux_profile = wave_amp * math.cos(phase) * cosh_z / cosh_kH  # (nz,)

    # Vertical velocity depth profile: -sinh(k*(z-z_bed)) / sinh(k*H)
    sinh_z = torch.sinh(wave_k * depth_from_bed.clamp(min=0.0))  # (nz,)
    uz_profile = -wave_amp * math.sin(phase) * sinh_z / sinh_kH  # (nz,)

    # Expand to (nz, ny)
    ux = (ux_profile.unsqueeze(1).expand(nz, ny) + u_mean)
    uy = torch.zeros(nz, ny, device=device, dtype=torch.float32)
    uz = uz_profile.unsqueeze(1).expand(nz, ny)

    return ux, uy, uz


def zou_he_inlet_velocity_profile_3d(
    f: torch.Tensor,
    ux_in: torch.Tensor,
    uy_in: torch.Tensor,
    uz_in: torch.Tensor,
) -> torch.Tensor:
    """Zou/He inlet velocity BC at x=0 with a non-uniform 2-D velocity field.

    This is a generalisation of :func:`zou_he_inlet_velocity_3d` that accepts
    spatially varying velocity tensors instead of scalar values.  It uses the
    non-equilibrium bounce-back (NEBB) method (Latt & Chopard 2008):

    .. math::

        f_k = f_k^{eq}(\\rho, \\mathbf{u}_{in}) - f_{\\bar{k}}^{eq}(\\rho,
              \\mathbf{u}_{in}) + f_{\\bar{k}}

    for every incoming direction *k* (cx > 0).

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        ux_in: Prescribed x-velocity field of shape ``(nz, ny)`` at the inlet.
        uy_in: Prescribed y-velocity field of shape ``(nz, ny)`` at the inlet.
        uz_in: Prescribed z-velocity field of shape ``(nz, ny)`` at the inlet.

    Returns:
        Updated distribution tensor of the same shape.
    """
    device = f.device

    # Sum of cx=0 directions and cx<0 directions at x=0
    sum_cx0 = (
        f[0, :, :, 0] + f[3, :, :, 0] + f[4, :, :, 0]
        + f[5, :, :, 0] + f[6, :, :, 0]
        + f[15, :, :, 0] + f[16, :, :, 0]
        + f[17, :, :, 0] + f[18, :, :, 0]
    )
    sum_cx_neg = f[2, :, :, 0] + f[8, :, :, 0] + f[10, :, :, 0] + f[12, :, :, 0] + f[14, :, :, 0]

    # Infer density from mass balance
    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_in)

    # Equilibrium at the inlet
    feq = equilibrium3d(rho, ux_in, uy_in, uz_in, device=device)  # (19, nz, ny)

    f_new = f.clone()
    opp = OPPOSITE.to(device)
    for k in (1, 7, 9, 11, 13):  # directions with cx > 0
        opp_k = int(opp[k].item())
        f_new[k, :, :, 0] = feq[k] - feq[opp_k] + f[opp_k, :, :, 0]
    return f_new


def apply_wave_inlet_3d(
    f: torch.Tensor,
    step: int,
    wall_mask: torch.Tensor,
    obstacle_mask: torch.Tensor,
    u_mean: float,
    wave_amp: float,
    wave_period: float,
    wave_k: float,
    water_depth: float,
    z_bed: float,
    rho_out: float = 1.0,
) -> torch.Tensor:
    """Apply Airy-wave inlet + pressure outlet + bounce-back in one call.

    Combines :func:`airy_wave_velocity_3d` and
    :func:`zou_he_inlet_velocity_profile_3d` to prescribe a regular wave
    velocity profile at the inlet, applies a pressure (Zou/He) outlet at
    the right face, and applies bounce-back to walls and the obstacle.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        step: Current simulation time step.
        wall_mask: Boolean wall mask of shape ``(nz, ny, nx)``.
        obstacle_mask: Boolean obstacle mask of shape ``(nz, ny, nx)``.
        u_mean: Mean current x-velocity.
        wave_amp: Horizontal velocity amplitude at the free surface.
        wave_period: Wave period in LBM time steps.
        wave_k: Wave number in units of 1/lattice spacing.
        water_depth: Water depth in lattice units.
        z_bed: z-index of the sea bed.
        rho_out: Prescribed density at the outlet (default 1.0).

    Returns:
        Updated distribution tensor of the same shape.
    """
    from .boundaries3d import zou_he_outlet_pressure_3d  # local import avoids circular

    nz, ny = f.shape[1], f.shape[2]
    device = f.device

    ux_in, uy_in, uz_in = airy_wave_velocity_3d(
        nz, ny, step, u_mean, wave_amp, wave_period, wave_k, water_depth, z_bed, device
    )
    f = zou_he_inlet_velocity_profile_3d(f, ux_in, uy_in, uz_in)
    f = zou_he_outlet_pressure_3d(f, rho_out=rho_out)
    f = bounce_back_cells_3d(f, wall_mask)
    f = bounce_back_cells_3d(f, obstacle_mask)
    return f


__all__ = [
    "airy_wave_velocity_3d",
    "zou_he_inlet_velocity_profile_3d",
    "apply_wave_inlet_3d",
]

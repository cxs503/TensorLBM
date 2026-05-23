from __future__ import annotations

import math

import torch

from .d3q19 import OPPOSITE, equilibrium3d


def airy_wave_velocity_3d(
    t: float,
    amplitude: float,
    wavelength: float,
    depth: float,
    iz_still_water: int,
    nz: int,
    ny: int,
    device: torch.device,
    g: float = 1.0 / 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Airy (linear) wave horizontal and vertical velocity profiles.

    The wave propagates in the +x direction.  Profiles are evaluated at
    x = 0 (the inlet plane) for time *t*.

    Airy theory (finite depth h):

        u_x(z, t) = A · ω · cosh(k · (z_phys + h)) / sinh(k·h) · cos(−ω·t)
        u_z(z, t) = A · ω · sinh(k · (z_phys + h)) / sinh(k·h) · sin(−ω·t)

    Dispersion relation: ω² = g · k · tanh(k·h).

    Velocity is set to zero for grid rows above the still-water level
    (``iz > iz_still_water``).

    Args:
        t: Current simulation time in LBM steps.
        amplitude: Wave amplitude *A* in lattice units.
        wavelength: Wavelength λ in lattice units.
        depth: Water depth *h* in lattice units (distance from tank bottom to
            still-water surface).
        iz_still_water: z-index of the still-water surface.
        nz: Number of grid points in the z-direction.
        ny: Number of grid points in the y-direction.
        device: Target torch device.
        g: Gravitational acceleration in LBM units
            (default c_s² = 1/3, the lattice speed of sound squared).

    Returns:
        Tuple ``(ux_profile, uz_profile)`` each of shape ``(nz, ny)``.
        Values are zero for rows above ``iz_still_water``.
    """
    k = 2.0 * math.pi / wavelength
    kh = k * depth
    omega = math.sqrt(g * k * math.tanh(kh))
    phase = -omega * t

    zz = torch.arange(nz, device=device, dtype=torch.float32)

    # Physical z measured upward from still-water level (≤0 below surface)
    z_phys = zz - float(iz_still_water)
    below = z_phys <= 0.0

    # z_phys + h: depth below still-water measured from tank bottom; ∈ [0, h]
    z_depth = (z_phys + depth).clamp(min=0.0)

    sinh_kh = math.sinh(kh)
    if abs(sinh_kh) < 1e-15:
        sinh_kh = 1e-15

    ux_z = torch.where(
        below,
        amplitude * omega * torch.cosh(k * z_depth) / sinh_kh * math.cos(phase),
        torch.zeros_like(zz),
    )
    uz_z = torch.where(
        below,
        amplitude * omega * torch.sinh(k * z_depth) / sinh_kh * math.sin(phase),
        torch.zeros_like(zz),
    )

    # Broadcast to (nz, ny)
    ux_profile = ux_z.unsqueeze(1).expand(nz, ny).contiguous()
    uz_profile = uz_z.unsqueeze(1).expand(nz, ny).contiguous()
    return ux_profile, uz_profile


def zou_he_inlet_velocity_profile_3d(
    f: torch.Tensor,
    ux_profile: torch.Tensor,
    uy_profile: torch.Tensor,
    uz_profile: torch.Tensor,
) -> torch.Tensor:
    """Zou/He inlet velocity BC at x = 0 with a spatially varying profile.

    Generalises :func:`~tensorlbm.zou_he_inlet_velocity_3d` to support
    non-uniform inlet profiles (e.g. boundary-layer or wave velocity
    profiles).  Uses the non-equilibrium bounce-back reconstruction
    (Latt & Chopard 2008).

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        ux_profile: x-velocity profile of shape ``(nz, ny)``.
        uy_profile: y-velocity profile of shape ``(nz, ny)``.
        uz_profile: z-velocity profile of shape ``(nz, ny)``.

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    opp = OPPOSITE.to(device)

    # ---- density from Zou/He x-momentum balance at inlet (x=0) ----
    # Populations with cx=0: directions 0,3,4,5,6,15,16,17,18
    # Populations with cx=-1 (known after streaming): directions 2,8,10,12,14
    sum_cx0 = (
        f[0, :, :, 0] + f[3, :, :, 0] + f[4, :, :, 0]
        + f[5, :, :, 0] + f[6, :, :, 0]
        + f[15, :, :, 0] + f[16, :, :, 0] + f[17, :, :, 0] + f[18, :, :, 0]
    )  # (nz, ny)
    sum_cx_neg = (
        f[2, :, :, 0] + f[8, :, :, 0] + f[10, :, :, 0]
        + f[12, :, :, 0] + f[14, :, :, 0]
    )  # (nz, ny)
    rho = (sum_cx0 + 2.0 * sum_cx_neg) / (1.0 - ux_profile)  # (nz, ny)

    # ---- compute equilibrium at the inlet as a (19, nz, ny, 1) column ----
    # Unsqueeze to add a dummy x-dimension so equilibrium3d returns 4D output.
    rho_col = rho.unsqueeze(-1)            # (nz, ny, 1)
    ux_col = ux_profile.unsqueeze(-1)      # (nz, ny, 1)
    uy_col = uy_profile.unsqueeze(-1)      # (nz, ny, 1)
    uz_col = uz_profile.unsqueeze(-1)      # (nz, ny, 1)
    feq_col = equilibrium3d(rho_col, ux_col, uy_col, uz_col, device=device)
    # feq_col shape: (19, nz, ny, 1)

    # ---- non-equilibrium bounce-back for cx>0 directions ----
    # Incoming directions (cx>0): 1,7,9,11,13
    f_new = f.clone()
    for k in (1, 7, 9, 11, 13):
        k_opp = int(opp[k].item())
        f_new[k, :, :, 0] = (
            feq_col[k, :, :, 0] - feq_col[k_opp, :, :, 0] + f[k_opp, :, :, 0]
        )
    return f_new


def apply_sponge_layer_3d(
    f: torch.Tensor,
    rho_target: torch.Tensor,
    ux_target: torch.Tensor,
    uy_target: torch.Tensor,
    uz_target: torch.Tensor,
    ix_start: int,
    sigma_max: float = 0.3,
    power: int = 2,
) -> torch.Tensor:
    """Apply an outlet sponge (damping) layer to suppress wave reflection.

    Within the sponge region x ∈ [``ix_start``, nx−1] the distributions are
    relaxed towards a target equilibrium:

        f ← f − σ(x) · (f − f_eq(ρ_t, **u**_t))

    The damping coefficient grows polynomially from zero:

        σ(x) = σ_max · ((x − ix_start) / (nx − 1 − ix_start))^power

    Cells at x < ``ix_start`` are not modified.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        rho_target: Target density of shape ``(nz, ny, nx)`` (broadcastable).
        ux_target: Target x-velocity of shape ``(nz, ny, nx)`` (broadcastable).
        uy_target: Target y-velocity of shape ``(nz, ny, nx)`` (broadcastable).
        uz_target: Target z-velocity of shape ``(nz, ny, nx)`` (broadcastable).
        ix_start: First x-index of the sponge region.
        sigma_max: Maximum damping coefficient at the outlet (default 0.3).
        power: Polynomial exponent for σ growth (default 2).

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    nx = f.shape[3]
    sponge_len = float(nx - 1 - ix_start)
    if sponge_len <= 0.0:
        return f

    feq_target = equilibrium3d(rho_target, ux_target, uy_target, uz_target, device=device)

    x_idx = torch.arange(nx, device=device, dtype=torch.float32)
    sigma = torch.where(
        x_idx >= float(ix_start),
        sigma_max * ((x_idx - float(ix_start)) / sponge_len) ** power,
        torch.zeros_like(x_idx),
    ).view(1, 1, 1, nx)

    return f - sigma * (f - feq_target)


def apply_wave_inlet_3d(
    f: torch.Tensor,
    t: float,
    amplitude: float,
    wavelength: float,
    depth: float,
    iz_still_water: int,
    mean_ux: float = 0.0,
) -> torch.Tensor:
    """Apply an Airy wave + mean current velocity BC at the inlet (x = 0).

    Combines Airy wave orbital velocities with an optional mean current and
    applies the result as a spatially varying Zou/He inlet profile BC
    (:func:`zou_he_inlet_velocity_profile_3d`).

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        t: Current simulation time in LBM steps.
        amplitude: Wave amplitude in lattice units.
        wavelength: Wavelength in lattice units.
        depth: Water depth in lattice units.
        iz_still_water: z-index of the still-water surface.
        mean_ux: Mean inflow x-velocity (e.g. ship advance speed) in LBM units.

    Returns:
        Updated distribution tensor (same shape).
    """
    device = f.device
    nz, ny = f.shape[1], f.shape[2]

    ux_wave, uz_wave = airy_wave_velocity_3d(
        t=t,
        amplitude=amplitude,
        wavelength=wavelength,
        depth=depth,
        iz_still_water=iz_still_water,
        nz=nz,
        ny=ny,
        device=device,
    )
    ux_profile = ux_wave + mean_ux
    uy_profile = torch.zeros(nz, ny, device=device, dtype=f.dtype)
    return zou_he_inlet_velocity_profile_3d(f, ux_profile, uy_profile, uz_wave)


__all__ = [
    "airy_wave_velocity_3d",
    "zou_he_inlet_velocity_profile_3d",
    "apply_sponge_layer_3d",
    "apply_wave_inlet_3d",
]

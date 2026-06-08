"""Wall-function boundary condition for LBM (simplified).
Computes slip velocity from fluid cells adjacent to the hull,
using the von Karman log-law.
"""
from __future__ import annotations
import torch
from .propeller_benchmark import moving_wall_bounce_back_3d

KAPPA = 0.41
B_CONST = 5.0


def compute_wall_slip_velocity(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    mask: torch.Tensor, nu: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute slip velocity for solid cells adjacent to fluid.

    Simple approach: find the first fluid neighbor for each wall cell,
    compute u_tan, solve log-law, return slip velocity grid.
    """
    device = ux.device
    nz, ny, nx = ux.shape
    ux_s = torch.zeros_like(ux)
    uy_s = torch.zeros_like(uy)
    uz_s = torch.zeros_like(uz)

    # Mask of fluid cells next to solid (this is where slip applies)
    m = mask
    fluid_nbr = torch.zeros_like(m)
    for dk, dj, di in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
        # Shift mask and check
        s1 = slice(1, None) if dk == 1 else (slice(-1) if dk == -1 else slice(None))
        s2 = slice(1, None) if dj == 1 else (slice(-1) if dj == -1 else slice(None))
        s3 = slice(1, None) if di == 1 else (slice(-1) if di == -1 else slice(None))
        t1 = slice(None, -1) if dk == 1 else (slice(1, None) if dk == -1 else slice(None))
        t2 = slice(None, -1) if dj == 1 else (slice(1, None) if dj == -1 else slice(None))
        t3 = slice(None, -1) if di == 1 else (slice(1, None) if di == -1 else slice(None))
        fluid_nbr[t1, t2, t3] |= ~m[s1, s2, s3] & m[t1, t2, t3]

    wall_adjacent = m & fluid_nbr
    if not wall_adjacent.any():
        return ux_s, uy_s, uz_s

    # For each wall cell, take velocity from the first fluid neighbor
    for dk, dj, di in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
        s1 = slice(1, None) if dk == 1 else (slice(-1) if dk == -1 else slice(None))
        s2 = slice(1, None) if dj == 1 else (slice(-1) if dj == -1 else slice(None))
        s3 = slice(1, None) if di == 1 else (slice(-1) if di == -1 else slice(None))
        t1 = slice(None, -1) if dk == 1 else (slice(1, None) if dk == -1 else slice(None))
        t2 = slice(None, -1) if dj == 1 else (slice(1, None) if dj == -1 else slice(None))
        t3 = slice(None, -1) if di == 1 else (slice(1, None) if di == -1 else slice(None))
        # Cell [t] is solid, cell [s] is fluid
        from_fluid = m[t1, t2, t3] & ~m[s1, s2, s3]
        if not from_fluid.any():
            continue
        ux_s[t1, t2, t3] = torch.where(from_fluid, ux[s1, s2, s3], ux_s[t1, t2, t3])
        uy_s[t1, t2, t3] = torch.where(from_fluid, uy[s1, s2, s3], uy_s[t1, t2, t3])
        uz_s[t1, t2, t3] = torch.where(from_fluid, uz[s1, s2, s3], uz_s[t1, t2, t3])

    # Compute slip ratio for wall-adjacent cells
    u_mag = torch.sqrt(ux_s**2 + uy_s**2 + uz_s**2)
    u_mag_w = u_mag[wall_adjacent]
    y_val = 1.5
    # Laminar estimate
    u_tau_lam = torch.sqrt(nu * u_mag_w / y_val)
    y_plus_lam = y_val * u_tau_lam / nu
    # Use laminar for y+ < 11.6, Newton log-law for y+ > 11.6
    is_laminar = y_plus_lam < 11.6
    u_tau_w = u_tau_lam.clone()

    # Newton for turbulent cells only
    turb_mask = ~is_laminar
    if turb_mask.any():
        u_tau_t = u_tau_lam[turb_mask].clone()
        u_mag_t = u_mag_w[turb_mask]
        for _ in range(8):
            log_yp = torch.log(y_val * u_tau_t / nu)
            f_val = u_tau_t * (log_yp / KAPPA + B_CONST) - u_mag_t
            f_prime = (log_yp / KAPPA + B_CONST) + 1.0 / KAPPA
            u_tau_t = u_tau_t - f_val / f_prime.clamp(min=1e-10)
            u_tau_t = torch.clamp(u_tau_t, min=1e-10)
        u_tau_w[turb_mask] = u_tau_t

    tau_w = u_tau_w**2
    # Laminar: sr=0 (full no-slip). Turbulent: sr=1 - u_tau^2 * y / (nu * u)
    sr_w = torch.zeros_like(u_mag_w)
    if turb_mask.any():
        sr_w[turb_mask] = torch.clamp(1.0 - tau_w[turb_mask] * y_val / (nu * u_mag_w[turb_mask].clamp(min=1e-10)), 0.0, 1.0)

    # Apply slip ratio: u_slip = (1-slip_ratio)*u_tan → NO. The slip ratio represents
    # the FRACTION of the wall-normal velocity that slips. Effective wall velocity
    # = u * (1 - slip_ratio) for tangential components.
    # Actually: the target WALL velocity (what the fluid sees) is u_wall = u_tan * sr
    # Then moving-wall bounce-back imposes u_wall at the wall.
    ux_full = torch.zeros_like(ux_s)
    uy_full = torch.zeros_like(uy_s)
    uz_full = torch.zeros_like(uz_s)
    ux_full[wall_adjacent] = ux_s[wall_adjacent] * sr_w
    uy_full[wall_adjacent] = uy_s[wall_adjacent] * sr_w
    uz_full[wall_adjacent] = uz_s[wall_adjacent] * sr_w

    return ux_full, uy_full, uz_full


def apply_wall_model_bounce_back(
    f: torch.Tensor, mask: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, nu: float,
) -> torch.Tensor:
    ux_s, uy_s, uz_s = compute_wall_slip_velocity(ux, uy, uz, mask, nu)
    return moving_wall_bounce_back_3d(f, mask, ux_s, uy_s, uz_s)


__all__ = ["compute_wall_slip_velocity", "apply_wall_model_bounce_back"]

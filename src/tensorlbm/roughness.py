"""Equivalent sand-grain wall roughness boundary condition for LBM.

In commercial LBM solvers (PowerFlow, XFlow) rough-wall boundary conditions
are modelled by modifying the near-wall slip velocity using an equivalent
sand-grain roughness height *ks*.  The modification follows the shifted
log-law (Nikuradse, 1933; Colebrook, 1939):

.. math::

    u^+ = \\frac{1}{\\kappa} \\ln\\!\\left(\\frac{y^+}{k_s^+}\\right) + B_r

where :math:`B_r` is the roughness-corrected additive constant and
:math:`k_s^+ = k_s u_\\tau / \\nu` is the dimensionless roughness height.

Three roughness regimes are automatically selected (Adams & Johnston, 1984):

* **Hydraulically smooth** (:math:`k_s^+ < 2.25`): no roughness correction.
* **Transitional** (:math:`2.25 \\leq k_s^+ \\leq 90`): blended correction.
* **Fully rough** (:math:`k_s^+ > 90`): :math:`B_r` from Colebrook formula.

The module extends the existing :mod:`~tensorlbm.wall_model` approach and
adds a roughness-aware slip velocity computation that can be passed directly
to :func:`tensorlbm.propeller_benchmark.moving_wall_bounce_back_3d`.

References
----------
* Nikuradse J. (1933) *Laws of Flow in Rough Pipes.* NACA TM 1292.
* Colebrook C.F. (1939) J. Inst. Civil Eng. 11 133.
* Knopp T. *et al.* (2006) J. Comput. Phys. 220 179.
"""
from __future__ import annotations

import torch

KAPPA: float = 0.41   # von Kármán constant
B_SMOOTH: float = 5.0  # smooth-wall additive constant


# ---------------------------------------------------------------------------
# Core roughness correction
# ---------------------------------------------------------------------------

def roughness_b_correction(ks_plus: torch.Tensor) -> torch.Tensor:
    """Compute additive constant correction ΔB due to wall roughness.

    Returns the effective B constant for the log-law after roughness
    correction according to the three-regime model.

    Args:
        ks_plus: Dimensionless roughness height ``ks * u_tau / nu``.

    Returns:
        Roughness correction ``ΔB`` (positive → log-law shifted downward).
        The effective B is ``B_smooth − ΔB``.
    """
    # Smooth regime: no correction
    smooth = torch.zeros_like(ks_plus)

    # Fully rough regime: Colebrook formula
    full_rough = (1.0 / KAPPA) * torch.log(ks_plus / 0.033)

    # Transitional blend (sin² weighting, Adams & Johnston 1984)
    blend_arg = torch.log10(ks_plus / 2.25) / torch.log10(
        torch.tensor(90.0 / 2.25, device=ks_plus.device)
    )
    blend_w = torch.sin(0.5 * torch.pi * torch.clamp(blend_arg, 0.0, 1.0))
    transitional = blend_w**2 * full_rough

    delta_b = torch.where(
        ks_plus < 2.25,
        smooth,
        torch.where(ks_plus > 90.0, full_rough, transitional),
    )
    return delta_b


def compute_rough_wall_slip_velocity(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    mask: torch.Tensor,
    nu: float,
    ks: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rough-wall slip velocity using shifted log-law.

    Extends :func:`tensorlbm.wall_model.compute_wall_slip_velocity` by
    applying a roughness correction to the log-law B constant based on
    the equivalent sand-grain roughness height *ks*.

    Args:
        ux: x-velocity field, shape ``(nz, ny, nx)``.
        uy: y-velocity field.
        uz: z-velocity field.
        mask: Boolean solid-cell mask, same shape as *ux*.
        nu: Kinematic viscosity (lattice units).
        ks: Equivalent sand-grain roughness height (lattice units).

    Returns:
        Tuple ``(ux_slip, uy_slip, uz_slip)`` — slip velocity components
        at wall-adjacent cells, zero elsewhere.
    """
    device = ux.device
    ux_s = torch.zeros_like(ux)
    uy_s = torch.zeros_like(uy)
    uz_s = torch.zeros_like(uz)

    # Identify fluid cells adjacent to solid
    fluid_nbr = torch.zeros_like(mask)
    shifts = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    for dk, dj, di in shifts:
        s0 = slice(1, None) if dk == 1 else (slice(None, -1) if dk == -1 else slice(None))
        s1 = slice(1, None) if dj == 1 else (slice(None, -1) if dj == -1 else slice(None))
        s2 = slice(1, None) if di == 1 else (slice(None, -1) if di == -1 else slice(None))
        t0 = slice(None, -1) if dk == 1 else (slice(1, None) if dk == -1 else slice(None))
        t1 = slice(None, -1) if dj == 1 else (slice(1, None) if dj == -1 else slice(None))
        t2 = slice(None, -1) if di == 1 else (slice(1, None) if di == -1 else slice(None))
        fluid_nbr[t0, t1, t2] = fluid_nbr[t0, t1, t2] | (~mask[s0, s1, s2] & mask[t0, t1, t2])

    wall_adj = mask & fluid_nbr
    if not wall_adj.any():
        return ux_s, uy_s, uz_s

    # Pull velocity from nearest fluid neighbour
    for dk, dj, di in shifts:
        s0 = slice(1, None) if dk == 1 else (slice(None, -1) if dk == -1 else slice(None))
        s1 = slice(1, None) if dj == 1 else (slice(None, -1) if dj == -1 else slice(None))
        s2 = slice(1, None) if di == 1 else (slice(None, -1) if di == -1 else slice(None))
        t0 = slice(None, -1) if dk == 1 else (slice(1, None) if dk == -1 else slice(None))
        t1 = slice(None, -1) if dj == 1 else (slice(1, None) if dj == -1 else slice(None))
        t2 = slice(None, -1) if di == 1 else (slice(1, None) if di == -1 else slice(None))
        from_fluid = mask[t0, t1, t2] & ~mask[s0, s1, s2]
        if not from_fluid.any():
            continue
        ux_s[t0, t1, t2] = torch.where(from_fluid, ux[s0, s1, s2], ux_s[t0, t1, t2])
        uy_s[t0, t1, t2] = torch.where(from_fluid, uy[s0, s1, s2], uy_s[t0, t1, t2])
        uz_s[t0, t1, t2] = torch.where(from_fluid, uz[s0, s1, s2], uz_s[t0, t1, t2])

    u_mag = torch.sqrt(ux_s**2 + uy_s**2 + uz_s**2)
    u_mag_w = u_mag[wall_adj].clamp(min=1e-12)
    y_val = 1.5  # distance to wall in lattice units (half-cell)

    # Initial u_tau estimate from laminar solution
    u_tau = torch.sqrt(torch.clamp(nu * u_mag_w / y_val, min=1e-12))

    # Newton iteration for log-law (with roughness)
    for _ in range(12):
        ks_plus = ks * u_tau / nu
        delta_b = roughness_b_correction(ks_plus.clamp(min=1e-12))
        b_eff = B_SMOOTH - delta_b
        yplus = y_val * u_tau / nu
        log_yp = torch.log(yplus.clamp(min=1e-12))
        f_val = u_tau * (log_yp / KAPPA + b_eff) - u_mag_w
        f_prime = (log_yp / KAPPA + b_eff) + 1.0 / KAPPA
        u_tau = (u_tau - f_val / f_prime.clamp(min=1e-10)).clamp(min=1e-12)

    # Laminar sub-layer correction: if y+ < 5, use viscous profile
    yplus_final = y_val * u_tau / nu
    is_viscous = yplus_final < 5.0
    u_tau = torch.where(is_viscous,
                         torch.sqrt(torch.clamp(nu * u_mag_w / y_val, min=1e-12)),
                         u_tau)

    tau_w = u_tau**2
    sr_w = torch.clamp(1.0 - tau_w * y_val / (nu * u_mag_w), 0.0, 1.0)

    ux_out = torch.zeros_like(ux_s)
    uy_out = torch.zeros_like(uy_s)
    uz_out = torch.zeros_like(uz_s)
    ux_out[wall_adj] = ux_s[wall_adj] * sr_w
    uy_out[wall_adj] = uy_s[wall_adj] * sr_w
    uz_out[wall_adj] = uz_s[wall_adj] * sr_w

    return ux_out, uy_out, uz_out


def apply_rough_wall_damping_2d(
    f: torch.Tensor,
    nu: float,
    ks: float,
    *,
    reference_u_tau: float | None = None,
    damping_limit: float = 0.75,
) -> tuple[torch.Tensor, float]:
    """Apply a 2-D rough-wall damping model on the first fluid rows.

    The method blends the first fluid row adjacent to the top/bottom walls
    towards a no-slip equilibrium state with a strength derived from the
    equivalent sand-grain roughness correction. This provides an inexpensive
    roughness-aware closure for D2Q9 channel-type runs.

    Returns:
        Tuple of ``(updated_distribution, mean_damping_strength)``.
    """
    from .d2q9 import equilibrium, macroscopic

    if ks <= 0.0 or nu <= 0.0 or f.shape[1] < 3:
        return f, 0.0

    rho, ux, uy = macroscopic(f)
    f_out = f.clone()
    mean_damping: list[float] = []
    rows = sorted({1, f.shape[1] - 2})

    for row in rows:
        speed = torch.sqrt(ux[row, :] ** 2 + uy[row, :] ** 2).clamp(min=1e-12)
        if reference_u_tau is None:
            u_tau = torch.sqrt(torch.clamp(nu * speed / 0.5, min=1e-12))
        else:
            u_tau = torch.full_like(speed, reference_u_tau)
        ks_plus = ks * u_tau / max(nu, 1e-12)
        delta_b = roughness_b_correction(ks_plus.clamp(min=1e-12))
        damping = (delta_b / (B_SMOOTH + delta_b + 1e-12)).clamp(
            min=0.0,
            max=damping_limit,
        )
        rho_row = rho[row : row + 1, :]
        zeros = torch.zeros_like(rho_row)
        f_target = equilibrium(rho_row, zeros, zeros)
        beta = damping.view(1, 1, -1)
        f_out[:, row : row + 1, :] = (
            (1.0 - beta) * f_out[:, row : row + 1, :]
            + beta * f_target
        )
        mean_damping.append(float(damping.mean().item()))

    return f_out, sum(mean_damping) / max(len(mean_damping), 1)


def apply_rough_wall_bounce_back(
    f: torch.Tensor,
    mask: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    nu: float,
    ks: float,
) -> torch.Tensor:
    """Apply rough-wall moving-bounce-back to distribution ``f``.

    Convenience wrapper combining :func:`compute_rough_wall_slip_velocity`
    with the moving-wall bounce-back from the propeller benchmark module.

    Args:
        f: Distribution tensor, shape ``(19, nz, ny, nx)``.
        mask: Solid-cell mask, shape ``(nz, ny, nx)``.
        ux, uy, uz: Velocity components, same shape as *mask*.
        nu: Kinematic viscosity (lattice units).
        ks: Sand-grain roughness height (lattice units).

    Returns:
        Updated distribution tensor.
    """
    from .propeller_benchmark import moving_wall_bounce_back_3d

    ux_s, uy_s, uz_s = compute_rough_wall_slip_velocity(ux, uy, uz, mask, nu, ks)
    return moving_wall_bounce_back_3d(f, mask, ux_s, uy_s, uz_s)


__all__ = [
    "roughness_b_correction",
    "compute_rough_wall_slip_velocity",
    "apply_rough_wall_damping_2d",
    "apply_rough_wall_bounce_back",
]

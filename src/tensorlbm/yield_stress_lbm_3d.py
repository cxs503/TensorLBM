"""Yield-stress fluid LBM kernel (D3Q19) with Papanastasiou / Herschel-Bulkley regularisation.

3D extension of the 2D yield-stress LBM kernel.  The effective viscosity is
computed from the local shear rate using either the Papanastasiou exponential
regularisation or the Herschel-Bulkley law, and the resulting relaxation time
enters the BGK collision.

The key difference from the 2D kernel is the full 3D stress tensor contraction
for shear-rate recovery and the D3Q19 equilibrium/streaming.

Usage
-----
    from tensorlbm.yield_stress_lbm_3d import collide_bingham_bgk_forced_3d

    # Papanastasiou-regularised Bingham
    f = collide_bingham_bgk_forced_3d(
        f, tau_y=0.01, mu_0=0.02, m=100.0,
        ax=1e-6, nu_min=1e-6, nu_max=0.3,
    )
"""

from __future__ import annotations

import math

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .yield_stress_rheology_3d import (
    papanastasiou_viscosity,
    herschel_bulkley_viscosity,
    effective_relaxation_time,
    shear_rate_from_stress_3d,
    neq_stress_3d,
    compute_tau_field_bingham_3d,
    compute_tau_field_hb_3d,
    YieldStressRheologyError,
)

__all__ = [
    "collide_bingham_bgk_3d",
    "collide_bingham_bgk_forced_3d",
    "collide_hb_bgk_3d",
    "collide_hb_bgk_forced_3d",
    "collide_yield_stress_bgk_3d",
    "collide_yield_stress_bgk_forced_3d",
    "guo_force_term_3d",
]


def guo_force_term_3d(
    f: torch.Tensor,
    ax: float | torch.Tensor,
    ay: float | torch.Tensor = 0.0,
    az: float | torch.Tensor = 0.0,
    tau: torch.Tensor | None = None,
    ux_star: torch.Tensor | None = None,
    uy_star: torch.Tensor | None = None,
    uz_star: torch.Tensor | None = None,
) -> torch.Tensor:
    """Guo (2002) discrete body-force term for D3Q19 with per-cell tau.

    For a body force with acceleration ``a`` (force per unit mass, so the
    force density is G = rho * a), the discrete force is

        F_i = w_i * (1 - 1/(2*tau)) * rho * [ 3*(c_i - u*) + 9*(c_i . u*)*c_i ] . a

    with c_s^2 = 1/3 and u* the force-corrected macroscopic velocity.

    This injects exactly rho * a of momentum per step, independent of tau,
    so the recovered steady momentum balance is correct.

    Args:
        f: Distribution tensor (19, nz, ny, nx).
        ax, ay, az: Body force accelerations (force per unit mass).
        tau: Relaxation time field.
        ux_star, uy_star, uz_star: Force-corrected velocities.

    Returns:
        Force term tensor of shape (19, nz, ny, nx).
    """
    from .d3q19 import C, W_EXACT64

    device = f.device
    dtype = f.dtype
    c = C.to(device=device, dtype=dtype)
    w = W_EXACT64.to(device=device, dtype=dtype).view(19, 1, 1, 1)

    rho, ux, uy, uz = macroscopic3d(f)

    if ux_star is None:
        ux_star = ux + 0.5 * ax if isinstance(ax, (int, float)) else ux + 0.5 * ax
    if uy_star is None:
        uy_star = uy + 0.5 * ay if isinstance(ay, (int, float)) else uy + 0.5 * ay
    if uz_star is None:
        uz_star = uz + 0.5 * az if isinstance(az, (int, float)) else uz + 0.5 * az

    # c_i . u_star
    cu = (c[:, 0].view(19, 1, 1, 1) * ux_star
          + c[:, 1].view(19, 1, 1, 1) * uy_star
          + c[:, 2].view(19, 1, 1, 1) * uz_star)

    # Force density = rho * a
    if isinstance(ax, torch.Tensor):
        fx, fy, fz = rho * ax, rho * ay, rho * az
    else:
        fx = torch.full_like(rho, float(ax)) * rho
        fy = torch.full_like(rho, float(ay)) * rho
        fz = torch.full_like(rho, float(az)) * rho

    # c_i . F
    c_dot_f = (c[:, 0].view(19, 1, 1, 1) * fx
               + c[:, 1].view(19, 1, 1, 1) * fy
               + c[:, 2].view(19, 1, 1, 1) * fz)

    # u* . F
    u_dot_f = ux_star * fx + uy_star * fy + uz_star * fz

    # Guo force term: F_i = w_i * [3*(c_i.F) + 9*(c_i.u*)(c_i.F) - 3*(u*.F)]
    force_term = w * (3.0 * c_dot_f + 9.0 * cu * c_dot_f - 3.0 * u_dot_f)

    if tau is not None:
        force_term = force_term * (1.0 - 1.0 / (2.0 * tau.unsqueeze(0)))

    return force_term


def collide_bingham_bgk_3d(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """BGK collision for Papanastasiou-regularised Bingham fluid (3D)."""
    if f.dim() != 4 or f.shape[0] != 19:
        raise ValueError("f must have shape (19, nz, ny, nx)")

    tau = (
        tau_field
        if tau_field is not None
        else compute_tau_field_bingham_3d(
            f, tau_y, mu_0, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )
    )

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau.unsqueeze(0)


def collide_bingham_bgk_forced_3d(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    ax: float | torch.Tensor,
    ay: float | torch.Tensor = 0.0,
    az: float | torch.Tensor = 0.0,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """BGK collision with Guo body force for Bingham fluid (3D)."""
    tau = compute_tau_field_bingham_3d(
        f, tau_y, mu_0, m,
        nu_min=nu_min, nu_max=nu_max,
        tau_min=tau_min, tau_max=tau_max,
        use_neq=use_neq,
    )

    rho, ux, uy, uz = macroscopic3d(f)
    ux_star = ux + 0.5 * ax if isinstance(ax, (int, float)) else ux + 0.5 * ax
    uy_star = uy + 0.5 * ay if isinstance(ay, (int, float)) else uy + 0.5 * ay
    uz_star = uz + 0.5 * az if isinstance(az, (int, float)) else uz + 0.5 * az

    feq = equilibrium3d(rho, ux_star, uy_star, uz_star)
    f_collided = f - (f - feq) / tau.unsqueeze(0)

    force = guo_force_term_3d(f, ax, ay, az, tau=tau,
                              ux_star=ux_star, uy_star=uy_star, uz_star=uz_star)
    return f_collided + force


def collide_hb_bgk_3d(
    f: torch.Tensor,
    tau_y: float,
    K: float,
    n: float,
    m: float,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """BGK collision for Papanastasiou-regularised Herschel-Bulkley fluid (3D)."""
    if f.dim() != 4 or f.shape[0] != 19:
        raise ValueError("f must have shape (19, nz, ny, nx)")

    tau = (
        tau_field
        if tau_field is not None
        else compute_tau_field_hb_3d(
            f, tau_y, K, n, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )
    )

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    return f - (f - feq) / tau.unsqueeze(0)


def collide_hb_bgk_forced_3d(
    f: torch.Tensor,
    tau_y: float,
    K: float,
    n: float,
    m: float,
    ax: float | torch.Tensor,
    ay: float | torch.Tensor = 0.0,
    az: float | torch.Tensor = 0.0,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """BGK collision with Guo body force for Herschel-Bulkley fluid (3D)."""
    tau = compute_tau_field_hb_3d(
        f, tau_y, K, n, m,
        nu_min=nu_min, nu_max=nu_max,
        tau_min=tau_min, tau_max=tau_max,
        use_neq=use_neq,
    )

    rho, ux, uy, uz = macroscopic3d(f)
    ux_star = ux + 0.5 * ax if isinstance(ax, (int, float)) else ux + 0.5 * ax
    uy_star = uy + 0.5 * ay if isinstance(ay, (int, float)) else uy + 0.5 * ay
    uz_star = uz + 0.5 * az if isinstance(az, (int, float)) else uz + 0.5 * az

    feq = equilibrium3d(rho, ux_star, uy_star, uz_star)
    f_collided = f - (f - feq) / tau.unsqueeze(0)

    force = guo_force_term_3d(f, ax, ay, az, tau=tau,
                              ux_star=ux_star, uy_star=uy_star, uz_star=uz_star)
    return f_collided + force


def collide_yield_stress_bgk_3d(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    *,
    hb_K: float | None = None,
    hb_n: float | None = None,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    tau_field: torch.Tensor | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """Unified BGK collision for yield-stress fluids (3D)."""
    if hb_K is not None and hb_n is not None:
        return collide_hb_bgk_3d(
            f, tau_y, hb_K, hb_n, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            tau_field=tau_field, use_neq=use_neq,
        )
    else:
        return collide_bingham_bgk_3d(
            f, tau_y, mu_0, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            tau_field=tau_field, use_neq=use_neq,
        )


def collide_yield_stress_bgk_forced_3d(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    ax: float | torch.Tensor,
    ay: float | torch.Tensor = 0.0,
    az: float | torch.Tensor = 0.0,
    *,
    hb_K: float | None = None,
    hb_n: float | None = None,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """Unified BGK collision with body force for yield-stress fluids (3D)."""
    if hb_K is not None and hb_n is not None:
        return collide_hb_bgk_forced_3d(
            f, tau_y, hb_K, hb_n, m, ax, ay, az,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )
    else:
        return collide_bingham_bgk_forced_3d(
            f, tau_y, mu_0, m, ax, ay, az,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )

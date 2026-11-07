"""Yield-stress fluid LBM kernel (D2Q9) with Papanastasiou / Herschel-Bulkley regularisation.

This module provides the lattice-Boltzmann collision step for yield-stress
fluids.  The effective viscosity is computed from the local shear rate using
either the Papanastasiou exponential regularisation or the Herschel-Bulkley
law, and the resulting relaxation time enters the BGK collision.

The key difference from the power-law kernel is the *yield stress* term:
below a critical stress the fluid behaves as a high-viscosity solid (plug
region), while above it flows with a shear-rate-dependent viscosity.

Usage
-----
    from tensorlbm.yield_stress_lbm import collide_bingham_bgk_forced

    # Papanastasiou-regularised Bingham
    f = collide_bingham_bgk_forced(
        f, tau_y=0.01, mu_0=0.02, m=100.0,
        ax=1e-6, nu_min=1e-6, nu_max=0.3,
    )

    # Herschel-Bulkley
    f = collide_hb_bgk_forced(
        f, tau_y=0.01, K=0.02, n=0.7, m=100.0,
        ax=1e-6, nu_min=1e-6, nu_max=0.3,
    )
"""

from __future__ import annotations

import math

import torch

from .d2q9 import C, W, equilibrium, macroscopic
from .powerlaw import central_difference, strain_rate_shear_rate_2d, guo_force_term
from .yield_stress_rheology import (
    papanastasiou_viscosity,
    herschel_bulkley_viscosity,
    effective_relaxation_time,
    shear_rate_from_stress,
    YieldStressRheologyError,
)

__all__ = [
    "neq_stress_2d",
    "collide_bingham_bgk",
    "collide_bingham_bgk_forced",
    "collide_hb_bgk",
    "collide_hb_bgk_forced",
    "collide_yield_stress_bgk",
    "collide_yield_stress_bgk_forced",
    "compute_tau_field_bingham",
    "compute_tau_field_hb",
]


def neq_stress_2d(
    f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Macroscopic state and non-equilibrium stress tensor.

    Returns (rho, ux, uy, Pi_xx, Pi_xy, Pi_yy) with
    Pi_ab = sum_i c_ia c_ib (f_i - f_i^eq).
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    fneq = f - feq
    c = C.to(device=f.device, dtype=f.dtype)
    cxx = (c[:, 0] * c[:, 0]).view(9, 1, 1)
    cxy = (c[:, 0] * c[:, 1]).view(9, 1, 1)
    cyy = (c[:, 1] * c[:, 1]).view(9, 1, 1)
    pi_xx = (cxx * fneq).sum(0)
    pi_xy = (cxy * fneq).sum(0)
    pi_yy = (cyy * fneq).sum(0)
    return rho, ux, uy, pi_xx, pi_xy, pi_yy


def compute_tau_field_bingham(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """Compute per-cell relaxation time for Papanastasiou-regularised Bingham.

    Args:
        f: Distribution tensor (9, ny, nx).
        tau_y: Yield stress.
        mu_0: Plastic viscosity.
        m: Papanastasiou regularisation parameter.
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        use_neq: If True, use consistent shear rate from non-equilibrium stress;
            if False, use finite-difference shear rate.

    Returns:
        Relaxation time field (ny, nx).
    """
    if use_neq:
        rho, _, _, pi_xx, pi_xy, pi_yy = neq_stress_2d(f)
        tau_guess = torch.full_like(pi_xx, 0.5 + 3.0 * mu_0)
        gamma = shear_rate_from_stress(pi_xx, pi_xy, pi_yy, rho, tau_guess)
    else:
        rho, ux, uy = macroscopic(f)
        gamma = strain_rate_shear_rate_2d(ux, uy)

    mu_eff = papanastasiou_viscosity(gamma, tau_y, mu_0, m, mu_plast=None)
    nu_eff = mu_eff  # rho = 1 in lattice units
    nu_eff = torch.clamp(nu_eff, min=nu_min, max=nu_max)
    tau = effective_relaxation_time(nu_eff, rho=1.0, tau_min=tau_min, tau_max=tau_max)
    return tau


def compute_tau_field_hb(
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
    use_neq: bool = True,
) -> torch.Tensor:
    """Compute per-cell relaxation time for Papanastasiou-regularised Herschel-Bulkley.

    Args:
        f: Distribution tensor (9, ny, nx).
        tau_y: Yield stress.
        K: Consistency index.
        n: Flow index.
        m: Papanastasiou regularisation parameter.
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        use_neq: If True, use consistent shear rate from non-equilibrium stress.

    Returns:
        Relaxation time field (ny, nx).
    """
    if use_neq:
        rho, _, _, pi_xx, pi_xy, pi_yy = neq_stress_2d(f)
        tau_guess = torch.full_like(pi_xx, 0.5 + 3.0 * K)
        gamma = shear_rate_from_stress(pi_xx, pi_xy, pi_yy, rho, tau_guess)
    else:
        rho, ux, uy = macroscopic(f)
        gamma = strain_rate_shear_rate_2d(ux, uy)

    mu_eff = herschel_bulkley_viscosity(gamma, tau_y, K, n, m, mu_plast=None)
    nu_eff = mu_eff
    nu_eff = torch.clamp(nu_eff, min=nu_min, max=nu_max)
    tau = effective_relaxation_time(nu_eff, rho=1.0, tau_min=tau_min, tau_max=tau_max)
    return tau


def collide_bingham_bgk(
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
    """BGK collision for Papanastasiou-regularised Bingham fluid.

    Args:
        f: Distribution tensor (9, ny, nx).
        tau_y: Yield stress.
        mu_0: Plastic viscosity.
        m: Papanastasiou regularisation parameter.
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        tau_field: Optional precomputed relaxation time field.
        use_neq: Use consistent shear rate from non-equilibrium stress.

    Returns:
        Updated distribution tensor.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError("f must have shape (9, ny, nx)")

    tau = (
        tau_field
        if tau_field is not None
        else compute_tau_field_bingham(
            f, tau_y, mu_0, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )
    )

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau.unsqueeze(0)


def collide_bingham_bgk_forced(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    ax: float | torch.Tensor,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """BGK collision with Guo body force for Bingham fluid.

    Args:
        f: Distribution tensor (9, ny, nx).
        tau_y: Yield stress.
        mu_0: Plastic viscosity.
        m: Papanastasiou regularisation parameter.
        ax: Body force acceleration in +x.
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        use_neq: Use consistent shear rate.

    Returns:
        Updated distribution tensor.
    """
    tau = compute_tau_field_bingham(
        f, tau_y, mu_0, m,
        nu_min=nu_min, nu_max=nu_max,
        tau_min=tau_min, tau_max=tau_max,
        use_neq=use_neq,
    )

    rho, ux, uy = macroscopic(f)
    ux_star = ux + 0.5 * ax  # force-corrected velocity
    feq = equilibrium(rho, ux_star, uy)
    f_collided = f - (f - feq) / tau.unsqueeze(0)

    # Guo force term
    force = guo_force_term(f, ax, tau, ux_star=ux_star)
    return f_collided + force


def collide_hb_bgk(
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
    """BGK collision for Papanastasiou-regularised Herschel-Bulkley fluid."""
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError("f must have shape (9, ny, nx)")

    tau = (
        tau_field
        if tau_field is not None
        else compute_tau_field_hb(
            f, tau_y, K, n, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )
    )

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau.unsqueeze(0)


def collide_hb_bgk_forced(
    f: torch.Tensor,
    tau_y: float,
    K: float,
    n: float,
    m: float,
    ax: float | torch.Tensor,
    *,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """BGK collision with Guo body force for Herschel-Bulkley fluid."""
    tau = compute_tau_field_hb(
        f, tau_y, K, n, m,
        nu_min=nu_min, nu_max=nu_max,
        tau_min=tau_min, tau_max=tau_max,
        use_neq=use_neq,
    )

    rho, ux, uy = macroscopic(f)
    ux_star = ux + 0.5 * ax
    feq = equilibrium(rho, ux_star, uy)
    f_collided = f - (f - feq) / tau.unsqueeze(0)

    force = guo_force_term(f, ax, tau, ux_star=ux_star)
    return f_collided + force


def collide_yield_stress_bgk(
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
    """Unified BGK collision for yield-stress fluids.

    If ``hb_K`` and ``hb_n`` are given, uses Herschel-Bulkley; otherwise
    uses Bingham (n=1, K=mu_0).

    Args:
        f: Distribution tensor (9, ny, nx).
        tau_y: Yield stress.
        mu_0: Plastic viscosity (Bingham) or reference viscosity (HB).
        m: Papanastasiou regularisation parameter.
        hb_K: Herschel-Bulkley consistency index (None → Bingham).
        hb_n: Herschel-Bulkley flow index (None → Bingham).
        nu_min: Minimum kinematic viscosity.
        nu_max: Maximum kinematic viscosity.
        tau_min: Minimum relaxation time.
        tau_max: Optional maximum relaxation time.
        tau_field: Optional precomputed relaxation time field.
        use_neq: Use consistent shear rate.

    Returns:
        Updated distribution tensor.
    """
    if hb_K is not None and hb_n is not None:
        return collide_hb_bgk(
            f, tau_y, hb_K, hb_n, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            tau_field=tau_field, use_neq=use_neq,
        )
    else:
        return collide_bingham_bgk(
            f, tau_y, mu_0, m,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            tau_field=tau_field, use_neq=use_neq,
        )


def collide_yield_stress_bgk_forced(
    f: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    ax: float | torch.Tensor,
    *,
    hb_K: float | None = None,
    hb_n: float | None = None,
    nu_min: float = 1e-9,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
    use_neq: bool = True,
) -> torch.Tensor:
    """Unified BGK collision with body force for yield-stress fluids."""
    if hb_K is not None and hb_n is not None:
        return collide_hb_bgk_forced(
            f, tau_y, hb_K, hb_n, m, ax,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )
    else:
        return collide_bingham_bgk_forced(
            f, tau_y, mu_0, m, ax,
            nu_min=nu_min, nu_max=nu_max,
            tau_min=tau_min, tau_max=tau_max,
            use_neq=use_neq,
        )

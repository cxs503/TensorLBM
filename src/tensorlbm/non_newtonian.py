from __future__ import annotations

import torch

from .d2q9 import equilibrium, macroscopic


def strain_rate_magnitude_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Estimate shear-rate magnitude from 2D velocity gradients on a unit grid."""
    du_dy, du_dx = torch.gradient(ux, dim=(0, 1))
    dv_dy, dv_dx = torch.gradient(uy, dim=(0, 1))

    s_xx = du_dx
    s_yy = dv_dy
    s_xy = 0.5 * (du_dy + dv_dx)
    second_invariant = s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy
    return torch.sqrt(torch.clamp(2.0 * second_invariant, min=0.0))


def apparent_viscosity_power_law(
    shear_rate: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    nu_min: float | None = None,
    nu_max: float | None = None,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Compute kinematic viscosity using a power-law rheology ν = K * γ^(n-1)."""
    if consistency_index <= 0.0:
        msg = "consistency_index must be > 0"
        raise ValueError(msg)
    if flow_index <= 0.0:
        msg = "flow_index must be > 0"
        raise ValueError(msg)
    if shear_rate_floor <= 0.0:
        msg = "shear_rate_floor must be > 0"
        raise ValueError(msg)

    shear_rate_safe = torch.clamp(shear_rate, min=shear_rate_floor)
    nu = consistency_index * torch.pow(shear_rate_safe, flow_index - 1.0)

    if nu_min is not None or nu_max is not None:
        min_v = nu_min if nu_min is not None else -torch.inf
        max_v = nu_max if nu_max is not None else torch.inf
        nu = torch.clamp(nu, min=min_v, max=max_v)
    return nu


def collide_power_law_bgk(
    f: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    nu_min: float = 1e-5,
    nu_max: float = 0.3,
    tau_min: float = 0.501,
    tau_max: float | None = None,
) -> torch.Tensor:
    """Variable-viscosity BGK collision for generalized Newtonian power-law fluids."""
    if f.dim() != 3 or f.shape[0] != 9:
        msg = "f must have shape (9, ny, nx)"
        raise ValueError(msg)
    if tau_min <= 0.5:
        msg = "tau_min must be > 0.5"
        raise ValueError(msg)
    if tau_max is not None and tau_max < tau_min:
        msg = "tau_max must be >= tau_min"
        raise ValueError(msg)

    rho, ux, uy = macroscopic(f)
    shear_rate = strain_rate_magnitude_2d(ux, uy)
    nu = apparent_viscosity_power_law(
        shear_rate,
        consistency_index=consistency_index,
        flow_index=flow_index,
        nu_min=nu_min,
        nu_max=nu_max,
    )

    tau = 3.0 * nu + 0.5
    tau = torch.clamp(tau, min=tau_min, max=tau_max if tau_max is not None else torch.inf)

    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau.unsqueeze(0)


__all__ = [
    "strain_rate_magnitude_2d",
    "apparent_viscosity_power_law",
    "collide_power_law_bgk",
]

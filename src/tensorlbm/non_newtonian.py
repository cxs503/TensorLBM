"""Non-Newtonian rheology models for LBM: Power-law, Carreau-Yasuda, Cross.

Provides apparent-viscosity closures for generalized Newtonian fluids and
a Poiseuille-flow analytical anchor for verification.

Models
------
1. **Power-law**: ν = K γ̇^(n-1)
2. **Carreau-Yasuda**: ν = ν_∞ + (ν_0 - ν_∞) [1 + (λ γ̇)^a]^((n-1)/a)
3. **Cross**: ν = ν_0 / [1 + (ν_0/K) γ̇^(1-n)]  (simplified Cross form)

All three reduce to Newtonian (ν = const) in appropriate limits and capture
shear-thinning (n<1) or shear-thickening (n>1) behaviour.

Analytical anchor: Poiseuille flow
-----------------------------------
For a power-law fluid in a 2D channel of half-height H with pressure gradient G:

    u(y) = (n/(n+1)) (G/K)^(1/n) [H^((n+1)/n) - |y|^((n+1)/n)]

For Carreau-Yasuda and Cross, no closed-form exists in general, but the
zero-shear-rate and infinite-shear-rate limits give Newtonian Poiseuille:

    u(y) = G/(2ν) (H² - y²)

which provides a verification anchor at the asymptotes.

References
----------
Bird, R. B., Armstrong, R. C. & Hassager, O. (1987). *Dynamics of Polymeric
    Liquids*, Vol. 1, Wiley.
Chhabra, R. P. & Richardson, J. F. (2008). *Non-Newtonian Flow and Applied
    Rheology*, 2nd ed., Butterworth-Heinemann.
Yasuda, K. (1979). PhD thesis, Kyoto Univ. (Carreau-Yasuda model origin).
Cross, M. M. (1965). *J. Colloid Sci.* 20, 417-437.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from .d2q9 import equilibrium, macroscopic


# ---------------------------------------------------------------------------
# Strain-rate magnitude
# ---------------------------------------------------------------------------

def strain_rate_magnitude_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Estimate shear-rate magnitude from 2D velocity gradients on a unit grid.

    Returns γ̇ = sqrt(2 S:S) where S is the strain-rate tensor.
    """
    du_dy, du_dx = torch.gradient(ux, dim=(0, 1))
    dv_dy, dv_dx = torch.gradient(uy, dim=(0, 1))

    s_xx = du_dx
    s_yy = dv_dy
    s_xy = 0.5 * (du_dy + dv_dx)
    second_invariant = s_xx * s_xx + s_yy * s_yy + 2.0 * s_xy * s_xy
    return torch.sqrt(torch.clamp(2.0 * second_invariant, min=0.0))


# ---------------------------------------------------------------------------
# Apparent viscosity models
# ---------------------------------------------------------------------------

def apparent_viscosity_power_law(
    shear_rate: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    nu_min: float | None = None,
    nu_max: float | None = None,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Power-law apparent viscosity: ν = K γ̇^(n-1).

    Args:
        shear_rate: Shear-rate magnitude γ̇.
        consistency_index: K (Pa·s^n or lattice equiv).
        flow_index: n (dimensionless; n<1 shear-thinning, n>1 thickening).
        nu_min: Optional lower clamp.
        nu_max: Optional upper clamp.
        shear_rate_floor: Minimum γ̇ to avoid singularity at γ̇=0.

    Returns:
        Apparent viscosity ν, same shape as shear_rate.
    """
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


def apparent_viscosity_carreau_yasuda(
    shear_rate: torch.Tensor,
    nu_0: float,
    nu_inf: float,
    relaxation_time: float,
    flow_index: float,
    yasuda_exponent: float = 2.0,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Carreau-Yasuda apparent viscosity.

    ν(γ̇) = ν_∞ + (ν_0 - ν_∞) [1 + (λ γ̇)^a]^((n-1)/a)

    where:
        ν_0   = zero-shear-rate viscosity (plateau at γ̇→0)
        ν_∞   = infinite-shear-rate viscosity (plateau at γ̇→∞)
        λ     = relaxation time (characterises onset of shear-thinning)
        n     = flow index (power-law exponent at intermediate γ̇)
        a     = Yasuda exponent (controls breadth of transition)

    Limits:
        γ̇ → 0:  ν → ν_0  (Newtonian plateau)
        γ̇ → ∞:  ν → ν_∞  (Newtonian plateau) or ν_∞ + (ν_0-ν_∞)(λγ̇)^(n-1) if ν_∞=0
        Intermediate: ν ~ K γ̇^(n-1) with K = (ν_0 - ν_∞) λ^(n-1)

    Args:
        shear_rate: Shear-rate magnitude γ̇.
        nu_0: Zero-shear viscosity (must be > nu_inf).
        nu_inf: Infinite-shear viscosity (≥ 0).
        relaxation_time: λ > 0.
        flow_index: n (typically 0 < n ≤ 1 for shear-thinning).
        yasuda_exponent: a > 0 (default 2.0 = Carreau model).
        shear_rate_floor: Minimum γ̇ to avoid issues.

    Returns:
        Apparent viscosity ν.
    """
    if nu_0 <= nu_inf:
        msg = f"nu_0 ({nu_0}) must exceed nu_inf ({nu_inf})"
        raise ValueError(msg)
    if relaxation_time <= 0.0:
        msg = "relaxation_time must be > 0"
        raise ValueError(msg)
    if flow_index <= 0.0:
        msg = "flow_index must be > 0"
        raise ValueError(msg)
    if yasuda_exponent <= 0.0:
        msg = "yasuda_exponent must be > 0"
        raise ValueError(msg)

    gamma_dot = torch.clamp(shear_rate, min=shear_rate_floor)
    lambda_gamma = relaxation_time * gamma_dot
    bracket = 1.0 + torch.pow(lambda_gamma, yasuda_exponent)
    exponent = (flow_index - 1.0) / yasuda_exponent
    nu = nu_inf + (nu_0 - nu_inf) * torch.pow(bracket, exponent)
    return nu


def apparent_viscosity_cross(
    shear_rate: torch.Tensor,
    nu_0: float,
    consistency_index: float,
    flow_index: float,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Cross apparent viscosity (simplified 3-parameter form).

    ν(γ̇) = ν_0 / [1 + (ν_0 / K) γ̇^(1-n)]

    This is the Cross (1965) model in the form commonly used for polymer
    melts and suspensions. At low γ̇ it gives ν → ν_0 (Newtonian plateau);
    at high γ̇ it gives ν ~ K γ̇^(n-1) (power-law region).

    Args:
        shear_rate: Shear-rate magnitude γ̇.
        nu_0: Zero-shear viscosity.
        consistency_index: K (sets the onset of shear-thinning).
        flow_index: n (power-law exponent in the thinning region).
        shear_rate_floor: Minimum γ̇.

    Returns:
        Apparent viscosity ν.
    """
    if nu_0 <= 0.0:
        msg = "nu_0 must be > 0"
        raise ValueError(msg)
    if consistency_index <= 0.0:
        msg = "consistency_index must be > 0"
        raise ValueError(msg)
    if flow_index <= 0.0:
        msg = "flow_index must be > 0"
        raise ValueError(msg)

    gamma_dot = torch.clamp(shear_rate, min=shear_rate_floor)
    denominator = 1.0 + (nu_0 / consistency_index) * torch.pow(gamma_dot, 1.0 - flow_index)
    nu = nu_0 / denominator
    return nu


# ---------------------------------------------------------------------------
# Yield-stress rheology (Bingham / Herschel-Bulkley) for ice slurry
# ---------------------------------------------------------------------------

def apparent_viscosity_bingham(
    shear_rate: torch.Tensor,
    plastic_viscosity: float,
    yield_stress: float,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Bingham apparent viscosity with regularisation.

    ν(γ̇) = μ_p + τ_y / γ̇

    where μ_p is the plastic viscosity and τ_y is the yield stress.
    Below the yield stress the material behaves as a rigid solid; above it,
    it flows with viscosity μ_p + τ_y/γ̇.

    For LBM we regularise with a small γ̇_floor to avoid singularity.

    Args:
        shear_rate: Shear-rate magnitude γ̇.
        plastic_viscosity: μ_p (viscosity above yield).
        yield_stress: τ_y (yield stress threshold).
        shear_rate_floor: Regularisation floor.

    Returns:
        Apparent viscosity ν.
    """
    if plastic_viscosity <= 0.0:
        msg = "plastic_viscosity must be > 0"
        raise ValueError(msg)
    if yield_stress < 0.0:
        msg = "yield_stress must be >= 0"
        raise ValueError(msg)

    gamma_dot = torch.clamp(shear_rate, min=shear_rate_floor)
    nu = plastic_viscosity + yield_stress / gamma_dot
    return nu


def apparent_viscosity_herschel_bulkley(
    shear_rate: torch.Tensor,
    consistency_index: float,
    flow_index: float,
    yield_stress: float,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Herschel-Bulkley apparent viscosity with regularisation.

    ν(γ̇) = K γ̇^(n-1) + τ_y / γ̇

    Generalises Bingham (n=1) to shear-thinning (n<1) or thickening (n>1)
    post-yield behaviour. Relevant for ice slurry where the ice crystal
    network gives a yield stress and the brine phase is shear-thinning.

    Args:
        shear_rate: Shear-rate magnitude γ̇.
        consistency_index: K.
        flow_index: n.
        yield_stress: τ_y.
        shear_rate_floor: Regularisation floor.

    Returns:
        Apparent viscosity ν.
    """
    if consistency_index <= 0.0:
        msg = "consistency_index must be > 0"
        raise ValueError(msg)
    if flow_index <= 0.0:
        msg = "flow_index must be > 0"
        raise ValueError(msg)
    if yield_stress < 0.0:
        msg = "yield_stress must be >= 0"
        raise ValueError(msg)

    gamma_dot = torch.clamp(shear_rate, min=shear_rate_floor)
    nu = consistency_index * torch.pow(gamma_dot, flow_index - 1.0) + yield_stress / gamma_dot
    return nu


# ---------------------------------------------------------------------------
# Poiseuille flow analytical solutions (verification anchors)
# ---------------------------------------------------------------------------

def poiseuille_power_law_profile(
    y: torch.Tensor,
    half_height: float,
    pressure_gradient: float,
    consistency_index: float,
    flow_index: float,
) -> torch.Tensor:
    """Analytical Poiseuille velocity profile for power-law fluid.

    For a 2D channel of half-height H with pressure gradient G = -dp/dx:

        u(y) = (n/(n+1)) (G/K)^(1/n) [H^((n+1)/n) - |y|^((n+1)/n)]

    Valid for n > 0, G > 0 (flow in +x direction).

    Args:
        y: Wall-normal coordinate, y ∈ [-H, H].
        half_height: H.
        pressure_gradient: G = -dp/dx > 0.
        consistency_index: K.
        flow_index: n.

    Returns:
        Velocity u(y), same shape as y.
    """
    if half_height <= 0.0:
        msg = "half_height must be > 0"
        raise ValueError(msg)
    if pressure_gradient <= 0.0:
        msg = "pressure_gradient must be > 0"
        raise ValueError(msg)
    if consistency_index <= 0.0:
        msg = "consistency_index must be > 0"
        raise ValueError(msg)
    if flow_index <= 0.0:
        msg = "flow_index must be > 0"
        raise ValueError(msg)

    n = flow_index
    K = consistency_index
    G = pressure_gradient
    H = half_height

    prefactor = (n / (n + 1.0)) * (G / K) ** (1.0 / n)
    y_abs = torch.abs(y)
    u = prefactor * (H ** ((n + 1.0) / n) - y_abs ** ((n + 1.0) / n))
    return u


def poiseuille_newtonian_profile(
    y: torch.Tensor,
    half_height: float,
    pressure_gradient: float,
    viscosity: float,
) -> torch.Tensor:
    """Analytical Poiseuille velocity profile for Newtonian fluid.

    u(y) = G/(2ν) (H² - y²)

    This is the n=1 limit of the power-law profile and the asymptotic limit
    of Carreau-Yasuda / Cross at γ̇→0 (ν→ν_0) or γ̇→∞ (ν→ν_∞).

    Args:
        y: Wall-normal coordinate.
        half_height: H.
        pressure_gradient: G = -dp/dx > 0.
        viscosity: ν (kinematic).

    Returns:
        Velocity u(y).
    """
    if half_height <= 0.0:
        msg = "half_height must be > 0"
        raise ValueError(msg)
    if pressure_gradient <= 0.0:
        msg = "pressure_gradient must be > 0"
        raise ValueError(msg)
    if viscosity <= 0.0:
        msg = "viscosity must be > 0"
        raise ValueError(msg)

    return pressure_gradient / (2.0 * viscosity) * (half_height**2 - y**2)


def poiseuille_max_velocity_power_law(
    half_height: float,
    pressure_gradient: float,
    consistency_index: float,
    flow_index: float,
) -> float:
    """Maximum (centreline) velocity for power-law Poiseuille flow.

    u_max = (n/(n+1)) (G/K)^(1/n) H^((n+1)/n)
    """
    n = flow_index
    K = consistency_index
    G = pressure_gradient
    H = half_height
    return float((n / (n + 1.0)) * (G / K) ** (1.0 / n) * H ** ((n + 1.0) / n))


def poiseuille_mean_velocity_power_law(
    half_height: float,
    pressure_gradient: float,
    consistency_index: float,
    flow_index: float,
) -> float:
    """Mean (bulk) velocity for power-law Poiseuille flow.

    u_mean = u_max * (n+1)/(2n+1)

    For n=1 (Newtonian): u_mean/u_max = 2/3.
    """
    n = flow_index
    u_max = poiseuille_max_velocity_power_law(
        half_height, pressure_gradient, consistency_index, flow_index
    )
    return float(u_max * (n + 1.0) / (2.0 * n + 1.0))


# ---------------------------------------------------------------------------
# BGK collision operators with variable viscosity
# ---------------------------------------------------------------------------

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


def collide_carreau_yasuda_bgk(
    f: torch.Tensor,
    nu_0: float,
    nu_inf: float,
    relaxation_time: float,
    flow_index: float,
    yasuda_exponent: float = 2.0,
    tau_min: float = 0.501,
    tau_max: float | None = None,
) -> torch.Tensor:
    """Variable-viscosity BGK collision for Carreau-Yasuda fluids."""
    if f.dim() != 3 or f.shape[0] != 9:
        msg = "f must have shape (9, ny, nx)"
        raise ValueError(msg)
    if tau_min <= 0.5:
        msg = "tau_min must be > 0.5"
        raise ValueError(msg)

    rho, ux, uy = macroscopic(f)
    shear_rate = strain_rate_magnitude_2d(ux, uy)
    nu = apparent_viscosity_carreau_yasuda(
        shear_rate,
        nu_0=nu_0,
        nu_inf=nu_inf,
        relaxation_time=relaxation_time,
        flow_index=flow_index,
        yasuda_exponent=yasuda_exponent,
    )

    tau = 3.0 * nu + 0.5
    tau = torch.clamp(tau, min=tau_min, max=tau_max if tau_max is not None else torch.inf)

    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau.unsqueeze(0)


def collide_cross_bgk(
    f: torch.Tensor,
    nu_0: float,
    consistency_index: float,
    flow_index: float,
    tau_min: float = 0.501,
    tau_max: float | None = None,
) -> torch.Tensor:
    """Variable-viscosity BGK collision for Cross-model fluids."""
    if f.dim() != 3 or f.shape[0] != 9:
        msg = "f must have shape (9, ny, nx)"
        raise ValueError(msg)
    if tau_min <= 0.5:
        msg = "tau_min must be > 0.5"
        raise ValueError(msg)

    rho, ux, uy = macroscopic(f)
    shear_rate = strain_rate_magnitude_2d(ux, uy)
    nu = apparent_viscosity_cross(
        shear_rate,
        nu_0=nu_0,
        consistency_index=consistency_index,
        flow_index=flow_index,
    )

    tau = 3.0 * nu + 0.5
    tau = torch.clamp(tau, min=tau_min, max=tau_max if tau_max is not None else torch.inf)

    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau.unsqueeze(0)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Strain rate
    "strain_rate_magnitude_2d",
    # Apparent viscosity models
    "apparent_viscosity_power_law",
    "apparent_viscosity_carreau_yasuda",
    "apparent_viscosity_cross",
    "apparent_viscosity_bingham",
    "apparent_viscosity_herschel_bulkley",
    # Poiseuille analytical solutions
    "poiseuille_power_law_profile",
    "poiseuille_newtonian_profile",
    "poiseuille_max_velocity_power_law",
    "poiseuille_mean_velocity_power_law",
    # BGK collision operators
    "collide_power_law_bgk",
    "collide_carreau_yasuda_bgk",
    "collide_cross_bgk",
]

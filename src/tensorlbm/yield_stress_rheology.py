"""Yield-stress fluid rheology: Bingham & Herschel-Bulkley with Papanastasiou regularisation.

Physical background
-------------------
Yield-stress fluids (Bingham plastics, Herschel-Bulkley fluids) behave as
rigid solids below a critical shear stress :math:`\\tau_y` and flow as
viscous fluids above it.  The constitutive law is

    Bingham:          τ = τ_y + μ₀·γ̇    for |τ| > τ_y
                      γ̇ = 0            for |τ| ≤ τ_y

    Herschel-Bulkley: τ = τ_y + K·γ̇ⁿ   for |τ| > τ_y
                      γ̇ = 0            for |τ| ≤ τ_y

Direct use of the discontinuous law in LBM causes numerical instability.
Two regularisation strategies are implemented:

1. **Papanastasiou exponential regularisation** (1987):

       μ_eff(γ̇) = μ₀ + τ_y·(1 − e^{−m·|γ̇|}) / |γ̇|

   where ``m`` is a regularisation parameter (large m → sharp yield surface).
   For Herschel-Bulkley the generalised form is

       μ_eff(γ̇) = K·|γ̇|^{n−1} + τ_y·(1 − e^{−m·|γ̇|}) / |γ̇|

   The effective kinematic viscosity is ν_eff = μ_eff / ρ.

2. **Bi-viscosity model**: when |τ| < τ_y the fluid is assigned a large
   "plastic" viscosity μ_plast (typically 10³–10⁶ × μ₀), creating a
   quasi-plug region while maintaining mass/momentum conservation.

The regularised viscosity enters the BGK collision through the local
relaxation time τ_eff = 3·ν_eff + ½.

References
----------
* Papanastasiou, T.C. (1987) *J. Rheol.* 31, 385.
* Bingham, E.C. (1922) *Fluidity and Plasticity*, McGraw-Hill.
* Herschel, W.H. & Bulkley, R. (1926) *Proc. Am. Soc. Test. Mater.* 26, 621.
* Mitsoulis, E. (2004) *J. Non-Newtonian Fluid Mech.* 121, 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "YieldStressRheologyError",
    "papanastasiou_viscosity",
    "herschel_bulkley_viscosity",
    "bingham_viscosity",
    "bi_viscosity_model",
    "effective_relaxation_time",
    "shear_rate_from_stress",
    "BinghamParams",
    "HerschelBulkleyParams",
]


class YieldStressRheologyError(ValueError):
    """Raised for unphysical yield-stress parameters."""


@dataclass(frozen=True)
class BinghamParams:
    """Bingham plastic parameters (SI or lattice units, caller's choice)."""
    tau_y: float  # yield stress
    mu_0: float   # plastic viscosity (post-yield)

    def __post_init__(self):
        if self.tau_y < 0:
            raise YieldStressRheologyError(f"tau_y must be >= 0, got {self.tau_y}")
        if self.mu_0 <= 0:
            raise YieldStressRheologyError(f"mu_0 must be > 0, got {self.mu_0}")


@dataclass(frozen=True)
class HerschelBulkleyParams:
    """Herschel-Bulkley parameters."""
    tau_y: float  # yield stress
    K: float      # consistency index
    n: float      # flow index (0 < n <= 1 for shear-thinning)

    def __post_init__(self):
        if self.tau_y < 0:
            raise YieldStressRheologyError(f"tau_y must be >= 0, got {self.tau_y}")
        if self.K <= 0:
            raise YieldStressRheologyError(f"K must be > 0, got {self.K}")
        if self.n <= 0 or self.n > 2:
            raise YieldStressRheologyError(f"n must be in (0, 2], got {self.n}")


def papanastasiou_viscosity(
    shear_rate: torch.Tensor,
    tau_y: float,
    mu_0: float,
    m: float,
    mu_plast: float | None = None,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Papanastasiou regularised effective viscosity for Bingham fluid.

    μ_eff(γ̇) = μ₀ + τ_y·(1 − e^{−m·|γ̇|}) / |γ̇|

    For |γ̇| → 0 the expression tends to μ₀ + τ_y·m (finite), avoiding
    the singularity of the unregularised Bingham law.

    Args:
        shear_rate: Local shear-rate field γ̇ ≥ 0, shape (ny, nx).
        tau_y: Yield stress.
        mu_0: Plastic viscosity (post-yield Newtonian viscosity).
        m: Regularisation parameter (larger → sharper yield surface).
        mu_plast: Optional maximum viscosity cap for the unyielded region.
        shear_rate_floor: Small positive floor for numerical stability.

    Returns:
        Effective viscosity field of shape (ny, nx).
    """
    if tau_y < 0:
        raise YieldStressRheologyError("tau_y must be >= 0")
    if mu_0 <= 0:
        raise YieldStressRheologyError("mu_0 must be > 0")
    if m <= 0:
        raise YieldStressRheologyError("m must be > 0")

    gamma = torch.clamp(shear_rate, min=shear_rate_floor)
    # Papanastasiou: mu_eff = mu_0 + tau_y * (1 - exp(-m*gamma)) / gamma
    exp_term = torch.exp(-m * gamma)
    mu_eff = mu_0 + tau_y * (1.0 - exp_term) / gamma

    if mu_plast is not None:
        mu_eff = torch.clamp(mu_eff, max=mu_plast)

    return mu_eff


def herschel_bulkley_viscosity(
    shear_rate: torch.Tensor,
    tau_y: float,
    K: float,
    n: float,
    m: float,
    mu_plast: float | None = None,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Papanastasiou-regularised Herschel-Bulkley effective viscosity.

    μ_eff(γ̇) = K·|γ̇|^{n−1} + τ_y·(1 − e^{−m·|γ̇|}) / |γ̇|

    Reduces to Bingham when n = 1 and K = μ₀.

    Args:
        shear_rate: Local shear-rate field γ̇ ≥ 0.
        tau_y: Yield stress.
        K: Consistency index.
        n: Flow index.
        m: Papanastasiou regularisation parameter.
        mu_plast: Optional maximum viscosity cap.
        shear_rate_floor: Small positive floor.

    Returns:
        Effective viscosity field.
    """
    if tau_y < 0:
        raise YieldStressRheologyError("tau_y must be >= 0")
    if K <= 0:
        raise YieldStressRheologyError("K must be > 0")
    if n <= 0:
        raise YieldStressRheologyError("n must be > 0")
    if m <= 0:
        raise YieldStressRheologyError("m must be > 0")

    gamma = torch.clamp(shear_rate, min=shear_rate_floor)
    # HB power-law part + Papanastasiou yield-stress regularisation
    mu_eff = K * torch.pow(gamma, n - 1.0) + tau_y * (1.0 - torch.exp(-m * gamma)) / gamma

    if mu_plast is not None:
        mu_eff = torch.clamp(mu_eff, max=mu_plast)

    return mu_eff


def bingham_viscosity(
    shear_rate: torch.Tensor,
    tau_y: float,
    mu_0: float,
    mu_plast: float | None = None,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Unregularised Bingham viscosity with bi-viscosity fallback.

    μ_eff = μ₀ + τ_y / γ̇   for γ̇ > 0
    μ_eff = μ_plast         for γ̇ ≈ 0

    This is the classical discontinuous model; use
    :func:`papanastasiou_viscosity` for smooth regularisation.

    Args:
        shear_rate: Local shear-rate field.
        tau_y: Yield stress.
        mu_0: Plastic viscosity.
        mu_plast: Viscosity in the unyielded (plug) region.
            Defaults to 1e6 * mu_0 if not given.
        shear_rate_floor: Threshold below which the fluid is considered unyielded.

    Returns:
        Effective viscosity field.
    """
    if tau_y < 0:
        raise YieldStressRheologyError("tau_y must be >= 0")
    if mu_0 <= 0:
        raise YieldStressRheologyError("mu_0 must be > 0")

    if mu_plast is None:
        mu_plast = 1e6 * mu_0

    gamma = torch.clamp(shear_rate, min=shear_rate_floor)
    mu_eff = mu_0 + tau_y / gamma
    mu_eff = torch.clamp(mu_eff, max=mu_plast)
    return mu_eff


def bi_viscosity_model(
    shear_rate: torch.Tensor,
    tau_y: float,
    mu_0: float,
    mu_ratio: float = 1e4,
    shear_rate_floor: float = 1e-12,
) -> torch.Tensor:
    """Bi-viscosity model: two viscosities separated by a yield-rate threshold.

    The yield-rate threshold is γ̇_y = τ_y / μ₀.  Below this rate the fluid
    uses μ_plast = mu_ratio * μ₀; above it uses the Bingham law.

    Args:
        shear_rate: Local shear-rate field.
        tau_y: Yield stress.
        mu_0: Plastic viscosity.
        mu_ratio: Ratio of plastic to post-yield viscosity (default 1e4).
        shear_rate_floor: Numerical floor.

    Returns:
        Effective viscosity field.
    """
    if tau_y < 0:
        raise YieldStressRheologyError("tau_y must be >= 0")
    if mu_0 <= 0:
        raise YieldStressRheologyError("mu_0 must be > 0")
    if mu_ratio <= 1:
        raise YieldStressRheologyError("mu_ratio must be > 1")

    mu_plast = mu_ratio * mu_0
    gamma_y = tau_y / mu_0  # yield shear rate
    gamma = torch.clamp(shear_rate, min=shear_rate_floor)

    # Below yield rate: use mu_plast; above: use Bingham
    mu_eff = torch.where(
        gamma < gamma_y,
        torch.full_like(gamma, mu_plast),
        mu_0 + tau_y / gamma,
    )
    return mu_eff


def effective_relaxation_time(
    mu_eff: torch.Tensor,
    rho: torch.Tensor | float = 1.0,
    tau_min: float = 0.501,
    tau_max: float | None = None,
) -> torch.Tensor:
    """Convert effective viscosity to BGK relaxation time.

    τ_eff = 3·ν_eff + ½ = 3·μ_eff/ρ + ½

    Args:
        mu_eff: Effective viscosity field.
        rho: Density (scalar or field).
        tau_min: Minimum relaxation time (> 0.5 for stability).
        tau_max: Optional maximum relaxation time.

    Returns:
        Relaxation time field.
    """
    if tau_min <= 0.5:
        raise YieldStressRheologyError("tau_min must be > 0.5")

    if isinstance(rho, (int, float)):
        nu_eff = mu_eff / float(rho)
    else:
        rho_safe = torch.clamp(rho, min=1e-12)
        nu_eff = mu_eff / rho_safe

    tau = 3.0 * nu_eff + 0.5
    tau = torch.clamp(tau, min=tau_min)
    if tau_max is not None:
        tau = torch.clamp(tau, max=tau_max)
    return tau


def shear_rate_from_stress(
    pi_xx: torch.Tensor,
    pi_xy: torch.Tensor,
    pi_yy: torch.Tensor,
    rho: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Recover shear rate from non-equilibrium stress (consistent with LBM).

    From Pi_ab^neq = −2·c_s²·τ·ρ·S_ab with c_s² = 1/3:

        γ̇ = sqrt(2·S:S) = 3·|Pi^neq| / (sqrt(2)·ρ·τ)

    This is the *consistent* shear rate that the LBM actually transports,
    avoiding finite-difference stencil errors.

    Args:
        pi_xx, pi_xy, pi_yy: Non-equilibrium stress components.
        rho: Density field.
        tau: Relaxation time field.

    Returns:
        Shear rate field.
    """
    pi_norm = torch.sqrt(
        torch.clamp(pi_xx * pi_xx + 2.0 * pi_xy * pi_xy + pi_yy * pi_yy, min=0.0)
    )
    rho_safe = torch.clamp(rho, min=1e-12)
    tau_safe = torch.clamp(tau, min=1e-12)
    gamma = 3.0 * pi_norm / (math.sqrt(2.0) * rho_safe * tau_safe)
    return gamma

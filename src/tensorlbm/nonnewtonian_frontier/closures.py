"""Generalised-Newtonian viscosity closures for the non-Newtonian verification surface.

Every closure is exposed as a frozen :class:`Closure` object that carries

* ``nu_of_gamma``  -- apparent viscosity ``ν(γ̇)`` (torch vectorised),
* ``tau_of_gamma`` -- ``τ(γ̇) = γ̇ · ν(γ̇)`` (monotone increasing),
* ``tau_y``        -- yield stress (0 for the yield-free closures),

so that the *identical* constitutive law is used by the lattice-Boltzmann
runner and by the one-dimensional reference solver in
:mod:`tensorlbm.nonnewtonian_frontier.reference`.  Because both codes consume
the same object, a residual error cannot hide inside a mismatched closure: it
is either discretisation (LBM) or nothing.

Models
------
``newtonian``        ν = μ
``power_law``        ν = K γ̇^(n-1)                     (Ostwald–de Waele)
``carreau_yasuda``   ν = ν∞ + (ν0-ν∞)[1+(λγ̇)^a]^((n-1)/a)
``cross``            ν = ν∞ + (ν0-ν∞)/(1+(λγ̇)^m)
``herschel_bulkley`` ν = τ_y(1-e^{-m γ̇})/γ̇ + K γ̇^(n-1)   (Papanastasiou regularisation)

Viscosities are lattice units (``lu²/ts``); a closure is defined by its
lattice-unit parameters, which is what makes the verification self-consistent
(the constitutive law is not invariant under an arbitrary rescaling of K unless
``K`` is expressed in the same units as ``ν``).

References
----------
Bird, Armstrong & Hassager (1987), *Dynamics of Polymeric Liquids* Vol. 1.
Papanastasiou, T.C. (1987) *J. Rheol.* 31, 385 (regularised yield stress).
Yasuda, K. (1979) PhD thesis, Kyoto University (Carreau–Yasuda exponent ``a``).
Cross, M.M. (1965) *J. Colloid Sci.* 20, 417.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

__all__ = [
    "Closure",
    "newtonian",
    "power_law",
    "carreau_yasuda",
    "cross",
    "herschel_bulkley",
    "GAMMA_FLOOR",
]

#: Smallest shear rate used inside the closures (avoids 0^negative powers).
GAMMA_FLOOR = 1.0e-14


def _floor(gamma: torch.Tensor) -> torch.Tensor:
    return torch.clamp(gamma, min=GAMMA_FLOOR)


@dataclass(frozen=True)
class Closure:
    """A generalised-Newtonian closure with lattice-unit parameters."""

    name: str
    nu_of_gamma: Callable[[torch.Tensor], torch.Tensor]
    params: dict = field(default_factory=dict)
    tau_y: float = 0.0
    nu_min: float = 0.0
    nu_max: float = math.inf
    description: str = ""

    # -- constitutive law -------------------------------------------------
    def tau_of_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        """Shear stress ``τ = ν(γ̇)·γ̇`` (monotone increasing in ``γ̇``)."""
        g = _floor(gamma)
        return g * self.nu_of_gamma(g)

    def relative_viscosity_error(self, gamma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """|ν(γ̇) - ν_ref(γ̇)| / ν_ref(γ̇), elementwise."""
        return (self.nu_of_gamma(gamma) - reference).abs() / reference.abs()


def newtonian(nu: float, nu_min: float = 0.0, nu_max: float = math.inf) -> Closure:
    """Constant-viscosity closure (``ν = μ``)."""

    def _nu(gamma: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.full_like(gamma, nu), min=nu_min, max=nu_max)

    return Closure(
        name="newtonian",
        nu_of_gamma=_nu,
        params={"nu": nu},
        nu_min=nu_min,
        nu_max=nu_max,
        description="ν = const",
    )


def power_law(
    consistency_index: float,
    flow_index: float,
    nu_min: float = 1.0e-6,
    nu_max: float = math.inf,
) -> Closure:
    """Ostwald–de Waele power law ``ν = K γ̇^(n-1)``."""
    if consistency_index <= 0.0:
        raise ValueError("consistency_index must be > 0")
    if flow_index <= 0.0:
        raise ValueError("flow_index must be > 0")
    K, n = float(consistency_index), float(flow_index)

    def _nu(gamma: torch.Tensor) -> torch.Tensor:
        nu = K * torch.pow(_floor(gamma), n - 1.0)
        return torch.clamp(nu, min=nu_min, max=nu_max)

    tag = "shear-thinning" if n < 1.0 else ("shear-thickening" if n > 1.0 else "Newtonian limit")
    return Closure(
        name="power_law",
        nu_of_gamma=_nu,
        params={"K": K, "n": n},
        nu_min=nu_min,
        nu_max=nu_max,
        description=f"ν = K γ̇^(n-1), {tag} (n={n:g})",
    )


def carreau_yasuda(
    nu_0: float,
    nu_inf: float,
    relaxation_time: float,
    flow_index: float,
    yasuda_exponent: float = 2.0,
) -> Closure:
    """Carreau–Yasuda ``ν = ν∞ + (ν0-ν∞)[1+(λγ̇)^a]^((n-1)/a)``."""
    if not 0.0 < nu_inf < nu_0:
        raise ValueError("need 0 < nu_inf < nu_0")
    if relaxation_time <= 0.0:
        raise ValueError("relaxation_time must be > 0")
    if flow_index <= 0.0:
        raise ValueError("flow_index must be > 0")
    if yasuda_exponent <= 0.0:
        raise ValueError("yasuda_exponent must be > 0")
    nu0, nuinf, lam, n, a = map(float, (nu_0, nu_inf, relaxation_time, flow_index, yasuda_exponent))

    def _nu(gamma: torch.Tensor) -> torch.Tensor:
        return nuinf + (nu0 - nuinf) * torch.pow(
            1.0 + torch.pow(lam * _floor(gamma), a), (n - 1.0) / a
        )

    return Closure(
        name="carreau_yasuda",
        nu_of_gamma=_nu,
        params={"nu_0": nu0, "nu_inf": nuinf, "lambda": lam, "n": n, "a": a},
        nu_min=nuinf,
        nu_max=nu0,
        description="ν = ν∞ + (ν0-ν∞)[1+(λγ̇)^a]^((n-1)/a)",
    )


def cross(
    nu_0: float,
    nu_inf: float,
    relaxation_time: float,
    exponent: float = 1.0,
) -> Closure:
    """Cross model ``ν = ν∞ + (ν0-ν∞)/(1+(λγ̇)^m)``."""
    if not 0.0 < nu_inf < nu_0:
        raise ValueError("need 0 < nu_inf < nu_0")
    if relaxation_time <= 0.0:
        raise ValueError("relaxation_time must be > 0")
    if exponent <= 0.0:
        raise ValueError("exponent must be > 0")
    nu0, nuinf, lam, m = map(float, (nu_0, nu_inf, relaxation_time, exponent))

    def _nu(gamma: torch.Tensor) -> torch.Tensor:
        return nuinf + (nu0 - nuinf) / (1.0 + torch.pow(lam * _floor(gamma), m))

    return Closure(
        name="cross",
        nu_of_gamma=_nu,
        params={"nu_0": nu0, "nu_inf": nuinf, "lambda": lam, "m": m},
        nu_min=nuinf,
        nu_max=nu0,
        description="ν = ν∞ + (ν0-ν∞)/(1+(λγ̇)^m)",
    )


def herschel_bulkley(
    yield_stress: float,
    consistency_index: float,
    flow_index: float,
    regularization: float = 1.0e4,
    nu_min: float = 1.0e-6,
    nu_max: float = math.inf,
) -> Closure:
    """Papanastasiou-regularised Herschel–Bulkley closure.

    ``ν(γ̇) = τ_y (1 - e^{-m γ̇}) / γ̇ + K γ̇^(n-1)``

    At ``γ̇ → 0`` the first term tends to the finite value ``τ_y·m`` (so the
    plug is a very viscous but numerically bounded fluid, ``ν_solid = τ_y m``);
    at ``γ̇ ≫ 1/m`` it recovers ``τ_y/γ̇``, i.e. the ideal Bingham branch.
    The plug half-width of a plane-Poiseuille flow is therefore *not* an input
    of the measurement: it is the locus where the stress ``τ = G|y|`` crosses
    ``τ_y``, blurred over a shear-rate band ``γ̇ ~ 1/m``.
    """
    if yield_stress < 0.0:
        raise ValueError("yield_stress must be >= 0")
    if consistency_index <= 0.0:
        raise ValueError("consistency_index must be > 0")
    if flow_index <= 0.0:
        raise ValueError("flow_index must be > 0")
    if regularization <= 0.0:
        raise ValueError("regularization must be > 0")
    ty, K, n, m = map(float, (yield_stress, consistency_index, flow_index, regularization))

    def _nu(gamma: torch.Tensor) -> torch.Tensor:
        g = _floor(gamma)
        # -(expm1(-m g))/g = (1 - exp(-m g))/g, accurate for small m g
        solid = -torch.expm1(-m * g) / g * ty
        return torch.clamp(solid + K * torch.pow(g, n - 1.0), min=nu_min, max=nu_max)

    return Closure(
        name="herschel_bulkley",
        nu_of_gamma=_nu,
        params={"tau_y": ty, "K": K, "n": n, "m": m, "nu_solid": ty * m},
        tau_y=ty,
        nu_min=nu_min,
        nu_max=nu_max,
        description="ν = τ_y(1-e^{-mγ̇})/γ̇ + K γ̇^(n-1)",
    )
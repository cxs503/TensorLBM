"""Generalised-Newtonian constitutive models and their exact Poiseuille solutions.

All quantities are in *lattice units* unless stated otherwise, i.e. the
kinematic viscosity plays the role of the dynamic viscosity with ``rho = 1``.

Constitutive laws
-----------------
Power law      ``mu(g) = K g**(n-1)``
Carreau-Yasuda ``mu(g) = mu_inf + (mu_0 - mu_inf) (1 + (lam g)**a)**((n-1)/a)``
Cross          ``mu(g) = mu_inf + (mu_0 - mu_inf) / (1 + (lam g)**m)``

Reference solutions
-------------------
For a fully developed pressure-driven flow the shear stress distribution is
known *exactly* from the momentum balance independent of the rheology::

    plane channel (half width h):  tau(s) = G (h - s),   s = |y|
    pipe (radius R, axis r=0):     tau(r) = G r / 2

with ``G = -dp/dx`` the driving pressure gradient.  The shear rate follows from
inverting ``mu(g) g = tau`` (scalar root find -- exact to quadrature error) and
the velocity profile from one more quadrature::

    u(s) = int_s^h g(eta) deta      (plane)
    u(r) = int_r^R g(eta) deta      (pipe)

For the power law the two quadratures are analytic (see
:meth:`PowerLaw.channel_profile_exact` / :meth:`PowerLaw.pipe_profile_exact`),
which lets the quadrature machinery itself be certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "PowerLaw",
    "CarreauYasuda",
    "Cross",
    "make_model",
    "poiseuille_channel_profile",
    "poiseuille_pipe_profile",
    "invert_shear_rate",
]


def invert_shear_rate(mu: Callable[[np.ndarray], np.ndarray], tau: np.ndarray,
                      iters: int = 80) -> np.ndarray:
    """Solve ``mu(g) * g = tau`` for the shear rate ``g`` (vectorised bisection).

    Bisection on ``log(g)`` is used instead of a Newton iteration because every
    model above is monotone in ``log g`` on ``[g_lo, g_hi]`` but its derivative
    can vanish (Carreau-Yasuda with ``a != 2``, Cross plateaus), which makes
    Newton fragile near the low/high shear rate plateaus.
    """
    tau = np.asarray(tau, dtype=float)
    lo = np.full_like(tau, 1e-12)
    hi = np.full_like(tau, 1e4)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)                       # geometric bisection
        f = mu(mid) * mid - tau
        lo = np.where(f < 0.0, mid, lo)
        hi = np.where(f < 0.0, hi, mid)
    return np.sqrt(lo * hi)


@dataclass(frozen=True)
class PowerLaw:
    """Ostwald-de Waele fluid. ``n < 1`` shear thinning, ``n > 1`` thickening."""

    K: float = 0.02
    n: float = 0.6
    mu_min: float = 1e-4
    mu_max: float = 10.0

    name = "power_law"

    def mu(self, gamma: np.ndarray) -> np.ndarray:
        g = np.maximum(np.asarray(gamma, dtype=float), 1e-12)
        return np.clip(self.K * g ** (self.n - 1.0), self.mu_min, self.mu_max)

    def mu_prime(self, gamma: np.ndarray) -> np.ndarray:
        """``d mu / d gamma`` (0 in the clamped branches)."""
        g = np.maximum(np.asarray(gamma, dtype=float), 1e-12)
        raw = self.K * g ** (self.n - 1.0)
        d = self.K * (self.n - 1.0) * g ** (self.n - 2.0)
        return np.where((raw > self.mu_min) & (raw < self.mu_max), d, 0.0)

    # -- analytic profiles -------------------------------------------------
    def channel_profile_exact(self, y: np.ndarray, h: float, G: float) -> np.ndarray:
        s = np.abs(np.asarray(y, dtype=float))
        s = np.minimum(s, h)
        n = self.n
        return (n / (n + 1.0)) * (G / self.K) ** (1.0 / n) * (
            h ** ((n + 1.0) / n) - s ** ((n + 1.0) / n))

    def pipe_profile_exact(self, r: np.ndarray, R: float, G: float) -> np.ndarray:
        r = np.minimum(np.abs(np.asarray(r, dtype=float)), R)
        n = self.n
        return (n / (n + 1.0)) * (G / (2.0 * self.K)) ** (1.0 / n) * (
            R ** ((n + 1.0) / n) - r ** ((n + 1.0) / n))


@dataclass(frozen=True)
class CarreauYasuda:
    mu_0: float = 0.1
    mu_inf: float = 0.01
    lam: float = 5.0
    n: float = 0.4
    a: float = 2.0

    name = "carreau_yasuda"

    def mu(self, gamma: np.ndarray) -> np.ndarray:
        g = np.maximum(np.asarray(gamma, dtype=float), 1e-12)
        return self.mu_inf + (self.mu_0 - self.mu_inf) * (
            1.0 + (self.lam * g) ** self.a) ** ((self.n - 1.0) / self.a)


@dataclass(frozen=True)
class Cross:
    mu_0: float = 0.1
    mu_inf: float = 0.01
    lam: float = 5.0
    m: float = 1.2

    name = "cross"

    def mu(self, gamma: np.ndarray) -> np.ndarray:
        g = np.maximum(np.asarray(gamma, dtype=float), 1e-12)
        return self.mu_inf + (self.mu_0 - self.mu_inf) / (
            1.0 + (self.lam * g) ** self.m)


# Lattice-unit parameters.  mu_inf is kept >= 0.02 so that tau = 1/2 + 3 mu
# stays above 0.55, which is the stability envelope of the BGK + bounce-back
# channel solver used in this repository (checked by the pytest guards).
_MODELS = {
    "power_law": (PowerLaw, dict(K=0.02, n=0.6, mu_min=1e-8, mu_max=1e6)),
    "power_law_shear_thickening": (PowerLaw, dict(K=0.5, n=1.4, mu_min=1e-8, mu_max=1e6)),
    "carreau_yasuda": (CarreauYasuda, dict(mu_0=0.2, mu_inf=0.02, lam=5.0, n=0.4, a=2.0)),
    "cross": (Cross, dict(mu_0=0.2, mu_inf=0.02, lam=5.0, m=1.2)),
}


def make_model(key: str):
    """Instantiate one of the registered rheology models by name."""
    if key not in _MODELS:
        raise KeyError(f"unknown rheology {key!r}; known: {sorted(_MODELS)}")
    cls, kw = _MODELS[key]
    return cls(**kw)


def model_registry() -> dict:
    return {k: dict(v[1]) | {"class": v[0].__name__} for k, v in _MODELS.items()}


def _quadrature_profile(model, coord: np.ndarray, L: float, tau_of_eta,
                        n_quad: int) -> np.ndarray:
    """u(s) = int_s^L gamma(eta) deta by trapezoid quadrature on ``eta``."""
    eta = np.linspace(0.0, L, n_quad)
    gam = invert_shear_rate(model.mu, tau_of_eta(eta))
    inner = np.concatenate([[0.0], np.cumsum(0.5 * (gam[1:] + gam[:-1]) * np.diff(eta))])
    total = inner[-1]
    return total - np.interp(np.minimum(coord, L), eta, inner)


def poiseuille_channel_profile(model, y: np.ndarray, h: float, G: float,
                               n_quad: int = 4001, force_quadrature: bool = False) -> np.ndarray:
    """Exact (to quadrature) plane-Poiseuille profile for any of the models.

    ``y`` is the wall-normal coordinate measured from the channel centre line,
    ``h`` the half width, ``G = -dp/dx`` the driving pressure gradient.
    """
    y = np.asarray(y, dtype=float)
    if (not force_quadrature) and hasattr(model, "channel_profile_exact"):
        return model.channel_profile_exact(y, h, G)
    return _quadrature_profile(model, np.abs(y), h, lambda eta: G * eta, n_quad)


def poiseuille_pipe_profile(model, r: np.ndarray, R: float, G: float,
                            n_quad: int = 4001, force_quadrature: bool = False) -> np.ndarray:
    """Exact (to quadrature) pipe-Poiseuille profile for any of the models."""
    r = np.asarray(r, dtype=float)
    if (not force_quadrature) and hasattr(model, "pipe_profile_exact"):
        return model.pipe_profile_exact(r, R, G)
    return _quadrature_profile(model, np.abs(r), R, lambda eta: 0.5 * G * eta, n_quad)

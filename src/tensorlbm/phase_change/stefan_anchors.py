"""Analytic anchors for the one-phase Stefan problem (planar / cylindrical / spherical).

The campaign brief quotes the reference growth law

    R_analytical(t) = 2 * Ja * sqrt(alpha_l * t / pi),      Ja = c_p * dT / h_fg

We show below (and verify numerically in ``tests/test_phase_change_lee.py``)
that this expression is exactly the **leading-order small-Ja expansion of the
one-phase planar Stefan similarity solution**, so it is only a valid anchor in
the limit ``Ja << 1`` and for a flat interface.  For finite ``Ja`` and for
other geometries (a 2-D plane "bubble" is a cylinder, a real bubble is a
sphere) the self-similar coefficient must be obtained from the exact
transcendental relation; those relations are implemented here.

Similarity ansatz
-----------------
``r`` is measured from the *initial* interface position (planar) or from the
centre (cylindrical/spherical), ``eta = r / (2 sqrt(alpha t))``, the interface
sits at ``eta = lambda`` and

    R(t) = 2 * lambda * sqrt(alpha * t) = C(Ja) * sqrt(alpha * t).

With ``theta = (T - T_sat) / dT``, ``theta(lambda) = 0``, ``theta(inf) = 1`` and
the interface energy balance ``rho*h_fg*dR/dt = k*dT*|theta'(lambda)|/(2 sqrt(alpha t))``
one gets, in the constant-density (density-equal) idealisation where
``Ja = c_p dT / h_fg``,

    |theta'(lambda)| = 2*lambda/Ja .

* planar:      theta = (erf(eta) - erf(lam)) / erfc(lam)
               => Ja = sqrt(pi) * lam * erfc(lam) * exp(lam**2)          (Neumann)
* cylindrical: theta' = A exp(-2 eta**2)/eta
               => exp(-2 lam**2)/E1(2 lam**2) = 2 lam**2 / Ja
* spherical:   theta' = A exp(-eta**2)/eta**2
               => (exp(-lam**2)/lam**2) / (exp(-lam**2)/lam - sqrt(pi)*erfc(lam))
                  = 2 lam / Ja

Small-``Ja`` behaviour: the planar branch has ``lam ~ Ja/sqrt(pi)`` (from the
planar relation above), so ``C = 2 lam -> 2 Ja / sqrt(pi)``, which is *exactly*
the campaign anchor.  The spherical branch behaves like ``C ~ sqrt(2 Ja)`` and
the cylindrical branch additionally carries a logarithmic correction
(``C ~ sqrt(2 Ja / ln(1/(2 lam^2)))``); both are O(sqrt(Ja)) and therefore
much larger than the anchor's ``1.128 Ja`` as ``Ja -> 0``.  Consequently the
campaign formula is a valid anchor **only** for a planar interface at small
``Ja`` -- for a bubble one must use the geometry-consistent exact solution
(this is why the 2-D bubble is benchmarked against the cylindrical branch).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import erfc, exp1, erf, erfinv  # noqa: F401

_EULER_GAMMA = 0.5772156649015329

_GEOMETRIES = ("planar", "cylindrical", "spherical")


def _planar_relation(lam: float, ja: float) -> float:
    """Residual of the Neumann relation ``Ja = sqrt(pi) lam erfc(lam) exp(lam^2)``."""
    return math.sqrt(math.pi) * lam * erfc(lam) * math.exp(lam * lam) - ja


def _cylindrical_relation(lam: float, ja: float) -> float:
    """Residual of the cylindrical similarity balance."""
    if lam <= 0.0:
        return -ja
    lhs = math.exp(-2.0 * lam * lam) / float(exp1(2.0 * lam * lam))
    return lhs - 2.0 * lam * lam / ja


def _spherical_relation(lam: float, ja: float) -> float:
    """Residual of the spherical similarity balance (density-equal)."""
    if lam <= 0.0:
        return -ja
    denom = math.exp(-lam * lam) / lam - math.sqrt(math.pi) * erfc(lam)
    lhs = (math.exp(-lam * lam) / (lam * lam)) / denom
    return lhs - 2.0 * lam / ja


_RELATIONS: dict[str, Callable[[float, float], float]] = {
    "planar": _planar_relation,
    "cylindrical": _cylindrical_relation,
    "spherical": _spherical_relation,
}

# Upper bound of the self-similar solution branch.  The cylindrical relation has
# no root above Ja ~ 0.2 (the cylindrical heat flux cannot sustain faster
# growth); the planar and spherical branches exist up to Ja ~ 0.564 / 1.0.
_JA_MAX = {"planar": 0.56, "cylindrical": 0.205, "spherical": 0.95}


def exact_stefan_coefficient(ja: float, geometry: str = "planar") -> float:
    """Self-similar coefficient ``C`` in ``R(t) = C sqrt(alpha t)``.

    Parameters
    ----------
    ja:
        Jacob (== Stefan, density-equal) number ``c_p dT / h_fg``.
    geometry:
        ``"planar"``, ``"cylindrical"`` (2-D plane bubble) or ``"spherical"``.

    Raises
    ------
    ValueError
        if no self-similar solution exists for the requested ``Ja``.
    """
    if geometry not in _RELATIONS:
        raise ValueError(f"unknown geometry {geometry!r}; use one of {_GEOMETRIES}")
    if ja <= 0.0:
        raise ValueError("Ja must be positive")
    if ja > _JA_MAX[geometry]:
        raise ValueError(
            f"no self-similar solution in {geometry} geometry for Ja={ja:g} "
            f"(limit {_JA_MAX[geometry]:g})"
        )
    rel = _RELATIONS[geometry]

    # Bisection on the monotone branch, robust and dependency free.
    lo, hi = 1e-10, 1e-3
    f_lo = rel(lo, ja)
    while rel(hi, ja) * f_lo > 0.0 and hi < 60.0:
        hi *= 1.7
    a, b = lo, hi
    fa = rel(a, ja)
    for _ in range(400):
        mid = 0.5 * (a + b)
        fm = rel(mid, ja)
        if fa * fm <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
        if b - a < 1e-14 * max(1.0, b):
            break
    lam = 0.5 * (a + b)
    return 2.0 * lam


def heat_flux_estimate_coefficient(ja: float) -> float:
    """Coefficient of the campaign anchor ``R = 2 Ja sqrt(alpha t / pi)``."""
    return 2.0 * ja / math.sqrt(math.pi)


def similarity_temperature_profile(
    eta: np.ndarray, ja: float, geometry: str = "planar"
) -> np.ndarray:
    """Exact self-similar liquid temperature profile ``theta(eta)``.

    ``theta = (T - T_sat)/dT``, ``theta(lambda) = 0``, ``theta(inf) = 1``.
    Values for ``eta < lambda`` (the vapour side) are returned as 0.
    """
    eta = np.asarray(eta, dtype=float)
    lam = 0.5 * exact_stefan_coefficient(ja, geometry)
    out = np.zeros_like(eta)
    above = eta >= lam
    e = eta[above]
    if geometry == "planar":
        val = (erf(e) - erf(lam)) / erfc(lam)
    elif geometry == "cylindrical":
        # theta = 1 - E1(2 eta^2)/E1(2 lam^2)
        val = 1.0 - np.array([float(exp1(2.0 * s * s)) for s in e]) / float(
            exp1(2.0 * lam * lam)
        )
    else:  # spherical
        def prof(s: float) -> float:
            g = lambda x: math.exp(-x * x) / x - math.sqrt(math.pi) * erfc(x)  # noqa: E731
            return 1.0 - g(s) / g(lam)

        val = np.array([prof(float(s)) for s in e])
    out[above] = val
    return out


def anchor_radius(
    t: np.ndarray, ja: float, alpha: float, geometry: str = "planar"
) -> np.ndarray:
    """Growth ``Delta R(t)`` measured from the initial interface position."""
    t = np.asarray(t, dtype=float)
    return exact_stefan_coefficient(ja, geometry) * np.sqrt(alpha * t)
"""Sheet -> cloud cavitation transition and the shedding Strouhal number.

Physics
-------
A sheet cavity stays attached while it is short compared with the chord.  Once
it becomes long enough, the adverse pressure gradient at its closure drives a
**re-entrant jet** that travels upstream, pinches the cavity off near the
detachment point and releases a **cloud** that convects downstream.  Two
engineering consequences are modelled here:

1. **Transition criterion** -- the sheet-to-cloud transition is placed where the
   predicted cavity length reaches a threshold fraction of the chord,

   .. math::  \\sigma_{crit} \\;\\text{such that}\\; L_{cav}(\\sigma_{crit})/c = (L/c)_{crit}

   with ``(L/c)_crit = 0.50`` (band 0.40-0.60).  That is the re-entrant-jet
   criterion in the form in which it is used for hydrofoil design: a sheet
   cavity covering more than about half the chord is no longer stable, the jet
   reaches the detachment point and the cavity is shed.
2. **Shedding frequency** -- the cloud shedding period is the re-entrant jet
   transit time over the cavity length, ``f_shed = U/(k L_cav)`` with
   ``k = 1/St_L``.  The cavity-length Strouhal number ``St_L = f L_cav/U`` is the
   quantity that is (nearly) universal for cloud cavitation shedding; the
   measured range is ``St_L = 0.2-0.3`` (equivalent to ``k = 3.3-5``), and the
   model uses ``k = 4`` (``St_L = 0.25``), i.e. the middle of that band.  The
   chord-based number follows as ``St_c = f c/U = St_L (c/L_cav)``, i.e. the
   chord-based value is *larger* than ``St_L`` whenever the cavity is shorter
   than the chord.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .section import SectionPressure, isobaric_cavity, suction_pressure
from .sigma import regime

__all__ = [
    "L_OVER_C_CRITICAL",
    "SHEDDING_K",
    "SheddingEstimate",
    "classify",
    "critical_sigma",
    "shedding",
    "strouhal_band",
]

L_OVER_C_CRITICAL = 0.50
L_OVER_C_CRITICAL_BAND = (0.40, 0.60)

SHEDDING_K = 4.0  # f_shed = U / (k L_cav);  St_L = 1/k = 0.25
SHEDDING_K_BAND = (3.3, 5.0)  # equivalent to St_L = 0.20 - 0.30


def critical_sigma(
    section: str = "naca0015",
    alpha_deg: float = 5.0,
    l_over_c_crit: float = L_OVER_C_CRITICAL,
    n_foil: int = 200,
    pressure: SectionPressure | None = None,
    sigma_lo: float = 0.05,
    sigma_hi: float | None = None,
    tol: float = 1e-4,
) -> dict:
    """``sigma`` at which ``L_cav/c`` reaches ``l_over_c_crit`` (bisection).

    ``L_cav/c`` decreases monotonically with ``sigma`` (the isobaric cavity
    shrinks as the pressure rises), so a bracketed bisection is well posed.  The
    upper bracket defaults to the inception number of the section.
    """
    sp = pressure if pressure is not None else suction_pressure(section, alpha_deg, n_foil)
    hi = float(sp.sigma_inception) if sigma_hi is None else float(sigma_hi)
    f = lambda s: isobaric_cavity(s, pressure=sp)["l_cav"] - float(l_over_c_crit)  # noqa: E731
    lo = float(sigma_lo)
    f_lo, f_hi = f(lo), f(hi)
    if f_hi >= 0.0:
        return {
            "sigma_critical": float("nan"),
            "l_over_c_crit": float(l_over_c_crit),
            "bracketed": False,
            "note": "cavity is already longer than the criterion at inception",
            "sigma_inception": sp.sigma_inception,
        }
    if f_lo <= 0.0:
        return {
            "sigma_critical": float("nan"),
            "l_over_c_crit": float(l_over_c_crit),
            "bracketed": False,
            "note": "criterion not reached at sigma_lo: transition lies below the explored range",
            "sigma_inception": sp.sigma_inception,
        }
    # f is decreasing in sigma (the isobaric cavity shrinks as the pressure rises)
    it = 0
    while hi - lo > tol and it < 200:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        it += 1
    sig = 0.5 * (lo + hi)
    cav = isobaric_cavity(sig, pressure=sp)
    return {
        "sigma_critical": float(sig),
        "l_over_c_crit": float(l_over_c_crit),
        "l_cav_at_critical": float(cav["l_cav"]),
        "sigma_inception": sp.sigma_inception,
        "bracketed": True,
        "iterations": it,
        "note": "",
    }


@dataclass
class SheddingEstimate:
    """Cloud-shedding estimate at one cavitation number."""

    sigma: float
    l_over_c: float
    u_inf: float
    chord: float
    k: float
    f_shed: float
    st_cavity: float
    st_chord: float
    regime: str

    def as_dict(self) -> dict:
        return {
            "sigma": self.sigma,
            "l_over_c": self.l_over_c,
            "u_inf": self.u_inf,
            "chord": self.chord,
            "k": self.k,
            "f_shed_hz": self.f_shed,
            "st_cavity": self.st_cavity,
            "st_chord": self.st_chord,
            "regime": self.regime,
        }


def shedding(
    sigma: float,
    u_inf: float = 10.0,
    chord: float = 0.1,
    section: str = "naca0015",
    alpha_deg: float = 5.0,
    k: float = SHEDDING_K,
    sigma_critical: float | None = None,
    pressure: SectionPressure | None = None,
) -> SheddingEstimate:
    """Re-entrant-jet shedding frequency at ``sigma``.

    ``f_shed = U / (k L_cav)`` with ``L_cav`` in metres.  The regime label uses
    the transition number of :func:`critical_sigma` (computed on the same
    pressure field when not supplied).
    """
    sp = pressure if pressure is not None else suction_pressure(section, alpha_deg)
    cav = isobaric_cavity(sigma, pressure=sp)
    if sigma_critical is None:
        sigma_critical = critical_sigma(pressure=sp)["sigma_critical"]
    l_c = float(cav["l_cav"])
    l_abs = l_c * chord
    f = u_inf / (k * l_abs) if l_abs > 0.0 else float("inf")
    return SheddingEstimate(
        sigma=float(sigma),
        l_over_c=l_c,
        u_inf=float(u_inf),
        chord=float(chord),
        k=float(k),
        f_shed=float(f),
        st_cavity=float(f * l_abs / u_inf) if l_abs > 0.0 else 0.0,
        st_chord=float(f * chord / u_inf),
        regime=regime(sigma, sp.sigma_inception, sigma_critical),
    )


def classify(sigma: float, section: str = "naca0015", alpha_deg: float = 5.0,
             pressure: SectionPressure | None = None,
             sigma_critical: float | None = None) -> str:
    """Regime label of the section at ``sigma`` (see :func:`tensorlbm.cavitation_chain.sigma.regime`)."""
    sp = pressure if pressure is not None else suction_pressure(section, alpha_deg)
    if sigma_critical is None:
        sigma_critical = critical_sigma(pressure=sp)["sigma_critical"]
    return regime(sigma, sp.sigma_inception, sigma_critical)


def strouhal_band(k: float = SHEDDING_K) -> dict:
    """``St_L`` implied by the re-entrant-jet constant ``k`` and the measured band."""
    st = 1.0 / float(k)
    lo, hi = 1.0 / SHEDDING_K_BAND[1], 1.0 / SHEDDING_K_BAND[0]
    return {
        "k": float(k),
        "st_cavity": st,
        "band_st_cavity": (lo, hi),
        "inside_band": bool(lo <= st <= hi),
    }
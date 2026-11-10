"""Cavitation / ventilation number: definitions, regimes and scaling laws.

Mission P151W1-N04 (cavitation + ventilation validation face).

Definitions
-----------
    sigma   = (p_inf - p_v) / (0.5 rho U^2)     cavitation number
    sigma_v = (p_inf - p_g) / (0.5 rho U^2)     ventilation number (injected gas)

``sigma >> 1`` is single phase, ``sigma ~ sigma_i`` inception, ``sigma << 1``
developed cavitation.  The module also holds the two scaling laws the chain is
validated on:

* **Inception scaling** -- for a section with a parabolic leading edge the
  minimum-pressure coefficient follows ``-Cp,min = sigma_i ∝ alpha^2 (c/r_LE)``
  (classical leading-edge suction-peak law).  The exponents (2 in ``alpha``,
  -1 in ``r_LE/c``) are the analytic references the panel-method prediction is
  held against.
* **Cavity-length scaling** -- ``L_cav/c = A sigma^-n``, the correlation form
  the sheet-cavity measurements are usually reported in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "REGIMES",
    "PowerLaw",
    "cavitation_number",
    "fit_power_law",
    "inception_sigma",
    "leading_edge_radius_ratio",
    "regime",
    "sigma_from_depth",
    "ventilation_number",
    "vapour_pressure",
]

REGIMES = ("single_phase", "inception", "sheet", "cloud")


def vapour_pressure(T_C: float = 20.0) -> float:
    """Saturation vapour pressure of water (Pa), Tetens form.

    ``p_v = 611.2 exp(17.62 T / (243.12 + T))`` with ``T`` in degC; better than
    0.2 % against the steam tables between 0 and 100 degC.
    """
    T = float(T_C)
    if not -1.0 < T < 100.0:
        msg = "vapour_pressure is restricted to water in -1..100 degC"
        raise ValueError(msg)
    return 611.2 * math.exp(17.62 * T / (243.12 + T))


def cavitation_number(p_inf: float, p_v: float, rho: float, U: float) -> float:
    """``sigma = (p_inf - p_v) / (0.5 rho U^2)``."""
    q = 0.5 * rho * U * U
    if q <= 0.0:
        msg = "U must be positive"
        raise ValueError(msg)
    return (p_inf - p_v) / q


def ventilation_number(p_inf: float, p_gas: float, rho: float, U: float) -> float:
    """Ventilation number: the cavitation-number form with the injected-gas pressure.

    Air injection raises the cavity pressure (``p_gas > p_v``) and therefore
    *lowers* the ventilation number relative to the cavitation number, which is
    the quantitative statement of "ventilation promotes a cavity": a ventilated
    cavity is sustained at a freestream pressure where a vapour cavity would
    collapse.
    """
    return cavitation_number(p_inf, p_gas, rho, U)


def sigma_from_depth(
    U: float,
    depth: float = 0.0,
    T_C: float = 20.0,
    rho: float = 998.2,
    p_atm: float = 101325.0,
    g: float = 9.80665,
) -> float:
    """``sigma`` for a body ``depth`` below the free surface, moving at ``U``."""
    p_inf = p_atm + rho * g * depth
    return cavitation_number(p_inf, vapour_pressure(T_C), rho, U)


def inception_sigma(cp_min: float) -> float:
    """Inception number from the section suction peak: ``sigma_i = -Cp,min``."""
    if cp_min > 0.0:
        msg = "Cp_min must be non-positive (suction peak)"
        raise ValueError(msg)
    return float(-cp_min)


def leading_edge_radius_ratio(thickness_ratio: float) -> float:
    """NACA 4-digit leading-edge radius ``r_LE/c = 1.1019 (t/c)^2``."""
    return 1.1019 * float(thickness_ratio) ** 2


@dataclass(frozen=True)
class PowerLaw:
    """Result of a log-log power-law fit ``y = A x^n``."""

    exponent: float
    prefactor: float
    rms_log_residual: float
    n_points: int

    def as_dict(self) -> dict:
        return {
            "exponent": self.exponent,
            "prefactor": self.prefactor,
            "rms_log_residual": self.rms_log_residual,
            "n_points": self.n_points,
        }


def fit_power_law(x, y) -> PowerLaw:
    """Least-squares fit of ``y = A x^n`` in log-log space."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size or xa.size < 2:
        msg = "fit_power_law needs at least two paired samples"
        raise ValueError(msg)
    if np.any(xa <= 0.0) or np.any(ya <= 0.0):
        msg = "power-law fit needs strictly positive samples"
        raise ValueError(msg)
    lx, ly = np.log(xa), np.log(ya)
    n, c = np.polyfit(lx, ly, 1)
    res = ly - (n * lx + c)
    return PowerLaw(
        exponent=float(n),
        prefactor=float(math.exp(c)),
        rms_log_residual=float(np.sqrt(np.mean(res**2))),
        n_points=int(xa.size),
    )


def regime(sigma: float, sigma_inception: float, sigma_critical: float | None = None) -> str:
    """Regime label (``'single_phase'|'inception'|'sheet'|'cloud'``).

    ``sigma > sigma_i``               single phase, no cavity
    ``sigma_crit < sigma <= sigma_i`` attached sheet cavity
    ``sigma <= sigma_crit``           cloud cavitation (sheet detaches and sheds)
    """
    if sigma_critical is None:
        sigma_critical = 0.5 * sigma_inception
    if sigma > sigma_inception * (1.0 + 1e-7):
        return "single_phase"
    if abs(sigma - sigma_inception) <= 0.05 * sigma_inception:
        return "inception"
    return "cloud" if sigma <= sigma_critical else "sheet"
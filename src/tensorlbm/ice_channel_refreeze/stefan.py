"""Analytic and semi-analytic ice-growth references.

Three reference levels, cheapest first:

1. ``neumann_thickness`` -- the exact one-phase Neumann/Stefan similarity
   solution for a constant surface temperature (the primary verification
   anchor, machine criterion a).
2. ``thickness_with_surface_resistance`` -- closed-form growth for a *Robin*
   surface (constant ``h_tot``, no ocean flux), obtained by integrating
   ``rhoL dh/dt = dT / (h/k + 1/h_s)``.
3. ``integrate_growth`` -- the same column energy balance with an arbitrary
   constant ocean heat flux ``q_o``, integrated with an adaptive RK45.  This is
   the cheap engine behind the ``T_window`` lookup table; it is validated
   against the enthalpy LBM (``column.py``) and against both closed forms.

All functions are in SI units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erf

from .constants import DEFAULT_PROPERTIES, IceWaterProperties


def neumann_lambda(ste: float) -> float:
    """Solve ``lam exp(lam^2) erf(lam) = Ste/sqrt(pi)`` for the Neumann root."""
    ste = float(ste)
    if not (0.0 < ste < 50.0):
        raise ValueError("Stefan number out of the supported range")
    target = ste / math.sqrt(math.pi)

    def f(lam: float) -> float:
        return lam * math.exp(lam * lam) * erf(lam) - target

    hi = 1.0
    while f(hi) < 0.0:
        hi *= 1.5
    return brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-14)


def neumann_thickness(
    t: float | np.ndarray,
    delta_t: float,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> np.ndarray | float:
    """Exact one-phase freezing front ``s(t) = 2 lam sqrt(alpha t)``, m.

    ``delta_t = T_freeze - T_wall > 0``; the liquid is held at ``T_freeze`` so
    only the solid conducts (Stefan/Neumann idealisation).
    """
    ste = props.cp_i * float(delta_t) / props.latent_heat
    lam = neumann_lambda(ste)
    t_arr = np.asarray(t, dtype=float)
    out = 2.0 * lam * np.sqrt(props.alpha_i * np.maximum(t_arr, 0.0))
    return float(out) if out.ndim == 0 else out


def neumann_temperature_profile(
    x: np.ndarray,
    t: float,
    delta_t: float,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> np.ndarray:
    """Similarity temperature field of the planar freezing problem.

    ``T(x) = T_f - delta_t erf(eta)/erf(lam)`` for ``x <= s(t)`` and
    ``T = T_f`` ahead of the front, with ``eta = x/(2 sqrt(alpha t))``.
    Used to initialise the LBM away from the ``t -> 0`` singularity.
    """
    x = np.asarray(x, dtype=float)
    lam = neumann_lambda(props.cp_i * float(delta_t) / props.latent_heat)
    eta = x / (2.0 * math.sqrt(props.alpha_i * max(float(t), 1e-9)))
    prof = props.t_freeze - float(delta_t) * (1.0 - erf(eta) / erf(lam))
    s = float(neumann_thickness(t, delta_t, props))
    return np.where(x <= s, prof, props.t_freeze)


def thickness_with_surface_resistance(
    t: float | np.ndarray,
    delta_t: float,
    h_surface: float,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> np.ndarray | float:
    """Closed-form Robin-surface growth (``h_s -> inf`` recovers Neumann).

    Integrating ``rhoL dh/dt = k dT / (k/h_s + h)`` gives

        h(t) = (k/h_s) ( sqrt(1 + 2 h_s^2 dT t / (rhoL k)) - 1 ).
    """
    t_arr = np.maximum(np.asarray(t, dtype=float), 0.0)
    b = props.k_i / float(h_surface)
    coef = 2.0 * float(h_surface) ** 2 * float(delta_t) * t_arr / (
        props.rho_latent * props.k_i
    )
    out = b * (np.sqrt(1.0 + coef) - 1.0)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class GrowthResult:
    """Sampled growth curve with helpers for navigation windows."""

    t: np.ndarray        # s
    h: np.ndarray        # m
    q_ocean: float       # W/m^2

    def time_to(self, thickness: float) -> float:
        """First time at which ``h >= thickness`` (s); ``inf`` if never."""
        idx = np.nonzero(self.h >= float(thickness))[0]
        if idx.size == 0:
            return math.inf
        i = int(idx[0])
        if i == 0:
            return float(self.t[0])
        h0, h1 = self.h[i - 1], self.h[i]
        t0, t1 = self.t[i - 1], self.t[i]
        if h1 <= h0:
            return float(t1)
        return float(t0 + (thickness - h0) * (t1 - t0) / (h1 - h0))

    def degree_day_coefficient(self, t_end: float) -> float:
        """Effective growth constant ``C = h / sqrt(FDD)`` in cm/degC^0.5/day^0.5."""
        h = float(self.h[-1]) if t_end >= self.t[-1] else float(
            np.interp(t_end, self.t, self.h)
        )
        fdd_days = max(self.t[-1] if t_end >= self.t[-1] else t_end, 1e-9) / 86400.0
        return 100.0 * h / math.sqrt(fdd_days)


def integrate_growth(
    t_end_days: float,
    t_eff: float,
    h_surface: float,
    q_ocean: float = 0.0,
    h0: float = 1e-5,
    n_out: int = 2001,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> GrowthResult:
    """Integrate the column energy balance with a constant ocean heat flux.

    ``rhoL dh/dt = (T_freeze - T_eff) / (h/k + 1/h_s) - q_o``

    where ``T_eff`` and ``h_s`` are the condensed atmospheric boundary
    condition of :mod:`surface`.  Growth starts from ``h0`` so that the
    conductive flux stays finite at ``t = 0``.
    """
    d_theta = props.t_freeze - float(t_eff)
    b = props.k_i / float(h_surface)
    t_end = float(t_end_days) * 86400.0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        h = max(float(y[0]), 1e-6)
        q_net = d_theta / (h / props.k_i + 1.0 / float(h_surface)) - float(q_ocean)
        return np.array([q_net / props.rho_latent])

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        np.array([float(h0)]),
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
        max_step=t_end / 50.0,
    )
    if not sol.success:
        raise RuntimeError(f"column growth integration failed: {sol.message}")
    t = np.linspace(0.0, t_end, int(n_out))
    h = np.maximum(sol.sol(t)[0], 0.0)
    del b  # documented above; kept for readability of the derivation
    return GrowthResult(t=t, h=h, q_ocean=float(q_ocean))


def equilibrium_thickness(
    t_eff: float,
    h_surface: float,
    q_ocean: float,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> float:
    """Thickness at which conduction exactly balances the ocean heat flux, m."""
    if q_ocean <= 0.0:
        return math.inf
    d_theta = props.t_freeze - float(t_eff)
    denom = float(q_ocean)
    h = props.k_i * d_theta / denom - props.k_i / float(h_surface)
    return max(h, 0.0)


def degree_day_thickness(fdd: float, coefficient_cm: float) -> float:
    """Empirical degree-day law ``h = C sqrt(FDD)`` (C in cm, FDD in degC day)."""
    arr = np.maximum(np.asarray(fdd, dtype=float), 0.0)
    out = coefficient_cm * np.sqrt(arr) / 100.0
    return float(out) if out.ndim == 0 else out


__all__ = [
    "neumann_lambda",
    "neumann_thickness",
    "thickness_with_surface_resistance",
    "neumann_temperature_profile",
    "GrowthResult",
    "integrate_growth",
    "equilibrium_thickness",
    "degree_day_thickness",
]
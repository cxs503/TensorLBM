"""Kunz-type pressure-driven cavitation mass-transfer model (mission P151W1-N04).

Model
-----
The Kunz *et al.* (2000) family writes the vaporisation / condensation rates per
unit volume as a *pressure-driven* transfer scaled by the freestream dynamic
pressure:

.. math::

    \\dot m_{evap} = C_{dest}\\,\\rho_v\\,\\alpha_l\\,\\frac{(p_v-p)^+}{t_\\infty\\,\\tfrac12\\rho_l U^2}

    \\dot m_{cond} = C_{prod}\\,\\rho_v\\,\\alpha_w\\,\\frac{(p-p_v)^+}{t_\\infty\\,\\tfrac12\\rho_l U^2}

with ``t_inf = c/U_inf`` the reference (flow) time.  Two weightings of the
condensation term ``alpha_w`` are implemented:

``weighting="physical"`` (default, used by the solver)
    ``alpha_w = alpha_v`` (and ``alpha_v_quad=True`` switches to the classical
    ``alpha_v^2 (1 - alpha_v)`` production factor).  The closure is then
    *closed*: no vapour implies no condensation, no liquid implies no
    evaporation, so a single-phase state stays single-phase.
``weighting="spec"``
    the literal form in the mission statement (condensation without an
    ``alpha_v`` prefactor).  It is kept only for the mass-transfer-rate
    acceptance check, because at ``alpha_v = 0`` it predicts condensation out of
    nothing.  The deviation is reported explicitly in the validation artefacts;
    the transport solver uses the physical weighting.

Verification anchors
--------------------
For a *constant* pressure departure the physical weighting reduces to

* ``p < p_v``:  ``d alpha_v/dt = k (1 - alpha_v)``, ``k = C_dest |p_v-p| / (t_inf q)``
* ``p > p_v``:  ``d alpha_v/dt = -k' alpha_v``,  ``k' = C_prod (p-p_v) / (t_inf q)``

both of which have closed-form solutions (:meth:`Kunz0D.analytic`), so the RK4
march is verified against an exact reference rather than against itself.  The
same closure gives the two-zone steady state of
:func:`steady_state_vapour_fraction`, where evaporation in a low-pressure zone
balances condensation in a high-pressure zone -- the "evaporated mass =
condensed mass" acceptance item.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Kunz0D",
    "KunzCoefficients",
    "kunz_mass_transfer",
    "kunz_timescale",
    "steady_state_vapour_fraction",
]

# saturated steam density at 20 degC (ideal gas at p_v = 2.34 kPa)
RHO_V_STEAM_20C = 0.01729


@dataclass(frozen=True)
class KunzCoefficients:
    """Empirical rate constants (Kunz et al. 2000 use ``C_dest = C_prod = 1``)."""

    c_dest: float = 1.0
    c_prod: float = 1.0
    weighting: str = "physical"
    alpha_v_quad: bool = False

    def __post_init__(self) -> None:
        if self.weighting not in ("physical", "spec"):
            msg = "weighting must be 'physical' or 'spec'"
            raise ValueError(msg)
        if self.c_dest <= 0.0 or self.c_prod <= 0.0:
            msg = "Kunz constants must be positive"
            raise ValueError(msg)

    def as_dict(self) -> dict:
        return {
            "c_dest": self.c_dest,
            "c_prod": self.c_prod,
            "weighting": self.weighting,
            "alpha_v_quad": self.alpha_v_quad,
        }


def kunz_mass_transfer(
    p,
    p_v: float,
    alpha_l,
    alpha_v,
    rho_l: float,
    rho_v: float,
    U: float,
    coeffs: KunzCoefficients | None = None,
    t_inf: float = 1.0,
):
    """Kunz mass-transfer rates in kg m^-3 s^-1: ``(m_dot_evap, m_dot_cond)``.

    Both outputs are non-negative and never simultaneously non-zero (the
    evaporation branch is active for ``p < p_v``, the condensation branch for
    ``p > p_v``).  Inputs broadcast.
    """
    c = coeffs or KunzCoefficients()
    q = 0.5 * rho_l * U * U
    if q <= 0.0:
        msg = "U must be positive"
        raise ValueError(msg)
    dp = np.asarray(p, dtype=float) - float(p_v)
    al = np.clip(np.asarray(alpha_l, dtype=float), 0.0, 1.0)
    av = np.clip(np.asarray(alpha_v, dtype=float), 0.0, 1.0)
    if c.weighting == "physical":
        w_cond = av**2 * al if c.alpha_v_quad else av
    else:
        w_cond = np.ones_like(dp)
    scale = rho_v / (t_inf * q)
    m_evap = c.c_dest * scale * al * np.clip(-dp, 0.0, None)
    m_cond = c.c_prod * scale * w_cond * np.clip(dp, 0.0, None)
    return m_evap, m_cond


def kunz_timescale(sigma: float, t_inf: float = 1.0, c_dest: float = 1.0) -> float:
    """Evaporation relaxation time ``tau = t_inf / (C_dest sigma)``.

    From ``d alpha_v/dt = (1 - alpha_v)/tau`` with the pressure departure
    ``p_v - p = sigma * 0.5 rho_l U^2``: the rate coefficient is
    ``C_dest |p_v-p| / (t_inf * 0.5 rho_l U^2) = C_dest sigma / t_inf``, i.e. one
    relaxation time is exactly ``t_inf / sigma`` for ``C_dest = 1`` -- the
    "cavity fills in one flow time at inception" behaviour of the closure.
    """
    if sigma <= 0.0:
        msg = "sigma must be positive"
        raise ValueError(msg)
    return t_inf / (c_dest * sigma)


def steady_state_vapour_fraction(
    dp_cavity: float,
    dp_condensation: float,
    alpha_v_cavity: float = 1.0e-3,
    coeffs: KunzCoefficients | None = None,
) -> dict:
    """Two-zone steady state: evaporation (low pressure) = condensation (high pressure).

    Zone A (``p < p_v``, departure ``dp_cavity = p_v - p > 0``) evaporates at
    ``C_dest alpha_l,A dp_cavity / (t_inf q)``; zone B (``p > p_v``, departure
    ``dp_condensation = p - p_v > 0``) condenses at
    ``C_prod alpha_v,B dp_cond / (t_inf q)``.  Equating the two gives the steady
    void fraction of zone B, which is what the "evaporated mass = condensed
    mass" acceptance item checks.
    """
    c = coeffs or KunzCoefficients()
    if dp_cavity <= 0.0 or dp_condensation <= 0.0:
        msg = "both pressure departures must be positive"
        raise ValueError(msg)
    alpha_l_a = 1.0 - float(alpha_v_cavity)
    evap = c.c_dest * alpha_l_a * dp_cavity
    alpha_v_b = evap / (c.c_prod * dp_condensation)
    alpha_v_b = float(min(1.0, max(0.0, alpha_v_b)))
    evap_rate = c.c_dest * alpha_l_a * dp_cavity
    cond_rate = c.c_prod * alpha_v_b * dp_condensation
    return {
        "alpha_v_condensation_zone": alpha_v_b,
        "evaporation_drive": evap_rate,
        "condensation_drive": cond_rate,
        "relative_imbalance": abs(evap_rate - cond_rate) / evap_rate,
    }


@dataclass
class Kunz0D:
    """Zero-dimensional homogeneous vapour-fraction transport with Kunz rates.

    ``d alpha_v/dt = (m_dot_evap - m_dot_cond) / rho_v``

    with the pressure prescribed in time.  Used for (i) the mass-transfer-rate
    acceptance check against the analytic solution, (ii) the periodic-driver
    mass balance and (iii) the single-phase / high-pressure negative controls.
    """

    rho_l: float = 998.2
    rho_v: float = RHO_V_STEAM_20C
    U: float = 10.0
    t_inf: float = 1.0
    coeffs: KunzCoefficients = field(default_factory=KunzCoefficients)
    alpha_v0: float = 1.0e-4

    # -- basics -----------------------------------------------------------
    def sigma_of_p(self, p, p_v: float):
        """``sigma`` implied by the local pressure (diagnostic, not a constant)."""
        return (np.asarray(p, dtype=float) - float(p_v)) / (0.5 * self.rho_l * self.U**2)

    def rate(self, alpha_v: float, p: float, p_v: float) -> float:
        """``d alpha_v / dt`` (1/s) at the given state and pressure."""
        me, mc = kunz_mass_transfer(
            p, p_v, 1.0 - alpha_v, alpha_v, self.rho_l, self.rho_v, self.U,
            self.coeffs, self.t_inf,
        )
        return float((float(me) - float(mc)) / self.rho_v)

    def rates(self, alpha_v, p, p_v: float):
        """``(m_dot_evap, m_dot_cond)`` arrays at the given state/pressure."""
        return kunz_mass_transfer(
            p, p_v, 1.0 - np.asarray(alpha_v, dtype=float),
            np.asarray(alpha_v, dtype=float), self.rho_l, self.rho_v, self.U,
            self.coeffs, self.t_inf,
        )

    # -- integrators ------------------------------------------------------
    def march(self, p_driver, p_v: float, dt: float, alpha_v0: float | None = None) -> dict:
        """Classical RK4 march over a pressure time series.

        ``p_driver[i]`` is the pressure at ``t = i*dt``; the state is advanced
        from ``i`` to ``i+1`` with the pressure frozen at the mid-step value,
        i.e. the driver is treated as piecewise constant over a step (second
        order for the smooth drivers used here).
        """
        p = np.asarray(p_driver, dtype=float)
        if p.ndim != 1 or p.size < 2:
            msg = "p_driver must be a 1-D array with at least two samples"
            raise ValueError(msg)
        a = float(self.alpha_v0 if alpha_v0 is None else alpha_v0)
        out = np.empty(p.size)
        out[0] = a
        for i in range(p.size - 1):
            pm = 0.5 * (p[i] + p[i + 1])
            k1 = self.rate(a, pm, p_v)
            k2 = self.rate(a + 0.5 * dt * k1, pm, p_v)
            k3 = self.rate(a + 0.5 * dt * k2, pm, p_v)
            k4 = self.rate(a + dt * k3, pm, p_v)
            a = float(np.clip(a + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0))
            out[i + 1] = a
        me, mc = self.rates(out, p, p_v)
        return {
            "t": np.arange(p.size) * dt,
            "alpha_v": out,
            "p": p,
            "sigma": self.sigma_of_p(p, p_v),
            "evap_mass": float(np.sum(me) * dt),
            "cond_mass": float(np.sum(mc) * dt),
            "net_vapour_mass": float(np.sum(me - mc) * dt),
        }

    def analytic(self, t, p: float, p_v: float, alpha_v0: float | None = None):
        """Closed-form solution for a *constant* pressure (physical weighting)."""
        if self.coeffs.alpha_v_quad:
            msg = "analytic() covers the linear (alpha_v_quad=False) closure only"
            raise ValueError(msg)
        tt = np.asarray(t, dtype=float)
        a0 = float(self.alpha_v0 if alpha_v0 is None else alpha_v0)
        q = 0.5 * self.rho_l * self.U * self.U
        dp = float(p) - float(p_v)
        if dp < 0.0:
            k = self.coeffs.c_dest * abs(dp) / (self.t_inf * q)
            return 1.0 - (1.0 - a0) * np.exp(-k * tt)
        k = self.coeffs.c_prod * dp / (self.t_inf * q)
        return a0 * np.exp(-k * tt)

    # -- negative controls -------------------------------------------------
    def single_phase_drift(self, p: float, p_v: float, alpha_v: float = 0.0) -> float:
        """``d alpha_v/dt`` for a state with **no vapour** (must be zero)."""
        return self.rate(float(alpha_v), p, p_v)

    def high_pressure_drift(self, p_v: float, sigma: float, alpha_v: float = 0.5) -> float:
        """``d alpha_v/dt`` at ``p = p_v + sigma * 0.5 rho U^2`` with large ``sigma``."""
        p = p_v + sigma * 0.5 * self.rho_l * self.U**2
        return self.rate(float(alpha_v), p, p_v)
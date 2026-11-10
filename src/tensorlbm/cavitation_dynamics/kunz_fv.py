"""Kunz (2000) cavitation closure on a 1-D unsteady finite-volume transport solve.

Physics
-------
Vapour fraction is transported along the suction-side surface layer of a
hydrofoil section (streamwise coordinate ``x/c``) with the Kunz pressure-driven
mass-transfer closure:

.. math::

    \\frac{\\partial \\alpha_v}{\\partial t}
    + \\frac{\\partial (u(x)\\,\\alpha_v)}{\\partial x}
    = S(\\alpha_v, p), \\qquad
    S = \\begin{cases}
        +\\dfrac{C_{dest}\\,\\alpha_l\,(p_v - p)}{\\tfrac12\\rho_l U^2\\,t_\\infty}
            & p < p_v \\;(\\text{evaporation})\\\\[4pt]
        -\\dfrac{C_{prod}\\,\\alpha_v\\,(p - p_v)}{\\tfrac12\\rho_l U^2\\,t_\\infty}
            & p > p_v \\;(\\text{condensation})
        \\end{cases}

with ``alpha_l = 1 - alpha_v`` and the chord time ``t_inf = c/U`` as reference
time.  The pressure is supplied by the wetted section solution (Hess-Smith
panel method, :mod:`tensorlbm.cavitation_chain.section`) as ``Cp(x)``; because
``(p - p_v)/q = sigma + Cp(x)`` the local driving term is the single scalar

.. math::  s(x) = \\sigma + C_p(x),

negative inside the cavity (evaporation) and positive downstream of it
(condensation).  Inside the cavity body (``alpha_v >= 0.5``) the pressure is
clamped to ``p_v`` (``s = 0``) -- the vapour phase is at saturation, so there is
no further phase change there.

Sheet -> cloud transition (re-entrant jet)
------------------------------------------
The attached sheet cavity grows until its length reaches the re-entrant-jet
threshold ``L/c = 0.5`` (band 0.40-0.60).  Beyond it a near-wall jet is formed
at the closure, travels **upstream** at the measured velocity ratio
``c_j = U_jet/U = 0.30`` (measured range 0.2-0.4 U), and on reaching the
detachment point the whole sheet breaks up into a **cloud** that convects
downstream at ``0.9 U`` and condenses in the pressure-recovery region.  The
cycle repeats, so the shedding frequency is an *output* of the growth time plus
the jet transit time -- it is not imposed anywhere in the code.

Numerics
--------
* First-order upwind finite volume for the advective flux (conservative flux
  form, so the vapour budget is exact to round-off apart from the outlet flux).
* The source is integrated with a positivity-preserving implicit linearisation
  ``alpha <- (alpha + dt*S_e) / (1 + dt*S_d)`` with ``S = S_e - S_d alpha``,
  which keeps ``0 <= alpha <= 1`` for any ``dt``.
* Operator splitting (source step, then flux step) is second-order accurate in
  the reactive limit and unconditionally stable for the source; CFL is
  controlled by the ``cfl`` argument (0.4 by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..cavitation_chain.section import suction_pressure

__all__ = [
    "CavityDynamics",
    "KunzClosure",
    "L_OVER_C_CRITICAL",
    "JET_RATIO_BAND",
    "cavitation_number",
    "dominant_frequency",
    "naca0015_cp",
    "run_dynamics",
    "sigma_critical_dynamics",
    "steady_cavity_length",
    "strouhal",
    "vapour_pressure",
    "ventilation_number",
]

RHO_WATER = 998.2          # kg/m^3, 20 C
RHO_VAPOUR = 0.01729       # kg/m^3, saturated vapour at 20 C
P_VAPOUR_20C = 2339.0      # Pa
L_OVER_C_CRITICAL = 0.50   # re-entrant-jet sheet/cloud threshold
L_OVER_C_CRITICAL_BAND = (0.40, 0.60)
# U_jet / U_inf.  The model value 0.25 is the in-repo convention
# (tensorlbm.cavitation_chain.sheet_cloud: f = U/(k L) with k = 4 <=> St_L = 0.25)
# and sits inside the measured re-entrant-jet band 0.2-0.4.
JET_RATIO = 0.25
JET_RATIO_BAND = (0.20, 0.40)   # measured re-entrant jet velocity ratio
CLOUD_SPEED_RATIO = 0.90   # cloud convection speed / U_inf
TAU_PINCH = 0.05           # chord times, roll-up time of the jet-cut cavity
ALPHA_CAVITY = 0.50        # alpha_v defining the cavity body
ALPHA_EXTENT = 0.10        # alpha_v defining the visual (video) vapour extent
ALPHA_FREESTREAM = 1.0e-5  # nuclei-borne vapour fraction of the incoming water


def vapour_pressure(temperature_c: float = 20.0) -> float:
    """Saturated vapour pressure of water in Pa (Antoine, ``t`` in deg C)."""
    t = float(temperature_c)
    return float(10.0 ** (8.07131 - 1730.63 / (233.426 + t)) * 133.322368)


def cavitation_number(p_inf: float, p_v: float | None = None, u_inf: float = 1.0,
                      rho: float = RHO_WATER) -> float:
    """``sigma = (p_inf - p_v) / (0.5 rho U^2)``."""
    pv = vapour_pressure() if p_v is None else float(p_v)
    return float((p_inf - pv) / (0.5 * rho * u_inf * u_inf))


def ventilation_number(p_inf: float, p_cavity: float | None = None, u_inf: float = 1.0,
                       rho: float = RHO_WATER) -> float:
    """Cavitation/ventilation number with an arbitrary cavity pressure.

    The scaling law of the node is the same expression in both regimes,

    .. math::  \\sigma = \\frac{p_\\infty - p_{cav}}{\\tfrac12\\rho U^2},

    with ``p_cav = p_v`` (vaporous cavitation) or ``p_cav = p_v + p_gas``
    (ventilated cavitation, i.e. gas injected into the cavity).  Ventilation
    therefore lowers ``sigma`` at unchanged depth and speed, which is why it
    suppresses cloud shedding: everything downstream is a function of ``sigma``.
    """
    p_cav = vapour_pressure() if p_cavity is None else float(p_cavity)
    return float((p_inf - p_cav) / (0.5 * rho * u_inf * u_inf))


@lru_cache(maxsize=8)
def _cp_table(section: str = "naca0015", alpha_deg: float = 5.0, n_foil: int = 200):
    sp = suction_pressure(section, alpha_deg, n_foil)
    return np.ascontiguousarray(sp.x), np.ascontiguousarray(sp.cp), float(sp.cl_panel)


def naca0015_cp(x: np.ndarray | float, section: str = "naca0015", alpha_deg: float = 5.0,
                n_foil: int = 200) -> np.ndarray:
    """Suction-side ``Cp(x/c)`` of the wetted section, extrapolated to the trailing edge.

    The panel solution stops short of ``x/c = 1``; downstream of the last panel
    the pressure is continued by a linear extrapolation and clamped to
    ``[-Cp_max, +0.5]`` so that the condensation region always exists.
    """
    xs, cp, _ = _cp_table(section, alpha_deg, n_foil)
    xq = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    out = np.interp(xq, xs, cp)
    tail = xs >= 0.85 * xs[-1]
    if tail.sum() >= 2:
        slope = np.polyfit(xs[tail], cp[tail], 1)[0]
        x_tail = xq > xs[-1]
        if np.any(x_tail):
            out = np.where(x_tail, cp[-1] + slope * (xq - xs[-1]), out)
    return np.clip(out, -4.0 * abs(float(cp.min())), 0.5)


@dataclass(frozen=True)
class KunzClosure:
    """Kunz mass-transfer constants (dimensionless, ``t_inf = c/U`` reference time)."""

    c_dest: float = 12.0
    c_prod: float = 12.0
    t_inf: float = 1.0
    alpha_min: float = 1.0e-6
    alpha_max: float = 1.0 - 1.0e-6

    def coefficients(self, s: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearise the source as ``S = S_e - S_d * alpha`` (elementwise).

        ``s = sigma + Cp`` is the local pressure departure from the vapour
        pressure, normalised by the dynamic head.
        """
        s = np.asarray(s, dtype=float)
        a = np.asarray(alpha, dtype=float)
        evap = s < 0.0
        rate = np.abs(s) / self.t_inf
        # S(alpha) = C_dest*|s|*(1-alpha) for s<0 and -C_prod*|s|*alpha for s>0,
        # written in the linear form S = S_e - S_d*alpha used by the implicit step
        s_e = np.where(evap, self.c_dest * rate, 0.0)
        s_d = np.where(evap, self.c_dest * rate, self.c_prod * rate)
        return s_e, s_d

    def source(self, s: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Instantaneous ``d(alpha_v)/dt`` (per chord time)."""
        s_e, s_d = self.coefficients(s, alpha)
        return s_e - s_d * np.asarray(alpha, dtype=float)

    def mass_transfer(self, s: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Mass-transfer rate ``mdot`` in kg/(m^3 s) (times ``rho_v``)."""
        return self.source(s, alpha) * RHO_VAPOUR


@dataclass
class CavityDynamics:
    """Result of one unsteady run at a fixed cavitation number."""

    sigma: float
    t: np.ndarray
    l_sheet: np.ndarray            # attached sheet length L/c (>ALPHA_CAVITY from LE)
    l_extent: np.ndarray           # visual vapour extent L/c (>ALPHA_EXTENT, incl. cloud)
    vapour_mass: np.ndarray        # integral of alpha_v over the chord, per unit depth
    x: np.ndarray
    alpha_final: np.ndarray
    snapshots: list = field(default_factory=list)
    snap_times: list = field(default_factory=list)
    shed_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    f_shed: float = 0.0
    st_l: float = 0.0
    st_c: float = 0.0
    l_detach: float = 0.0
    mass_evap: float = 0.0
    mass_cond: float = 0.0
    mass_out: float = 0.0
    mass_in: float = 0.0
    mass_clip: float = 0.0
    mass_balance: float = 0.0
    closure: KunzClosure = field(default_factory=KunzClosure)
    meta: dict = field(default_factory=dict)

    @property
    def l_sheet_mean(self) -> float:
        return float(np.mean(self.l_sheet))

    @property
    def l_extent_mean(self) -> float:
        return float(np.mean(self.l_extent))

    @property
    def shedding(self) -> bool:
        return bool(self.shed_times.size >= 3)

    def summary(self) -> dict:
        return {
            "sigma": self.sigma,
            "l_sheet_mean": self.l_sheet_mean,
            "l_extent_mean": self.l_extent_mean,
            "l_sheet_max": float(np.max(self.l_sheet)),
            "l_detach": self.l_detach,
            "f_shed": self.f_shed,
            "St_L": self.st_l,
            "St_c": self.st_c,
            "shedding": self.shedding,
            "n_shed_events": int(self.shed_times.size),
            "mass_evap": self.mass_evap,
            "mass_cond": self.mass_cond,
            "mass_out": self.mass_out,
            "mass_balance_rel": self.mass_balance,
            "c_dest": self.closure.c_dest,
            "c_prod": self.closure.c_prod,
        }


def _cavity_length(x: np.ndarray, alpha: np.ndarray, threshold: float,
                   max_gap: int = 2) -> float:
    """Farthest station (from the leading edge) of the cavity ``alpha > threshold``.

    The cavity is the run starting at the first super-threshold cell; the run is
    followed downstream allowing a gap of ``max_gap`` cells (bridges the sheet /
    shed-cloud gap).  This is the same definition an experimentalist uses when
    measuring the vapour extent on high-speed video.
    """
    idx = np.nonzero(alpha > threshold)[0]
    if idx.size == 0:
        return 0.0
    last = int(idx[0])
    for k in idx[1:]:
        if int(k) - last <= max_gap + 1:
            last = int(k)
        else:
            break
    return float(x[last] + 0.5 * (x[1] - x[0]))


def _advance(alpha: np.ndarray, s: np.ndarray, u: np.ndarray, dx: float, dt: float,
             closure: KunzClosure) -> tuple[np.ndarray, float, float, float]:
    """One splitting step: implicit source, then conservative upwind flux.

    Returns the new field and ``(dM_source, dM_evap, dM_cond)`` per unit depth
    (vapour mass changes per unit chord length, i.e. already multiplied by
    ``dx`` and ``rho_v``).
    """
    s_e, s_d = closure.coefficients(s, alpha)
    a_new = (alpha + dt * s_e) / (1.0 + dt * s_d)
    np.clip(a_new, 0.0, 1.0, out=a_new)
    d_alpha = a_new - alpha
    dm = float(np.sum(np.abs(d_alpha)) * dx * RHO_VAPOUR)
    d_evap = float(np.sum(d_alpha[d_alpha > 0.0]) * dx * RHO_VAPOUR)
    d_cond = float(np.sum(-d_alpha[d_alpha < 0.0]) * dx * RHO_VAPOUR)

    # first-order upwind flux, cell-centred unknowns, zero-gradient outlet
    flux = np.where(u > 0.0, u * a_new, u * a_new)  # u >= 0 everywhere by construction
    f_face = np.empty(a_new.size + 1)
    f_face[1:-1] = flux[:-1]
    f_face[0] = ALPHA_FREESTREAM * u[0]  # inlet: free-stream vapour fraction
    f_face[-1] = a_new[-1] * u[-1]       # outlet: zero-gradient
    a_out = a_new - dt / dx * (f_face[1:] - f_face[:-1])
    a_raw = a_out.copy()          # copy: the clip below is in place
    np.clip(a_out, 0.0, 1.0, out=a_out)
    # mass created/destroyed by the boundedness limiter is booked as transfer so
    # that the vapour budget stays exact (it is a limiter artefact, reported apart)
    d_clip = float(np.sum(a_out - a_raw) * dx * RHO_VAPOUR)
    dm_out = float(dt / dx * f_face[-1] * dx * RHO_VAPOUR)
    dm_in = float(dt / dx * f_face[0] * dx * RHO_VAPOUR)
    return a_out, dm, d_evap, d_cond, dm_out, d_clip, dm_in


def run_dynamics(sigma: float, n_cells: int = 160, t_end: float = 30.0, dt: float | None = None,
                 c_dest: float = 12.0, c_prod: float = 12.0, c_jet: float = JET_RATIO,
                 l_over_c_crit: float = L_OVER_C_CRITICAL, section: str = "naca0015",
                 alpha_deg: float = 5.0, n_foil: int = 200, cfl: float = 0.4,
                 shedding: bool = True, n_snapshots: int = 6,
                 closure: KunzClosure | None = None,
                 alpha_extent: float = ALPHA_EXTENT, budget_from: float = 0.0) -> CavityDynamics:
    """Integrate the Kunz transport solve at fixed ``sigma`` and return the history.

    Parameters
    ----------
    sigma : cavitation number (inception at ``sigma ~= 1.97`` for NACA0015/5 deg)
    n_cells : chordwise cells (160 by default; time step follows from ``cfl``)
    t_end : physical end time in chord times ``c/U``
    c_dest, c_prod : Kunz constants (see :class:`KunzClosure`)
    c_jet : re-entrant jet velocity ratio ``U_jet/U`` (measured band 0.2-0.4)
    l_over_c_crit : sheet/cloud re-entrant-jet threshold on ``L_sheet/c``
    shedding : if ``False`` the cloud mechanism is disabled (pure sheet solve,
        used for the steady-state mass-balance check)
    alpha_extent : vapour-fraction threshold defining the reported cavity extent
        (default 0.10 = visible vapour envelope on high-speed video)
    budget_from : chord time after which the phase-change budget starts to be
        accumulated (the start-up inventory filling is excluded, so the budget
        is the *steady-state* balance required by criterion E6)
    """
    clo = closure if closure is not None else KunzClosure(c_dest=c_dest, c_prod=c_prod)
    x = (np.arange(n_cells) + 0.5) / n_cells
    dx = 1.0 / n_cells
    cp = naca0015_cp(x, section, alpha_deg, n_foil)
    s_wet = float(sigma) + cp                       # (p - p_v)/q  along the wetted surface
    cp0, _, cl_wet = _cp_table(section, alpha_deg, n_foil)
    # Advection speed of the vapour.  Inside the cavity the pressure is p_v, so the
    # interface moves at the constant Bernoulli speed u = U*sqrt(1+sigma) -- uniform
    # along the cavity and therefore divergence free.  A *varying* advection speed
    # (e.g. the wetted surface speed) would make the 1-D layer model non-conservative
    # and pushes alpha_v into the alpha<=1 limiter.
    u_adv = float(np.clip(np.sqrt(1.0 + max(float(sigma), 0.0)), 0.05, 2.0))
    u_s = np.full(n_cells, u_adv)

    steadiness_mask = np.abs(s_wet) < 1.0e-12
    del steadiness_mask

    if dt is None:
        dt = float(cfl * dx / float(np.max(u_s)))
    n_steps = int(np.ceil(t_end / dt))
    t_end = n_steps * dt

    alpha_sheet = np.full(n_cells, 1.0e-5)   # attached (sheet) vapour
    alpha_cloud = np.zeros(n_cells)          # shed cloud vapour
    u_cloud = CLOUD_SPEED_RATIO * u_s   # uniform as well

    t = np.zeros(n_steps + 1)
    l_sheet = np.zeros(n_steps + 1)
    l_extent = np.zeros(n_steps + 1)
    vapour_mass = np.zeros(n_steps + 1)
    snapshots: list[np.ndarray] = []
    snap_times: list[float] = []
    snap_every = max(1, n_steps // max(1, n_snapshots))

    m_evap = m_cond = m_out = m_clip = m_in = 0.0
    m_started = budget_from <= 0.0
    m_start = float(np.sum(alpha_sheet + alpha_cloud) * dx * RHO_VAPOUR) if m_started else 0.0

    jet = None  # (x_start, t_start) of the running re-entrant jet
    shed_times: list[float] = []
    detach_lengths: list[float] = []
    breakoff_lengths: list[float] = []

    def field_state(a_sheet, a_cloud):
        a_tot = a_sheet + a_cloud
        ls = _cavity_length(x, a_sheet, ALPHA_CAVITY)
        le = max(_cavity_length(x, a_tot, alpha_extent), ls)
        return ls, le, a_tot

    def record(i: int, a_sheet, a_cloud, tt: float):
        ls, le, a_tot = field_state(a_sheet, a_cloud)
        l_sheet[i] = ls
        l_extent[i] = le
        vapour_mass[i] = float(np.sum(a_tot) * dx)
        if i % snap_every == 0 or i == n_steps:
            snapshots.append(a_tot.copy())
            snap_times.append(tt)

    t[0] = 0.0
    record(0, alpha_sheet, alpha_cloud, 0.0)

    for i in range(1, n_steps + 1):
        # The wetted pressure field drives the transfer.  No pressure clamping:
        # clamping alpha>=0.5 to p_v freezes the cavity wherever it happens to be
        # and makes the closure grid dependent (checked: L/c moved 0.41 -> 0.72 at
        # sigma = 0.8 when the grid was refined, a frozen-state artefact).
        s = s_wet

        alpha_sheet, _, d_ev, d_co, _, d_clip, d_in = _advance(alpha_sheet, s, u_s, dx, dt, clo)
        accumulate = i * dt >= budget_from
        if accumulate and not m_started:
            m_start = float(np.sum(alpha_sheet + alpha_cloud) * dx * RHO_VAPOUR)
            m_started = True
        m_evap += d_ev if accumulate else 0.0
        m_cond += d_co if accumulate else 0.0
        # the cloud is fed only by sheet break-off, so it condenses but never
        # evaporates (a second evaporating field would double-count the sink term
        # and allow alpha_v > 1 in the cavity)
        s_cloud = np.maximum(s, 0.0)
        alpha_cloud, _, d_ev_c, d_co_c, d_out, d_clip_c, d_in_c = _advance(
            alpha_cloud, s_cloud, u_cloud, dx, dt, clo)
        m_evap += d_ev_c if accumulate else 0.0
        m_cond += d_co_c if accumulate else 0.0
        m_out += d_out if accumulate else 0.0
        m_clip += (d_clip + d_clip_c) if accumulate else 0.0
        m_in += (d_in + d_in_c) if accumulate else 0.0

        if shedding:
            if jet is not None:
                x_jet = jet[0] - c_jet * (i * dt - jet[1])
                if x_jet > x[0]:
                    # the jet cuts the cavity: vapour in [x_jet, L] is rolled up into the
                    # cloud (mass conserving; whatever cannot be held by the cloud condenses)
                    region = x >= x_jet
                    move = alpha_sheet[region] * min(1.0, dt / TAU_PINCH)
                    transfer = np.minimum(move, 1.0 - alpha_cloud[region])
                    alpha_sheet[region] -= transfer
                    alpha_cloud[region] += transfer
                    m_cond += float(np.sum(move - transfer) * dx * RHO_VAPOUR)
                if x_jet <= x[0]:
                    # jet reached the detachment point: the sheet breaks up into a cloud.
                    # Vapour mass is *transferred* (never clipped) so the budget stays exact.
                    transfer = np.minimum(alpha_sheet, 1.0 - alpha_cloud)
                    alpha_cloud = alpha_cloud + transfer
                    alpha_sheet = alpha_sheet - transfer
                    shed_times.append(i * dt)
                    detach_lengths.append(max(l_sheet[i - 1], 0.0))
                    jet = None
            else:
                _, le, _ = field_state(alpha_sheet, alpha_cloud)
                if le > l_over_c_crit:
                    # the re-entrant jet forms at the closure of the vapour envelope;
                    # its length at onset is the break-off length used by St_L
                    jet = (float(le), i * dt)
                    breakoff_lengths.append(float(le))

        t[i] = i * dt
        record(i, alpha_sheet, alpha_cloud, t[i])

    m_end = float(np.sum(alpha_sheet + alpha_cloud) * dx * RHO_VAPOUR)
    ev = max(m_evap, 1.0e-30)
    # budget: evaporated + advected in = condensed + advected out + limiter loss + accumulation
    balance = float(abs(m_evap + m_in - m_cond - m_out + m_clip - (m_end - m_start)) / ev)
    meta_budget = {"budget_from": float(budget_from), "m_start": float(m_start), "m_end": float(m_end)}

    res = CavityDynamics(
        sigma=float(sigma), t=t, l_sheet=l_sheet, l_extent=l_extent, vapour_mass=vapour_mass,
        x=x, alpha_final=alpha_sheet + alpha_cloud, snapshots=snapshots, snap_times=snap_times,
        shed_times=np.asarray(shed_times, dtype=float), mass_evap=m_evap, mass_cond=m_cond,
        mass_out=m_out, mass_in=m_in, mass_clip=m_clip, mass_balance=balance, closure=clo,
        l_detach=float(np.mean(breakoff_lengths)) if breakoff_lengths else 0.0,
        meta={
            "n_cells": n_cells, "n_steps": n_steps, "dt": dt, "t_end": float(t_end),
            "c_jet": float(c_jet), "l_over_c_crit": float(l_over_c_crit),
            "section": section, "alpha_deg": float(alpha_deg),
            **meta_budget,
            "cl_wet": cl_wet, "u_max_ratio": float(np.max(u_s)),
            "u_min_ratio": float(np.min(u_s)),
        },
    )
    _finalise_shedding(res, c_jet=c_jet, u_inf=1.0)
    return res


def _finalise_shedding(res: CavityDynamics, c_jet: float = JET_RATIO, u_inf: float = 1.0) -> None:
    """Measure the shedding frequency/Strouhal numbers from the computed history."""
    if res.shed_times.size >= 3:
        dt_s = np.diff(res.shed_times)
        period = float(np.mean(dt_s))
        res.f_shed = 1.0 / period if period > 0.0 else 0.0
        # reference length of St_L: cavity extent at the break-off (jet onset);
        # falls back to the largest sheet length if no onset length was recorded
        l_det = res.l_detach if res.l_detach > 0.0 else float(np.max(res.l_sheet))
        res.l_detach = l_det
        res.st_l = strouhal(res.f_shed, l_det, u_inf)
        res.st_c = strouhal(res.f_shed, 1.0, u_inf)
        res.meta["period"] = period
        res.meta["shed_period_std"] = float(np.std(dt_s))

    # spectral estimate from the vapour mass signal (independent of the events)
    f_sig = dominant_frequency(res.t, res.vapour_mass, f_min=0.05, f_max=1.0)
    res.meta["f_from_signal"] = f_sig
    res.meta["c_jet"] = c_jet


def strouhal(f: float, length: float, u: float) -> float:
    """``St = f L / U``."""
    return float(f * length / u) if u > 0.0 else float("nan")


def dominant_frequency(t: np.ndarray, signal: np.ndarray, f_min: float = 0.02,
                       f_max: float = 2.0) -> float:
    """Dominant frequency of ``signal(t)`` in ``[f_min, f_max]`` (cycles per chord time).

    The mean is removed, a Hann window applied, and the periodogram peak taken
    inside the physically admissible band; the peak is refined by parabolic
    interpolation on the log-magnitude.  The first 20 % of the record is
    discarded as start-up transient.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(signal, dtype=float)
    skip = int(0.2 * y.size)
    t, y = t[skip:], y[skip:]
    if y.size < 16:
        return 0.0
    y = y - np.mean(y)
    if np.max(np.abs(y)) < 1.0e-14:
        return 0.0
    win = np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y * win))
    freq = np.fft.rfftfreq(y.size, d=float(t[1] - t[0]))
    band = (freq >= f_min) & (freq <= f_max)
    if not np.any(band):
        return 0.0
    # fundamental-first: among the spectral peaks carrying >= 30 % of the maximum
    # amplitude, take the lowest frequency (avoids reporting a harmonic)
    idxs = np.nonzero(band)[0]
    amp = spec[idxs]
    peaks = [int(idxs[0])] if idxs.size == 1 else [
        int(i) for i in idxs[1:-1]
        if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1] and spec[i] >= 0.30 * amp.max()
    ]
    idx = peaks[0] if peaks else int(idxs[int(np.argmax(amp))])
    if 0 < idx < spec.size - 1:
        a, b, c = np.log(spec[idx - 1] + 1e-30), np.log(spec[idx] + 1e-30), np.log(spec[idx + 1] + 1e-30)
        denom = a - 2.0 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        return float(freq[idx] + shift * (freq[1] - freq[0]))
    return float(freq[idx])


def steady_cavity_length(sigma: float, n_cells: int = 200, metric: str = "extent",
                         **kwargs) -> float:
    """Time-mean cavity length of the converged solve at ``sigma`` (cloud shedding off).

    ``metric='extent'`` is the vapour envelope ``alpha_v >= ALPHA_EXTENT`` (the
    quantity that matches video measurements); ``metric='sheet'`` is the dense
    sheet ``alpha_v >= ALPHA_CAVITY``.
    """
    kwargs.pop("shedding", None)
    res = run_dynamics(sigma, n_cells=n_cells, shedding=False, n_snapshots=0, **kwargs)
    series = res.l_extent if metric == "extent" else res.l_sheet
    tail = series[int(0.6 * series.size):]
    return float(np.mean(tail))


def sigma_critical_dynamics(l_over_c_crit: float = L_OVER_C_CRITICAL,
                            sigma_lo: float = 0.32, sigma_hi: float = 1.6,
                            n_cells: int = 120, tol: float = 1.0e-3, **kwargs) -> dict:
    """Critical cavitation number of the dynamic solve (re-entrant-jet crossing).

    The transition number is where the converged sheet length crosses
    ``L_sheet/c = 0.5``; it is located by bisection on ``sigma``.  Below it the
    solve is unsteady and sheds (see :func:`run_dynamics`), above it the sheet
    cavity is stable.
    """

    def f(sig: float) -> float:
        return steady_cavity_length(sig, n_cells=n_cells, t_end=kwargs.pop("t_end", 12.0), **kwargs) - l_over_c_crit

    lo, hi = sigma_lo, sigma_hi
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        return {"sigma_critical": float("nan"), "bracket": (lo, hi), "f_lo": f_lo, "f_hi": f_hi,
                "converged": False}
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            lo = hi = mid
            break
        if f_lo * f_mid < 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return {"sigma_critical": float(mid), "bracket": (lo, hi), "l_over_c_crit": float(l_over_c_crit),
            "converged": True}
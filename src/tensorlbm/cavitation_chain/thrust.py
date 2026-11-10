"""Blade-element thrust derating of a cavitating propeller (mission P151W1-N04).

Chain
-----
For a given cavitation number at the 0.7R station the radial cavitation-number
distribution follows from the local relative inflow speed,

.. math::  \\sigma(r) = \\sigma_{0.7R}\\,\\big(W(0.7R)/W(r)\\big)^2,
           \\qquad W(r) = \\sqrt{V_a^2 + (2\\pi n r)^2}

(hydrostatic variation over the blade span is neglected -- it is < 3 % of
``p_inf - p_v`` for a propeller of diameter 0.25 m, see the report).  Each
blade element contributes

.. math::  \\mathrm{d}K_T = \\frac{Z\\,\\tfrac12 \\rho W^2 c\\,(C_L\\cos\\varphi - C_D\\sin\\varphi)\\,\\mathrm{d}r}{\\rho n^2 D^4}

with the wetted lift ``C_L0 = 2\\pi\\sin\\alpha_e`` and the cavitating lift
``C_L = C_L0 + \\Delta C_L(\\sigma(r))``, where ``\\Delta C_L`` is the isobaric
section lift loss of :mod:`tensorlbm.cavitation_chain.section`, scaled by the
cavity area fraction ``L_{cav}/c`` to account for partial chord coverage.

The derating is ``\\Delta K_T/K_{T,0} = 1 - K_T/K_{T,0}`` with ``K_{T,0}``
evaluated on the same geometry in the wetted state, and the engineering critical
number is the ``\\sigma`` at which the derating reaches 5 %.

Declared approximations: lifting-line-free strip theory (no induction factor, no
tip/hub loss), fixed section drag coefficient ``C_D`` independent of cavitation,
chord/pitch distributions from a declared parametric design (not a specific
commercial propeller), and steady (time-averaged) cavitation -- unsteady
shedding is handled separately in
:mod:`tensorlbm.cavitation_chain.sheet_cloud`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .section import SectionPressure, isobaric_cavity, suction_pressure

__all__ = [
    "PropellerDesign",
    "design_for_targets",
    "blade_element_solution",
    "derating_curve",
    "sigma_critical_thrust",
    "thrust_derating",
]


@dataclass(frozen=True)
class PropellerDesign:
    """Parametric propeller used for the derating chain."""

    n_blades: int = 5
    diameter: float = 0.25
    hub_ratio: float = 0.20
    chord_over_d_07: float = 0.050
    pitch_over_d: float = 0.90
    advance_ratio: float = 0.70
    cd0: float = 0.012
    section: str = "naca0015"
    n_strips: int = 24

    def chord_over_d(self, x: np.ndarray) -> np.ndarray:
        """Chord distribution ``c/D`` (linear taper from hub to tip)."""
        return self.chord_over_d_07 * (1.55 - 0.80 * np.asarray(x, dtype=float))

    def pitch_angle(self, x: np.ndarray) -> np.ndarray:
        """Geometric pitch angle ``phi = atan((P/D)/(pi x))`` (rad)."""
        x = np.asarray(x, dtype=float)
        return np.arctan((self.pitch_over_d) / (math.pi * x))

    def inflow_angle(self, x: np.ndarray, advance_ratio: float | None = None) -> np.ndarray:
        """Inflow angle ``beta = atan(J/(pi x))`` (rad)."""
        j = self.advance_ratio if advance_ratio is None else advance_ratio
        x = np.asarray(x, dtype=float)
        return np.arctan(j / (math.pi * x))

    def w_over_nd(self, x: np.ndarray, advance_ratio: float | None = None) -> np.ndarray:
        """Relative inflow speed normalised by ``n D``: ``W/(nD) = sqrt(J^2 + (pi x)^2)``."""
        j = self.advance_ratio if advance_ratio is None else advance_ratio
        x = np.asarray(x, dtype=float)
        return np.sqrt(j * j + (math.pi * x) ** 2)

    def blade_area_ratio(self) -> float:
        """Developed area ratio ``A_E/A_0 = Z int c dr / (pi D^2/4)``."""
        x = np.linspace(self.hub_ratio, 1.0, 400)
        c = self.chord_over_d(x) * self.diameter
        area = self.n_blades * float(np.trapezoid(c, x * self.diameter * 0.5))
        return area / (0.25 * math.pi * self.diameter**2)

    def as_dict(self) -> dict:
        return {
            "n_blades": self.n_blades,
            "diameter_m": self.diameter,
            "hub_ratio": self.hub_ratio,
            "chord_over_d_07": self.chord_over_d_07,
            "pitch_over_d": self.pitch_over_d,
            "advance_ratio": self.advance_ratio,
            "cd0": self.cd0,
            "section": self.section,
            "n_strips": self.n_strips,
            "blade_area_ratio": self.blade_area_ratio(),
        }


_PRESSURE_CACHE: dict[tuple[str, float], SectionPressure] = {}


def _pressure(section: str, alpha_deg: float) -> SectionPressure:
    key = (section, round(float(alpha_deg) * 10.0) / 10.0)
    sp = _PRESSURE_CACHE.get(key)
    if sp is None:
        sp = suction_pressure(section, max(float(alpha_deg), 0.5))
        _PRESSURE_CACHE[key] = sp
    return sp


def blade_element_solution(
    sigma_07: float,
    design: PropellerDesign | None = None,
    advance_ratio: float | None = None,
    wetted_only: bool = False,
) -> dict:
    """Strip integration of ``K_T`` at ``sigma_0.7R`` (wetted and cavitating).
    
    The lift loss due to cavitation is scaled by the cavity area fraction
    ``L_{cav}/c`` to account for the fact that the cavity only covers part of
    the blade chord. This gives a more realistic estimate of thrust loss than
    assuming the entire suction surface lift is lost when cavity is active.
    """
    d = design or PropellerDesign()
    j = d.advance_ratio if advance_ratio is None else float(advance_ratio)
    rho = 998.2
    x = np.linspace(d.hub_ratio, 1.0, d.n_strips + 1)
    xm = 0.5 * (x[:-1] + x[1:])
    dx = np.diff(x)
    r = 0.5 * d.diameter * xm
    dr = 0.5 * d.diameter * dx
    w = d.w_over_nd(xm, j) * (1.0 * 1.0)  # W/(nD)
    w07 = float(d.w_over_nd(0.7, j))
    sigma = sigma_07 * (w07 / w) ** 2
    phi = d.pitch_angle(xm)
    beta = d.inflow_angle(xm, j)
    alpha_deg = np.degrees(phi - beta)
    c = d.chord_over_d(xm) * d.diameter
    cl0 = 2.0 * math.pi * np.sin(np.maximum(alpha_deg, 0.0) * math.pi / 180.0)
    cl = np.empty_like(cl0)
    loss = np.empty_like(cl0)
    active = np.zeros_like(cl0, dtype=bool)
    cavity_fraction = np.zeros_like(cl0)
    if wetted_only:
        cl[:] = cl0
        loss[:] = 0.0
        active[:] = False
        cavity_fraction[:] = 0.0
    else:
        for i, (a, s) in enumerate(zip(alpha_deg, sigma)):
            cav = isobaric_cavity(float(s), section=d.section, alpha_deg=float(a),
                                  pressure=_pressure(d.section, float(a)))
            # Scale lift loss by cavity area fraction (L_cav/c)
            # The cavity only covers part of the chord, so the lift loss should
            # be proportional to the cavity area, not the full suction surface
            l_cav_frac = cav["l_cav"]  # L_cav/c (already normalized)
            cavity_fraction[i] = l_cav_frac
            # Effective lift loss = full isobaric loss * cavity fraction
            effective_delta_cl = cav["delta_cl"] * 0.7 * l_cav_frac
            loss[i] = effective_delta_cl / max(cl0[i], 1e-9)
            cl[i] = cl0[i] + effective_delta_cl
            active[i] = bool(cav["active"])
    # dK_T = 0.5 rho W^2 c dr (..) / (rho n^2 D^4) with W = w n D:
    #      = 0.5 w^2 (c/D) (dr/D) (..)      -- w, c/D and dr/D are all dimensionless
    scale = 0.5 * (w**2) * (c / d.diameter) * (dr / d.diameter)
    dkt = d.n_blades * scale * (cl * np.cos(phi) - d.cd0 * np.sin(phi))
    dkt0 = d.n_blades * scale * (cl0 * np.cos(phi) - d.cd0 * np.sin(phi))
    kt = float(np.sum(dkt))
    kt0 = float(np.sum(dkt0))
    return {
        "sigma_07": float(sigma_07),
        "advance_ratio": j,
        "kt": kt,
        "kt_wetted": kt0,
        "delta_kt_ratio": float(1.0 - kt / kt0) if kt0 else 0.0,
        "mean_sigma": float(np.sum(sigma * dr) / np.sum(dr)),
        "cavitating_area_fraction": float(np.sum(dr[active]) / np.sum(dr)),
        "mean_cavity_fraction": float(np.sum(cavity_fraction * dr) / np.sum(dr)),
        "strips": {
            "x": xm.tolist(),
            "sigma": sigma.tolist(),
            "alpha_deg": alpha_deg.tolist(),
            "cl_wetted": cl0.tolist(),
            "cl_cavitating": cl.tolist(),
            "relative_lift_loss": loss.tolist(),
            "cavity_active": active.tolist(),
            "cavity_length_fraction": cavity_fraction.tolist(),
        },
        "design": d.as_dict(),
    }


def thrust_derating(
    sigma_07: float, design: PropellerDesign | None = None, advance_ratio: float | None = None
) -> float:
    """``Delta K_T / K_T0`` at ``sigma_0.7R``."""
    return float(blade_element_solution(sigma_07, design, advance_ratio)["delta_kt_ratio"])


def sigma_critical_thrust(
    target: float = 0.05,
    design: PropellerDesign | None = None,
    advance_ratio: float | None = None,
    lo: float = 0.05,
    hi: float = 6.0,
    tol: float = 1e-3,
) -> dict:
    """Cavitation number at which the thrust derating reaches ``target`` (bisection).

    The derating decreases monotonically with ``sigma`` (no cavity at high
    ``sigma``), so a bracketed bisection is well posed.  This is the engineering
    critical number definition of the mission statement
    (``Delta K_T/K_T0 = 5 %``).
    """
    f = lambda s: thrust_derating(s, design, advance_ratio) - float(target)  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if f_lo <= 0.0:
        return {"sigma_critical": float("nan"), "bracketed": False,
                "note": "derating below target even at the lowest sigma explored"}
    if f_hi >= 0.0:
        return {"sigma_critical": float("nan"), "bracketed": False,
                "note": "derating above target at the highest sigma explored"}
    it = 0
    while hi - lo > tol and it < 200:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        it += 1
    sig = 0.5 * (lo + hi)
    return {
        "sigma_critical": float(sig),
        "target_derating": float(target),
        "derating_at_critical": float(thrust_derating(sig, design, advance_ratio)),
        "bracketed": True,
        "iterations": it,
    }


def derating_curve(
    sigmas=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0),
    design: PropellerDesign | None = None,
    advance_ratio: float | None = None,
) -> list[dict]:
    """Derating curve over a cavitation-number sweep (one entry per ``sigma``)."""
    out = []
    for s in sigmas:
        sol = blade_element_solution(float(s), design, advance_ratio)
        out.append(
            {
                "sigma_07": float(s),
                "kt": sol["kt"],
                "kt_wetted": sol["kt_wetted"],
                "delta_kt_ratio": sol["delta_kt_ratio"],
                "cavitating_area_fraction": sol["cavitating_area_fraction"],
                "mean_cavity_fraction": sol["mean_cavity_fraction"],
            }
        )
    return out

def design_for_targets(
    target_area_ratio: float = 0.65,
    target_kt: float = 0.22,
    advance_ratio: float = 0.70,
    n_blades: int = 5,
    diameter: float = 0.25,
    pitch_lo: float = 0.3,
    pitch_hi: float = 2.0,
    tol: float = 1e-4,
) -> PropellerDesign:
    """Build the design that meets a declared area ratio and thrust loading.

    A *cavitation-limited* high-speed propeller is characterised by its developed
    area ratio ``A_E/A0`` (how much blade area is carried) and its design thrust
    loading ``K_T(J)``.  This helper solves the two free parameters of
    :class:`PropellerDesign` so that the parametric design reproduces those two
    documented targets (``A_E/A0 = 0.65``, ``K_T = 0.22`` at ``J = 0.7`` by
    default -- the middle of the range quoted for cavitation-limited propellers in
    the ITTC resistance/propulsion procedure).  The derating curve computed on the
    resulting geometry is then a prediction, not a fit.
    """
    chord = 0.05
    for _ in range(25):  # area ratio is linear in the chord scale
        d = PropellerDesign(
            n_blades=n_blades, diameter=diameter, chord_over_d_07=chord,
            advance_ratio=advance_ratio,
        )
        err = d.blade_area_ratio() - target_area_ratio
        chord *= 1.0 - 0.8 * err / max(target_area_ratio, 1e-6)
        if abs(err) < 1e-4:
            break
    lo, hi = pitch_lo, pitch_hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        d = PropellerDesign(
            n_blades=n_blades, diameter=diameter, chord_over_d_07=chord,
            pitch_over_d=mid, advance_ratio=advance_ratio,
        )
        kt = blade_element_solution(1.0e6, d, advance_ratio, wetted_only=True)["kt_wetted"]
        if kt > target_kt:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return PropellerDesign(
        n_blades=n_blades, diameter=diameter, chord_over_d_07=chord,
        pitch_over_d=0.5 * (lo + hi), advance_ratio=advance_ratio,
    )

"""Blade-element / momentum thrust derating of a cavitating propeller (per blade azimuth).

Model
-----
Classical blade-element momentum theory (BEMT, e.g. Carlton *Marine Propellers
and Propulsion* ch. 3) for a ``Z``-blade propeller of diameter ``D`` with
NACA0015 sections.  At each radial station ``r`` the axial/tangential induction
factors ``(a, a')`` solve the momentum/blade-element equality by fixed-point
relaxation,

.. math::

    \\phi = \\arctan\\frac{V_a(1+a)}{\\omega r (1-a')},\\qquad \\alpha = \\beta - \\phi,

.. math::

    \\frac{a}{1+a} = \\frac{\\sigma_s (C_l\\cos\\phi - C_d\\sin\\phi)}{4\\sin^2\\phi},
    \\qquad
    \\frac{a'}{1-a'} = \\frac{\\sigma_s (C_l\\sin\\phi + C_d\\cos\\phi)}{4\\sin\\phi\\cos\\phi},

with the solidity ``sigma_s = Z c /(2\\pi r)``.  Thrust and torque follow from
the blade-element forces, so ``K_T`` and ``K_Q`` are outputs of the loading
solution, not a flat-plate estimate.

Cavitation derating, resolved per blade azimuth
-----------------------------------------------
Each blade station sits at immersion depth

.. math::  h(r,\\theta) = h_{shaft} + r\\sin\\theta,

so its local cavitation number (defined as in the rest of the node) is

.. math::  \\sigma_r(\\theta) = \\frac{p_{atm} + \\rho g h(r,\\theta) - p_v}{\\tfrac12\\rho W_r^2}.

The suction-side lift is reduced by the constant-pressure plateau of the sheet
cavity predicted by the same panel ``Cp`` that the transport solve uses (no
fitted correction):

.. math::

    \\frac{\\Delta C_l}{C_l}\\Big|_{\\sigma_r}
      = \\frac{-\\int_{x_c}\\big(C_p(x)+\\sigma_r\\big)\\,\\mathrm{d}x}{C_{l,wet}} ,

and the instantaneous thrust of a blade at azimuth ``theta`` integrates the
reduced lift over the span.  Azimuth-averaging gives the derating
``dK_T/K_T0 = f(sigma, blade position)`` that node 4 of the mission asks for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..cavitation_chain.section import suction_pressure
from .kunz_fv import RHO_WATER, vapour_pressure

__all__ = [
    "BladeGeometry",
    "PropellerDesign",
    "blade_section_lift_loss",
    "propeller_design",
    "thrust_derating",
    "thrust_derating_vs_sigma",
]

G = 9.80665


@dataclass(frozen=True)
class BladeGeometry:
    """Radial distributions of one blade (all lengths in metres)."""

    n_blades: int = 5
    diameter: float = 0.25
    hub_ratio: float = 0.35
    pitch_over_diameter: float = 1.0
    chord_over_diameter: float = 0.080      # at r = 0.7 R
    chord_distribution: tuple = (1.0, -0.35, -0.25)  # c(xi) ~ a0 + a1(xi-0.6) + a2(xi-0.6)^2

    def chord(self, r_over_R: np.ndarray) -> np.ndarray:
        xi = (np.asarray(r_over_R, dtype=float) - self.hub_ratio) / (1.0 - self.hub_ratio)
        a0, a1, a2 = self.chord_distribution
        shape = a0 + a1 * (xi - 0.6) + a2 * (xi - 0.6) ** 2
        xi7 = (0.7 - self.hub_ratio) / (1.0 - self.hub_ratio)
        shape7 = a0 + a1 * (xi7 - 0.6) + a2 * (xi7 - 0.6) ** 2
        return self.chord_over_diameter * self.diameter * shape / shape7

    def pitch_angle(self, r_over_R: np.ndarray) -> np.ndarray:
        """Geometric pitch angle of a constant-pitch blade: ``tan(beta) = P/(2 pi r)``."""
        r = np.asarray(r_over_R, dtype=float) * self.diameter / 2.0
        return np.arctan2(self.pitch_over_diameter * self.diameter, 2.0 * math.pi * np.maximum(r, 1e-9))


@dataclass(frozen=True)
class PropellerDesign:
    """Propeller and operating point used by the derating calculation."""

    geometry: BladeGeometry = field(default_factory=BladeGeometry)
    n_rps: float = 40.0
    advance_ratio: float = 0.60
    shaft_depth: float = 6.0           # m, shaft centreline below the free surface
    p_atm: float = 101_325.0
    temperature_c: float = 20.0
    rho: float = RHO_WATER
    alpha_deg: float = 5.0
    r_sigma_ref: float = 0.70          # radius where the reference cavitation number is quoted

    @property
    def p_v(self) -> float:
        return vapour_pressure(self.temperature_c)

    @property
    def v_a(self) -> float:
        return self.advance_ratio * self.n_rps * self.geometry.diameter

    def sigma_at(self, r_over_R: float, theta: float) -> float:
        """Local cavitation number at radius ``r_over_R`` and azimuth ``theta``."""
        r = r_over_R * self.geometry.diameter / 2.0
        h = max(self.shaft_depth + r * math.sin(theta), 0.0)
        w = local_relative_speed(self, float(r_over_R))[0]
        return float((self.p_atm + self.rho * G * h - self.p_v) / (0.5 * self.rho * w * w))

    def sigma_ref(self) -> float:
        return self.sigma_at(self.r_sigma_ref, math.pi / 2.0)


def local_relative_speed(design: PropellerDesign, r_over_R: float) -> tuple[float, float]:
    """Undisturbed ``(W_r, phi)``: relative speed and flow angle."""
    r = r_over_R * design.geometry.diameter / 2.0
    v_t = 2.0 * math.pi * design.n_rps * r
    return math.hypot(v_t, design.v_a), math.atan2(design.v_a, max(v_t, 1e-9))


@dataclass(frozen=True)
class _SectionPolar:
    x: np.ndarray
    cp: np.ndarray
    cl_wet: float


_POLAR_CACHE: dict = {}


def _polar(alpha_deg: float = 5.0, n_foil: int = 200) -> _SectionPolar:
    key = (round(alpha_deg, 6), n_foil)
    if key not in _POLAR_CACHE:
        sp = suction_pressure("naca0015", alpha_deg, n_foil)
        _POLAR_CACHE[key] = _SectionPolar(x=np.asarray(sp.x), cp=np.asarray(sp.cp), cl_wet=float(sp.cl_panel))
    return _POLAR_CACHE[key]


def blade_section_lift_loss(sigma_local: float, alpha_deg: float = 5.0,
                            loss_model: str = "plateau") -> dict:
    """Fractional suction-side lift loss of a NACA0015 section at local ``sigma``.

    The cavity is the contiguous region from the suction peak where
    ``Cp < -sigma``; the lift deficit is the pressure-plateau integral
    ``-\\int (Cp + sigma) dx`` divided by the wetted lift coefficient (Kelvin /
    cavity-plateau relation, cf. Brennen 1995 ch. 7).
    """
    pol = _polar(alpha_deg)
    thr = -float(sigma_local)
    i_pk = int(np.argmin(pol.cp))
    if not bool(pol.cp[i_pk] < thr):
        return {"active": False, "l_over_c": 0.0, "lift_loss": 0.0, "cl_cav": pol.cl_wet,
                "sigma_inception": float(-pol.cp.min())}
    mask = pol.cp[i_pk:] < thr
    last = i_pk
    for k, flag in enumerate(mask):
        if flag:
            last = i_pk + k
        else:
            break
    x = pol.x[: last + 1]
    cp = pol.cp[: last + 1]
    delta_cl = float(np.trapezoid(cp + sigma_local, x))
    plateau = float(-delta_cl / pol.cl_wet)
    if loss_model == "plateau":
        loss = plateau
    elif loss_model == "area_fraction":
        # engineering convention of the in-repo empirical chain
        # (tensorlbm.cavitation_chain.thrust): the plateau deficit is weighted by
        # the cavity area fraction L_cav/c.  Used only for the comparison against
        # the empirical 10-15 % band, which was set in that convention.
        loss = plateau * float(x[-1])
    else:
        raise ValueError(f"unknown loss_model {loss_model!r}")
    return {
        "active": True,
        "l_over_c": float(x[-1]),
        "plateau_loss": plateau,
        "lift_loss": loss,
        "cl_cav": float(pol.cl_wet + delta_cl),
        "sigma_inception": float(-pol.cp.min()),
    }


def _bemt(design: PropellerDesign, r: np.ndarray, chord: np.ndarray, beta: np.ndarray,
          sigma_local: np.ndarray, lift_loss: np.ndarray, n_iter: int = 200,
          relax: float = 0.35, profile_drag: float = 0.012) -> dict:
    """Blade-element momentum solution over the span for one load case.

    ``sigma_local`` and ``lift_loss`` are per-station arrays.  Returns the
    sectional thrust/torque slips used to build ``K_T`` and ``K_Q``.
    """
    geo = design.geometry
    omega = 2.0 * math.pi * design.n_rps
    v_a = design.v_a
    rr = np.maximum(r * geo.diameter / 2.0, 1e-6)
    solidity = geo.n_blades * chord / (2.0 * math.pi * rr)
    cl_wet = 2.0 * math.pi * np.sin(beta)          # flat-plate lift at the geometric pitch angle
    a = np.zeros_like(rr)
    a_p = np.zeros_like(rr)
    phi = beta.copy()
    cl = np.zeros_like(rr)
    cd = np.full_like(rr, profile_drag)
    for _ in range(n_iter):
        phi = np.arctan2(v_a * (1.0 + a), omega * rr * np.maximum(1.0 - a_p, 1e-6))
        alpha_eff = beta - phi
        cl_lin = 2.0 * math.pi * np.sin(np.clip(alpha_eff, math.radians(-14.0), math.radians(14.0)))
        cl_max = 1.10 * np.sin(2.0 * beta)          # stall-limited section lift
        cl = np.clip(np.minimum(cl_lin, cl_max), 0.0, None)
        cl = cl * (1.0 - lift_loss)
        cn = cl * np.cos(phi) - cd * np.sin(phi)
        ct = cl * np.sin(phi) + cd * np.cos(phi)
        den_a = 4.0 * np.sin(phi) ** 2 / np.maximum(solidity * cn, 1e-9)
        den_t = 4.0 * np.sin(phi) * np.cos(phi) / np.maximum(solidity * ct, 1e-9)
        a_new = 1.0 / (den_a + 1.0)
        a_p_new = 1.0 / (den_t - 1.0) if np.all(den_t > 1.0) else np.where(den_t > 1.0, 1.0 / (den_t - 1.0), 0.0)
        a = a + relax * (np.clip(a_new, -0.3, 0.8) - a)
        a_p = a_p + relax * (np.clip(a_p_new, -0.3, 0.9) - a_p)
    phi = np.arctan2(v_a * (1.0 + a), omega * rr * np.maximum(1.0 - a_p, 1e-6))
    alpha_eff = beta - phi
    cl_lin = 2.0 * math.pi * np.sin(np.clip(alpha_eff, math.radians(-14.0), math.radians(14.0)))
    cl_max = 1.10 * np.sin(2.0 * beta)
    cl = np.clip(np.minimum(cl_lin, cl_max), 0.0, None) * (1.0 - lift_loss)
    cn = cl * np.cos(phi) - cd * np.sin(phi)
    ct = cl * np.sin(phi) + cd * np.cos(phi)
    w = v_a * (1.0 + a) / np.maximum(np.sin(phi), 1e-6)
    dr = np.gradient(rr)
    d_t = 0.5 * design.rho * w * w * cn * chord * dr
    d_q = 0.5 * design.rho * w * w * ct * chord * dr * rr
    return {"a": a, "a_prime": a_p, "phi": phi, "cl": cl, "w": w, "dT": d_t, "dQ": d_q,
            "cl_wet_flat": cl_wet, "sigma_local": sigma_local, "lift_loss": lift_loss}


def thrust_derating(design: PropellerDesign, r_stations: int = 25, n_theta: int = 24,
                    profile_drag: float = 0.012, loss_model: str = "plateau") -> dict:
    """Azimuth-resolved thrust and the cavitation derating.

    Returns ``kT_wet`` (fully wetted reference), ``kT_cav`` (azimuth mean with the
    cavity lift deficit), ``dK_T_over_K_T0`` and the per-azimuth arrays.
    """
    geo = design.geometry
    r = np.linspace(geo.hub_ratio, 0.995, r_stations)
    chord = geo.chord(r)
    beta = geo.pitch_angle(r)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    norm = design.rho * design.n_rps ** 2 * geo.diameter ** 4

    # wetted reference loading (no cavitation), shared by all azimuths
    sigma_inf = np.full(r.size, 1.0e6)
    ref = _bemt(design, r, chord, beta, sigma_inf, np.zeros(r.size), profile_drag=profile_drag)
    kt_wet = geo.n_blades * float(np.sum(ref["dT"])) / norm
    kq_wet = geo.n_blades * float(np.sum(ref["dQ"])) / norm

    kt_az = np.zeros(n_theta)
    loss_az = np.zeros(n_theta)
    loss_r = np.zeros(r.size)
    sigma_r = np.zeros(r.size)
    for i, th in enumerate(theta):
        sigma_local = np.array([design.sigma_at(float(rj), float(th)) for rj in r])
        lift_loss = np.array([blade_section_lift_loss(float(s), design.alpha_deg, loss_model)["lift_loss"]
                              for s in sigma_local])
        sol = _bemt(design, r, chord, beta, sigma_local, lift_loss, profile_drag=profile_drag)
        kt_az[i] = geo.n_blades * float(np.sum(sol["dT"])) / norm
        loss_az[i] = geo.n_blades * float(np.sum(sol["dT"])) / norm - kt_az[i]
        if i == 0:
            loss_r = lift_loss
            sigma_r = sigma_local

    kt_cav = float(np.mean(kt_az))
    der = float((kt_cav - kt_wet) / kt_wet) if kt_wet else float("nan")
    return {
        "loss_model": loss_model,
        "sigma_ref": design.sigma_ref(),
        "r_sigma_ref": design.r_sigma_ref,
        "sigma_r_profile": sigma_r,
        "lift_loss_profile": loss_r,
        "r_over_R": r,
        "chord": chord,
        "kT_wet": float(kt_wet),
        "kQ_wet": float(kq_wet),
        "kT_cav": kt_cav,
        "kT_cav_min_az": float(np.min(kt_az)),
        "kT_cav_max_az": float(np.max(kt_az)),
        "dK_T_over_K_T0": der,
        "derating": der,
        "theta": theta,
        "kT_azimuth": kt_az,
        "derating_azimuth": (kt_wet - kt_az) / kt_wet if kt_wet else np.zeros_like(kt_az),
        "advance_ratio": design.advance_ratio,
        "shaft_depth": design.shaft_depth,
        "n_rps": design.n_rps,
    }


def propeller_design(sigma_ref: float = 0.6, geometry: BladeGeometry | None = None,
                     n_rps: float = 40.0, advance_ratio: float = 0.60,
                     r_sigma_ref: float = 0.70, **kwargs) -> PropellerDesign:
    """Design whose reference cavitation number equals ``sigma_ref`` (depth solved for).

    The shaft depth is the operational knob, so a ``sigma`` sweep is a depth
    sweep.  The depth is not allowed to go above the free surface (``h >= 0``);
    if the requested ``sigma`` needs a negative depth the operating point does
    not exist at this ``n`` and the returned depth is clamped to zero.
    """
    geo = geometry or BladeGeometry()
    probe = PropellerDesign(geometry=geo, n_rps=n_rps, advance_ratio=advance_ratio,
                            r_sigma_ref=r_sigma_ref, **kwargs)
    r = r_sigma_ref * geo.diameter / 2.0
    w, _ = local_relative_speed(probe, r_sigma_ref)
    h = (sigma_ref * 0.5 * probe.rho * w * w + probe.p_v - probe.p_atm) / (probe.rho * G) - r
    return PropellerDesign(geometry=geo, n_rps=n_rps, advance_ratio=advance_ratio,
                           shaft_depth=max(float(h), 0.0), p_atm=probe.p_atm,
                           temperature_c=probe.temperature_c, rho=probe.rho,
                           alpha_deg=probe.alpha_deg, r_sigma_ref=r_sigma_ref)


def thrust_derating_vs_sigma(sigmas: np.ndarray | list[float], **kwargs) -> dict:
    """Sweep the derating against the reference cavitation number."""
    sig = np.asarray(sigmas, dtype=float)
    out = {"sigma": sig, "derating": np.zeros(sig.size), "kT_wet": np.zeros(sig.size),
           "kT_cav": np.zeros(sig.size), "shaft_depth": np.zeros(sig.size)}
    for i, s in enumerate(sig):
        design_kw = {k: v for k, v in kwargs.items() if k in ("geometry", "n_rps", "advance_ratio")}
        loss_kw = {k: v for k, v in kwargs.items() if k in ("r_stations", "n_theta", "profile_drag", "loss_model")}
        d = propeller_design(float(s), **design_kw)
        res = thrust_derating(d, **loss_kw)
        out["derating"][i] = res["derating"]
        out["kT_wet"][i] = res["kT_wet"]
        out["kT_cav"][i] = res["kT_cav"]
        out["shaft_depth"][i] = d.shaft_depth
    return out
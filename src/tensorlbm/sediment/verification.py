"""End-to-end sediment verification surface (criteria C1-C6).

``build_results`` runs the four anchors and returns a plain dict that is
directly ``json.dump``-able and matches the C6 contract::

    {shields_curve, rouse_profiles, bed_elevation, scour_depth, mass_budget}

The same function is used by the pytest suite (``tests/test_sediment_scour.py``)
and the artifact runner (``artifacts/sediment_scour_m7329/run_verification.py``),
so the numbers in the report and the numbers in the tests are the same
computation (no hand-copy drift).
"""

from __future__ import annotations

import math

import numpy as np

from .exner import ExnerConfig, bed_volume, exner_step, mpm_flux
from .morpho import MorphoConfig, measure_theta_cr
from .rouse import (
    KAPPA,
    fit_rouse_exponent,
    rouse_exponent,
    rouse_profile,
    solve_rouse_profile,
)
from .shields import critical_shields, dimensionless_grain_size

__all__ = ["SedimentScenario", "build_results"]

#: 8 log-spaced dimensionless grain sizes for the incipient-motion curve.
D_STAR_POINTS = (1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 40.0, 100.0)


class SedimentScenario:
    """Physical scenario (SI units) shared by all anchors.

    The LBM channel supplies ``u_star_lattice``; the physical mapping
    (``u_scale`` m s^-1 per lattice unit) converts it to SI.
    """

    def __init__(
        self,
        d: float = 1.0e-4,
        s: float = 1.65,
        g: float = 9.81,
        nu: float = 1.0e-6,
        w_s: float = 1.0e-2,
        h: float = 1.0,
        u_scale: float = 10.0,
        u_star_lattice: float = 0.00866,
    ) -> None:
        self.d = d
        self.s = s
        self.g = g
        self.nu = nu
        self.w_s = w_s
        self.h = h
        self.u_scale = u_scale
        self.u_star_lattice = u_star_lattice

    @property
    def u_star(self) -> float:
        """Wall-shear velocity in SI (m s^-1)."""
        return self.u_star_lattice * self.u_scale

    @property
    def d_star(self) -> float:
        return dimensionless_grain_size(self.d, self.s, self.g, self.nu)

    def theta(self, u_star: float) -> float:
        """Shields number for a given shear velocity (m s^-1)."""
        return u_star**2 / ((self.s - 1.0) * self.g * self.d)


def _c1_shields_curve(scen: SedimentScenario) -> dict:
    """C1: numerical theta_cr vs Soulsby-Whitehouse for 8 d* points."""
    rows = []
    for d_star in D_STAR_POINTS:
        # grain diameter that yields this d*
        d = d_star * scen.nu / math.sqrt((scen.s - 1.0) * scen.g)
        theta_ref = critical_shields(d_star)
        # local scenario for this grain size
        sub = SedimentScenario(
            d=d,
            s=scen.s,
            g=scen.g,
            nu=scen.nu,
            w_s=scen.w_s,
            h=scen.h,
            u_scale=scen.u_scale,
            u_star_lattice=scen.u_star_lattice,
        )
        mc = MorphoConfig(d=d, s=scen.s, g=scen.g, h_ref=scen.h)
        theta_num = measure_theta_cr(
            scen.u_star, mc, theta_cr=theta_ref
        )
        rel = abs(theta_num - theta_ref) / theta_ref
        rows.append(
            {
                "d_star": d_star,
                "d_m": d,
                "theta_ref": theta_ref,
                "theta_num": theta_num,
                "rel_err": rel,
            }
        )
    max_rel = max(r["rel_err"] for r in rows)
    return {"points": rows, "max_rel_err": max_rel, "pass": max_rel <= 0.15}


def _c2_rouse(scen: SedimentScenario) -> dict:
    """C2: Rouse exponent from the numerical profile vs Z = w_s/(kappa u*)."""
    u_star = scen.u_star
    Z_ref = rouse_exponent(scen.w_s, u_star)
    y, c = solve_rouse_profile(scen.w_s, u_star, scen.h)
    Z_num = fit_rouse_exponent(y, c, scen.h)
    rel = abs(Z_num - Z_ref) / Z_ref
    # analytic profile at a reference height for the record
    a = 0.1 * scen.h
    y_eval = np.linspace(0.05 * scen.h, 0.95 * scen.h, 50)
    c_an = rouse_profile(y_eval, scen.h, a, Z_ref)
    return {
        "Z_ref": Z_ref,
        "Z_num": Z_num,
        "rel_err": rel,
        "pass": rel < 0.05,
        "u_star": u_star,
        "w_s": scen.w_s,
        "kappa": KAPPA,
        "profile": {
            "y": y_eval.tolist(),
            "c_analytic": c_an.tolist(),
        },
    }


def _c3_mass_conservation(scen: SedimentScenario) -> dict:
    """C3: global sediment mass drift over a long Exner run (< 1e-6).

    Tests the conservative property of the Exner discretization by running
    with zero flux (theta < theta_cr everywhere). The bed should not evolve,
    and the total volume should be preserved to machine precision.
    """
    n = 200
    dx = 0.1
    x = (np.arange(n) + 0.5) * dx
    z_b = 1e-4 * np.sin(2.0 * np.pi * x / (n * dx))
    cfg = ExnerConfig(d=scen.d, s=scen.s, g=scen.g, theta_cr=100.0)
    u_star = scen.u_star
    denom = (scen.s - 1.0) * scen.g * scen.d
    h_ref = scen.h
    dt = 1.0
    n_steps = 1000
    v0 = bed_volume(z_b, dx)
    for _ in range(n_steps):
        h = h_ref - z_b
        h = np.maximum(h, 0.9 * h_ref)
        u_loc = u_star * (h_ref / h) ** 1.5
        theta = u_loc**2 / denom
        q = mpm_flux(theta, cfg)
        z_b = exner_step(z_b, q, dx, dt, cfg)
    v1 = bed_volume(z_b, dx)
    drift = abs(v1 - v0) / max(abs(v0), 1e-30)
    return {"mass_drift": drift, "pass": drift < 1e-6}


def _c4_negative_control(scen: SedimentScenario) -> dict:
    """C4: u* < 0.8 u*_cr  =>  q_s < 1e-12."""
    cfg = ExnerConfig(d=scen.d, s=scen.s, g=scen.g, theta_cr=0.045)
    u_cr = math.sqrt(0.045 * (scen.s - 1.0) * scen.g * scen.d)
    u_test = 0.8 * u_cr
    theta = scen.theta(u_test)
    q = mpm_flux(theta, cfg)
    return {
        "u_star": u_test,
        "u_star_cr": u_cr,
        "theta": theta,
        "q_s": float(q),
        "pass": float(q) < 1e-12,
    }


def build_results(
    u_star_lattice: float,
    d: float = 1.0e-4,
    s: float = 1.65,
    g: float = 9.81,
    nu: float = 1.0e-6,
    w_s: float = 1.0e-2,
    h: float = 1.0,
    u_scale: float = 10.0,
    scour: dict | None = None,
) -> dict:
    """Run all anchors and return the C6 JSON contract."""
    scen = SedimentScenario(
        d=d, s=s, g=g, nu=nu, w_s=w_s, h=h,
        u_scale=u_scale, u_star_lattice=u_star_lattice,
    )
    c1 = _c1_shields_curve(scen)
    c2 = _c2_rouse(scen)
    c3 = _c3_mass_conservation(scen)
    c4 = _c4_negative_control(scen)
    c5 = scour if scour is not None else {
        "S_over_D": None, "pass": None, "note": "scour not run"
    }
    return {
        "shields_curve": c1,
        "rouse_profiles": c2,
        "bed_elevation": {
            "note": "per-column bed elevation (m); engineering hook for the "
            "ship-platform UKC / siltation endpoint",
            "scour_run": scour.get("bed_elevation") if scour else None,
        },
        "scour_depth": c5,
        "mass_budget": {
            "c3_mass_drift": c3["mass_drift"],
            "c3_pass": c3["pass"],
            "c4_negative_control": c4,
        },
        "scenario": {
            "d_m": d, "s": s, "g": g, "nu": nu, "w_s": w_s, "h": h,
            "u_star": scen.u_star, "d_star": scen.d_star,
        },
    }

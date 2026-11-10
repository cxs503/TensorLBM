"""End-to-end sediment verification surface (criteria C1-C6).

``build_results`` runs the anchors and returns a plain dict that is directly
``json.dump``-able and matches the C6 contract::

    {shields_curve, rouse_profiles, bed_elevation, scour_depth, mass_budget}

The same functions are used by the pytest suite (``tests/test_sediment_scour.py``)
and by ``tensorlbm.sediment_benchmarks``, so the numbers in the report and the
numbers in the tests come from one computation (no hand-copy drift).

Anchors
-------
C1  incipient-motion curve      ``theta_cr(d*)`` measured in physical units
                                against Soulsby & Whitehouse (2005)
C2  Rouse profile               passive-scalar D2Q9 LBM in Rouse-log
                                coordinates, ``Z = w_s/(kappa u*)`` for 3 u*
C3  mass budget                 Exner bed volume + coupled suspended/bed drift
C4  negative controls           no-motion gate, ``w_s -> 0``, Exner off
C5  single-pile scour           2-D LBM flow + movable-bed relaxation, ``S/D``
C6  JSON contract               the dict below, plus the UKC engineering hook
"""

from __future__ import annotations

import math

import numpy as np

from .exner import ExnerConfig, bed_volume, exner_step, mpm_flux, stable_dt
from .morpho import MorphoConfig, measure_onset_shields
from .passive_scalar import (
    RouseLBMConfig,
    ScalarLBMConfig,
    column_exchange_budget,
    run_rouse_lbm,
)
from .rouse import KAPPA, rouse_exponent, rouse_profile
from .scour import PileConfig, ScourConfig, relax_scour, run_pile_flow, run_single_pile_scour
from .shields import (
    critical_shields,
    dimensionless_grain_size,
    grain_diameter_from_dstar,
    settling_velocity,
    shear_velocity_from_shields,
)

__all__ = [
    "D_STAR_POINTS",
    "SedimentScenario",
    "build_results",
    "c1_incipient_curve",
    "c2_rouse_profiles",
    "c3_mass_budget",
    "c4_negative_controls",
    "c5_pile_scour",
]

#: 8 log-spaced dimensionless grain sizes for the incipient-motion curve.
D_STAR_POINTS = (1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 40.0, 100.0)

#: sediment permeability / porosity used by the Exner equation.
POROSITY = 0.4


class SedimentScenario:
    """Physical scenario (SI units) shared by the anchors.

    The settling velocity is *not* a free input: it comes from the grain
    diameter through the van Rijn (1984) relation, so the Rouse exponent is
    a physical consequence of the sand, not a knob.
    """

    def __init__(
        self,
        d: float = 3.85e-4,
        s: float = 1.65,
        g: float = 9.81,
        nu: float = 1.0e-6,
        h: float = 0.15,
    ) -> None:
        self.d = d
        self.s = s
        self.g = g
        self.nu = nu
        self.h = h
        self.w_s = settling_velocity(d, s, g, nu)
        self.d_star = dimensionless_grain_size(d, s, g, nu)
        self.theta_cr = critical_shields(self.d_star)

    @property
    def g_prime(self) -> float:
        return (self.s - 1.0) * self.g

    def u_star_cr(self) -> float:
        """Critical shear velocity (m s^-1) from the critical Shields stress."""
        return shear_velocity_from_shields(self.theta_cr, self.d, self.s, self.g)

    def theta(self, u_star: float) -> float:
        """Shields number for a shear velocity (m s^-1)."""
        return u_star**2 / (self.g_prime * self.d)

    def as_dict(self) -> dict:
        return {
            "d_m": self.d,
            "d_star": self.d_star,
            "s": self.s,
            "g": self.g,
            "nu": self.nu,
            "h_m": self.h,
            "w_s_m_s": self.w_s,
            "theta_cr": self.theta_cr,
            "u_star_cr_m_s": self.u_star_cr(),
        }


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------
def c1_incipient_curve(
    scen: SedimentScenario, d_star_points=D_STAR_POINTS, iters: int = 30
) -> dict:
    """C1: measured onset Shields number vs Soulsby-Whitehouse at 8 ``d*``.

    For each ``d*`` the physical grain diameter is computed from the *correct*
    ``d*`` definition, the reference ``theta_cr`` is evaluated with the
    critically evaluated curve, and the numerical onset is measured by
    bisection on the base Shields number of a morphodynamic bed response
    (see :mod:`tensorlbm.sediment.morpho`).

    Also reported: ``Re*_cr = u*_cr d / nu`` (the Shields-diagram abscissa) and
    the deviation that the *previous, dimensionally inconsistent* ``d*``
    definition (``d sqrt((s-1)g)/nu``, removed in this mission) would produce --
    the demonstration that the criterion discriminates.
    """
    rows = []
    for d_star in d_star_points:
        d = grain_diameter_from_dstar(d_star, scen.s, scen.g, scen.nu)
        theta_ref = critical_shields(d_star)
        u_cr = shear_velocity_from_shields(theta_ref, d, scen.s, scen.g)
        # the onset probe is a 1 m deep numerical channel: h_ref only sets
        # the (weak) depth-scaling amplification of the bed perturbation
        mc = MorphoConfig(d=d, s=scen.s, g=scen.g, h_ref=1.0)
        onset = measure_onset_shields(mc, theta_ref, iters=iters)
        theta_num = onset["theta_num"]
        rel = abs(theta_num - theta_ref) / theta_ref

        # sensitivity: what the removed dimensionally-wrong d* would give
        d_star_old = d * math.sqrt((scen.s - 1.0) * scen.g) / scen.nu
        theta_ref_old = critical_shields(d_star_old)
        rel_old = abs(theta_ref_old - theta_ref) / theta_ref

        rows.append(
            {
                "d_star": d_star,
                "d_m": d,
                "theta_ref": theta_ref,
                "theta_num": theta_num,
                "rel_err": rel,
                "converged": onset["converged"],
                "bisection_iters": onset["iters"],
                "re_star_cr": u_cr * d / scen.nu,
                "d_star_old_formula": d_star_old,
                "old_formula_rel_err": rel_old,
            }
        )
    max_rel = max(r["rel_err"] for r in rows)
    return {
        "points": rows,
        "max_rel_err": max_rel,
        "tol": 0.15,
        "pass": bool(max_rel <= 0.15),
        "n_points": len(rows),
        "note": (
            "implementation-consistency check of the chain d -> d* -> theta_cr "
            "-> gate -> Exner response; not an independent measurement of the "
            "empirical curve (see morpho.py docstring)"
        ),
    }


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def c2_rouse_profiles(
    scen: SedimentScenario,
    z_targets=(0.3, 0.6, 1.2),
    cfg: RouseLBMConfig | None = None,
) -> dict:
    """C2: Rouse exponent from the passive-scalar LBM for three shear velocities.

    The three cases span the weakly to strongly suspended regime by targeting
    Rouse numbers ``Z = w_s/(kappa u*)`` = 0.3 / 0.6 / 1.2; the corresponding
    shear velocities are ``u* = w_s/(kappa Z)``.  For the 0.385 mm sand of the
    scenario (``w_s = 0.030 m/s``) that is ``u* = 0.24 / 0.12 / 0.06 m/s`` --
    strong currents, deliberately chosen because a Rouse number of several
    units (the harbour-siltation regime) collapses the profile into a layer a
    few lattice nodes thick and cannot be verified numerically with a
    fixed-resolution LBM.  Each case is one LBM run.
    """
    base = cfg or RouseLBMConfig()
    cases = []
    for Z_target in z_targets:
        u_star = scen.w_s / (KAPPA * Z_target)
        c = RouseLBMConfig(
            h=scen.h,
            a_frac=base.a_frac,
            u_star=u_star,
            w_s=scen.w_s,
            kappa=KAPPA,
            dxi=base.dxi,
            D_lat=base.D_lat,
            max_steps=base.max_steps,
            tol=base.tol,
            check_every=base.check_every,
        )
        res = run_rouse_lbm(c)
        a = base.a_frac * scen.h
        y_an = np.linspace(1.1 * a, 0.6 * scen.h, 40)
        cases.append(
            {
                "u_star": u_star,
                "Z_target": Z_target,
                "Z_ref": res["Z_ref"],
                "Z_num": res["Z_num"],
                "rel_err": res["rel_err"],
                "fit_asymmetry": res["fit_asymmetry"],
                "profile_rel_err_vs_analytic": res["profile_rel_err_vs_analytic"],
                "steps": res["steps"],
                "n_nodes": res["n_nodes"],
                "v_lat": res["v_lat"],
                "y": res["y"].tolist(),
                "c_normalised": res["c_normalised"].tolist(),
                "c_analytic": rouse_profile(y_an, scen.h, a, res["Z_ref"]).tolist(),
                "y_analytic": y_an.tolist(),
            }
        )
    max_rel = max(c["rel_err"] for c in cases)
    return {
        "cases": cases,
        "max_rel_err": max_rel,
        "tol": 0.05,
        "pass": bool(max_rel < 0.05),
        "w_s": scen.w_s,
        "kappa": KAPPA,
        "note": "Z targets 0.3/0.6/1.2; u* = w_s/(kappa Z)",
    }


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------
def c3_mass_budget(
    scen: SedimentScenario, n_steps: int = 2000, t_end: float = 600.0
) -> dict:
    """C3: sediment mass budget.

    (a) **Exner bed volume**: a genuinely evolving bed (non-uniform Shields
        number from the local-depth feedback) is advanced ``n_steps``
        conservative updates over ``t_end`` physical seconds; the total bed
        volume must be conserved to round-off while the bed itself changes by
        a measurable fraction of a grain diameter.  The step is CFL-limited,
        and the elapsed time is chosen so the bed stays in the small-perturbation
        regime (no depth runaway).
    (b) **Coupled suspended + bed**: the vertical column exchange
        (:func:`tensorlbm.sediment.passive_scalar.column_exchange_budget`)
        must conserve suspended + bed sediment to round-off.
    """
    n = 200
    dx = 0.1
    x = (np.arange(n) + 0.5) * dx
    z_b = 0.05 * scen.d * np.sin(2.0 * np.pi * x / (n * dx))
    cfg = ExnerConfig(d=scen.d, s=scen.s, g=scen.g, porosity=POROSITY,
                      theta_cr=scen.theta_cr)
    # A smooth, non-uniform *prescribed* bed-load flux (kinematic Exner test):
    # the flux comes from a base Shields number 1.5 theta_cr modulated by 50 %
    # over the domain.  Prescribing the flux isolates the property under test
    # (conservation of the discrete Exner update) from the well-known
    # instability of the closed feedback loop "no slope term in q_s" -- see the
    # limitations section of the report.
    q_base = float(mpm_flux(1.5 * scen.theta_cr, cfg))
    q_s = q_base * (1.0 + 0.5 * np.sin(2.0 * np.pi * 2.0 * x / (n * dx)))
    dt_cfl = stable_dt(
        z_b, dx, cfg, lambda z: np.full_like(z, 1.5 * scen.theta_cr), cfl=0.2
    )
    dt = min(t_end / n_steps, 0.9 * dt_cfl)
    v0 = bed_volume(z_b, dx)
    z_init = z_b.copy()
    # reference volume: the absolute bed volume of the initial perturbation
    # (the signed volume of a pure sinusoid is zero, so it cannot be used as a
    # relative-error denominator)
    ref_vol = float(np.sum(np.abs(z_init)) * dx)
    for _ in range(n_steps):
        z_b = exner_step(z_b, q_s, dx, dt, cfg)
    v1 = bed_volume(z_b, dx)
    drift_bed = abs(v1 - v0) / max(ref_vol, 1e-30)
    bed_change = float(np.max(np.abs(z_b - z_init)))

    exch = column_exchange_budget(
        ScalarLBMConfig(ny=24, h=24.0, u_star=0.05, w_s=0.006, nx=1),
        n_steps=min(n_steps, 3000),
        porosity=POROSITY,
    )
    return {
        "exner": {
            "mass_drift": drift_bed,
            "bed_change_m": bed_change,
            "bed_change_over_d": bed_change / scen.d,
            "steps": n_steps,
            "ddt_s": dt,
            "q_base_m2_s": q_base,
            "dt_cfl_s": dt_cfl,
            "t_end_s": n_steps * dt,
            "pass": bool(drift_bed < 1e-6 and np.isfinite(bed_change)),
        },
        "coupled_column": exch,
        "pass": bool(drift_bed < 1e-6 and exch["mass_drift"] < 1e-6),
        "tol": 1e-6,
    }


# ---------------------------------------------------------------------------
# C4
# ---------------------------------------------------------------------------
def c4_negative_controls(scen: SedimentScenario) -> dict:
    """C4: (a) no motion below 0.8 u*_cr, (b) w_s -> 0 uniform, (c) Exner off."""
    cfg = ExnerConfig(d=scen.d, s=scen.s, g=scen.g, porosity=POROSITY,
                      theta_cr=scen.theta_cr)
    u_cr = scen.u_star_cr()
    u_test = 0.8 * u_cr
    theta = scen.theta(u_test)
    q = mpm_flux(theta, cfg)
    # (a) also: the bed must be bit-identical after an Exner step with q = 0
    z = np.linspace(-1e-4, 1e-4, 64)
    z_after = exner_step(z, np.full_like(z, float(q)), 0.01, 1.0, cfg)
    no_motion_identical = bool(np.array_equal(z_after, z))

    # (b) w_s -> 0: uniform concentration field
    r0 = run_rouse_lbm(
        RouseLBMConfig(h=1.0, u_star=0.05, w_s=0.0, dxi=0.2, D_lat=0.1)
    )

    # (c) Exner off: bed elevation unchanged, bit-identical
    q_hi = mpm_flux(np.full_like(z, 5.0 * scen.theta_cr), cfg)
    z_on = exner_step(z, q_hi, 0.01, 1.0, cfg, enabled=True)
    z_off = exner_step(z, q_hi, 0.01, 1.0, cfg, enabled=False)
    return {
        "a_no_motion": {
            "u_star": u_test,
            "u_star_cr": u_cr,
            "ratio": u_test / u_cr,
            "theta": theta,
            "q_s": float(q),
            "bed_identical": no_motion_identical,
            "pass": bool(abs(float(q)) < 1e-12 and no_motion_identical),
        },
        "b_zero_settling": {
            "Z_num": r0["Z_num"],
            "uniformity": r0["profile_uniformity"],
            "pass": bool(r0["profile_uniformity"] < 1e-3),
        },
        "c_exner_off": {
            "bed_on_change": float(np.max(np.abs(z_on - z))),
            "bed_off_change": float(np.max(np.abs(z_off - z))),
            "pass": bool(np.array_equal(z_off, z) and not np.array_equal(z_on, z)),
        },
    }


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------
def _transect(scour: dict, n: int = 25) -> dict:
    """Bed elevation along the streamline through the deepest scour.

    The section is taken along the flow direction (x) through the location of
    maximum scour, which is the line a vessel would cross -- that is what the
    UKC consumer needs.  The lateral offset of that line is reported.
    """
    xd = np.asarray(scour["x_over_D"])[0, :]
    yd = np.asarray(scour["y_over_D"])[:, 0]
    S = np.asarray(scour["scour_field"])
    j = int(np.unravel_index(np.argmax(S), S.shape)[0])
    prof = S[j, :]
    idx = np.linspace(0, prof.size - 1, min(n, prof.size)).astype(int)
    return {
        "x_over_D": xd[idx].tolist(),
        "y_over_D": float(yd[j]),
        "scour_depth_m": prof[idx].tolist(),
        "bed_elevation_m": (-prof[idx]).tolist(),
    }


def c5_pile_scour(
    pile: PileConfig | None = None,
    scour_cfg: ScourConfig | None = None,
    trend: bool = True,
) -> dict:
    """C5: equilibrium ``S/D`` of a single pile + the trend when D doubles."""
    pile = pile or PileConfig(nx=96, ny=72, diameter=12.0, n_steps=8000, n_avg=2000)
    scour_cfg = scour_cfg or ScourConfig()
    flow = run_pile_flow(pile)

    def _with(cfg_i: ScourConfig) -> dict:
        bed = relax_scour(
            flow["speed_field"], cfg_i, flow["U_far"], flow["A"],
            flow["x_over_D"], flow["y_over_D"],
        )
        return {
            "S_over_D": bed["S_over_D"],
            "S_max": bed["S_max"],
            "A": flow["A"],
            "Re": flow["Re"],
            "U_far": flow["U_far"],
            "theta0": bed["theta0"],
            "u_over_uc": bed["u_over_uc"],
            "h_over_D": cfg_i.h / cfg_i.D,
            "closed_form_S_over_D": bed["closed_form_S_over_D"],
            "relax_steps": bed["relax_steps"],
            "history": bed["history"],
            "scour_field": bed["scour_field"],
            "x_over_D": flow["x_over_D"],
            "y_over_D": flow["y_over_D"],
        }

    base = _with(scour_cfg)
    out = {
        "S_over_D": base["S_over_D"],
        "S_max_m": base["S_max"],
        "A": base["A"],
        "Re": base["Re"],
        "U_far": base["U_far"],
        "theta0": base["theta0"],
        "u_over_uc": base["u_over_uc"],
        "h_over_D": base["h_over_D"],
        "closed_form_S_over_D": base["closed_form_S_over_D"],
        "band": [1.0, 2.4],
        "in_band": bool(1.0 <= base["S_over_D"] <= 2.4),
        "relax_steps": base["relax_steps"],
        "history": base["history"],
        "transect": _transect(base),
        "pile_cfg": {
            "nx": pile.nx,
            "ny": pile.ny,
            "diameter_lat": pile.diameter,
            "nu": pile.nu,
            "inlet_velocity": pile.inlet_velocity,
            "n_steps": pile.n_steps,
            "n_avg": pile.n_avg,
        },
        "scenario": {
            "D_m": scour_cfg.D,
            "h_m": scour_cfg.h,
            "d_m": scour_cfg.d,
            "theta_cr": scour_cfg.theta_cr,
            "u_over_uc": scour_cfg.u_over_uc,
            "depth_exponent": scour_cfg.depth_exponent,
        },
    }
    # sensitivity of the equilibrium depth to the approach-flow intensity
    sens = []
    for uuc in (0.80, 0.90, 0.95, 1.00):
        c_i = ScourConfig(**{**scour_cfg.__dict__, "u_over_uc": uuc})
        r_i = _with(c_i)
        sens.append({
            "u_over_uc": uuc,
            "S_over_D": r_i["S_over_D"],
            "relax_steps": r_i["relax_steps"],
        })
    out["sensitivity_u_over_uc"] = sens
    if trend:
        cfg2 = ScourConfig(**{**scour_cfg.__dict__, "D": 2.0 * scour_cfg.D})
        # doubling D keeps the *lattice* flow identical (same Re and h/D ratio
        # is broken on purpose: h is fixed, so h/D halves)
        trend_res = _with(cfg2)
        out["trend"] = {
            "D_m": cfg2.D,
            "h_over_D": trend_res["h_over_D"],
            "S_over_D": trend_res["S_over_D"],
            "direction": "decrease" if trend_res["S_over_D"] < base["S_over_D"]
            else "increase",
            "empirical_direction": "decrease",
            "pass": bool(trend_res["S_over_D"] < base["S_over_D"]),
        }
        out["pass"] = bool(out["in_band"] and out["trend"]["pass"])
    else:
        out["pass"] = out["in_band"]
    return out


# ---------------------------------------------------------------------------
# C6
# ---------------------------------------------------------------------------
def build_results(
    scen: SedimentScenario | None = None,
    c1: dict | None = None,
    c2: dict | None = None,
    c3: dict | None = None,
    c4: dict | None = None,
    c5: dict | None = None,
    cheap_scour: bool = False,
) -> dict:
    """Run/serve the anchors and return the C6 JSON contract.

    Any anchor can be injected (used by the benchmark runner to cache stages);
    the rest are computed here.
    """
    scen = scen or SedimentScenario()
    c1 = c1 if c1 is not None else c1_incipient_curve(scen)
    c2 = c2 if c2 is not None else c2_rouse_profiles(scen)
    c3 = c3 if c3 is not None else c3_mass_budget(scen)
    c4 = c4 if c4 is not None else c4_negative_controls(scen)
    if c5 is None:
        if cheap_scour:
            c5 = c5_pile_scour(
                PileConfig(nx=80, ny=60, diameter=10.0, n_steps=5000, n_avg=1500),
                trend=True,
            )
        else:
            c5 = c5_pile_scour()
    return {
        "schema": "tensorlbm.sediment.verification/v1",
        "scenario": scen.as_dict(),
        "shields_curve": c1,
        "rouse_profiles": c2,
        "bed_elevation": {
            "units": "m (positive up, relative to the undisturbed bed)",
            "x_over_D": c5["transect"]["x_over_D"],
            "bed_elevation_m": c5["transect"]["bed_elevation_m"],
            "scour_depth_m": c5["transect"]["scour_depth_m"],
            "note": "per-column bed elevation; engineering hook for the ship "
            "platform UKC / siltation endpoint",
        },
        "scour_depth": c5,
        "mass_budget": c3,
        "negative_controls": c4,
        "pass": {
            "C1": c1["pass"],
            "C2": c2["pass"],
            "C3": c3["pass"],
            "C4": bool(
                c4["a_no_motion"]["pass"]
                and c4["b_zero_settling"]["pass"]
                and c4["c_exner_off"]["pass"]
            ),
            "C5": bool(c5["pass"]),
        },
    }
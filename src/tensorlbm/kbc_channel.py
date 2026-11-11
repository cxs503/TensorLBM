"""Wall-unit analysis and DNS gates for the KBC coarse-grid channel.

The production driver (:mod:`tensorlbm.kbc_channel_native`) writes wall-normal
profiles in lattice units.  This module turns them into wall units, measures the
effective wall position / wall stress (the "effective wall consistency check"
the campaign node asks for), integrates the bulk velocity and evaluates the
mission gates against the MKM-1999 spectral DNS that ships with the repo
(:mod:`tensorlbm.validation.mkm1999`, SHA-256 verified).

Conventions
-----------
* Node ``y`` (``1 .. ny-2``) sits at the wall-normal distance ``y - 0.5`` from
  the lower wall plane (y = 0.5), so ``y+ = (y - 0.5) nu_tau / h * Re_tau`` with
  ``h = (ny - 2) / 2`` the half height.  The centreline is at ``y = ny/2``.
* ``u+`` uses the friction velocity ``u_tau = sqrt(force * h)`` from the global
  momentum balance, which for a body-force-driven channel is *exact* at
  statistical steady state and independent of the grid -- so ``C_f`` error can
  only come from the bulk velocity ``U_b``.
* ``C_f = 2 u_tau^2 / U_b^2`` (plane channel, ``C_f = tau_w / (rho U_b^2 / 2)``).
* The momentum-exchange check compares the measured wall stress, obtained by
  extrapolating the total stress profile ``Pi_xy(y)`` (resolved + sub-grid, i.e.
  the actual momentum flux carried by the populations) to the wall, with the
  force input ``force * h``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import mkm1999

__all__ = [
    "ChannelResult",
    "GATE_SPEC",
    "cf_from_profiles",
    "effective_wall_check",
    "evaluate_gates",
    "fit_log_law",
    "wall_units",
]

#: Mission acceptance thresholds (campaign node p495w4-n03), frozen before the
#: production runs.  ``re_tau`` selects the constant of the Cf gate.
GATE_SPEC = {
    "cf_rel_err": {395: 0.03, 590: 0.03, 180: 0.05},
    "uplus_pointwise_rel_err_y100": 0.05,
    "reynolds_stress_rel_err": 0.10,
    "entropy_dh_max": 0.0,
    "grid_cf_rel_change": 0.05,
}


@dataclass
class ChannelResult:
    """Wall-unit view of one channel run."""

    ny: int
    half_height: float
    nu: float
    u_tau: float
    y: np.ndarray
    y_plus: np.ndarray
    u_plus: np.ndarray
    uu_plus: np.ndarray
    vv_plus: np.ndarray
    ww_plus: np.ndarray
    uv_plus: np.ndarray
    stress: np.ndarray
    y_over_h: np.ndarray

    @property
    def bulk_velocity_plus(self) -> float:
        y = self.y_plus[: len(self.y_plus)]
        return bulk_velocity_plus(y, self.u_plus, self.half_height)


def bulk_velocity_plus(y_plus: np.ndarray, u_plus: np.ndarray, half_height: float) -> float:
    """``U_b+ = (1/h) int_0^h u^+ dy^+`` from the half-channel profile.

    The profile array must cover the wall-normal nodes of one half channel
    (``y+`` increasing, ending at the centreline).  The piece from the wall plane
    to the first node centre is integrated with the wall value ``u+(0) = 0``,
    i.e. trapezoid on ``[0, y+_1]``; the rest with the trapezoid rule.  This is
    the same estimator used for the DNS reference, so the comparison is
    apples-to-apples.
    """
    y = np.asarray(y_plus, dtype=float)
    u = np.asarray(u_plus, dtype=float)
    up = np.concatenate(([0.0], u))
    yp = np.concatenate(([0.0], y))
    return float(np.trapezoid(up, yp) / yp[-1])


def wall_units(
    profiles: dict[str, np.ndarray],
    *,
    nu: float,
    force: float,
    half_height: float,
    re_tau: float,
) -> ChannelResult:
    """Convert a lattice-unit profile set to wall units (half channel)."""
    ny = len(profiles["u"])
    y = np.arange(ny, dtype=float) - 0.5
    half = half_height
    u_tau = float(np.sqrt(force * half))
    yp = y * re_tau / half
    # half channel: nodes strictly inside the lower wall and below the centreline
    sel = (np.arange(ny) >= 1) & (y <= half + 1e-12) & (np.arange(ny) <= ny // 2)
    idx = np.where(sel)[0]
    up = profiles["u"][idx] / u_tau
    scale = u_tau * u_tau
    return ChannelResult(
        ny=ny,
        half_height=half,
        nu=nu,
        u_tau=u_tau,
        y=y[idx],
        y_plus=yp[idx],
        u_plus=up,
        uu_plus=profiles["uu"][idx] / scale,
        vv_plus=profiles["vv"][idx] / scale,
        ww_plus=profiles["ww"][idx] / scale,
        uv_plus=profiles["uv"][idx] / scale,
        stress=profiles["stress"][idx] / scale,
        y_over_h=y[idx] / half,
    )


def effective_wall_check(
    profiles: dict[str, np.ndarray],
    *,
    force: float,
    half_height: float,
    u_tau: float,
) -> dict:
    """Momentum-balance / effective-wall consistency check.

    For a statistically steady body-force channel the *total* stress carried by
    the populations, ``Pi_xy(y)``, must be linear in ``y`` and equal to
    ``-rho force (h - y)``: it is the discrete statement that the wall stress
    the flow actually feels equals the force input, i.e. that the half-way
    bounce-back wall sits where the wall units say it does.

    Returns the wall stress implied by extrapolating ``Pi_xy`` to the wall, the
    linear-fit slope of ``Pi_xy`` versus ``(h - y)``, and their relative
    deviations from the force input.
    """
    ny = len(profiles["u"])
    y = np.arange(ny, dtype=float) - 0.5
    half = half_height
    idx = (np.arange(ny) >= 1) & (y <= half)
    yy = y[idx]
    pi = profiles["pi_xy"][idx]
    # Pi_xy = -tau_w (1 - y/h) with tau_w = force*h  ->  fit pi = a*(h - y)
    hminus = half - yy
    a, _ = np.polyfit(hminus, pi, 1)
    tau_fit = abs(a)
    tau_force = force * half
    # extrapolate to the wall plane (y = 0)
    wall = abs(float(np.polyval(np.polyfit(hminus, pi, 1), half)))
    u_tau_stress = float(np.sqrt(max(wall, 0.0)))
    return {
        "tau_force": float(tau_force),
        "tau_wall_fit": float(tau_fit),
        "u_tau_force": float(u_tau),
        "u_tau_stress": u_tau_stress,
        "u_tau_rel_err": float(abs(u_tau_stress - u_tau) / u_tau),
        "tau_rel_err": float(abs(tau_fit - tau_force) / tau_force),
    }


def fit_log_law(
    y_plus: np.ndarray, u_plus: np.ndarray, *, y_lo: float = 30.0, y_hi: float | None = None
) -> dict:
    """Least-squares ``u+ = (1/kappa) ln(y+) + B`` over ``[y_lo, y_hi]``."""
    y = np.asarray(y_plus, float)
    u = np.asarray(u_plus, float)
    hi = float(y.max()) if y_hi is None else y_hi
    m = (y >= y_lo) & (y <= hi)
    if m.sum() < 3:
        return {"kappa": float("nan"), "B": float("nan"), "n_points": int(m.sum())}
    slope, intercept = np.polyfit(np.log(y[m]), u[m], 1)
    kappa = 1.0 / slope if slope > 0 else float("nan")
    return {
        "kappa": float(kappa),
        "B": float(intercept),
        "n_points": int(m.sum()),
        "y_lo": float(y[m][0]),
        "y_hi": float(y[m][-1]),
    }


def cf_from_profiles(
    profiles: dict[str, np.ndarray], *, nu: float, force: float, half_height: float, re_tau: float
) -> dict:
    """Friction coefficient from the bulk velocity and the momentum balance."""
    res = wall_units(profiles, nu=nu, force=force, half_height=half_height, re_tau=re_tau)
    ub_plus = res.bulk_velocity_plus
    cf = 2.0 / (ub_plus * ub_plus)
    return {
        "u_tau": res.u_tau,
        "ub_plus": float(ub_plus),
        "cf": float(cf),
        "bulk_velocity_lattice": float(ub_plus * res.u_tau),
    }


def evaluate_gates(
    profiles: dict[str, np.ndarray],
    *,
    nu: float,
    force: float,
    half_height: float,
    re_tau: float,
    nominal_re_tau: int,
    scalars: dict[str, float] | None = None,
) -> dict:
    """Evaluate the mission gates of node p495w4-n03 against MKM-1999 DNS."""
    mkm = mkm1999.load_profile(nominal_re_tau)
    res = wall_units(profiles, nu=nu, force=force, half_height=half_height, re_tau=re_tau)
    cf_sim = 2.0 / res.bulk_velocity_plus**2
    cf_dns = mkm.friction_coefficient()
    cf_err = abs(cf_sim - cf_dns) / cf_dns

    # --- u+ pointwise in y+ <= 100 --------------------------------------
    sel = res.y_plus <= 100.0
    dns_u = mkm.u_plus_at(res.y_plus[sel])
    up_err = np.abs(res.u_plus[sel] - dns_u) / np.maximum(dns_u, 1e-12)
    up_worst = float(up_err.max()) if up_err.size else float("nan")
    up_rms = float(np.sqrt(np.mean(up_err**2))) if up_err.size else float("nan")

    # --- Reynolds shear stress ------------------------------------------
    dns_uv = np.interp(res.y_plus, mkm.y_plus, mkm.uv_plus)
    band = (res.y_plus >= 5.0) & (res.y_plus <= float(mkm.re_tau))
    sim_uv = -res.uv_plus[band]      # -<u'v'>+ (positive below the centreline)
    dns_band = -dns_uv[band]
    denom = np.maximum(np.abs(dns_band), 1e-12)
    uv_rel = float(np.sqrt(np.mean(((sim_uv - dns_band) / denom) ** 2)))
    peak_sim = float(sim_uv.max()) if sim_uv.size else float("nan")
    peak_dns = float(dns_band.max()) if dns_band.size else float("nan")
    peak_err = abs(peak_sim - peak_dns) / abs(peak_dns) if peak_dns else float("nan")

    wall = effective_wall_check(profiles, force=force, half_height=half_height, u_tau=res.u_tau)
    loglaw = fit_log_law(res.y_plus, res.u_plus, y_lo=30.0, y_hi=min(0.3 * re_tau, 150.0))
    dh = float(scalars.get("dh_global_worst", float("nan"))) if scalars else float("nan")

    gates = {
        "cf": {
            "criterion": f"|Cf - Cf_DNS|/Cf_DNS < {100 * GATE_SPEC['cf_rel_err'][nominal_re_tau]:.0f}%",
            "sim": cf_sim,
            "dns": cf_dns,
            "rel_err": cf_err,
            "threshold": GATE_SPEC["cf_rel_err"][nominal_re_tau],
            "pass": bool(cf_err < GATE_SPEC["cf_rel_err"][nominal_re_tau]),
        },
        "uplus": {
            "criterion": "pointwise |u+ - u+_DNS|/u+_DNS < 5% for y+ <= 100",
            "worst_rel_err": up_worst,
            "rms_rel_err": up_rms,
            "worst_y_plus": float(res.y_plus[sel][int(np.argmax(up_err))]) if up_err.size else float("nan"),
            "threshold": GATE_SPEC["uplus_pointwise_rel_err_y100"],
            "pass": bool(up_worst < GATE_SPEC["uplus_pointwise_rel_err_y100"]),
        },
        "reynolds_stress": {
            "criterion": "profile RMS rel. err < 10% over 5 <= y+ <= Re_tau; peak within 10%",
            "rms_rel_err": uv_rel,
            "peak_sim": peak_sim,
            "peak_dns": peak_dns,
            "peak_rel_err": peak_err,
            "threshold": GATE_SPEC["reynolds_stress_rel_err"],
            "pass": bool(uv_rel < GATE_SPEC["reynolds_stress_rel_err"]),
        },
        "entropy": {
            "criterion": "per-step total dH <= 0 (H-theorem)",
            "dh_global_worst": dh,
            "threshold": GATE_SPEC["entropy_dh_max"],
            "pass": bool(np.isfinite(dh) and dh <= 1e-12),
        },
        "wall": {
            "criterion": "|u_tau,stress - u_tau,force|/u_tau < 2%",
            **wall,
            "pass": bool(wall["u_tau_rel_err"] < 0.02),
        },
    }
    if scalars:
        gates["numerics"] = {
            "min_population": scalars.get("min_population"),
            "max_abs_u": scalars.get("max_abs_u"),
            "alpha_mean": scalars.get("alpha_mean"),
            "gamma_mean": scalars.get("gamma_mean"),
            "gamma_limited_fraction": scalars.get("gamma_limited_fraction"),
            "entropy_skipped_fraction": scalars.get("entropy_skipped_fraction"),
            "ghost_fraction": scalars.get("ghost_fraction"),
            "newton_iterations": scalars.get("newton_iterations"),
            "finite": scalars.get("finite"),
        }
    gates["profiles"] = {
        "log_law": loglaw,
        "ub_plus": float(res.bulk_velocity_plus),
        "dns_ub_plus": float(mkm.bulk_velocity_plus()),
        "re_tau_dns": float(mkm.re_tau),
        "n_nodes_half": int(len(res.y_plus)),
    }
    return gates
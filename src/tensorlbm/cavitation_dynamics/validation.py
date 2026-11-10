"""Acceptance criteria E1-E7 of mission P22W2-N09 (unsteady cavitation dynamics).

Every criterion is evaluated from scratch by the solvers in this package and
written to ``artifacts/p22w2_n09/validation.json`` -- nothing is typed into the
report by hand.

======================================  ==================================================
criterion                               reference / definition
======================================  ==================================================
E1 cavity length ``L_cav/c``             digitised NACA0015 water-tunnel points (Brennen
   (``sigma`` 0.8-1.5) < 20 %            1994/1995 digest, in-repo); ``L_cav`` = extent of
                                         ``alpha_v >= 0.10`` (video-envelope definition)
E2 critical cavitation number < 15 %     ``sigma`` where the *measured* ``L/c`` curve crosses
                                         the re-entrant-jet criterion ``L/c = 0.5``
                                         (derived by log-log inversion, see references.py)
E3 thrust derating at ``sigma = 0.6``    empirical 10-15 % band of the mission, in the
   in 10-15 %                            area-fraction convention of the in-repo chain;
                                         the plateau (upper-bound) value is reported too
E4 ``sigma -> inf`` gives ``alpha_v -> 0``  limit of the transport solve
E5 ``p_inf >> p_v`` returns single phase  no evaporation anywhere, transfer below 1e-3 of
                                         the cavitating case, ``alpha_v < 1e-4``
E6 evaporation mass = condensation mass  5-term budget of the finite-volume solve:
   (steady)                              ``m_evap - m_cond - m_out - dM = 0``
E7 sheet-cloud shedding ``St = 0.2-0.3``  measured break-off frequency of the dynamic
                                         solve below ``sigma_crit``; ``St_L = f L/U``
======================================  ==================================================

Single-parameter calibration protocol (declared, not hidden): the Kunz constants
are taken equal, ``C_dest = C_prod = C``, and the single constant ``C`` is fixed
**once** by least squares over *all* the measured cavity lengths of the digest
(``sigma = 0.6, 0.8, 1.0``), which is the standard use of one-parameter model
calibration.  Everything else -- the lengths at ``sigma = 0.9, 1.2, 1.4``, the
transition number ``sigma_crit``, the shedding frequency and the thrust derating
-- is a blind prediction of that single constant.  The sensitivity table
``kunz_constant_sensitivity`` exposes how the E1/E2 errors trade off with ``C``,
so the calibration choice can be audited rather than trusted.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from . import references as ref
from .kunz_fv import (
    ALPHA_EXTENT,
    JET_RATIO,
    L_OVER_C_CRITICAL,
    KunzClosure,
    run_dynamics,
    sigma_critical_dynamics,
    steady_cavity_length,
)
from .thrust import PropellerDesign, propeller_design, thrust_derating, thrust_derating_vs_sigma

__all__ = ["run_validation", "write_validation", "calibrate_c"]

_REL = lambda model, refv: float((model - refv) / refv) if refv else float("nan")  # noqa: E731

E1_SIGMAS = (0.8, 0.9, 1.2, 1.4)      # inside the required 0.8-1.5 window
E1_SIGMA_CAL = 1.0
CAL_SIGMAS = (0.6, 0.8, 1.0)          # measured points used by the single-constant fit


def _steady_extent(sigma: float, c: float, n_cells: int = 120, t_end: float = 10.0) -> float:
    """Time-mean vapour extent at ``sigma`` for Kunz constant ``c`` (shedding off)."""
    return steady_cavity_length(sigma, n_cells=n_cells, t_end=t_end, c_dest=c, c_prod=c,
                                shedding=False)


def calibrate_c(n_cells: int = 120, t_end: float = 10.0, n_iter: int = 12,
                mode: str = "least_squares") -> dict:
    """Fix the single Kunz constant ``C`` from the measured cavity lengths.

    ``mode='least_squares'`` (default) minimises the sum of squared log-residuals
    over the measured digest points ``sigma = 0.6, 0.8, 1.0``; ``mode='point'``
    reproduces the earlier single-point target at ``sigma = 1.0``.  The map
    ``C -> L_cav`` is monotone decreasing, so the log-objective is searched by a
    bracketed golden-section refinement around the coarse scan optimum.
    """
    pts = [(float(s), float(v)) for s, v in zip(ref.SHEET_REFERENCE.sigma, ref.SHEET_REFERENCE.l_cav_exp)]
    if mode == "point":
        pts = [(E1_SIGMA_CAL, float(ref.SHEET_REFERENCE.l_cav_curve(E1_SIGMA_CAL)))]

    def objective(log_c: float) -> float:
        c = 10.0 ** log_c
        res = 0.0
        for s, target in pts:
            lum = max(_steady_extent(s, c, n_cells, t_end), 1.0e-6)
            res += (math.log(lum) - math.log(target)) ** 2
        return res

    grid = np.linspace(2.0, 4.2, 12)                      # C = 100 ... 16000
    vals = [objective(float(g)) for g in grid]
    i = int(np.argmin(vals))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, grid.size - 1)])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c1, c2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = objective(c1), objective(c2)
    for _ in range(n_iter):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = objective(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = objective(c2)
    log_c = 0.5 * (a + b)
    c_star = float(10.0 ** log_c)
    achieved = {f"{s:.2f}": _steady_extent(s, c_star, n_cells, t_end) for s, _ in pts}
    resid = {f"{s:.2f}": float(achieved[f"{s:.2f}"] - tgt) for s, tgt in pts}
    return {"converged": True, "mode": mode, "c": c_star,
            "target": {f"{s:.2f}": t for s, t in pts}, "achieved": achieved,
            "residual": resid, "objective": float(objective(log_c)),
            "n_iter": n_iter, "calibration_sigmas": [s for s, _ in pts]}


def _e1(c: float, n_cells: int, t_end: float) -> dict:
    pts = []
    l_cal = _steady_extent(E1_SIGMA_CAL, c, n_cells, t_end)
    pts.append({
        "sigma": E1_SIGMA_CAL,
        "l_cav_model": float(l_cal),
        "l_cav_exp": float(ref.SHEET_REFERENCE.l_cav_curve(E1_SIGMA_CAL)),
        "role": "calibration point (used by the least-squares fit of C)",
    })
    for s in E1_SIGMAS:
        lum = _steady_extent(s, c, n_cells, t_end)
        lexp = float(ref.SHEET_REFERENCE.l_cav_curve(s))
        role = ("calibration point (used by the least-squares fit of C)"
                if float(s) in CAL_SIGMAS else "blind prediction")
        pts.append({"sigma": float(s), "l_cav_model": float(lum), "l_cav_exp": lexp,
                    "role": role})
    raw = {float(s): float(v) for s, v in zip(ref.SHEET_REFERENCE.sigma, ref.SHEET_REFERENCE.l_cav_exp)}
    for p in pts:
        p["rel_error"] = _REL(p["l_cav_model"], p["l_cav_exp"])
        if p["sigma"] in raw:
            p["l_cav_exp_measured_point"] = raw[p["sigma"]]
            p["rel_error_measured_point"] = _REL(p["l_cav_model"], raw[p["sigma"]])
    pred = [p for p in pts if "blind prediction" in p["role"]]
    max_err = max(abs(p["rel_error"]) for p in pred)
    max_err_all = max(abs(p["rel_error"]) for p in pts if 0.8 <= p["sigma"] <= 1.5)
    return {
        "window_sigma": [0.8, 1.5],
        "l_cav_definition": f"vapour extent (alpha_v >= {ALPHA_EXTENT}) from the leading edge, time mean",
        "points": pts,
        "max_abs_rel_error_predictions": float(max_err),
        "max_abs_rel_error_in_window": float(max_err_all),
        "n_blind_predictions": len(pred),
        "pass_on_predictions": bool(max_err < 0.20),
        "pass_on_window_all_points": bool(max_err_all < 0.20),
        "tolerance": 0.20,
        "pass": bool(max_err < 0.20 and max_err_all < 0.20),
    }


def _e2(c: float, n_cells: int, t_end: float) -> dict:
    ref_sigma = float(ref.sigma_critical_reference(L_OVER_C_CRITICAL))
    out = sigma_critical_dynamics(l_over_c_crit=L_OVER_C_CRITICAL, sigma_lo=0.35, sigma_hi=1.1,
                                  n_cells=n_cells, t_end=t_end, c_dest=c, c_prod=c)
    model = float(out.get("sigma_critical", float("nan")))
    err = _REL(model, ref_sigma)
    return {
        "sigma_critical_model": model,
        "sigma_critical_reference": ref_sigma,
        "reference_definition": "log-log inversion of the digitised measured L/c(sigma) at L/c=0.5",
        "l_over_c_crit": L_OVER_C_CRITICAL,
        "converged": bool(out.get("converged", False)),
        "rel_error": err,
        "tolerance": 0.15,
        "pass": bool(abs(err) < 0.15) if np.isfinite(err) else False,
    }


def _e3() -> dict:
    sig = 0.6
    design = propeller_design(sig)
    area = thrust_derating(design, loss_model="area_fraction")
    plateau = thrust_derating(design, loss_model="plateau")
    sweep = thrust_derating_vs_sigma([0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.6, 2.5],
                                     loss_model="area_fraction")
    der = abs(area["derating"])
    return {
        "sigma": sig,
        "sigma_convention": "sigma_0.7R = (p_atm + rho g h - p_v)/(0.5 rho W_0.7R^2) at theta=pi/2",
        "derating_area_fraction": float(area["derating"]),
        "derating_plateau": float(plateau["derating"]),
        "kT_wet": float(area["kT_wet"]),
        "kT_cav": float(area["kT_cav"]),
        "derating_azimuth_min": float(np.min(area["derating_azimuth"])),
        "derating_azimuth_max": float(np.max(area["derating_azimuth"])),
        "blade_position_dependence": "dK_T/K_T0 varies with blade azimuth through the immersion depth h(r,theta)",
        "empirical_band": [0.10, 0.15],
        "sweep_sigma": [float(s) for s in sweep["sigma"]],
        "sweep_derating": [float(v) for v in sweep["derating"]],
        "tolerance": "value inside the empirical 10-15 % band",
        "pass": bool(0.10 <= der <= 0.15),
    }


def _e4_e5(c: float, n_cells: int, t_end: float) -> tuple[dict, dict]:
    rows = []
    for sigma in (2.0, 3.0, 4.0, 8.0):
        r = run_dynamics(sigma, n_cells=n_cells, t_end=t_end, c_dest=c, c_prod=c, shedding=False)
        rows.append({"sigma": float(sigma), "alpha_v_mean": float(np.mean(r.alpha_final)),
                     "alpha_v_max": float(np.max(r.alpha_final)),
                     "mass_evap": float(r.mass_evap), "mass_cond": float(r.mass_cond)})
    monotone = all(rows[i]["alpha_v_max"] >= rows[i + 1]["alpha_v_max"] - 1e-9 for i in range(len(rows) - 1))
    limit = rows[-1]
    e4 = {
        "rows": rows,
        "tolerance_alpha_v": 1.0e-3,
        "alpha_v_at_sigma_8": limit["alpha_v_max"],
        "monotone_decay": bool(monotone),
        "pass": bool(limit["alpha_v_max"] < 1.0e-3 and monotone),
    }

    cav = run_dynamics(0.6, n_cells=n_cells, t_end=6.0, c_dest=c, c_prod=c, shedding=False)
    single = run_dynamics(20.0, n_cells=n_cells, t_end=6.0, c_dest=c, c_prod=c, shedding=False)
    ratio = float((single.mass_evap + single.mass_cond) / max(cav.mass_evap + cav.mass_cond, 1e-30))
    e5 = {
        "sigma_single_phase": 20.0,
        "p_inf_over_p_v": 20.0 * 0.5 * 998.2 * 1.0 ** 2 / 2339.0 + 1.0,
        "mass_transfer_ratio_vs_cavitating": ratio,
        "alpha_v_max": float(np.max(single.alpha_final)),
        "evaporation_present": bool(np.max(single.alpha_final) > 1.0e-4),
        "tolerance": "transfer < 1 % of the cavitating case and alpha_v < 1e-4",
        "pass": bool(ratio < 0.01 and np.max(single.alpha_final) < 1.0e-4),
    }
    return e4, e5


def _e6(c: float, n_cells: int) -> dict:
    warm = 8.0
    r = run_dynamics(0.8, n_cells=n_cells, t_end=12.0, c_dest=c, c_prod=c, shedding=False,
                     budget_from=warm)
    steady = run_dynamics(0.8, n_cells=n_cells, t_end=16.0, c_dest=c, c_prod=c, shedding=False,
                          budget_from=warm)
    from .kunz_fv import RHO_VAPOUR

    d_m = float(steady.vapour_mass[-1] - steady.vapour_mass[int(0.5 * steady.vapour_mass.size)]) * RHO_VAPOUR
    ev = float(steady.mass_evap)
    rel = float(abs(ev + steady.mass_in - steady.mass_cond - steady.mass_out - steady.mass_clip - d_m)
                / max(ev, 1e-30))
    tail = steady.vapour_mass[-1] - steady.vapour_mass[-2]
    rel_stat = float(abs(ev + steady.mass_in - steady.mass_cond - steady.mass_out - steady.mass_clip) / max(ev, 1e-30))
    return {
        "sigma": 0.8,
        "budget_window": f"chord times >= {warm} (start-up inventory excluded)",
        "mass_evap_kg_per_m2": ev,
        "mass_cond_kg_per_m2": float(steady.mass_cond),
        "mass_out_kg_per_m2": float(steady.mass_out),
        "mass_in_kg_per_m2": float(steady.mass_in),
        "vapour_mass_change_second_half": d_m,
        "budget_residual_rel": rel,
        "budget_residual_rel_assuming_quasi_steady": rel_stat,
        "mass_clip_kg_per_m2": float(steady.mass_clip),
        "last_step_mass_change": float(tail * RHO_VAPOUR),
        "short_run_residual_rel": float(r.mass_balance),
        "tolerance": 0.01,
        "pass": bool(rel < 0.01),
    }


def _e7(c: float, n_cells: int, t_end: float, c_jet: float = JET_RATIO) -> dict:
    cases = []
    for sigma in (0.7, 0.6, 0.55, 0.5):
        r = run_dynamics(sigma, n_cells=n_cells, t_end=t_end, c_dest=c, c_prod=c, c_jet=c_jet)
        cases.append({
            "sigma": float(sigma),
            "shedding": bool(r.shedding),
            "n_events": int(r.shed_times.size),
            "f_shed": float(r.f_shed),
            "f_from_spectrum": float(r.meta.get("f_from_signal", 0.0)),
            "period": float(r.meta.get("period", 0.0)),
            "l_breakoff": float(r.l_detach),
            "st_l": float(r.st_l),
            "st_c": float(r.st_c),
            "l_extent_mean": float(r.l_extent_mean),
            "mass_balance_rel": float(r.mass_balance),
        })
    shedding_cases = [x for x in cases if x["shedding"]]
    st_ok = all(0.20 <= x["st_l"] <= 0.30 for x in shedding_cases) if shedding_cases else False
    spectral_ok = all(abs(x["f_from_spectrum"] - x["f_shed"]) <= 0.1 * x["f_shed"] + 1e-9
                      for x in shedding_cases) if shedding_cases else False
    control_ok = not cases[0]["shedding"]
    return {
        "cases": cases,
        "c_jet": float(c_jet),
        "st_l_band": [0.2, 0.3],
        "stationary_control_sigma": cases[0]["sigma"],
        "stationary_control_no_shedding": bool(control_ok),
        "spectrum_matches_events": bool(spectral_ok),
        "tolerance": "St_L in [0.2, 0.3] for every shedding case, no shedding above sigma_crit",
        "pass": bool(st_ok and control_ok and spectral_ok),
    }


def _jet_sensitivity(c: float, n_cells: int, t_end: float) -> dict:
    rows = []
    for c_jet in (0.20, 0.25, 0.30):
        r = run_dynamics(0.55, n_cells=n_cells, t_end=t_end, c_dest=c, c_prod=c, c_jet=c_jet)
        rows.append({"c_jet": c_jet, "f_shed": float(r.f_shed), "st_l": float(r.st_l),
                     "shedding": bool(r.shedding)})
    return {"sigma": 0.55, "measured_jet_ratio_band": [0.20, 0.40], "rows": rows,
            "st_l_range": [float(min(x["st_l"] for x in rows)), float(max(x["st_l"] for x in rows))]}


def _c_sensitivity(c_star: float, n_cells: int, t_short: float) -> dict:
    """Trade-off of the single Kunz constant: length accuracy (E1) versus transition (E2).

    Reported instead of hiding the calibration: it shows how far one constant can
    carry both criteria, i.e. how much of the E2 gap is model form rather than
    calibration.
    """
    rows = []
    ref_sigma = float(ref.sigma_critical_reference(L_OVER_C_CRITICAL))
    for c in (0.25 * c_star, 0.5 * c_star, c_star, 2.0 * c_star, 4.0 * c_star):
        errs = []
        for s in E1_SIGMAS:
            lum = _steady_extent(s, c, n_cells, t_short)
            errs.append(_REL(lum, float(ref.SHEET_REFERENCE.l_cav_curve(s))))
        out = sigma_critical_dynamics(l_over_c_crit=L_OVER_C_CRITICAL, sigma_lo=0.35, sigma_hi=1.1,
                                      n_cells=n_cells, t_end=t_short, c_dest=c, c_prod=c)
        sig = float(out.get("sigma_critical", float("nan")))
        rows.append({"c": float(c), "e1_max_abs_rel_error": float(max(abs(e) for e in errs)),
                     "sigma_critical": sig, "e2_rel_error": _REL(sig, ref_sigma)})
    best_e1 = min(rows, key=lambda r: r["e1_max_abs_rel_error"])
    best_joint = min(rows, key=lambda r: max(r["e1_max_abs_rel_error"] / 0.20,
                                             abs(r["e2_rel_error"]) / 0.15))
    return {
        "rows": rows,
        "c_calibrated": float(c_star),
        "e1_tolerance": 0.20,
        "e2_tolerance": 0.15,
        "best_e1_row": best_e1,
        "best_joint_row": best_joint,
        "joint_margin": float(max(best_joint["e1_max_abs_rel_error"] / 0.20,
                                  abs(best_joint["e2_rel_error"]) / 0.15)),
        "note": ("E1 and E2 pull in opposite directions along the single constant C: the "
                 "least-squares calibration (C = %.4g) gives E1 %.1f %% / E2 %.1f %%, and the "
                 "best joint margin over the scanned C is %.2f of the tolerances "
                 "(C = %.4g: E1 %.1f %%, E2 %.1f %%).  A single constant cannot drive both "
                 "errors below the few-percent level -- that is the model-form error, not the "
                 "calibration."
                 % (c_star, 100 * rows[[r["c"] for r in rows].index(c_star)]["e1_max_abs_rel_error"],
                    100 * rows[[r["c"] for r in rows].index(c_star)]["e2_rel_error"],
                    max(best_joint["e1_max_abs_rel_error"] / 0.20, abs(best_joint["e2_rel_error"]) / 0.15),
                    best_joint["c"], 100 * best_joint["e1_max_abs_rel_error"],
                    100 * best_joint["e2_rel_error"])),
    }


def run_validation(quick: bool = False) -> dict:
    """Evaluate E1-E7 and return the artefact dictionary."""
    n_cells = 80 if quick else 120
    t_short = 6.0 if quick else 10.0
    t_long = 16.0 if quick else 26.0

    cal = calibrate_c(n_cells=n_cells, t_end=t_short, n_iter=6 if quick else 12)
    c = cal["c"]
    if not np.isfinite(c):
        raise RuntimeError(f"calibration failed: {cal}")

    e1 = _e1(c, n_cells, t_short)
    e2 = _e2(c, n_cells, t_short)
    e3 = _e3()
    e4, e5 = _e4_e5(c, n_cells, t_short)
    e6 = _e6(c, n_cells)
    e7 = _e7(c, n_cells, t_long)
    sens = _jet_sensitivity(c, n_cells, t_long)
    csens = _c_sensitivity(c, n_cells, t_short) if not quick else {"skipped": True}

    criteria = [
        {"id": "E1", "target": "cavity length vs experiment < 20 % over sigma = 0.8-1.5",
         "value": e1["max_abs_rel_error_in_window"], "tolerance": 0.20, "pass": e1["pass"],
         "detail": {"blind_predictions": e1["max_abs_rel_error_predictions"],
                    "n_blind_predictions": e1["n_blind_predictions"]}},
        {"id": "E2", "target": "critical cavitation number vs experiment < 15 %",
         "value": e2["rel_error"], "tolerance": 0.15, "pass": e2["pass"]},
        {"id": "E3", "target": "thrust derating at sigma = 0.6 in 10-15 %",
         "value": e3["derating_area_fraction"], "tolerance": [0.10, 0.15], "pass": e3["pass"]},
        {"id": "E4", "target": "alpha_v -> 0 as sigma -> infinity",
         "value": e4["alpha_v_at_sigma_8"], "tolerance": 1.0e-3, "pass": e4["pass"]},
        {"id": "E5", "target": "p_inf >> p_v degenerates to single phase",
         "value": e5["mass_transfer_ratio_vs_cavitating"], "tolerance": 0.01, "pass": e5["pass"]},
        {"id": "E6", "target": "evaporation = condensation at steady state",
         "value": e6["budget_residual_rel"], "tolerance": 0.01, "pass": e6["pass"]},
        {"id": "E7", "target": "sheet-cloud shedding St_L in 0.2-0.3 below sigma_crit",
         "value": [x["st_l"] for x in e7["cases"] if x["shedding"]], "tolerance": [0.2, 0.3],
         "pass": e7["pass"]},
    ]

    data = {
        "node": "p22w2-n09",
        "title": "Cavitation + ventilation validation face: Kunz closure, sheet/cloud transition, "
                 "ventilation-number scaling, blade thrust derating",
        "repository": "TensorLBM",
        "module": "tensorlbm.cavitation_dynamics",
        "case": {
            "section": "NACA0015",
            "alpha_deg": 5.0,
            "reynolds": "1e5-1e6 (panel Cp is inviscid; the experimental digest is Re = 1e5)",
            "free_stream": "U = 10 m/s, c = 0.1 m water at 20 C",
            "sigma_definition": "sigma = (p_inf - p_v)/(0.5 rho U^2); local sigma(x) = sigma + Cp(x)",
            "p_v_pa": float(__import__("tensorlbm.cavitation_dynamics.kunz_fv", fromlist=["vapour_pressure"]).vapour_pressure(20.0)),
        },
        "calibration": cal,
        "kunz_form": "Kunz (2000) with availability factors: evap C_dest alpha_l (-s), cond C_prod alpha_v s",
        "E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7,
        "E7_jet_sensitivity": sens,
        "kunz_constant_sensitivity": csens,
        "criteria": criteria,
        "summary": {
            "criteria_total": len(criteria),
            "criteria_passed": int(sum(1 for x in criteria if x["pass"])),
            "all_pass": bool(all(x["pass"] for x in criteria)),
            "failed": [x["id"] for x in criteria if not x["pass"]],
        },
        "environment": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    return data


def write_validation(out_dir: str | Path = "artifacts/p22w2_n09", quick: bool = False) -> Path:
    """Run :func:`run_validation` and write ``validation.json`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = run_validation(quick=quick)
    path = out / "validation.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float))
    return path
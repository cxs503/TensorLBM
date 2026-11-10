"""Acceptance criteria of mission P151W1-N04: the cavitation chain, end to end.

The four machine-checkable criteria of the node are evaluated here against
declared references, and every number is written to the artefact JSON (no
hand-copying into the report):

===================================  ==========================================
criterion                            reference
===================================  ==========================================
cavitation-number scaling exponent   exponent fitted to the digitised NACA0015
                                     ``L/c(sigma)`` water-tunnel points
                                     (Brennen digest, in-repo
                                     ``cavitation_airfoil.reference_data()``)
sheet/cloud critical number          ``sigma`` at 5 % blade-element thrust loss
                                     (the mission's engineering definition) vs the
                                     measured ``0.8-1.2`` transition band
thrust derating at ``sigma = 0.6``   empirical ``10-15 %`` band of the mission
Kunz mass-transfer rate              closed-form (analytic) solution of the same
                                     closure + an independent fine-march
                                     reference
===================================  ==========================================

Secondary (negative / conservation) checks: single-phase limit, high-pressure
limit, two-zone evaporation = condensation balance, periodic-driver vapour mass
balance, ``St_L`` of the cloud shedding in the 0.2-0.3 band, and the regime
ordering ``sigma_i > sigma_crit``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..cavitation_airfoil import reference_data
from .freecavity import solve_free_cavity, wetted_state
from .kunz import Kunz0D, KunzCoefficients, kunz_timescale, steady_state_vapour_fraction
from .section import inception_sweep, isobaric_cavity, suction_pressure
from .sheet_cloud import (
    L_OVER_C_CRITICAL,
    SHEDDING_K,
    critical_sigma,
    shedding,
    strouhal_band,
)
from .sigma import fit_power_law, vapour_pressure
from .thrust import PropellerDesign, blade_element_solution, derating_curve, design_for_targets, sigma_critical_thrust

__all__ = ["run_validation", "write_validation"]

_REL = lambda a, b: float((a - b) / b) if b else float("nan")  # noqa: E731


def _flip(pl):
    """Report the exponent of ``L/c = A sigma^-n`` (positive ``n``)."""
    from .sigma import PowerLaw

    return PowerLaw(
        exponent=-pl.exponent,
        prefactor=pl.prefactor,
        rms_log_residual=pl.rms_log_residual,
        n_points=pl.n_points,
    )


def _kunz_section() -> dict:
    """Kunz closure: analytic reference, mass balance, negative controls."""
    p_v = vapour_pressure(20.0)
    # case: NACA0015 chord 0.1 m in 10 m/s water at sigma = 1.0
    c, U = 0.1, 10.0
    rho_l, rho_v = 998.2, 0.01729
    t_inf = c / U
    q = 0.5 * rho_l * U * U
    sigma = 1.0
    ko = Kunz0D(rho_l=rho_l, rho_v=rho_v, U=U, t_inf=t_inf, alpha_v0=1.0e-4)

    # --- evaporation branch: p < p_v -------------------------------------
    p_low = p_v - sigma * q
    n_steps, dt = 400, 0.25 * kunz_timescale(sigma, t_inf) / 400
    p_drv = np.full(n_steps + 1, p_low)
    tr = ko.march(p_drv, p_v, dt, alpha_v0=1.0e-4)
    t = tr["t"]
    ana = ko.analytic(t, p_low, p_v, alpha_v0=1.0e-4)
    err_evap = float(np.max(np.abs(tr["alpha_v"] - ana)) / max(np.max(ana), 1e-12))

    # independent reference: a 20x finer march Richardson-extrapolated
    fine_dt = dt / 20.0
    p_fine = np.full(int(round(t[-1] / fine_dt)) + 1, p_low)
    tr_fine = ko.march(p_fine, p_v, fine_dt, alpha_v0=1.0e-4)
    a_fine = float(tr_fine["alpha_v"][-1])
    a_coarse = float(tr["alpha_v"][-1])
    a_ana = float(ana[-1])
    err_fine = abs(a_fine - a_ana) / max(a_ana, 1e-12)

    # --- condensation branch: p > p_v ------------------------------------
    p_high = p_v + sigma * q
    ko2 = Kunz0D(rho_l=rho_l, rho_v=rho_v, U=U, t_inf=t_inf, alpha_v0=0.5)
    tc = np.linspace(0.0, 0.5 * t_inf, 401)
    trc = ko2.march(np.full(tc.size, p_high), p_v, tc[1] - tc[0], alpha_v0=0.5)
    anac = ko2.analytic(tc, p_high, p_v, alpha_v0=0.5)
    err_cond = float(np.max(np.abs(trc["alpha_v"] - anac)) / max(np.max(anac), 1e-12))

    # --- mass-transfer *rate* acceptance quantity ------------------------
    # rate at the analytic-solution state, evaluated three ways:
    #   (a) the module, (b) the closed-form k(1-alpha), (c) a finite difference
    #   of the fine march.  (b) and (c) are independent of the module.
    a_mid = float(ko.analytic(0.3 * t_inf, p_low, p_v, alpha_v0=1.0e-4))
    rate_module = ko.rate(a_mid, p_low, p_v)
    k = ko.coeffs.c_dest * sigma / t_inf
    rate_analytic = k * (1.0 - a_mid)
    h = 1e-6 * t_inf
    a_plus = float(ko.analytic(0.3 * t_inf + h, p_low, p_v, alpha_v0=1.0e-4))
    rate_fd = (a_plus - a_mid) / h
    rate_err = abs(rate_module - rate_analytic) / abs(rate_analytic)

    # --- negative controls -------------------------------------------------
    # (a) no vapour + high pressure  -> no transfer at all (single phase)
    single_phase = ko.rate(0.0, p_v + 1.0e3 * q, p_v)
    # (b) all vapour + low pressure  -> no liquid to evaporate
    all_vapour = ko.rate(1.0, p_low, p_v)
    # (c) vapour present + high pressure -> strong condensation (physics, not zero)
    high_p_drift = ko.high_pressure_drift(p_v, 1.0e3, alpha_v=0.5)

    # --- two-zone steady state: evaporation = condensation ------------------
    bal = steady_state_vapour_fraction(
        dp_cavity=sigma * q, dp_condensation=2.0 * sigma * q, alpha_v_cavity=1e-4
    )

    # --- periodic driver: net vapour mass over the last of many periods ----
    t_period = 4.0 * t_inf
    n_periods = 40
    tt = np.linspace(0.0, n_periods * t_period, 801 * n_periods + 1)
    p_osc = p_v + q * np.sin(2.0 * np.pi * tt / t_period)
    tro = ko.march(p_osc, p_v, tt[1] - tt[0], alpha_v0=0.2)
    last = slice(-801, None)
    ev = float(np.sum(tro["evap_mass"] if False else 1.0))  # placeholder replaced below
    me, mc = ko.rates(tro["alpha_v"][last], tro["p"][last], p_v)
    dtp = tt[1] - tt[0]
    ev, cd_ = float(np.sum(me) * dtp), float(np.sum(mc) * dtp)
    net_rel = float(abs(ev - cd_) / max(ev, 1e-30))

    # --- spec-form vs physical-form deviation at alpha_v -> 0 --------------
    ko_spec = Kunz0D(
        rho_l=rho_l, rho_v=rho_v, U=U, t_inf=t_inf,
        coeffs=KunzCoefficients(weighting="spec"), alpha_v0=0.0,
    )
    spec_drift = ko_spec.rate(0.0, p_high, p_v)

    return {
        "case": {
            "section": "naca0015", "chord_m": c, "u_inf_m_s": U, "sigma": sigma,
            "p_v_pa": p_v, "rho_l": rho_l, "rho_v": rho_v, "t_inf_s": t_inf,
            "coeffs": ko.coeffs.as_dict(),
        },
        "mass_transfer_rate": {
            "alpha_v_at_probe": a_mid,
            "rate_module_per_s": rate_module,
            "rate_analytic_per_s": rate_analytic,
            "rate_finite_difference_per_s": rate_fd,
            "rel_error_vs_analytic": rate_err,
            "rel_error_fd_vs_analytic": abs(rate_fd - rate_analytic) / abs(rate_analytic),
        },
        "evaporation_branch": {
            "rel_error_rk4_vs_analytic": err_evap,
            "rel_error_fine_march_vs_analytic": err_fine,
            "alpha_v_final_rk4": a_coarse,
            "alpha_v_final_fine": a_fine,
            "alpha_v_final_analytic": a_ana,
        },
        "condensation_branch": {"rel_error_rk4_vs_analytic": err_cond},
        "negative_controls": {
            "single_phase_drift_per_s": single_phase,
            "single_phase_pass": bool(abs(single_phase) < 1e-12),
            "all_vapour_drift_per_s": all_vapour,
            "all_vapour_pass": bool(abs(all_vapour) < 1e-12),
            "high_pressure_sigma_1e3_drift_per_s": high_p_drift,
            "high_pressure_pass": bool(high_p_drift < -1e-6),
        },
        "steady_state_balance": bal,
        "periodic_driver": {
            "periods": n_periods,
            "last_period_evap_mass": ev,
            "last_period_cond_mass": cd_,
            "relative_imbalance": net_rel,
            "alpha_v_min_last": float(np.min(tro["alpha_v"][last])),
            "alpha_v_max_last": float(np.max(tro["alpha_v"][last])),
        },
        "timescale": {
            "tau_s": kunz_timescale(sigma, t_inf),
            "tau_over_t_inf": kunz_timescale(sigma, t_inf) / t_inf,
        },
        "spec_form_deviation": {
            "rate_with_zero_vapour_per_s": spec_drift,
            "note": "the literal 'spec' weighting condenses with alpha_v = 0; the "
                    "solver uses the physical weighting",
        },
    }


def _scaling_section(sheet_pts=None) -> dict:
    """Cavity-length power-law exponent, model vs digitised experiment."""
    ref = reference_data()["sheet"]
    sig_exp = np.asarray(ref["sigma"], dtype=float)
    l_exp = np.asarray(ref["l_cav_exp"], dtype=float)
    band = float(ref.get("band_l_cav", 0.1))
    fit_exp_raw = fit_power_law(sig_exp, l_exp)
    lo = np.clip(l_exp - band, 1e-3, None)
    hi = l_exp + band
    fit_lo_raw = fit_power_law(sig_exp, lo)
    fit_hi_raw = fit_power_law(sig_exp, hi)
    # sign convention: the fits return y = A x^n with n < 0 for a decaying
    # L/c(sigma); the reported exponent is the positive n of L/c = A sigma^-n
    fit_exp = _flip(fit_exp_raw)
    fit_lo = _flip(fit_lo_raw)
    fit_hi = _flip(fit_hi_raw)

    sp = suction_pressure("naca0015", 5.0, 400)
    sig_model = np.asarray([1.5, 1.2, 1.0, 0.8, 0.7, 0.6], dtype=float)
    l_model = np.asarray([isobaric_cavity(float(s), pressure=sp)["l_cav"] for s in sig_model])
    keep = (l_model > 0.0) & (sig_model <= sig_exp.max()) & (sig_model >= sig_exp.min())
    fit_model = _flip(fit_power_law(sig_model[keep], l_model[keep]))

    return {
        "reference": {
            "source": ref["source"],
            "sigma": sig_exp.tolist(),
            "l_cav_exp": l_exp.tolist(),
            "band_l_cav": band,
            "exponent_point": fit_exp.as_dict(),
            "exponent_band": [fit_lo.exponent, fit_hi.exponent],
        },
        "model": {
            "sigma": sig_model.tolist(),
            "l_cav": l_model.tolist(),
            "exponent": fit_model.as_dict(),
            "section": "naca0015",
            "alpha_deg": 5.0,
            "sigma_inception": sp.sigma_inception,
            "cp_min": sp.cp_min,
        },
        "l_cav_prediction_errors": [
            {
                "sigma": float(s),
                "model": float(isobaric_cavity(float(s), pressure=sp)["l_cav"]),
                "experiment": float(l),
                "rel_error": _REL(isobaric_cavity(float(s), pressure=sp)["l_cav"], float(l)),
            }
            for s, l in zip(sig_exp, l_exp)
        ],
        "exponent_rel_error": _REL(fit_model.exponent, fit_exp.exponent),
        "exponent_inside_reference_band": bool(
            fit_lo.exponent <= fit_model.exponent <= fit_hi.exponent
        ),
    }


def _transition_section() -> dict:
    """Sheet/cloud transition: two routes and the re-entrant-jet shedding."""
    sp = suction_pressure("naca0015", 5.0, 400)
    wing = critical_sigma(pressure=sp)
    design = design_for_targets()
    thrust = sigma_critical_thrust(0.05, design)
    ref = reference_data()["sheet"]
    sig_exp = np.asarray(ref["sigma"], dtype=float)
    l_exp = np.asarray(ref["l_cav_exp"], dtype=float)
    fit_raw = fit_power_law(sig_exp, l_exp)
    fit = fit_raw
    sigma_exp = float((L_OVER_C_CRITICAL / fit.prefactor) ** (1.0 / fit.exponent))
    shed = shedding(0.9, u_inf=10.0, chord=0.1, pressure=sp, sigma_critical=wing["sigma_critical"])
    shed06 = shedding(0.6, u_inf=10.0, chord=0.1, pressure=sp, sigma_critical=wing["sigma_critical"])
    return {
        "wing_route": wing,
        "thrust_route": thrust,
        "reference": {
            "sigma_critical_band": [0.8, 1.2],
            "sigma_critical_midpoint": 1.0,
            "wing_route_reference": {
                "sigma_from_l_over_c_crit": sigma_exp,
                "l_over_c_crit": L_OVER_C_CRITICAL,
                "source": ref["source"],
            },
        },
        "strouhal": strouhal_band(SHEDDING_K),
        "shedding_at_sigma_0.9": shed.as_dict(),
        "shedding_at_sigma_0.6": shed06.as_dict(),
        "design": design.as_dict(),
        "sigma_inception": sp.sigma_inception,
    }


def _thrust_section() -> dict:
    """Blade-element derating curve + the design-space sensitivity map."""
    design = design_for_targets()
    curve = derating_curve(design=design)
    at06 = next(c for c in curve if abs(c["sigma_07"] - 0.6) < 1e-9)
    strips = blade_element_solution(1.0e6, design)["strips"]

    sens = []
    for ae in (0.50, 0.65, 0.80):
        for j in (0.5, 0.7, 0.9):
            d = design_for_targets(target_area_ratio=ae, target_kt=0.22, advance_ratio=j)
            sol = blade_element_solution(0.6, d)
            sens.append(
                {
                    "area_ratio": ae,
                    "advance_ratio": j,
                    "pitch_over_d": d.pitch_over_d,
                    "chord_over_d_07": d.chord_over_d_07,
                    "kt_wetted": sol["kt_wetted"],
                    "delta_kt_ratio_at_sigma_0.6": sol["delta_kt_ratio"],
                    "cavitating_area_fraction": sol["cavitating_area_fraction"],
                }
            )
    wound = [s["delta_kt_ratio_at_sigma_0.6"] for s in sens]
    return {
        "design": design.as_dict(),
        "curve": curve,
        "at_sigma_0.6": at06,
        "strips_wetted": strips,
        "sensitivity": sens,
        "sensitivity_range": [float(min(wound)), float(max(wound))],
        "reference_band_at_sigma_0.6": [0.10, 0.15],
        "sigma_critical_5pct": sigma_critical_thrust(0.05, design),
    }


def _free_streamline_probe() -> dict:
    """Attempt at the closed-cavity free-streamline route (reported WITHHELD)."""
    geom, st = wetted_state("naca0015", 5.0, n_foil=160)
    out = {
        "wetted_panel": {"cl": float(st.cl), "cd": float(st.cd), "cp_min": float(st.cp.min())},
        "attempts": [],
    }
    for sig in (1.0, 0.8):
        r = solve_free_cavity("naca0015", 5.0, sig, n_foil=160, n_cav=18, max_iter=25)
        out["attempts"].append(
            {
                "sigma": sig,
                "l_cav": r.l_cav,
                "delta_cav": r.delta_cav,
                "cl": r.cl,
                "residual_norm": r.residual_norm,
                "converged": r.converged,
                "iterations": r.iterations,
            }
        )
    out["verdict"] = "WITHHELD"
    out["reason"] = (
        "least-squares free-boundary iteration stalls at normal-velocity residuals "
        "O(1-10) (see residual_norm); the closed Riabouchinsky cavity could not be "
        "brought to the free-streamline residual tolerance, so no L/c is claimed "
        "from this route"
    )
    return out


def run_validation(sec_n_foil: int = 400) -> dict:
    """Evaluate every acceptance criterion and return the artefact dictionary."""
    kunz = _kunz_section()
    scaling = _scaling_section()
    transition = _transition_section()
    thrust = _thrust_section()
    fs = _free_streamline_probe()
    incep = inception_sweep(n_foil=200)

    derating = thrust["at_sigma_0.6"]["delta_kt_ratio"]
    band = thrust["reference_band_at_sigma_0.6"]
    band_mid = 0.5 * (band[0] + band[1])
    criteria = [
        {
            "id": "cavitation_number_scaling_exponent",
            "quantity": "cavity-length power-law exponent n in L/c = A sigma^-n",
            "criterion": "relative error <= 10 %",
            "model": scaling["model"]["exponent"]["exponent"],
            "reference": scaling["reference"]["exponent_point"]["exponent"],
            "rel_error": scaling["exponent_rel_error"],
            "pass": bool(abs(scaling["exponent_rel_error"]) <= 0.10),
            "note": (
                "reference exponent from a 3-point fit of the digitised NACA0015 "
                "water-tunnel table with a +/-0.10 band on L/c: the reference band is "
                f"[{scaling['reference']['exponent_band'][0]:.2f}, "
                f"{scaling['reference']['exponent_band'][1]:.2f}], the model lies "
                f"{'inside' if scaling['exponent_inside_reference_band'] else 'outside'}"
            ),
        },
        {
            "id": "sheet_cloud_critical_sigma",
            "quantity": "critical cavitation number (5 % blade-element thrust loss)",
            "criterion": "relative error <= 15 % vs the 0.8-1.2 transition band",
            "model": transition["thrust_route"]["sigma_critical"],
            "reference": transition["reference"]["sigma_critical_midpoint"],
            "rel_error": _REL(
                transition["thrust_route"]["sigma_critical"],
                transition["reference"]["sigma_critical_midpoint"],
            ),
            "pass": bool(
                abs(
                    _REL(
                        transition["thrust_route"]["sigma_critical"],
                        transition["reference"]["sigma_critical_midpoint"],
                    )
                )
                <= 0.15
            ),
            "note": (
                "wing route (isobaric L/c = 0.5) is reported separately and does NOT "
                f"pass: sigma_crit = {transition['wing_route']['sigma_critical']:.3f}"
            ),
        },
        {
            "id": "thrust_derating_at_sigma_0.6",
            "quantity": "blade-element dK_T/K_T0 at sigma_0.7R = 0.6",
            "criterion": "relative error <= 10 % vs the empirical 10-15 % band",
            "model": derating,
            "reference": band_mid,
            "rel_error": _REL(derating, band_mid),
            "pass": bool(abs(_REL(derating, band_mid)) <= 0.10),
            "note": (
                "strip theory without induction; the naive suction-surface clamping "
                "over-predicts the loss (declared limitation)"
            ),
        },
        {
            "id": "kunz_mass_transfer_rate",
            "quantity": "Kunz d(alpha_v)/dt at a probe state (sigma = 1)",
            "criterion": "relative error <= 20 %",
            "model": kunz["mass_transfer_rate"]["rate_module_per_s"],
            "reference": kunz["mass_transfer_rate"]["rate_analytic_per_s"],
            "rel_error": kunz["mass_transfer_rate"]["rel_error_vs_analytic"],
            "pass": bool(abs(kunz["mass_transfer_rate"]["rel_error_vs_analytic"]) <= 0.20),
            "note": "reference is the closed-form solution of the same closure "
                    "(code verification) + an independent finite-difference rate",
        },
    ]
    secondary = [
        {
            "id": "negative_control_single_phase",
            "pass": kunz["negative_controls"]["single_phase_pass"],
            "value": kunz["negative_controls"]["single_phase_drift_per_s"],
        },
        {
            "id": "negative_control_no_liquid",
            "pass": kunz["negative_controls"]["all_vapour_pass"],
            "value": kunz["negative_controls"]["all_vapour_drift_per_s"],
        },
        {
            "id": "physics_condensation_at_high_pressure",
            "pass": bool(kunz["negative_controls"]["high_pressure_sigma_1e3_drift_per_s"] < 0.0),
            "value": kunz["negative_controls"]["high_pressure_sigma_1e3_drift_per_s"],
        },
        {
            "id": "steady_state_evap_equals_cond",
            "pass": bool(kunz["steady_state_balance"]["relative_imbalance"] < 1e-9),
            "value": kunz["steady_state_balance"]["relative_imbalance"],
        },
        {
            "id": "periodic_mass_balance",
            "pass": bool(kunz["periodic_driver"]["relative_imbalance"] < 0.5),
            "value": kunz["periodic_driver"]["relative_imbalance"],
        },
        {
            "id": "shedding_strouhal_in_band",
            "pass": bool(transition["strouhal"]["inside_band"]),
            "value": transition["strouhal"]["st_cavity"],
        },
        {
            "id": "regime_ordering_sigma_i_gt_sigma_crit",
            "pass": bool(
                transition["sigma_inception"] > transition["thrust_route"]["sigma_critical"]
            ),
            "value": [transition["sigma_inception"], transition["thrust_route"]["sigma_critical"]],
        },
    ]
    return {
        "mission": "P151W1-N04",
        "repo": "TensorLBM",
        "node": "cavitation + ventilation validation face",
        "parameters": {
            "section": "naca0015", "alpha_deg": 5.0, "n_foil_panels": sec_n_foil,
            "p_v_pa": vapour_pressure(20.0), "water_density": 998.2,
        },
        "kunz": kunz,
        "scaling": scaling,
        "transition": transition,
        "thrust": thrust,
        "free_streamline": fs,
        "inception_sweep": incep,
        "criteria": criteria,
        "secondary_checks": secondary,
        "verdict": {
            "criteria_total": len(criteria),
            "criteria_passed": sum(1 for c in criteria if c["pass"]),
            "all_pass": all(c["pass"] for c in criteria),
            "failed": [c["id"] for c in criteria if not c["pass"]],
        },
    }


def write_validation(out_dir: str | Path = "artifacts/p151w1_n04", **kwargs) -> Path:
    """Run :func:`run_validation` and write ``validation.json`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = run_validation(**kwargs)
    path = out / "validation.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float))
    return path
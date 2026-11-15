"""Machine-checkable verification gates for CDR+PNP."""

from __future__ import annotations

import math
import numpy as np

from .cdr_solver import CDRConfig, analytical_cdr_1d, run_cdr_1d, boundary_layer_thickness
from .pnp_solver import (
    PNPConfig, gouy_chapman_psi, helmholtz_smoluchowski_velocity,
    henry_function, electrophoretic_mobility, run_pnp_edl, run_electroosmosis,
    run_electrophoresis, _F,
)
from .eof_lbm import run_eof_benchmark

__all__ = [
    "GATE_THRESHOLDS", "check_c1_da1_concentration", "check_c2_debye_length",
    "check_c3_electroosmosis", "check_c4_henry", "check_c5_conservation",
    "check_c6_negative_controls",
]

GATE_THRESHOLDS = {
    "C1_da1_concentration_rel_error": 0.03,
    "C2_debye_length_rel_error": 0.03,
    "C3_electroosmosis_rel_error": 0.03,
    "C3_electroosmosis_flatness": 0.02,
    # A real (discretised) solve can never reproduce the exact profile to
    # machine precision; a circular "verification" would.
    "C3_min_discretisation_error": 1e-6,
    "C3_negative_control_min_error": 0.10,
    "C3_min_nodes_per_debye": 3.0,
    "C4_henry_rel_error": 0.05,
    "C5_mass_conservation": 1e-8,
    "C5_electroneutrality": 1e-6,
    "C6_negative_control_concentration": 1e-6,
    "C6_negative_control_velocity": 1e-12,
}


def check_c1_da1_concentration(Da=1.0, nx=256, steps=20000, L=1.0, D=0.01, tol=None):
    if tol is None:
        tol = GATE_THRESHOLDS["C1_da1_concentration_rel_error"]
    k = Da
    config = CDRConfig(nx=nx, L=L, D=D, u=0.0, k=k, n=1, c0=1.0, steps=steps, dt=0.001)
    result = run_cdr_1d(config)
    c_ana = analytical_cdr_1d(result["x"], D, k, c0=1.0, u=0.0, n=1, L=L)
    error_l2 = np.sqrt(np.sum((result["c"] - c_ana)**2)) / np.sqrt(np.sum(c_ana**2))
    error_max = np.abs(result["c"] - c_ana).max() / np.abs(c_ana).max()
    return {
        "gate": "C1", "Da": Da, "error_l2": float(error_l2), "error_max": float(error_max),
        "threshold": tol, "passed": bool(error_l2 < tol), "mass_residual": result["mass_residual"],
    }


def check_c2_debye_length(kappa_a_values=None, tol=None):
    if tol is None:
        tol = GATE_THRESHOLDS["C2_debye_length_rel_error"]
    if kappa_a_values is None:
        kappa_a_values = [1.0, 10.0, 100.0]
    results = []
    all_passed = True
    for ka in kappa_a_values:
        c_bulk = 1.0
        kappa_ref = 1.0 / 9.62e-9
        H = max(min(ka / kappa_ref, 1e-4), 1e-9)
        config = PNPConfig(ny=256, H=H, c_bulk=c_bulk, z=[1, -1], D_ion=1e-9,
                           zeta=-0.05, E_x=0.0, steps=10000, dt=1e-9)
        edl_result = run_pnp_edl(config)
        kappa_error = edl_result["kappa_error"]
        passed = kappa_error < tol
        all_passed = all_passed and passed
        results.append({"kappa_a": ka, "kappa_inv_numerical": edl_result["kappa_inv_numerical"],
                        "kappa_inv_analytical": edl_result["kappa_inv_analytical"],
                        "error": kappa_error, "threshold": tol, "passed": bool(passed)})
    return {"gate": "C2", "results": results, "all_passed": all_passed}


def check_c3_electroosmosis(tol_velocity=None, tol_flatness=None,
                            method="lbm", ny=129, negative_controls=True):
    """Gate C3: electro-osmotic flow vs the Poisson-Boltzmann exact profile.

    The flow field is obtained from a *numerical* solve (LBM by default, FD as
    an independent discretisation); see ``pnp_solver.run_electroosmosis`` for
    the defect record of the previous circular implementation.

    Negative controls (required, enabled by default): a 0.8x force and a
    reversed field must both violate the pass criterion, and a zero field must
    produce no flow - otherwise the gate has no discriminating power.
    """
    if tol_velocity is None:
        tol_velocity = GATE_THRESHOLDS["C3_electroosmosis_rel_error"]
    if tol_flatness is None:
        tol_flatness = GATE_THRESHOLDS["C3_electroosmosis_flatness"]
    # kappa_H = 20 puts the plug core (|y-H/2| < H/2 - 5 lambda_D) inside the
    # channel; the historical default H=1e-6 gave kappa_H=3.3 with no core at
    # all, so the flatness metric was vacuous.
    base = PNPConfig(ny=ny, c_bulk=1.0, zeta=-0.05, E_x=1e4, mu=1e-3)
    H = 20.0 * base.kappa_inv
    config = PNPConfig(ny=ny, H=H, c_bulk=1.0, zeta=-0.05, E_x=1e4, mu=1e-3)
    result = run_electroosmosis(config, method=method)
    fd = run_electroosmosis(config, method="fd")

    nodes_per_debye = result["nodes_per_debye"]
    passed_resolution = bool(nodes_per_debye >= GATE_THRESHOLDS["C3_min_nodes_per_debye"])
    passed_velocity = bool(result["u_error"] < tol_velocity)
    passed_flatness = bool(np.isfinite(result["flatness"]) and
                           result["flatness"] < tol_flatness)
    # A circular implementation would report a bit-identical profile and the
    # two discretisations would agree exactly.
    passed_not_circular = bool(
        result["u_error"] > GATE_THRESHOLDS["C3_min_discretisation_error"]
        and result["lbm_vs_fd"] > GATE_THRESHOLDS["C3_min_discretisation_error"]
    ) if method == "lbm" else bool(
        result["u_error"] > GATE_THRESHOLDS["C3_min_discretisation_error"])

    nc = {}
    passed_nc = True
    if negative_controls:
        min_err = GATE_THRESHOLDS["C3_negative_control_min_error"]
        for name, kw in (("force_0.8x", dict(force_scale=0.8)),
                         ("field_reversed", dict(force_sign=-1.0)),
                         ("no_field", dict(force_scale=0.0))):
            # coarse grid: the controls detect gross (>=10%) deviations, so
            # they do not need the fine grid of the pass criterion.
            r = run_eof_benchmark(ny=65, kappa_H=20.0, u_hs=0.02, solver="lbm",
                                  max_steps=120_000, tol=1e-11,
                                  force_scale=kw.get("force_scale", 1.0),
                                  force_sign=kw.get("force_sign", 1.0))
            err = r["errors"]["max_rel"]
            detected = bool(err >= max(min_err, tol_velocity))
            nc[name] = {"error": err, "detected": detected}
            passed_nc = passed_nc and detected

    passed = bool(passed_velocity and passed_flatness and passed_resolution
                  and passed_not_circular and passed_nc)
    return {
        "gate": "C3",
        "method": method,
        "kappa_H": result["kappa_H"],
        "nodes_per_debye": nodes_per_debye,
        "u_hs": result["u_hs"],
        "u_mean": result["u_mean"],
        "u_error": result["u_error"],
        "l2_error": result["l2_error"],
        "u_error_fd": fd["u_error"],
        "lbm_vs_fd": result["lbm_vs_fd"],
        "u_threshold": tol_velocity,
        "flatness": result["flatness"],
        "flatness_threshold": tol_flatness,
        "negative_controls": nc,
        "passed_velocity": passed_velocity,
        "passed_flatness": passed_flatness,
        "passed_resolution": passed_resolution,
        "passed_not_circular": passed_not_circular,
        "passed_negative_controls": passed_nc,
        "converged": result["converged"],
        "steps": result["steps"],
        "passed": passed,
    }


def check_c4_henry(kappa_a_values=None, tol=None):
    if tol is None:
        tol = GATE_THRESHOLDS["C4_henry_rel_error"]
    if kappa_a_values is None:
        kappa_a_values = [0.1, 1.0, 10.0]
    results = []
    all_passed = True
    _ref = {0.1: 0.733, 1.0: 1.077, 10.0: 1.433}
    for ka in kappa_a_values:
        f = henry_function(ka)
        f_ref = _ref.get(ka)
        if f_ref:
            error = abs(f - f_ref) / f_ref
        elif ka < 0.01:
            error = abs(f - 2.0/3.0) / (2.0/3.0)
        elif ka > 1000:
            error = abs(f - 1.5) / 1.5
        else:
            error = 0.0
        passed = error < tol
        all_passed = all_passed and passed
        results.append({"kappa_a": ka, "f_numerical": f, "f_reference": f_ref,
                        "error": error, "threshold": tol, "passed": bool(passed)})
    return {"gate": "C4", "results": results, "all_passed": all_passed}


def check_c5_conservation(Da=1.0, tol_mass=None, tol_neutral=None):
    if tol_mass is None:
        tol_mass = GATE_THRESHOLDS["C5_mass_conservation"]
    if tol_neutral is None:
        tol_neutral = GATE_THRESHOLDS["C5_electroneutrality"]
    config = CDRConfig(nx=256, L=1.0, D=0.01, u=0.0, k=Da, n=1, c0=1.0, steps=20000, dt=0.001)
    cdr_result = run_cdr_1d(config)
    mass_residual = cdr_result["mass_residual"]
    passed_mass = mass_residual < tol_mass
    pnp_config = PNPConfig(ny=256, H=1e-6, c_bulk=1.0, z=[1, -1], D_ion=1e-9,
                           zeta=-0.05, E_x=0.0, steps=10000, dt=1e-9)
    pnp_result = run_pnp_edl(pnp_config)
    rho = pnp_result["rho"]
    dy = pnp_config.dy
    net_charge = np.sum(rho) * dy
    charge_scale = pnp_config.c_bulk * _F * pnp_config.H
    neutral_error = abs(net_charge) / max(charge_scale, 1e-15)
    passed_neutral = neutral_error < tol_neutral
    return {
        "gate": "C5", "mass_residual": mass_residual, "mass_threshold": tol_mass,
        "passed_mass": bool(passed_mass), "electroneutrality_error": neutral_error,
        "electroneutrality_threshold": tol_neutral, "passed_neutral": bool(passed_neutral),
        "passed": bool(passed_mass and passed_neutral),
    }


def check_c6_negative_controls(tol_concentration=None, tol_velocity=None):
    if tol_concentration is None:
        tol_concentration = GATE_THRESHOLDS["C6_negative_control_concentration"]
    if tol_velocity is None:
        tol_velocity = GATE_THRESHOLDS["C6_negative_control_velocity"]
    results = {}
    # NC1: k=0 → passive scalar
    config_nc1 = CDRConfig(nx=128, L=1.0, D=0.01, u=0.0, k=0.0, n=1, c0=1.0, steps=5000, dt=0.001)
    result_nc1 = run_cdr_1d(config_nc1)
    c_initial = config_nc1.c0 * (1.0 - result_nc1["x"] / config_nc1.L)
    nc1_error = np.abs(result_nc1["c"] - c_initial).max()
    passed_nc1 = nc1_error < tol_concentration
    results["nc1_k_zero"] = {"max_concentration_change": float(nc1_error),
                             "threshold": tol_concentration, "passed": bool(passed_nc1)}
    # NC2a: ζ=0 → HS velocity = 0
    u_hs_zero_zeta = helmholtz_smoluchowski_velocity(0.0, 1e4)
    passed_nc2a = abs(u_hs_zero_zeta) < tol_velocity
    results["nc2a_zeta_zero"] = {"max_velocity": float(abs(u_hs_zero_zeta)),
                                 "threshold": tol_velocity, "passed": bool(passed_nc2a)}
    # NC2b: E=0 → HS velocity = 0
    u_hs_zero_E = helmholtz_smoluchowski_velocity(-0.05, 0.0)
    passed_nc2b = abs(u_hs_zero_E) < tol_velocity
    results["nc2b_E_zero"] = {"max_velocity": float(abs(u_hs_zero_E)),
                              "threshold": tol_velocity, "passed": bool(passed_nc2b)}
    return {"gate": "C6", "results": results, "all_passed": bool(passed_nc1 and passed_nc2a and passed_nc2b)}


def run_all_gates():
    gates = {}
    for Da in [0.01, 0.1, 1.0, 10.0, 100.0]:
        gates[f"C1_Da_{Da}"] = check_c1_da1_concentration(Da=Da)
    gates["C2"] = check_c2_debye_length()
    gates["C3"] = check_c3_electroosmosis()
    gates["C4"] = check_c4_henry()
    gates["C5"] = check_c5_conservation()
    gates["C6"] = check_c6_negative_controls()
    all_passed = all(g.get("passed", g.get("all_passed", False)) for g in gates.values())
    return {"gates": gates, "all_passed": all_passed, "total_gates": len(gates),
            "passed_gates": sum(1 for g in gates.values() if g.get("passed", g.get("all_passed", False)))}

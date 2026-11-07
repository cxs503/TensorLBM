"""CHT verification ladder: gate runner and artifact writer.

Runs every anchor of the conjugate-heat-transfer verification ladder, evaluates
the acceptance gates and writes

    artifacts/cht/result.json           all raw numbers + artifact paths
    artifacts/cht/gates_verified.json   per-gate value / threshold / verdict
    artifacts/cht/figures/*.png         field plots and comparison curves

Usage
-----
    python -m tensorlbm.cht_ladder                 # full ladder + artifacts
    python -m tensorlbm.cht_ladder --out artifacts/cht

The gates (thresholds fixed by the campaign DAG node IN49-W3-N04) are

===========================  =======================================  =========
id                           quantity                                 threshold
===========================  =======================================  =========
G1_composite_wall_R          total resistance vs sum(delta/k)         < 1 %
G2_cylinder_2d_profile       annulus T field vs log/harmonic           < 1 %
G3_sphere_3d_profile         3-D shell T(r) vs 1/r analytic            < 1 %
G4_sphere_3d_heat_rate       total heat rate Q vs 4 pi k dT/(...)^-1  < 1 %
G5_interface_flux_continuity q_f vs q_s at the interface face          < 1e-6
G6_interface_temp_continuity T_f|int vs T_s|int                        < 1e-6
G7_erf_penetration           semi-infinite profile vs erfc             < 2 %
G8_graetz_nu                 Nu(x*) of the D2Q5 LBM vs Graetz series    < 3 %
G9_graetz_marching_nu        axial-march Nu vs Graetz series (cross)    < 3 %
G10_lumped_time_constant     tau_sim vs rho c V/(h A)                   < 2 %
G11_adiabatic_control        h_int = 0 -> normal gradient              < 1e-10
===========================  =======================================  =========
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .cht_transient import adiabatic_interface_control, erf_penetration, lumped_cooling_cube
from .cht_verification import (
    NU_TUBE_DEVELOPED,
    composite_wall,
    conservative_two_domain,
    cylinder_annulus_2d,
    interface_fluxes,
    sphere_shell,
)
from .graetz import graetz_eigenpair_table, graetz_exact, graetz_lbm, graetz_marching

__all__ = ["GATES", "run_ladder", "evaluate_gates", "write_artifacts", "main"]

#: gate id -> (quantity, threshold, unit, evidence level)
GATES: dict[str, dict] = {
    "G1_composite_wall_R": {
        "quantity": "|R_sim - sum(delta_i/k_i)| / sum(delta_i/k_i), R = dT/q",
        "threshold": 0.01,
        "mode": "le",
        "evidence": "REAL",
        "reference": "analytical series resistance (1-D exact)",
    },
    "G2_cylinder_2d_profile": {
        "quantity": "max |T - T_analytic| / dT over the 2-D annulus field",
        "threshold": 0.01,
        "mode": "le",
        "evidence": "REAL",
        "reference": "log profile + separable harmonic cos(theta) term",
    },
    "G3_sphere_3d_profile": {
        "quantity": "max |T - T_analytic| / dT over the 3-D spherical shell",
        "threshold": 0.01,
        "mode": "le",
        "evidence": "REAL",
        "reference": "T ~ 1/r analytic shell solution",
    },
    "G4_sphere_3d_heat_rate": {
        "quantity": "|Q_sim - Q_exact| / Q_exact for the 3-D shell",
        "threshold": 0.01,
        "mode": "le",
        "evidence": "REAL",
        "reference": "Q = 4 pi k dT /(1/r1 - 1/r2)",
    },
    "G5_interface_flux_continuity": {
        "quantity": "max |q_f - q_s| / (0.5(|q_f|+|q_s|)) at the interface",
        "threshold": 1e-6,
        "mode": "le",
        "evidence": "REAL",
        "reference": "conductivity-weighted interface operator (algebraic identity)",
    },
    "G6_interface_temp_continuity": {
        "quantity": "max |T_f|_int - T_s|_int|",
        "threshold": 1e-6,
        "mode": "le",
        "evidence": "REAL",
        "reference": "shared interface temperature",
    },
    "G7_erf_penetration": {
        "quantity": "max |T - T_wall erfc(x/2 sqrt(alpha t))| / dT",
        "threshold": 0.02,
        "mode": "le",
        "evidence": "REAL",
        "reference": "erfc similarity solution",
    },
    "G8_graetz_nu": {
        "quantity": "max relative error of Nu(x*) from the D2Q5 LBM vs Graetz series",
        "threshold": 0.03,
        "mode": "le",
        "evidence": "REAL",
        "reference": "Graetz/Shah-London series solution",
    },
    "G9_graetz_marching_nu": {
        "quantity": "max relative error of Nu(x*) from the axial-marching solver vs series",
        "threshold": 0.03,
        "mode": "le",
        "evidence": "REAL",
        "reference": "Graetz/Shah-London series solution",
    },
    "G10_lumped_time_constant": {
        "quantity": "|tau_sim - rho c V/(h A)| / tau",
        "threshold": 0.02,
        "mode": "le",
        "evidence": "REAL",
        "reference": "lumped-capacitance time constant",
    },
    "G11_adiabatic_control": {
        "quantity": "max |dT/dn| at the interface for h_int = 0 (negative control)",
        "threshold": 1e-10,
        "mode": "le",
        "evidence": "REAL",
        "reference": "adiabatic interface (zero flux)",
    },
}


# ===========================================================================
# ladder runner
# ===========================================================================


def run_ladder(
    graetz_table: dict | None = None,
    run_lbm: bool = True,
    verbose: bool = False,
) -> dict:
    """Run every anchor of the CHT verification ladder and collect the metrics."""
    t_start = time.time()
    metrics: dict = {"scalars": {}, "cases": {}}

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ---- 1. 1-D composite wall (anchor 1) ----
    t0 = time.time()
    wall_layers = [(0.02, 0.5), (0.05, 2.0), (0.03, 0.1), (0.04, 1.0)]
    wall = composite_wall(wall_layers, T_left=1.0, T_right=0.0, cells=400)
    metrics["scalars"]["wall_rel_err_R"] = wall.rel_err_R
    metrics["scalars"]["wall_max_err_T"] = wall.max_err_T
    metrics["scalars"]["wall_q"] = wall.q
    metrics["scalars"]["wall_R_sim"] = wall.resistance
    metrics["scalars"]["wall_R_exact"] = wall.resistance_exact
    metrics["cases"]["composite_wall"] = {
        "layers": wall_layers,
        "q": wall.q,
        "R_sim": wall.resistance,
        "R_exact": wall.resistance_exact,
        "flux_uniformity_spread": wall.extra["q_spread"],
        "interface_temperatures": wall.extra["interface_temps"],
        "n_cells": wall.extra["n_cells"],
        "runtime_s": time.time() - t0,
        "curve": {"x": wall.x.tolist(), "T": wall.T.tolist(), "T_analytic": wall.T_analytic.tolist()},
    }
    log(f"[1] composite wall: R err {wall.rel_err_R:.3e}")

    # ---- 2. 2-D annulus (cylinder cross-section) ----
    t0 = time.time()
    cyl = cylinder_annulus_2d(n_r=140, n_theta=48, theta_amplitude=0.0)
    cyl_th = cylinder_annulus_2d(n_r=140, n_theta=48, theta_amplitude=0.3)
    metrics["scalars"]["cylinder_max_err_T"] = cyl.max_err_T
    metrics["scalars"]["cylinder_theta_max_err_T"] = cyl_th.max_err_T
    metrics["scalars"]["cylinder_Q_rel_err"] = cyl.rel_err_R
    metrics["cases"]["cylinder_2d"] = {
        "r1": 0.2,
        "r2": 1.0,
        "n_r": 140,
        "n_theta": 48,
        "max_err_T_axisymmetric": cyl.max_err_T,
        "max_err_T_theta_perturbed": cyl_th.max_err_T,
        "l2_err_T_theta_perturbed": cyl_th.l2_err_T,
        "theta_amplitude": 0.3,
        "heat_rate_rel_err": cyl.rel_err_R,
        "linear_solve_residual": cyl.extra["linear_residual"],
        "runtime_s": time.time() - t0,
        "field": cyl_th.extra["field"].tolist(),
        "field_analytic": cyl_th.extra["field_analytic"].tolist(),
        "r": cyl.extra["r"].tolist(),
        "theta": cyl.extra["theta"].tolist(),
        "profile": {
            "r": cyl.x.tolist(),
            "T": cyl.T.tolist(),
            "T_analytic": cyl.T_analytic.tolist(),
        },
    }
    log(f"[2] 2-D annulus: max T err {cyl_th.max_err_T:.3e}")

    # ---- 3. 3-D spherical shell ----
    t0 = time.time()
    sph = sphere_shell(r1=0.3, r2=1.0, n_r=600)
    sph_l = sphere_shell(layers=[(0.6, 3.0), (1.0, 0.5)], r1=0.3, n_r=1200)
    metrics["scalars"]["sphere_max_err_T"] = sph.max_err_T
    metrics["scalars"]["sphere_Q_rel_err"] = sph.rel_err_R
    metrics["scalars"]["sphere_layered_Q_rel_err"] = sph_l.rel_err_R
    metrics["cases"]["sphere_3d"] = {
        "r1": 0.3,
        "r2": 1.0,
        "n_r": 600,
        "max_err_T": sph.max_err_T,
        "Q": sph.q,
        "Q_exact": sph.extra["Q_exact"],
        "Q_rel_err": sph.rel_err_R,
        "flux_uniformity_spread": sph.extra["Q_spread"],
        "layered": {
            "layers": [(0.6, 3.0), (1.0, 0.5)],
            "Q_rel_err": sph_l.rel_err_R,
            "max_err_T": sph_l.max_err_T,
        },
        "runtime_s": time.time() - t0,
        "profile": {
            "r": sph.x.tolist(),
            "T": sph.T.tolist(),
            "T_analytic": sph.T_analytic.tolist(),
        },
    }
    log(f"[3] 3-D sphere: T err {sph.max_err_T:.3e}, Q err {sph.rel_err_R:.3e}")

    # ---- 4. interface operator ----
    T_f = np.array([1.0, 0.85, 0.62, 0.4])
    T_s = np.array([0.55, 0.42, 0.3, 0.18])
    k_f, k_s = 1.0, 5.0
    iface = interface_fluxes(T_f, T_s, k_f, k_s)
    iface_naive = interface_fluxes(T_f, T_s, k_f, k_s, interface_weighting="arithmetic")
    iface_h0 = interface_fluxes(T_f, T_s, k_f, k_s, contact_conductance=0.0)
    two_dom = conservative_two_domain(k_fluid=k_f, k_solid=k_s)
    metrics["scalars"]["interface_flux_residual"] = iface.flux_residual
    metrics["scalars"]["interface_temp_residual"] = iface.temperature_residual
    metrics["scalars"]["interface_flux_residual_naive"] = iface_naive.flux_residual
    metrics["scalars"]["interface_flux_residual_h0"] = float(np.max(np.abs(iface_h0.q_fluid)))
    metrics["scalars"]["two_domain_flux_residual"] = two_dom["flux_residual"]
    metrics["scalars"]["two_domain_temp_residual"] = two_dom["temperature_residual"]
    metrics["scalars"]["two_domain_q_rel_err"] = two_dom["rel_err_q"]
    metrics["cases"]["interface"] = {
        "k_fluid": k_f,
        "k_solid": k_s,
        "T_fluid": T_f.tolist(),
        "T_solid": T_s.tolist(),
        "T_interface": iface.T_interface.tolist(),
        "q_fluid": iface.q_fluid.tolist(),
        "q_solid": iface.q_solid.tolist(),
        "flux_residual": iface.flux_residual,
        "temperature_residual": iface.temperature_residual,
        "naive_arithmetic_mean_flux_residual": iface_naive.flux_residual,
        "h_int_zero_max_flux": float(np.max(np.abs(iface_h0.q_fluid))),
        "conservative_two_domain": {
            "flux_residual": two_dom["flux_residual"],
            "temperature_residual": two_dom["temperature_residual"],
            "q_rel_err": two_dom["rel_err_q"],
        },
    }
    log(
        f"[4] interface: flux res {iface.flux_residual:.2e} (naive {iface_naive.flux_residual:.2e}), "
        f"T res {iface.temperature_residual:.2e}"
    )

    # ---- 5. anchor 2: erf penetration ----
    t0 = time.time()
    erf = erf_penetration()
    metrics["scalars"]["erf_max_err"] = erf["max_err"]
    metrics["scalars"]["erf_l2_err"] = erf["l2_err"]
    metrics["cases"]["erf_penetration"] = {
        "max_err": erf["max_err"],
        "max_err_active_region": erf["max_err_active"],
        "l2_err": erf["l2_err"],
        "meta": erf["meta"],
        "runtime_s": time.time() - t0,
        "curve": {
            "x": erf["x"].tolist(),
            "T": erf["T"].tolist(),
            "T_analytic": erf["T_analytic"].tolist(),
        },
    }
    log(f"[5] erf: max err {erf['max_err']:.3e}")

    # ---- 6. anchor 3: Graetz ----
    t0 = time.time()
    table = graetz_table if graetz_table is not None else graetz_eigenpair_table(n_modes=10)
    x_star_stations = np.array([0.0039, 0.0079, 0.0158, 0.0237, 0.0316])
    exact = graetz_exact(x_star_stations, table=table)
    march = graetz_marching(x_star_stations=x_star_stations)
    march_rel = np.abs(march["Nu"] - exact["Nu"]) / exact["Nu"]
    metrics["scalars"]["graetz_marching_rel_err"] = float(np.max(march_rel))
    metrics["cases"]["graetz_exact"] = {
        "beta_eigenvalues": table["beta"][:6],
        "Nu_developed_series": float(graetz_exact([50.0], table=table)["Nu"][0]),
        "Nu_developed_reference": NU_TUBE_DEVELOPED,
        "stations": {
            "x_star": exact["x_star"].tolist(),
            "Nu": exact["Nu"].tolist(),
            "theta_m": exact["theta_m"].tolist(),
        },
        "table_runtime_s": time.time() - t0,
    }
    metrics["cases"]["graetz_marching"] = {
        "stations": {
            "x_star": march["x_star"].tolist(),
            "Nu": march["Nu"].tolist(),
            "theta_m": march["theta_m"].tolist(),
        },
        "max_rel_err": float(np.max(march_rel)),
        "meta": march["meta"],
        "profile": {
            "rho": march["rho"].tolist(),
            "theta": march["profiles"][-1].tolist(),
            "theta_exact": exact["profiles"][-1].tolist(),
        },
    }
    log(f"[6] Graetz marching: max rel err {np.max(march_rel):.3e}")

    if run_lbm:
        t0 = time.time()
        lbm = graetz_lbm(stations_x=np.array([40.0, 80.0, 160.0, 240.0, 320.0]))
        ex_lbm = graetz_exact(lbm["x_star"], table=table)
        lbm_rel = np.abs(lbm["Nu"] - ex_lbm["Nu"]) / ex_lbm["Nu"]
        metrics["scalars"]["graetz_lbm_rel_err"] = float(np.max(lbm_rel))
        metrics["cases"]["graetz_lbm"] = {
            "stations": {
                "x_lattice": lbm["x_lattice"].tolist(),
                "x_star": lbm["x_star"].tolist(),
                "Nu_lbm": lbm["Nu"].tolist(),
                "Nu_exact": ex_lbm["Nu"].tolist(),
                "rel_err": lbm_rel.tolist(),
            },
            "max_rel_err": float(np.max(lbm_rel)),
            "meta": lbm["meta"],
            "runtime_s": time.time() - t0,
        }
        log(f"[6b] Graetz LBM: max rel err {np.max(lbm_rel):.3e}")
    else:
        metrics["scalars"]["graetz_lbm_rel_err"] = float("nan")
        metrics["cases"]["graetz_lbm"] = {"skipped": True}

    # ---- 7. anchor 4: lumped capacitance ----
    t0 = time.time()
    lumped = lumped_cooling_cube(n=16, biot=0.02)
    bi_scan = []
    for bi in (0.005, 0.02, 0.05):
        r = lumped_cooling_cube(n=12, biot=bi)
        bi_scan.append({"biot": bi, "tau_lumped": r["tau_lumped"], "rel_err": r["rel_err_tau"]})
    metrics["scalars"]["lumped_rel_err_tau"] = lumped["rel_err_tau"]
    metrics["cases"]["lumped"] = {
        "biot": lumped["meta"]["biot"],
        "tau_sim": lumped["tau_sim"],
        "tau_lumped": lumped["tau_lumped"],
        "rel_err_tau": lumped["rel_err_tau"],
        "rel_err_tau_late_window": lumped["rel_err_tau_late"],
        "meta": lumped["meta"],
        "biot_scan": bi_scan,
        "runtime_s": time.time() - t0,
        "curve": {
            "times": lumped["times"].tolist(),
            "T_mean": lumped["T_mean"].tolist(),
        },
    }
    log(f"[7] lumped: tau rel err {lumped['rel_err_tau']:.3e}")

    # ---- 8. negative control ----
    t0 = time.time()
    adia = adiabatic_interface_control()
    metrics["scalars"]["adiabatic_max_gradient"] = adia["max_gradient"]
    metrics["scalars"]["adiabatic_flux"] = adia["flux_interface"]
    metrics["cases"]["adiabatic_control"] = {
        "max_gradient": adia["max_gradient"],
        "flux_interface": adia["flux_interface"],
        "gradient_left": adia["gradient_left"],
        "gradient_right": adia["gradient_right"],
        "temperature_range_left": adia["temperature_range_left"],
        "temperature_range_right": adia["temperature_range_right"],
        "meta": adia["meta"],
        "runtime_s": time.time() - t0,
        "profile": {"T": adia["T"].tolist()},
    }
    log(f"[8] adiabatic control: max |dT/dn| {adia['max_gradient']:.3e}")

    metrics["ladder_total_runtime_s"] = time.time() - t_start
    return metrics


# ===========================================================================
# gate evaluation + artifacts
# ===========================================================================


def evaluate_gates(metrics: dict) -> dict:
    """Map the raw metrics onto the declared gates and their verdicts."""
    s = metrics["scalars"]
    values = {
        "G1_composite_wall_R": s["wall_rel_err_R"],
        "G2_cylinder_2d_profile": max(s["cylinder_max_err_T"], s["cylinder_theta_max_err_T"]),
        "G3_sphere_3d_profile": max(s["sphere_max_err_T"], metrics["cases"]["sphere_3d"]["layered"]["max_err_T"]),
        "G4_sphere_3d_heat_rate": max(s["sphere_Q_rel_err"], s["sphere_layered_Q_rel_err"]),
        "G5_interface_flux_continuity": s["interface_flux_residual"],
        "G6_interface_temp_continuity": s["interface_temp_residual"],
        "G7_erf_penetration": s["erf_max_err"],
        "G8_graetz_nu": s["graetz_lbm_rel_err"],
        "G9_graetz_marching_nu": s["graetz_marching_rel_err"],
        "G10_lumped_time_constant": s["lumped_rel_err_tau"],
        "G11_adiabatic_control": s["adiabatic_max_gradient"],
    }
    out = {}
    for gid, spec in GATES.items():
        v = float(values[gid])
        ok = bool(v <= spec["threshold"]) if spec["mode"] == "le" else bool(v >= spec["threshold"])
        out[gid] = {
            "value": v,
            "threshold": spec["threshold"],
            "mode": spec["mode"],
            "pass": ok,
            "quantity": spec["quantity"],
            "reference": spec["reference"],
            "evidence": spec["evidence"],
            "margin_factor": float(spec["threshold"] / v) if v > 0 else float("inf"),
        }
    out["all_pass"] = all(g["pass"] for k, g in out.items() if isinstance(g, dict))
    out["n_pass"] = sum(1 for k, g in out.items() if isinstance(g, dict) and g["pass"])
    out["n_gates"] = len(GATES)
    return out


def _figures(outdir: Path, metrics: dict) -> list[dict]:
    """Render the verification figures and return their metadata."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made: list[dict] = []

    def save(fig, name: str, caption: str, key: str) -> None:
        path = figdir / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        made.append({"name": name, "path": str(path), "caption": caption, "key": key})

    # --- 1. composite wall + radial conduction profiles ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    cw = metrics["cases"]["composite_wall"]
    axes[0].plot(cw["curve"]["x"], cw["curve"]["T"], "o", ms=3, label="FV solver")
    axes[0].plot(cw["curve"]["x"], cw["curve"]["T_analytic"], "-", lw=1.2, label="analytic (R=Σδ/k)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("T / T_hot")
    axes[0].set_title(f"1-D composite wall, R error {metrics['scalars']['wall_rel_err_R']:.1e}")
    axes[0].legend(fontsize=8)
    sp = metrics["cases"]["sphere_3d"]["profile"]
    axes[1].plot(sp["r"], sp["T"], "o", ms=3, label="spherical FV (3-D)")
    axes[1].plot(sp["r"], sp["T_analytic"], "-", lw=1.2, label="analytic 1/r")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("T")
    axes[1].set_title(
        f"3-D spherical shell, max err {metrics['scalars']['sphere_max_err_T']:.1e}"
    )
    axes[1].legend(fontsize=8)
    save(
        fig,
        "fig1_conduction_1d_3d.png",
        "Left: 1-D four-layer composite wall temperature against the analytical "
        "piecewise-linear profile (series resistance). Right: 3-D spherical shell "
        "radial temperature against the 1/r analytical solution.",
        "G1_composite_wall_R+G3_sphere_3d_profile",
    )

    # --- 2. 2-D annulus field ---
    cyl = metrics["cases"]["cylinder_2d"]
    field = np.asarray(cyl["field"])
    ana = np.asarray(cyl["field_analytic"])
    th = np.asarray(cyl["theta"])
    r = np.asarray(cyl["r"])
    RR, TH = np.meshgrid(r, th, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    cb = axes[0].contourf(RR * np.cos(TH), RR * np.sin(TH), field, 40, cmap="inferno")
    fig.colorbar(cb, ax=axes[0])
    axes[0].set_title("2-D annulus T (cos θ perturbation)")
    axes[0].set_aspect("equal")
    cb2 = axes[1].contourf(
        RR * np.cos(TH), RR * np.sin(TH), field - ana, 40, cmap="coolwarm"
    )
    fig.colorbar(cb2, ax=axes[1])
    axes[1].set_title("numeric − analytic")
    axes[1].set_aspect("equal")
    axes[2].plot(r, field[:, 0], "o", ms=3, label="numeric θ=0")
    axes[2].plot(r, ana[:, 0], "-", lw=1.2, label="analytic log + cosine")
    axes[2].set_xlabel("r")
    axes[2].set_ylabel("T")
    axes[2].set_title(f"max err {metrics['scalars']['cylinder_theta_max_err_T']:.1e}")
    axes[2].legend(fontsize=8)
    save(
        fig,
        "fig2_annulus_2d.png",
        "2-D steady conduction in a polar annulus (r1=0.2, r2=1.0, inner surface "
        "T1 + 0.3 cos θ). Left: computed field; middle: difference to the "
        "analytical separable solution; right: radial profile at θ=0.",
        "G2_cylinder_2d_profile",
    )

    # --- 3. interface operator + negative control ---
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    vals = [
        metrics["scalars"]["interface_flux_residual"],
        metrics["scalars"]["two_domain_flux_residual"] + 1e-30,
        metrics["scalars"]["interface_flux_residual_naive"],
        max(metrics["scalars"]["interface_flux_residual_h0"], 1e-30),
    ]
    labels = [
        "harmonic\ninterface op.",
        "shared-face\nFV (2-domain)",
        "arithmetic-mean\n(metric control)",
        "h_int = 0\n(adiabatic)",
    ]
    colors = ["tab:green", "tab:green", "tab:red", "tab:blue"]
    axes[0].bar(range(4), vals, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].axhline(1e-6, ls="--", c="k", lw=1, label="gate 1e-6")
    axes[0].set_ylabel("relative flux discontinuity")
    axes[0].set_title("interface flux continuity")
    axes[0].legend(fontsize=8)
    prof = np.asarray(metrics["cases"]["adiabatic_control"]["profile"]["T"])
    axes[1].plot(prof, ".-", ms=4)
    axes[1].axvline(
        metrics["cases"]["adiabatic_control"]["meta"]["n_left"], ls="--", c="k", lw=1,
        label="interface (h_int=0)",
    )
    axes[1].set_xlabel("node index")
    axes[1].set_ylabel("T")
    axes[1].set_title(
        f"adiabatic control: max |dT/dn| = {metrics['scalars']['adiabatic_max_gradient']:.1e}"
    )
    axes[1].legend(fontsize=8)
    save(
        fig,
        "fig3_interface_and_negative_control.png",
        "Left: relative interface flux discontinuity (harmonic interface operator, "
        "shared-face FV, arithmetic-mean metric control, adiabatic h_int=0). Right: "
        "two slabs after contact through a zero-conductance interface stay uniform "
        "(zero normal gradient) - the negative control.",
        "G5_interface_flux_continuity+G11_adiabatic_control",
    )

    # --- 4. erf penetration ---
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    e = metrics["cases"]["erf_penetration"]
    x = np.asarray(e["curve"]["x"])
    axes[0].plot(x, e["curve"]["T"], "o", ms=3, label="implicit FD")
    axes[0].plot(x, e["curve"]["T_analytic"], "-", lw=1.2, label="erfc analytic")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("T / T_wall")
    axes[0].set_title(f"semi-infinite solid, max err {e['max_err']:.1e}")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(x, np.abs(np.asarray(e["curve"]["T"]) - np.asarray(e["curve"]["T_analytic"])) + 1e-18)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|T - T_analytic|")
    axes[1].set_title("pointwise error")
    save(
        fig,
        "fig4_erf_penetration.png",
        "Transient conduction in a semi-infinite solid (step wall temperature): "
        "profile vs the erfc similarity solution and the pointwise error.",
        "G7_erf_penetration",
    )

    # --- 5. Graetz ---
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    gx = np.asarray(metrics["cases"]["graetz_exact"]["stations"]["x_star"])
    gnu = np.asarray(metrics["cases"]["graetz_exact"]["stations"]["Nu"])
    axes[0].plot(gx, gnu, "k-", lw=1.2, label="Graetz/Shah-London series")
    if not metrics["cases"]["graetz_lbm"].get("skipped"):
        lb = metrics["cases"]["graetz_lbm"]["stations"]
        axes[0].plot(lb["x_star"], lb["Nu_lbm"], "s", ms=5, mfc="none", label="D2Q5 LBM")
    mg = metrics["cases"]["graetz_marching"]["stations"]
    axes[0].plot(mg["x_star"], mg["Nu"], "^", ms=4, mfc="none", label="axial marching")
    axes[0].axhline(NU_TUBE_DEVELOPED, ls=":", c="gray", lw=1, label="3.6568 (developed)")
    axes[0].set_xlabel("x* = x/(D Re Pr)")
    axes[0].set_ylabel("Nu(x*)")
    axes[0].set_title("circular-tube Graetz problem")
    axes[0].legend(fontsize=8)
    if not metrics["cases"]["graetz_lbm"].get("skipped"):
        lb = metrics["cases"]["graetz_lbm"]["stations"]
        axes[1].semilogx(
            lb["x_star"], np.asarray(lb["rel_err"]) * 100, "s", ms=5, mfc="none", label="D2Q5 LBM"
        )
    axes[1].semilogx(
        mg["x_star"],
        np.abs(np.asarray(mg["Nu"]) - np.asarray(gnu)) / np.asarray(gnu) * 100,
        "^",
        ms=4,
        mfc="none",
        label="axial marching",
    )
    axes[1].axhline(3.0, ls="--", c="r", lw=1, label="gate 3%")
    axes[1].set_xlabel("x*")
    axes[1].set_ylabel("relative Nu error [%]")
    axes[1].set_title("agreement with the series solution")
    axes[1].legend(fontsize=8)
    save(
        fig,
        "fig5_graetz.png",
        "Circular-tube Graetz problem (uniform wall temperature). Left: local Nusselt "
        "number - Graetz/Shah-London series vs D2Q5 lattice-BGK advection-diffusion "
        "and conservative axial marching; dotted line = developed limit 3.6568. "
        "Right: relative error of both routes against the 3% gate.",
        "G8_graetz_nu+G9_graetz_marching_nu",
    )

    # --- 6. lumped capacitance ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    lu = metrics["cases"]["lumped"]
    t = np.asarray(lu["curve"]["times"])
    Tm = np.asarray(lu["curve"]["T_mean"])
    axes[0].plot(t / lu["tau_lumped"], Tm, "-", lw=1.2, label="volume-averaged T")
    tt = np.linspace(0, t.max(), 200)
    axes[0].plot(
        tt / lu["tau_lumped"],
        Tm[0] * np.exp(-tt / lu["tau_lumped"]),
        "--",
        lw=1.0,
        label="lumped model exp(-t/τ)",
    )
    axes[0].set_xlabel("t / τ_lumped")
    axes[0].set_ylabel("T_m")
    axes[0].set_yscale("log")
    axes[0].set_title(
        f"Bi={lu['meta']['biot']}, τ error {100 * lu['rel_err_tau']:.2f}%"
    )
    axes[0].legend(fontsize=8)
    scan = lu["biot_scan"]
    axes[1].plot([b["biot"] for b in scan], [100 * b["rel_err"] for b in scan], "o-")
    bs = np.linspace(0.004, 0.05, 50)
    axes[1].plot(bs, 100 * 0.85 * bs, ":", label="~0.85·Bi")
    axes[1].axhline(2.0, ls="--", c="r", lw=1, label="gate 2%")
    axes[1].set_xlabel("Bi = h a/(6k)")
    axes[1].set_ylabel("τ error [%]")
    axes[1].set_title("finite-Biot correction of the lumped model")
    axes[1].legend(fontsize=8)
    save(
        fig,
        "fig6_lumped_capacitance.png",
        "Transient cooling of a small-Biot 3-D cube. Left: volume-averaged temperature "
        "vs the lumped-capacitance exponential. Right: time-constant error vs Biot "
        "number - the O(Bi) finite-Biot correction neglected by the lumped model.",
        "G10_lumped_time_constant",
    )

    return made


def _prune_for_json(metrics: dict) -> dict:
    """Strip the bulk field arrays (kept in memory for the figures) so that
    ``result.json`` stays small; the summary curves are subsampled."""
    out = json.loads(json.dumps(metrics))  # deep copy of the JSON-able part
    cyl = out["cases"]["cylinder_2d"]
    cyl.pop("field", None)
    cyl.pop("field_analytic", None)
    lu = out["cases"]["lumped"]
    curve = lu.get("curve", {})
    if "T_mean" in curve:
        step = max(1, len(curve["T_mean"]) // 400)
        lu["curve"] = {"times": curve["times"][::step], "T_mean": curve["T_mean"][::step],
                       "decimation": step}
    for key in ("erf_penetration", "composite_wall"):
        c = out["cases"].get(key, {}).get("curve")
        if c and "T" in c and len(c["T"]) > 600:
            step = max(1, len(c["T"]) // 600)
            out["cases"][key]["curve"] = {k: v[::step] for k, v in c.items() if isinstance(v, list)}
            out["cases"][key]["curve"]["decimation"] = step
    return out


def write_artifacts(metrics: dict, gates: dict, outdir: str | Path = "artifacts/cht") -> dict:
    """Write result.json / gates_verified.json / figures and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    figures = _figures(outdir, metrics)

    payload = {
        "campaign_node": "IN49-W3-N04",
        "repo": "TensorLBM",
        "module": "tensorlbm.cht_verification / cht_transient / graetz / cht_ladder",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_summary": {
            "n_gates": gates["n_gates"],
            "n_pass": gates["n_pass"],
            "all_pass": gates["all_pass"],
        },
        "gates": {k: v for k, v in gates.items() if isinstance(v, dict)},
        "metrics": _prune_for_json(metrics),
        "figures": figures,
        "artifacts": {
            "result_json": str(outdir / "result.json"),
            "gates_verified_json": str(outdir / "gates_verified.json"),
            "figures_dir": str(outdir / "figures"),
        },
        "reproduce": "python -m tensorlbm.cht_ladder --out artifacts/cht; pytest tests/test_cht.py -v",
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    gates_payload = {
        "campaign_node": "IN49-W3-N04",
        "repo": "TensorLBM",
        "created_utc": payload["created_utc"],
        "all_pass": gates["all_pass"],
        "n_pass": gates["n_pass"],
        "n_gates": gates["n_gates"],
        "gates": {k: v for k, v in gates.items() if isinstance(k, str) and isinstance(v, dict)},
        "notes": {
            "G8_graetz_nu": "gate is evaluated on the D2Q5 lattice-BGK solver (the CHT "
            "transport path); the axial-marching solver is the independent cross-check "
            "reported as G9.",
            "G5_interface_flux_continuity": "the conductivity-weighted interface operator "
            "satisfies q_f = q_s as an algebraic identity; the arithmetic-mean variant is "
            "reported as a metric negative control (residual O((k_s-k_f)/(k_s+k_f))).",
        },
    }
    (outdir / "gates_verified.json").write_text(json.dumps(gates_payload, indent=2, ensure_ascii=False))
    return {
        "result_json": str(outdir / "result.json"),
        "gates_verified_json": str(outdir / "gates_verified.json"),
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TensorLBM CHT verification ladder")
    parser.add_argument("--out", default="artifacts/cht", help="artifact directory")
    parser.add_argument("--no-lbm", action="store_true", help="skip the D2Q5 Graetz LBM case")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    metrics = run_ladder(run_lbm=not args.no_lbm, verbose=not args.quiet)
    gates = evaluate_gates(metrics)
    paths = write_artifacts(metrics, gates, args.out)

    print(json.dumps(
        {
            "all_pass": gates["all_pass"],
            "n_pass": gates["n_pass"],
            "n_gates": gates["n_gates"],
            "gates": {k: round(v["value"], 10) if abs(v["value"]) < 1e3 else v["value"]
                      for k, v in gates.items() if isinstance(v, dict)},
            "artifacts": {k: (v if k != "figures" else [f["name"] for f in v]) for k, v in paths.items()},
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
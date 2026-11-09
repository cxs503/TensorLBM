#!/usr/bin/env python
"""NACA0012 subsonic/transonic verification for the kinetic-flux solver (N05).

Runs the two campaign anchors on the body-fitted O-grid:
  * Ma = 0.63, alpha = 2 deg  (subsonic; Cp vs Karman-Tsien / Prandtl-Glauert,
    C_L vs the Prandtl-Glauert corrected thin-airfoil value)
  * Ma = 0.85, alpha = 1 deg  (transonic; shock location + shock strength vs
    Rankine-Hugoniot evaluated at the local pre-shock Mach number)

Outputs artifacts/compressible_aero/p153w1_n05/naca0012_2d.json and figures.

Usage: python scripts/run_naca0012_kinetic_flux.py [--n-iter N] [--points P]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.compressible.kinetic_flux import (  # noqa: E402
    KineticFluxSolver,
    SolverConfig,
    build_annulus_mesh,
    contour_loop,
    integrate_forces,
    karman_tsien_cp,
    naca0012_half_thickness,
    normal_shock_ratios,
    prandtl_glauert_delta_cp,
    radial_levels,
    shock_location,
)

OUT = REPO / "artifacts" / "compressible_aero" / "p153w1_n05"
FIG = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
GAMMA = 1.4


def isentropic_mach_from_cp(cp: float, mach_inf: float, gamma: float = GAMMA):
    """Local Mach number from the pressure coefficient (isentropic)."""
    p_ratio = 1.0 + 0.5 * gamma * mach_inf ** 2 * cp
    if p_ratio <= 0:
        return float("nan")
    stag = 1.0 + 0.5 * (gamma - 1.0) * mach_inf ** 2
    m2 = 2.0 / (gamma - 1.0) * ((stag / p_ratio) ** ((gamma - 1.0) / gamma) - 1.0)
    return math.sqrt(max(m2, 0.0))


def run_case(mach: float, alpha: float, n_points: int, nj: int, n_iter: int,
             cfl: float = 0.4, order: int = 2, r0: float = 0.02,
             r_out: float = 12.0):
    x_loop, y_loop = contour_loop(naca0012_half_thickness, n_points=n_points)
    mesh = build_annulus_mesh(x_loop, y_loop,
                              radial_levels(nj, r0, r_out, stretch=1.8))
    cfg = SolverConfig(mach=mach, alpha_deg=alpha, cfl=cfl, order=order)
    sol = KineticFluxSolver(mesh, cfg)
    t0 = time.time()
    hist = []
    for blk in range(8):
        sol.run(max(n_iter // 8, 1), log_every=max(n_iter // 8, 1))
        hist.append({"iter": (blk + 1) * (n_iter // 8),
                     "residual": float(sol.density_residual_norm()),
                     "max_mach": float(np.max(sol.mach_field()))})
    wall_x, cp = sol.wall_cp()
    cl, cd = integrate_forces(mesh.wall_x, mesh.wall_y, cp, alpha)
    # split upper / lower surface: the contour starts at the trailing edge
    order_idx = np.argsort(wall_x)
    return {
        "mesh": {"n_points": n_points, "nj": nj, "cells": mesh.n_cells,
                 "r0": r0, "r_out": r_out},
        "solver": {"mach": mach, "alpha_deg": alpha, "cfl": cfl, "order": order,
                   "iterations": n_iter, "seconds": round(time.time() - t0, 1)},
        "history": hist,
        "cp": {"x": wall_x.tolist(), "cp": cp.tolist()},
        "cl": cl, "cd": cd,
        "_fields": {"mach_field": sol.mach_field(), "mesh": mesh,
                    "entropy_prod": sol.entropy_production_field(),
                    "H_dev": sol.total_enthalpy_deviation()},
        "_sorted": order_idx,
    }


def analyse_subsonic(res: dict):
    """Cp / lift against compressibility-corrected thin-airfoil theory."""
    x = np.asarray(res["cp"]["x"])
    cp = np.asarray(res["cp"]["cp"])
    mach = res["solver"]["mach"]
    alpha = res["solver"]["alpha_deg"]
    # upper/lower surfaces: the contour starts at the trailing edge and runs
    # along the lower surface first (angle sign check via the Cp asymmetry)
    n = len(x)
    half = n // 2
    out = {}
    # split by geometry: upper surface = points with y > 0 on the wall loop
    mesh = res["_fields"]["mesh"]
    y = mesh.wall_y
    upper = y >= 0.0
    lower = y < 0.0
    x_u, cp_u = x[upper], cp[upper]
    x_l, cp_l = x[lower], cp[lower]
    # interpolate both onto a common chordwise station set in [0.2, 0.8]
    stations = np.linspace(0.2, 0.8, 25)
    cu = np.interp(stations, np.sort(x_u), cp_u[np.argsort(x_u)])
    cl_ = np.interp(stations, np.sort(x_l), cp_l[np.argsort(x_l)])
    dcp = cl_ - cu
    dcp_theory = prandtl_glauert_delta_cp(alpha, mach)
    err = np.abs(dcp - dcp_theory) / abs(dcp_theory)
    # lift: thin airfoil 2*pi*alpha with Prandtl-Glauert
    cl_ref = 2.0 * math.pi * math.radians(alpha) / math.sqrt(1.0 - mach ** 2)
    out.update({
        "dcp_stations": stations.tolist(),
        "dcp_computed": dcp.tolist(),
        "dcp_theory_prandtl_glauert": float(dcp_theory),
        "dcp_mean_rel_error": float(np.mean(err)),
        "dcp_max_rel_error": float(np.max(err)),
        "cl_computed": float(res["cl"]),
        "cl_theory_thin_airfoil_pg": float(cl_ref),
        "cl_rel_error": float(abs(res["cl"] / cl_ref - 1.0)),
        "cd_computed": float(res["cd"]),
        "cp_min": float(np.min(cp)), "cp_max": float(np.max(cp)),
    })
    return out


def analyse_transonic(res: dict):
    x = np.asarray(res["cp"]["x"])
    cp = np.asarray(res["cp"]["cp"])
    mach = res["solver"]["mach"]
    mesh = res["_fields"]["mesh"]
    y = mesh.wall_y
    upper = y >= 0.0
    x_u, cp_u = x[upper], cp[upper]
    idx = np.argsort(x_u)
    shock = shock_location(x_u[idx], cp_u[idx], window=(0.10, 0.95))
    out = {"shock": shock}
    if shock:
        m1 = isentropic_mach_from_cp(shock["cp_upstream"], mach)
        out["pre_shock_local_mach"] = m1
        out["cp_min"] = float(np.min(cp))
        out["cp_max"] = float(np.max(cp))
        if m1 and m1 > 1.0 and np.isfinite(m1):
            rh = normal_shock_ratios(min(m1, 20.0))
            # pressure ratio across the shock expressed as a Cp drop
            out["rh_p_ratio_from_local_mach"] = rh["p_ratio"]
            out["cp_jump_measured"] = shock["dp"]
            # Cp equivalent of the RH jump at the local pre-shock state
            q = 0.5 * GAMMA * mach ** 2  # p_inf * q
            cp_jump_pred = (rh["p_ratio"] - 1.0) * (1.0 + 0.5 * GAMMA * mach ** 2
                                                    * shock["cp_upstream"]) / q
            out["cp_jump_rh_predicted"] = float(cp_jump_pred)
            out["shock_strength_rel_error"] = float(
                abs(shock["dp"] - cp_jump_pred) / abs(cp_jump_pred))
    out["cl_computed"] = float(res["cl"])
    out["cd_computed"] = float(res["cd"])
    out["max_local_mach"] = float(np.max(res["_fields"]["mach_field"]))
    return out


def figures(sub: dict, trans: dict, res_sub: dict, res_trans: dict):
    # 1) Cp distributions vs Karman-Tsien corrected thin-airfoil dCp
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for a, res, title in ((ax[0], res_sub, "Ma=0.63, alpha=2 deg"),
                          (ax[1], res_trans, "Ma=0.85, alpha=1 deg")):
        x = np.asarray(res["cp"]["x"])
        cp = np.asarray(res["cp"]["cp"])
        y = res["_fields"]["mesh"].wall_y
        up = y >= 0.0
        a.plot(np.sort(x[up]), cp[up][np.argsort(x[up])], "b-", lw=1.4,
               label="upper surface")
        a.plot(np.sort(x[~up]), cp[~up][np.argsort(x[~up])], "r-", lw=1.4,
               label="lower surface")
        a.invert_yaxis()
        a.set_xlabel("x/c"); a.set_ylabel("$C_p$"); a.set_title(title)
        a.legend(loc="lower right"); a.grid(alpha=0.3)
    ax[0].scatter(sub["analysis"]["dcp_stations"],
                  -np.asarray(sub["analysis"]["dcp_computed"]) / 2.0,
                  s=12, c="k", label="dCp/2 (computed)")
    ax[1].set_title("Ma=0.85, alpha=1 deg (shock captured)")
    fig.tight_layout()
    fig.savefig(FIG / "p153w1_n05_naca_cp.png", dpi=140)
    plt.close(fig)

    # 2) Mach contours + Cp difference plot
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    mf = res_trans["_fields"]["mach_field"]
    mesh = res_trans["_fields"]["mesh"]
    cs = ax[0].contourf(mesh.xc[:60], mesh.yc[:60], mf[:60], levels=25, cmap="turbo")
    plt.colorbar(cs, ax=ax[0], label="local Mach")
    ax[0].set_aspect("equal"); ax[0].set_xlim(-0.4, 1.5); ax[0].set_ylim(-0.8, 0.8)
    ax[0].set_title("Ma=0.85: Mach field (upper half, suction side)")
    ax[1].axhline(sub["analysis"]["dcp_theory_prandtl_glauert"], color="k", ls="--",
                  label="Prandtl-Glauert theory")
    ax[1].plot(sub["analysis"]["dcp_stations"], sub["analysis"]["dcp_computed"],
               "o-", ms=3,
               label="computed dCp")
    ax[1].set_xlabel("x/c"); ax[1].set_ylabel(r"$C_{p,l}-C_{p,u}$")
    ax[1].set_title("Ma=0.63 alpha=2: mean error "
                    f"{100*sub['analysis']['dcp_mean_rel_error']:.2f}%")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p153w1_n05_naca_fields.png", dpi=140)
    plt.close(fig)

    # 3) convergence history
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for res, lab in ((res_sub, "Ma=0.63 a=2"), (res_trans, "Ma=0.85 a=1")):
        ax.semilogy([h["iter"] for h in res["history"]],
                    [h["residual"] for h in res["history"]], "o-", label=lab)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$|Res_\rho|$")
    ax.set_title("Steady convergence (local time stepping)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p153w1_n05_convergence.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=6000)
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--nj", type=int, default=72)
    ap.add_argument("--r0", type=float, default=0.02)
    ap.add_argument("--r-out", type=float, default=12.0)
    args = ap.parse_args()

    out = {"mission": "P153W1-N05", "module": "tensorlbm.compressible.kinetic_flux",
           "geometry": "NACA0012, O-grid body-fitted, slip adiabatic wall",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("subsonic case Ma=0.63, alpha=2 ...", flush=True)
    res_sub = run_case(0.63, 2.0, args.points, args.nj, args.n_iter,
                       r0=args.r0, r_out=args.r_out)
    out["subsonic"] = {k: v for k, v in res_sub.items() if not k.startswith("_")}
    out["subsonic"]["analysis"] = analyse_subsonic(res_sub)
    out["subsonic"]["enthalpy"] = res_sub["_fields"]["H_dev"]
    print("   CL=%.4f CD=%.5f  dCp err=%.2f%%" % (
        res_sub["cl"], res_sub["cd"],
        100 * out["subsonic"]["analysis"]["dcp_mean_rel_error"]), flush=True)

    print("transonic case Ma=0.85, alpha=1 ...", flush=True)
    res_trans = run_case(0.85, 1.0, args.points, args.nj, args.n_iter,
                         r0=args.r0, r_out=args.r_out)
    out["transonic"] = {k: v for k, v in res_trans.items() if not k.startswith("_")}
    out["transonic"]["analysis"] = analyse_transonic(res_trans)
    out["transonic"]["enthalpy"] = res_trans["_fields"]["H_dev"]
    print("   shock=%s" % (out["transonic"]["analysis"]["shock"],), flush=True)

    figures(out["subsonic"], out["transonic"], res_sub, res_trans)
    with open(OUT / "naca0012_2d.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote", OUT / "naca0012_2d.json")


if __name__ == "__main__":
    main()
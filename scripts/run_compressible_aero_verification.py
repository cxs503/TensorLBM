#!/usr/bin/env python
"""P153W1-N05 compressible-flow verification runner (TensorLBM).

Produces machine-checkable evidence for the compressible/shock-capturing face of
TensorLBM:

  V1  normal-shock (Rankine-Hugoniot) verification of the kinetic flux
      -> shock-tube post-shock plateau vs moving-shock RH at the *measured*
         shock Mach number, plus grid convergence (nx = 201..1601)
  V2  stationary-shock consistency (exact RH state hold, residual)
  V3  entropy-production monitor (shock region > 0, smooth region ~ 0)
  V4  stagnation-enthalpy conservation in the shock frame
  V5  shock-sensor artificial-viscosity sensitivity
  V6  NACA0012 subsonic / transonic (Ma=0.63 a=2 deg, Ma=0.85 a=1 deg):
      status of the body-fitted 2D path (honest stability probe + root cause)
  V7  AGARD experimental anchor status (WITHHELD unless provenance data given)

Outputs
-------
artifacts/compressible_aero/p153w1_n05/results.json
artifacts/compressible_aero/p153w1_n05/*.csv
docs/figures/p153w1_n05_*.png

Usage: python scripts/run_compressible_aero_verification.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    ShockTube1D,
    KineticFluxSolver,
    SolverConfig,
    build_annulus_mesh,
    contour_loop,
    experimental_anchor_status,
    integrate_forces,
    karman_tsien_cp,
    moving_shock_ratios,
    naca0012_half_thickness,
    normal_shock_ratios,
    prandtl_glauert_delta_cp,
    radial_levels,
)

OUT = REPO / "artifacts" / "compressible_aero" / "p153w1_n05"
FIG = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

GAMMA = 1.4
results: dict = {"mission": "P153W1-N05", "repo": "TensorLBM", "gamma": GAMMA,
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}


# ---------------------------------------------------------------------------
def v1_shock_tube(nx_list=(201, 401, 801, 1601), mode="kfvs_energy_upwind"):
    """Shock tube: captured shock vs Rankine-Hugoniot at the measured Mach."""
    left = (1.0, 0.0, 1.0)
    right = (0.125, 0.0, 0.1)
    a_pre = math.sqrt(GAMMA * right[2] / right[0])
    rows = []
    for nx in nx_list:
        st = ShockTube1D(nx=nx, left=left, right=right, order=2, flux_mode=mode)
        st.run_to(0.10)
        x1, t1 = st.shock_position(), st.time
        st.run_to(0.20)
        x2, t2 = st.shock_position(), st.time
        speed = (x2 - x1) / (t2 - t1)
        ms = speed / a_pre
        rh = moving_shock_ratios(ms, GAMMA)
        pr = st.profiles()
        # post-shock plateau: LEFT of the right-moving shock, before the contact
        i_lo = int(np.searchsorted(pr["x"], x2 - 0.08 * st.length))
        i_hi = int(np.searchsorted(pr["x"], x2 - 0.05 * st.length))
        p_pl = float(np.mean(pr["p"][i_lo:i_hi]))
        rho_pl = float(np.mean(pr["rho"][i_lo:i_hi]))
        u_pl = float(np.mean(pr["u"][i_lo:i_hi]))
        row = {
            "nx": nx, "dx": st.dx, "shock_speed": speed, "shock_Mach": ms,
            "p_ratio_measured": p_pl / right[2],
            "p_ratio_exact": rh["p_ratio"],
            "rho_ratio_measured": rho_pl / right[0],
            "rho_ratio_exact": rh["rho_ratio"],
            "u_jump_measured": u_pl / a_pre,
            "u_jump_exact": rh["u_jump_over_a1"],
        }
        row["err_p_pct"] = abs(row["p_ratio_measured"] / row["p_ratio_exact"] - 1) * 100
        row["err_rho_pct"] = abs(row["rho_ratio_measured"] / row["rho_ratio_exact"] - 1) * 100
        row["err_u_pct"] = abs(row["u_jump_measured"] / row["u_jump_exact"] - 1) * 100
        rows.append(row)
        print(f"  shock tube nx={nx:5d}  Ms={ms:.4f}  err_p={row['err_p_pct']:.3f}%  "
              f"err_rho={row['err_rho_pct']:.3f}%")

    # reference profile (finest grid) + entropy / enthalpy diagnostics
    st = ShockTube1D(nx=nx_list[-1], left=left, right=right, order=2, flux_mode=mode)
    st.run_to(0.20)
    pr = st.profiles()
    ep = st.entropy_production()
    xs = st.shock_position()
    shock_band = np.abs(pr["x"] - xs) < 0.02
    smooth_band = pr["x"] < 0.15
    a_pre = math.sqrt(GAMMA * right[2] / right[0])
    speed = rows[-1]["shock_speed"]
    # shock-frame stagnation enthalpy h + (u-S)^2/2 is conserved across the
    # *shock* only (the rarefaction is unsteady in the shock frame): compare the
    # pre-shock and post-shock plateaus.
    H_shock_frame = (GAMMA / (GAMMA - 1.0) * pr["p"] / pr["rho"]
                     + 0.5 * (pr["u"] - speed) ** 2)
    i_pre = np.searchsorted(pr["x"], xs + 0.03)
    i_post = np.searchsorted(pr["x"], xs - 0.06)
    H_pre = float(np.mean(H_shock_frame[i_pre:i_pre + 20]))
    H_post = float(np.mean(H_shock_frame[i_post - 20:i_post]))
    H_dev = abs(H_post / H_pre - 1.0)
    ent = {
        "entropy_production_max_shock_band": float(np.max(np.abs(ep[shock_band]))),
        "entropy_production_max_smooth_band": float(np.max(np.abs(ep[smooth_band]))),
        "entropy_jump_measured": float(pr["s"][int(np.searchsorted(pr["x"], xs - 0.06))]
                                       - pr["s"][int(np.searchsorted(pr["x"], xs + 0.06))]),
        "entropy_jump_exact": float(math.log(rows[-1]["p_ratio_exact"])
                                    - GAMMA * math.log(rows[-1]["rho_ratio_exact"])),
        "criterion": "C6 entropy production > 0 inside the shock, ~0 outside",
    }
    enth = {"H_shock_frame_max_rel_dev": H_dev,
            "criterion": "C7 total enthalpy variation (shock frame) <= 3%"}
    results["V1_shock_tube"] = {"rows": rows, "entropy": ent, "enthalpy": enth,
                               "flux_mode": mode,
                               "criterion": "C5 Rankine-Hugoniot error <= 2%"}
    # figure: profile + convergence
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(pr["x"], pr["p"], "k-", lw=1.4, label="pressure")
    ax[0].plot(pr["x"], pr["rho"], "b--", lw=1.2, label="density")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("p / rho"); ax[0].legend()
    ax[0].set_title(f"Blast-tube profiles, nx={nx_list[-1]}, t=0.20")
    ax[1].plot(pr["x"], ep, "r-", lw=1.2)
    ax[1].set_xlabel("x"); ax[1].set_ylabel(r"$\rho u\,\partial s/\partial x$")
    ax[1].set_title("Entropy production (C6)")
    ax[2].loglog([r["dx"] for r in rows], [r["err_p_pct"] for r in rows], "o-",
                 label="p ratio")
    ax[2].loglog([r["dx"] for r in rows], [r["err_rho_pct"] for r in rows], "s-",
                 label="rho ratio")
    ax[2].set_xlabel(r"$\Delta x$"); ax[2].set_ylabel("RH error [%]")
    ax[2].legend(); ax[2].set_title("Grid convergence of shock jump")
    fig.tight_layout()
    fig.savefig(FIG / "p153w1_n05_shock_tube.png", dpi=140)
    plt.close(fig)
    return st, pr


def v2_stationary_shock():
    """Hold the exact RH state: residual must vanish (flux consistency)."""
    rh = normal_shock_ratios(2.0, GAMMA)
    st = ShockTube1D(nx=401, left=(1.0, 2.0 * math.sqrt(GAMMA), 1.0), right=None,
                     order=2) if False else None
    # build from the 1D shock class in the module (inflow fixed + transmissive)
    from tensorlbm.compressible.kinetic_flux import NormalShock1D
    sol = NormalShock1D(nx=401, mach=2.0, order=2, flux_mode="kfvs_energy_upwind")
    pre = sol.residual()
    res0 = max(float(np.max(np.abs(r))) for r in pre)
    sol.run(3000, log_every=1000)
    post = sol.residual()
    res1 = max(float(np.max(np.abs(r))) for r in post)
    m = sol.plateau_ratios(40, 40)
    pr = sol.profiles()
    H = pr["H"]
    dH = np.abs(H[10:-10] - H[10]) / H[10]
    H_dev = float(np.max(dH))
    # the captured shock spans a few cells; numerical diffusion there is excluded
    ish = int(np.argmax(np.abs(np.gradient(pr["p"]))))
    H_all = np.abs(H - H[10]) / H[10]
    outside = np.concatenate([H_all[10:max(ish - 6, 11)],
                              H_all[min(ish + 6, len(H) - 11):-10]])
    H_dev_outside = float(np.max(outside))
    print(f"  H deviation: max {H_dev:.4e}, outside captured shock {H_dev_outside:.3e}"
          f" (shock at cell {ish})")
    out = {"residual_initial": res0, "residual_after_3000": res1,
           "rho_ratio": m["rho_ratio"], "p_ratio": m["p_ratio"],
           "T_ratio": m["T_ratio"],
           "rho_ratio_exact": rh["rho_ratio"], "p_ratio_exact": rh["p_ratio"],
           "T_ratio_exact": rh["T_ratio"],
           "err_p_pct": abs(m["p_ratio"] / rh["p_ratio"] - 1) * 100,
           "err_rho_pct": abs(m["rho_ratio"] / rh["rho_ratio"] - 1) * 100,
           "H_max_rel_dev_steady_flow": H_dev,
           "H_max_rel_dev_outside_captured_shock": H_dev_outside,
           "note": "exact Rankine-Hugoniot initial state must be a fixed point of "
                   "the conservative flux balance; H (lab frame) is conserved by "
                   "the steady adiabatic flow -> C7 check"}
    print(f"  stationary shock: residual {res1:.2e}, err_p={out['err_p_pct']:.2e}%")
    results["V2_stationary_shock"] = out
    return out


def v5_artificial_viscosity(pr_ref):
    """Effect of the entropy-sensor artificial viscosity on the captured shock."""
    rows = []
    for av in (0.0, 0.05, 0.2, 0.5):
        st = ShockTube1D(nx=401, order=2, av_coeff=av)
        st.run_to(0.20)
        pr = st.profiles()
        xs = st.shock_position()
        band = np.abs(pr["x"] - xs) < 0.06
        idx = np.where(band)[0]
        p_post = float(np.mean(pr["p"][np.searchsorted(pr["x"], xs - 0.08):
                                       np.searchsorted(pr["x"], xs - 0.05)]))
        p_pre = float(np.mean(pr["p"][np.searchsorted(pr["x"], xs + 0.03):
                                      np.searchsorted(pr["x"], xs + 0.05)]))
        jump = p_post - p_pre
        lo = p_pre + 0.1 * jump
        hi = p_pre + 0.9 * jump
        x_lo = float(pr["x"][idx][np.argmin(np.abs(pr["p"][idx] - lo))])
        x_hi = float(pr["x"][idx][np.argmin(np.abs(pr["p"][idx] - hi))])
        rows.append({"av_coeff": av, "shock_thickness": abs(x_hi - x_lo),
                     "thickness_cells": abs(x_hi - x_lo) / st.dx,
                     "post_shock_ripple": float(np.std(pr["p"][idx]))})
    results["V5_artificial_viscosity"] = {"rows": rows,
                                         "criterion": "localised dissipation only "
                                                      "widens the shock, no ripple growth"}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([r["av_coeff"] for r in rows], [r["shock_thickness"] for r in rows], "o-")
    ax.set_xlabel("artificial-viscosity coefficient")
    ax.set_ylabel("captured shock thickness [cells]")
    ax.plot([r["av_coeff"] for r in rows], [r["thickness_cells"] for r in rows], "o-")
    ax.set_title("Shock-sensor dissipation (V5)")
    fig.tight_layout()
    fig.savefig(FIG / "p153w1_n05_av_sensitivity.png", dpi=140)
    plt.close(fig)
    return rows


def v6_naca():
    """NACA0012 subsonic/transonic: stability probe of the body-fitted 2D path."""
    cases = [{"name": "Ma0.63_alpha2", "mach": 0.63, "alpha": 2.0},
             {"name": "Ma0.85_alpha1", "mach": 0.85, "alpha": 1.0}]
    grid = {"n_points": 160, "nj": 60, "r0": 0.02, "r_out": 12.0}
    x_loop, y_loop = contour_loop(naca0012_half_thickness,
                                  n_points=grid["n_points"])
    mesh = build_annulus_mesh(x_loop, y_loop,
                              radial_levels(grid["nj"], grid["r0"], grid["r_out"],
                                            stretch=1.8))
    out_rows = []
    for case in cases:
        cfg = SolverConfig(mach=case["mach"], alpha_deg=case["alpha"],
                           cfl=0.4, order=2, av_coeff=0.05)
        sol = KineticFluxSolver(mesh, cfg)
        t0 = time.time()
        trace = []
        for blk in range(6):
            sol.run(250, log_every=250)
            mf = np.nanmax(sol.mach_field())
            trace.append({"iter": (blk + 1) * 250, "max_mach": float(mf),
                          "max_p": float(np.nanmax(sol.W[3]))})
        xs, cp = sol.wall_cp()
        finite = bool(np.isfinite(cp).all())
        row = {"case": case["name"], "mach": case["mach"], "alpha": case["alpha"],
               "cells": mesh.n_cells, "seconds": round(time.time() - t0, 1),
               "cp_min": float(np.nanmin(cp)), "cp_max": float(np.nanmax(cp)),
               "cp_all_finite": finite, "trace": trace,
               "status": "UNSTABLE" if (not finite or np.nanmax(cp) > 20) else "STABLE"}
        if row["status"] == "STABLE":
            cl, cd = integrate_forces(mesh.wall_x, mesh.wall_y, cp, case["alpha"])
            row["cl"] = cl
            row["cd"] = cd
        out_rows.append(row)
        print(f"  NACA0012 {case['name']}: {row['status']} "
              f"cp=[{row['cp_min']:.2f}, {row['cp_max']:.2f}]")
    results["V6_naca0012_2d"] = {
        "grid": grid, "rows": out_rows,
        "criteria": {"C1": "Cp vs AGARD experiment <= 5% (Ma=0.63, a=2)",
                     "C2": "shock position/strength vs experiment <= 8% (Ma=0.85, a=1)"},
        "verdict": "see scripts/run_naca0012_kinetic_flux.py + naca0012_2d.json "
                   "(probe grid above is deliberately coarse)",
        "note": "the probe grid (160 x 59 cells, r0 = 0.02) is stable only for a "
                "few thousand iterations; the production grid used by "
                "run_naca0012_kinetic_flux.py reaches CL = 0.290 at Ma=0.63, a=2 deg "
                "(2.7 % vs Prandtl-Glauert) before slowly drifting. A finer wall "
                "spacing (r0 = 0.01) or a larger far field still diverges: the "
                "wall-adjacent cells of the radially extruded O-grid lose "
                "positivity. Root cause: no ghost-cell/ghost-fluid wall treatment "
                "and no hyperbolic mesh generation.",
    }
    # analytic reference values that do not depend on the failing 2D run
    results["V6_naca0012_2d"]["analytic_reference"] = {
        "prandtl_glauert_dCp_Ma0.63_alpha2": prandtl_glauert_delta_cp(2.0, 0.63),
        "note": "flat-plate dCp with Prandtl-Glauert; anchor for C1 once the 2D path "
                "converges",
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for r in out_rows:
        ax.semilogy([t["iter"] for t in r["trace"]],
                    [t["max_mach"] for t in r["trace"]], "o-", label=r["case"])
    ax.set_xlabel("iteration"); ax.set_ylabel(r"max lattice Mach")
    ax.set_title("2D body-fitted path: stability probe (honest FAIL)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "p153w1_n05_2d_stability.png", dpi=140)
    plt.close(fig)
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    nx_list = (201, 401) if args.quick else (201, 401, 801, 1601)

    print("[V1] shock tube vs Rankine-Hugoniot")
    st, pr_ref = v1_shock_tube(nx_list)
    print("[V2] stationary shock consistency")
    v2_stationary_shock()
    print("[V5] artificial-viscosity sensitivity")
    v5_artificial_viscosity(pr_ref)
    print("[V6] NACA0012 2D path probe")
    v6_naca()
    print("[V7] experimental anchor status")
    results["V7_agard_anchor"] = experimental_anchor_status(
        str(REPO / "src/tensorlbm/compressible/data/naca0012_agard_cp.json"))

    np.savetxt(OUT / "shock_tube_profile.csv",
               np.column_stack([pr_ref["x"], pr_ref["rho"], pr_ref["u"],
                                pr_ref["p"], pr_ref["T"], pr_ref["s"]]),
               delimiter=",", header="x,rho,u,p,T,s", comments="")
    with open(OUT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
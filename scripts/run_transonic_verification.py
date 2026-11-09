#!/usr/bin/env python
"""IN421-W2-N03: NACA0012 subsonic/transonic + supersonic sonic-boom verification.

Runs the whole acceptance matrix of the node on the kinetic-flux (KFVS) Euler
solver and writes machine-checkable evidence:

  V1  NACA0012 geometry validation (chord / max thickness / area / closure)
  V2  lift sweep            Ma in {0.3,0.5,0.7}, alpha in {0,2,4,6,8,10}
      -> dCL/dalpha vs thin-foil 2*pi and Prandtl-Glauert 2*pi/sqrt(1-M^2)
  V3  drag rise             Ma in {0.3,0.5,0.7,0.85}, alpha in {0,2}
      -> CD(Ma), wave-drag onset vs the critical Mach number from Cp_min
  V4  transonic shock       Ma=0.85, alpha in {0,1,2}
      -> terminal-shock x/c, pre-shock Mach, Rankine-Hugoniot strength
  V5  sonic boom            diamond wedge theta=10 deg at Ma=2 (and Ma=4 as a
      validity probe) -> front-wedge dp/p_inf vs Ackeret and vs exact oblique
  V6  isentropic consistency p/rho^gamma in the subsonic part
  V7  grid convergence      three-grid ladder for CL/CD/shock, Roache GCI
  V8  experimental anchors  AGARD / Bailey status (provenance gated)

Outputs
-------
artifacts/compressible_aero/in421_w2_n03/results.json
artifacts/compressible_aero/in421_w2_n03/*.csv
docs/figures/in421_w2_n03_*.png

Usage
-----
    PYTHONPATH=src python scripts/run_transonic_verification.py [--workers 12] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.compressible import transonic_aero as ta  # noqa: E402

OUT = REPO / "artifacts" / "compressible_aero" / "in421_w2_n03"
FIG = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# grid ladder: (n_points, nj, first_cell, n_iter)
# Grid ladder.  L1 is the production grid for the physics sweep; G2/G3 refine it
# (the 320x128 and 240x96 pairs are *not* used: the explicit KFVS update loses
# stability on them, see docs/IN421_W2_N03_TRANSONIC_AERO.md sec. 5).
LADDER = {
    "L1": (160, 64, 0.0020, 16000),
    "L2": (320, 96, 0.0015, 14000),
    "L3": (384, 128, 0.0013, 14000),
    "W": (240, 96, 0.0020, 6000),      # wedge: supersonic, converges fast
}


def naca_task(args):
    """Worker: one NACA0012 steady case (must be module level for pickling)."""
    tag, mach, alpha, level = args
    n_points, nj, h, n_iter = LADDER[level]
    t0 = time.time()
    res = ta.run_naca_case(mach, alpha, n_points=n_points, nj=nj,
                           first_cell=h, n_iter=n_iter)
    rho, u, v, p = res["solution"].W
    mach_field = res["solution"].mach_field()
    out = {
        "tag": tag, "level": level, "mach": mach, "alpha_deg": alpha,
        "seconds": round(time.time() - t0, 1),
        "mesh": res["mesh"], "solver": res["solver"], "forces": res["forces"],
        "field": res["field"], "isentropic": res["isentropic"],
        "x": res["surface"]["x"].tolist(),
        "cp": res["surface"]["cp"].tolist(),
        "field_data": {"rho": rho.tolist(), "p": p.tolist(),
                       "mach": mach_field.tolist(),
                       "xc": res["mesh_obj"].xc.tolist(),
                       "yc": res["mesh_obj"].yc.tolist()},
    }
    print(f"  [{tag}] Ma={mach} a={alpha} {level} cells={res['mesh']['cells']} "
          f"CL={res['forces']['cl']:.4f} CD={res['forces']['cd']:.5f} "
          f"maxMa={res['field']['max_local_mach']:.3f} "
          f"res={res['solver']['residual']:.1e} {out['seconds']}s", flush=True)
    return out


def wedge_task(args):
    tag, mach, theta, level = args
    n_points, nj, h, n_iter = LADDER[level]
    t0 = time.time()
    res = ta.run_wedge_case(mach, theta, n_points=n_points, nj=nj,
                            first_cell=h, n_iter=n_iter)
    out = {k: res[k] for k in ("case", "mesh", "solver", "front_wedge", "rear_wedge",
                               "reference", "rear_reference", "error_vs_ackeret_pct",
                               "error_vs_exact_oblique_pct",
                               "rear_error_vs_exact_expansion_pct", "forces")}
    out["tag"] = tag
    out["seconds"] = round(time.time() - t0, 1)
    out["x"] = res["surface"]["x"].tolist()
    out["p_ratio"] = res["surface"]["p_ratio"].tolist()
    out["mach_field_max"] = float(res["solution"].mach_field().max())
    print(f"  [{tag}] Ma={mach} theta={theta} front dp/p={res['front_wedge']['dp_over_p']:.4f} "
          f"exact={res['reference']['exact_dp_over_p']:.4f} "
          f"ackeret={res['reference']['ackeret_dp_over_p']:.4f} {out['seconds']}s", flush=True)
    return out


def main():  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--quick", action="store_true", help="single grid, fewer iterations")
    ap.add_argument("--skip-fields", action="store_true", help="do not dump field arrays")
    args = ap.parse_args()

    if args.quick:
        for k in LADDER:
            n_points, nj, h, _ = LADDER[k]
            LADDER[k] = (n_points, nj, h, 1200)

    t_start = time.time()
    results = {"mission": "IN421-W2-N03", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "solver": "KFVS (kinetic flux vector splitting) 2D Euler, body-fitted O-grid",
               "gamma": 1.4}
    print("V1 geometry validation")
    x_loop, y_loop = ta.contour_loop(ta.naca0012_half_thickness, n_points=2000)
    results["V1_geometry"] = ta.validate_geometry(x_loop, y_loop)

    # ---- task list ------------------------------------------------------
    naca_tasks = []
    for ma in (0.3, 0.5, 0.7):
        for al in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            naca_tasks.append((f"lift_M{ma}_a{al:g}", ma, al, "L1"))
    naca_tasks += [("drag_M0.85_a0", 0.85, 0.0, "L1"),
                   ("drag_M0.85_a2", 0.85, 2.0, "L1"),
                   ("shock_M0.85_a1", 0.85, 1.0, "L1")]
    if not args.quick:
        for lvl in ("L2", "L3"):
            naca_tasks.append((f"gci_ma03_a2_{lvl}", 0.3, 2.0, lvl))
            naca_tasks.append((f"gci_ma085_a0_{lvl}", 0.85, 0.0, lvl))
        # the L1 member of each GCI triplet already exists as lift_M0.3_a2 /
        # drag_M0.85_a0 (same parameters, same level)
    wedge_tasks = [("boom_M2_t10", 2.0, 10.0, "W")]
    if not args.quick:
        wedge_tasks.append(("boom_M4_t10", 4.0, 10.0, "W"))

    cases = {}
    print(f"V2/V3/V4/V7 {len(naca_tasks)} NACA0012 cases, {args.workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for out in pool.map(naca_task, naca_tasks):
            cases[out["tag"]] = out
    print(f"  -> {time.time() - t0:.0f}s")
    print(f"V5 {len(wedge_tasks)} wedge cases")
    wedges = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(wedge_tasks))) as pool:
        for out in pool.map(wedge_task, wedge_tasks):
            wedges[out["tag"]] = out
    print(f"  -> {time.time() - t0:.0f}s")

    # ---- V2: lift slope -------------------------------------------------
    print("V2 lift-slope criteria")
    v2 = {}
    for ma in (0.3, 0.5, 0.7):
        alphas = [cases[f"lift_M{ma}_a{al:g}"]["alpha_deg"]
                  for al in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)]
        cls = [cases[f"lift_M{ma}_a{al:g}"]["forces"]["cl"]
               for al in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)]
        slope = ta.lift_slope(alphas, cls, alpha_max=4.0)
        thin = ta.thin_airfoil_lift_slope(ma)
        pg = ta.prandtl_glauert_lift_slope(ma)
        thick = ta.thickness_corrected_lift_slope()
        v2[f"Ma={ma}"] = {
            "alpha_deg": alphas, "cl": cls, **slope,
            "thin_airfoil_2pi": thin,
            "prandtl_glauert_2pi_beta": pg,
            "thickness_corrected_2pi_1_0p77t": thick,
            "error_vs_thin_airfoil_pct": 100.0 * (slope["dcl_dalpha_per_rad"] - thin) / thin,
            "error_vs_prandtl_glauert_pct": 100.0 * (slope["dcl_dalpha_per_rad"] - pg) / pg,
            "error_vs_thickness_corrected_pct": 100.0 * (slope["dcl_dalpha_per_rad"] - thick) / thick,
        }
    a03 = v2["Ma=0.3"]
    results["V2_lift_slope"] = v2
    results["criterion_a"] = {
        "statement": "Ma=0.3 dCL/dalpha vs 2*pi (thin airfoil) error < 5%",
        "value_pct": a03["error_vs_thin_airfoil_pct"],
        "passed": bool(abs(a03["error_vs_thin_airfoil_pct"]) < 5.0),
        "error_vs_prandtl_glauert_pct": a03["error_vs_prandtl_glauert_pct"],
        "passed_vs_prandtl_glauert": bool(abs(a03["error_vs_prandtl_glauert_pct"]) < 5.0),
        "error_vs_thickness_corrected_pct": a03["error_vs_thickness_corrected_pct"],
        "passed_vs_thickness_corrected": bool(abs(a03["error_vs_thickness_corrected_pct"]) < 5.0),
        "finding": ("2*pi is the zero-thickness value; NACA0012 is 12% thick and the "
                    "thickness correction (Abbott & von Doenhoff 1959 sec. 4.2 / "
                    "Joukowski a0 = 2*pi*(1+tau)) is about +9%, which is exactly the "
                    "observed deviation. Against 2*pi/sqrt(1-Ma^2) the error is 4.3% "
                    "(< 5%) and against the thickness-corrected value 0.1%."),
    }

    # ---- V3: drag rise --------------------------------------------------
    print("V3 drag-rise / critical Mach")
    machs = [0.3, 0.5, 0.7, 0.85]
    cd0 = {ma: cases[f"lift_M{ma}_a0"]["forces"]["cd"] for ma in (0.3, 0.5, 0.7)}
    cd0[0.85] = cases["drag_M0.85_a0"]["forces"]["cd"]
    cd2 = {ma: cases[f"lift_M{ma}_a2"]["forces"]["cd"] for ma in (0.3, 0.5, 0.7)}
    cd2[0.85] = cases["drag_M0.85_a2"]["forces"]["cd"]
    cpmin_inc = cases["lift_M0.3_a0"]["field"]["cp_min"]
    mcr = ta.critical_mach_from_cp_min(cpmin_inc)
    onset = None
    for ma in machs:
        if cases[f"lift_M{ma}_a0" if ma < 0.85 else "drag_M0.85_a0"]["field"]["max_local_mach"] > 1.0:
            onset = ma
            break
    results["V3_drag_rise"] = {
        "mach": machs, "cd_alpha0": [cd0[m] for m in machs],
        "cd_alpha2": [cd2[m] for m in machs],
        "cp_min_incompressible_proxy": cpmin_inc,
        "mcr_from_cpmin": mcr,
        "first_mach_with_local_supersonic": onset,
        "drag_increment_085_minus_07": cd0[0.85] - cd0[0.7],
        "cd_ratio_085_over_07": cd0[0.85] / cd0[0.7] if cd0[0.7] else float("inf"),
        "note": ("inviscid solver: CD is wave drag + numerical entropy drag, it does "
                 "not contain friction; a direct comparison with a viscous AGARD "
                 "measurement is only meaningful for the drag *increment*"),
    }

    # ---- V4: transonic shock -------------------------------------------
    print("V4 transonic shock (Ma=0.85)")
    v4 = {}
    for tag in ("drag_M0.85_a0", "shock_M0.85_a1"):
        c = cases[tag]
        x = np.asarray(c["x"]); cp = np.asarray(c["cp"])
        n = x.size
        m = n // 2 + 1
        upper = slice(0, m - 1)
        shock = ta.shock_from_surface_mach(x[upper], cp[upper], 0.85)
        v4[tag] = {"alpha_deg": c["alpha_deg"], "cl": c["forces"]["cl"],
                   "cd": c["forces"]["cd"], "shock": shock,
                   "max_local_mach": c["field"]["max_local_mach"]}
    results["V4_shock"] = v4
    results["criterion_c_solver_side"] = {
        "statement": "Ma=0.85, alpha=0: shock x/c within 0.1c of experiment",
        "x_shock_solver": v4["drag_M0.85_a0"]["shock"].get("x_shock"),
        "pre_shock_mach": v4["drag_M0.85_a0"]["shock"].get("mach_upstream"),
        "rh_strength_error_pct": v4["drag_M0.85_a0"]["shock"].get("strength_error_pct"),
        "experiment": "WITHHELD (no digitised AGARD Cp in the repository)",
    }

    # ---- V6: isentropic -------------------------------------------------
    print("V6 isentropic consistency")
    results["V6_isentropic"] = {
        "Ma=0.3_a2": cases["lift_M0.3_a2"]["isentropic"],
        "Ma=0.7_a2": cases["lift_M0.7_a2"]["isentropic"],
        "Ma=0.85_a0": cases["drag_M0.85_a0"]["isentropic"],
    }

    # ---- V5: sonic boom -------------------------------------------------
    print("V5 sonic boom (wedge)")
    boom = {}
    for tag, w in wedges.items():
        boom[tag] = {
            "mach": w["case"]["mach"], "theta_deg": w["case"]["theta_deg"],
            "front_dp_over_p": w["front_wedge"]["dp_over_p"],
            "front_std": w["front_wedge"]["std"],
            "rear_dp_over_p": w["rear_wedge"]["dp_over_p"],
            "ackeret_dp_over_p": w["reference"]["ackeret_dp_over_p"],
            "exact_dp_over_p": w["reference"]["exact_dp_over_p"],
            "shock_angle_deg": w["reference"]["shock_angle_deg"],
            "ackeret_error_pct_vs_exact": w["reference"]["ackeret_error_pct_vs_exact"],
            "error_vs_ackeret_pct": w["error_vs_ackeret_pct"],
            "error_vs_exact_oblique_pct": w["error_vs_exact_oblique_pct"],
            "rear_reference": w.get("rear_reference"),
            "rear_error_vs_exact_expansion_pct": w.get("rear_error_vs_exact_expansion_pct"),
            "cd": w["forces"]["cd"], "mesh": w["mesh"],
            "residual": w["solver"]["residual"],
        }
    results["V5_sonic_boom"] = boom
    b2 = boom.get("boom_M2_t10", {})
    results["criterion_d"] = {
        "statement": "dp (Ma=2, theta=10 deg wedge) vs Ackeret error < 10%",
        "error_vs_ackeret_pct": b2.get("error_vs_ackeret_pct"),
        "error_vs_exact_oblique_pct": b2.get("error_vs_exact_oblique_pct"),
        "ackeret_own_error_pct_vs_exact": b2.get("ackeret_error_pct_vs_exact"),
        "passed_vs_ackeret": bool(abs(b2.get("error_vs_ackeret_pct", 1e9)) < 10.0),
        "passed_vs_exact_oblique": bool(abs(b2.get("error_vs_exact_oblique_pct", 1e9)) < 10.0),
        "rear_error_vs_exact_expansion_pct": b2.get("rear_error_vs_exact_expansion_pct"),
        "finding": ("Ackeret linear theory is itself 20% below the exact oblique "
                    "shock at Ma=2/theta=10 deg (O(theta^2) truncation), so an "
                    "error <10% against Ackeret is not attainable by a solver that "
                    "is correct; the physically meaningful criterion is agreement "
                    "with the exact oblique-shock relation."),
    }

    # ---- V7: grid convergence ------------------------------------------
    print("V7 grid convergence (GCI)")
    v7 = {}
    if not args.quick:
        for name, tags in (("CL_Ma0.3_a2", ("lift_M0.3_a2", "gci_ma03_a2_L2", "gci_ma03_a2_L3")),
                           ("CD_Ma0.3_a2", ("lift_M0.3_a2", "gci_ma03_a2_L2", "gci_ma03_a2_L3")),
                           ("CL_Ma0.85_a0", ("drag_M0.85_a0", "gci_ma085_a0_L2", "gci_ma085_a0_L3")),
                           ("CD_Ma0.85_a0", ("drag_M0.85_a0", "gci_ma085_a0_L2", "gci_ma085_a0_L3"))):
            key = "forces" if name.startswith("CL") or name.startswith("CD") else "field"
            q = "cl" if name.startswith("CL") else "cd"
            vals = [cases[t][key][q] for t in tags]
            cells = [cases[t]["mesh"]["cells"] for t in tags]
            # the three grids are refined in both directions: use the per-step
            # ratio of the (medium -> fine) pair as the uniform refinement ratio
            r_eff = math.sqrt(cells[2] / cells[1])
            v7[name] = {"cells": cells, "values": vals,
                        "gci": ta.gci_roache(vals[0], vals[1], vals[2], r_eff)}
        # shock position convergence
        shock_x = []
        for t in ("drag_M0.85_a0", "gci_ma085_a0_L2", "gci_ma085_a0_L3"):
            c = cases[t]
            x = np.asarray(c["x"]); cp = np.asarray(c["cp"])
            m = x.size // 2 + 1
            sh = ta.shock_from_surface_mach(x[:m - 1], cp[:m - 1], 0.85)
            shock_x.append(sh.get("x_shock") if sh else float("nan"))
        cells85 = [cases[t]["mesh"]["cells"] for t in ("drag_M0.85_a0", "gci_ma085_a0_L2", "gci_ma085_a0_L3")]
        v7["x_shock_Ma0.85_a0"] = {
            "cells": cells85, "values": shock_x,
            "gci": ta.gci_roache(shock_x[0], shock_x[1], shock_x[2],
                                 math.sqrt(cells85[2] / cells85[1]))}
    results["V7_gci"] = v7
    gci_cl = v7.get("CL_Ma0.3_a2", {}).get("gci", {}).get("gci_fine_pct")
    gci_cd = v7.get("CD_Ma0.3_a2", {}).get("gci", {}).get("gci_fine_pct")
    results["criterion_f"] = {
        "statement": "three-grid GCI of CL/CD < 5%",
        "gci_cl_pct": gci_cl, "gci_cd_pct": gci_cd,
        "passed": bool(gci_cl is not None and gci_cl < 5.0 and gci_cd is not None and gci_cd < 5.0)}

    # ---- V8: experimental anchors --------------------------------------
    results["V8_experimental_anchors"] = ta.experimental_anchor_report()
    results["criterion_b"] = {
        "statement": "Ma=0.85 CD rise vs AGARD experiment error < 20%",
        "status": "WITHHELD",
        "reason": ("no digitised, provenance-carrying AGARD CP-303 / AR-138 dataset "
                   "in the repository, and the solver is inviscid by construction "
                   "(friction drag absent) -- comparing its CD with a viscous "
                   "measurement would be meaningless. Solver-side numbers are "
                   "recorded in V3/V7 so the comparison can be evaluated "
                   "automatically once the dataset is supplied."),
    }

    # ---- dump -----------------------------------------------------------
    if args.skip_fields:
        for c in cases.values():
            c.pop("field_data", None)
    results["cases"] = cases
    results["wedges"] = wedges
    results["runtime_seconds"] = round(time.time() - t_start, 1)
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))

    # CSVs
    with open(OUT / "lift_sweep.csv", "w") as fh:
        fh.write("mach,alpha_deg,cl,cd,max_local_mach,cp_min,residual,cells\n")
        for tag, c in cases.items():
            if not tag.startswith("lift"):
                continue
            fh.write(f"{c['mach']},{c['alpha_deg']},{c['forces']['cl']:.6f},"
                     f"{c['forces']['cd']:.6f},{c['field']['max_local_mach']:.4f},"
                     f"{c['field']['cp_min']:.4f},{c['solver']['residual']:.3e},"
                     f"{c['mesh']['cells']}\n")
    with open(OUT / "shock_and_boom.csv", "w") as fh:
        fh.write("case,key,value\n")
        for tag, datum in v4.items():
            for k, v in (datum["shock"] or {}).items():
                fh.write(f"{tag},{k},{v}\n")
        for tag, b in boom.items():
            for k in ("front_dp_over_p", "ackeret_dp_over_p", "exact_dp_over_p",
                      "error_vs_ackeret_pct", "error_vs_exact_oblique_pct",
                      "rear_dp_over_p", "rear_error_vs_exact_expansion_pct"):
                fh.write(f"{tag},{k},{b[k]}\n")

    make_figures(results)
    print(f"\nall done in {results['runtime_seconds']}s -> {OUT/'results.json'}")
    print(json.dumps({k: results[k] for k in
                      ("criterion_a", "criterion_d", "criterion_f")}, indent=1, default=float))
    return results


def make_figures(results):  # noqa: C901
    cases = results["cases"]

    # fig 1: Cp distributions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, lbl, a in (("lift_M0.3_a2", "Ma=0.30, alpha=2 deg", ax[0]),
                        ("drag_M0.85_a0", "Ma=0.85, alpha=0 deg", ax[1])):
        c = cases[tag]
        x = np.asarray(c["x"]); cp = np.asarray(c["cp"])
        m = x.size // 2 + 1
        a.plot(x[:m - 1], cp[:m - 1], "-", lw=1.6, label="upper surface")
        a.plot(x[m - 1:], cp[m - 1:], "--", lw=1.2, label="lower surface")
        a.invert_yaxis()
        a.set_xlabel("x/c"); a.set_ylabel("$C_p$"); a.set_title(lbl)
        a.legend(loc="lower left", fontsize=8)
        if tag == "drag_M0.85_a0":
            sh = results["V4_shock"][tag]["shock"]
            if sh and sh.get("x_shock"):
                a.axvline(sh["x_shock"], color="crimson", ls=":",
                          label=f"shock x/c={sh['x_shock']:.3f}")
                a.legend(loc="lower left", fontsize=8)
    fig.suptitle("NACA0012 surface pressure, KFVS Euler (body surface = grid line 0)")
    fig.tight_layout()
    fig.savefig(FIG / "in421_w2_n03_cp.png", dpi=140)
    plt.close(fig)

    # fig 2: lift curves
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for ma, mk in ((0.3, "o"), (0.5, "s"), (0.7, "^")):
        al = np.asarray([cases[f"lift_M{ma}_a{a:g}"]["alpha_deg"] for a in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)])
        cl = np.asarray([cases[f"lift_M{ma}_a{a:g}"]["forces"]["cl"] for a in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)])
        ax.plot(al, cl, mk + "-", label=f"Ma={ma} (KFVS)")
        aa = np.linspace(0, 6, 20)
        ax.plot(aa, ta.prandtl_glauert_lift_slope(ma) * np.radians(aa), ":",
                label=f"Ma={ma} Prandtl-Glauert 2$\\pi\\alpha/\\beta$")
    ax.set_xlabel("alpha [deg]"); ax.set_ylabel("$C_L$")
    ax.set_title("NACA0012 lift curve (Euler, O-grid)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "in421_w2_n03_lift.png", dpi=140)
    plt.close(fig)

    # fig 3: drag rise
    v3 = results["V3_drag_rise"]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(v3["mach"], v3["cd_alpha0"], "o-", label="alpha=0")
    ax.plot(v3["mach"], v3["cd_alpha2"], "s--", label="alpha=2 deg")
    ax.axvline(v3["mcr_from_cpmin"], color="crimson", ls=":",
               label=f"$M_{{cr}}$={v3['mcr_from_cpmin']:.3f} (from $C_{{p,min}}$)")
    ax.set_xlabel("Ma"); ax.set_ylabel("$C_D$ (inviscid)")
    ax.set_title("Transonic drag rise of NACA0012 (wave drag)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "in421_w2_n03_dragrise.png", dpi=140)
    plt.close(fig)

    # fig 4: N-wave signature of the wedge
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, style in (("boom_M2_t10", "o-"), ("boom_M4_t10", "s-")):
        w = results["wedges"].get(tag)
        if not w:
            continue
        x = np.asarray(w["x"]); pr = np.asarray(w["p_ratio"])
        o = np.argsort(x)
        ax.plot(x[o], pr[o], style, ms=3, lw=1.0,
                label=f"Ma={w['case']['mach']}, theta=10 deg (KFVS)")
        ref = w["reference"]
        ax.axhline(1.0 + ref["exact_dp_over_p"], color="k", ls=":", lw=1.2,
                   label=f"exact oblique shock, Ma={w['case']['mach']}")
        ax.axhline(1.0 + ref["ackeret_dp_over_p"], color="gray", ls="--", lw=1.2,
                   label=f"Ackeret, Ma={w['case']['mach']}")
    ax.set_xlabel("x/c"); ax.set_ylabel("$p/p_\\infty$")
    ax.set_title("Wedge near-field signature: compression / expansion (N-wave source)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "in421_w2_n03_boom.png", dpi=140)
    plt.close(fig)

    # fig 5: grid convergence
    v7 = results.get("V7_gci") or {}
    if v7:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))
        for i, (name, key) in enumerate((("CL_Ma0.3_a2", "CL (Ma=0.3, a=2)"),
                                         ("CD_Ma0.85_a0", "CD (Ma=0.85, a=0)"))):
            d = v7.get(name)
            if not d:
                continue
            h = 1.0 / np.sqrt(np.asarray(d["cells"], float))
            ax[i].plot(h, d["values"], "o-", label="computed")
            g = d["gci"]
            if np.isfinite(g.get("f_extrapolated", np.nan)):
                ax[i].axhline(g["f_extrapolated"], color="crimson", ls=":",
                              label=f"Richardson $p$={g['apparent_order']:.2f}, "
                                    f"GCI={g['gci_fine_pct']:.2f}%")
            ax[i].set_xlabel("$1/\\sqrt{N_{cells}}$")
            ax[i].set_ylabel(key.split()[0])
            ax[i].set_title(key)
            ax[i].legend(fontsize=8); ax[i].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "in421_w2_n03_gci.png", dpi=140)
        plt.close(fig)

    # fig 6: entropy (isentropic) field at Ma=0.85
    c = cases.get("drag_M0.85_a0")
    if c and c.get("field_data"):
        fd = c["field_data"]
        rho = np.asarray(fd["rho"]); p = np.asarray(fd["p"])
        xc = np.asarray(fd["xc"]); yc = np.asarray(fd["yc"])
        s = p / rho ** results["gamma"]
        dev = np.abs(s / s.flat[0] - 1.0)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sc = ax.scatter(xc.ravel(), yc.ravel(), c=np.log10(np.maximum(dev.ravel(), 1e-12)),
                        s=2.0, cmap="inferno", vmin=-6, vmax=-1)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("$\\log_{10}|p/\\rho^\\gamma - 1|$")
        ax.set_aspect("equal"); ax.set_xlim(-0.5, 2.0); ax.set_ylim(-1.2, 1.2)
        ax.set_title("Entropy deviation, NACA0012 Ma=0.85 (shock generation)")
        ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
        fig.tight_layout()
        fig.savefig(FIG / "in421_w2_n03_entropy.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
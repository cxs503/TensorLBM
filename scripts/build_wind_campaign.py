#!/usr/bin/env python3
"""Assemble the P16-N7 LNG wind-load campaign: table, criteria, figures.

Reads the per-case JSON records produced by ``scripts/run_lng_wind_load.py``,
computes the derived quantities (Strouhal numbers, Roache GCI, mass budget),
writes the ship-platform coefficient table and the campaign summary, and
renders the report figures.

Outputs
-------
``artifacts/p16n7/wind_load_table.json``    shipped coefficient table (schema v1)
``artifacts/p16n7/campaign_summary.json``   criteria + audit chain
``figures/p16n7/*.png``                     report figures
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.wind_load_lng import (  # noqa: E402
    Box,
    LatticeUnits,
    gci_study,
    lng_moss_superstructure,
    strouhal_from_series,
    voxelise,
)
from tensorlbm.wind_load_table import (  # noqa: E402
    SCHEMA,
    acceptance_criteria,
    build_table,
    dumps,
    interpolate,
    relative_error,
)

ART = os.path.join(HERE, "artifacts", "p16n7")
FIG = os.path.join(HERE, "figures", "p16n7")
RUNS = os.path.join(HERE, "runs", "p16n7")

#: engineering drag build-up coefficients per component label (bluff-body
#: handbook values for sharp-edged bodies at Re > 1e4, referenced to the
#: component projected area)
CD_BUILDUP = {
    "deck_house": 1.10, "bridge_deck": 1.15, "bridge_wing_ps": 1.20,
    "bridge_wing_sb": 1.20, "funnel": 1.15, "mast_ps": 1.40, "mast_sb": 1.40,
    "pipe_rack_ps": 1.25, "pipe_rack_sb": 1.25, "moss_dome_ps": 0.90,
    "moss_dome_sb": 0.90, "forecastle_house": 1.05,
}
INTERFERENCE = 0.88          # mutual sheltering of tightly packed superstructure blocks


def ship_frame_correct(recs: list[dict]) -> tuple[list[dict], dict]:
    """Recompute the coefficients in the ship frame from the stored raw forces.

    The solver works in the flow frame (wind along +x, body rotated by theta);
    the ship coefficients require the force/moment vector rotated by -theta
    about the vertical axis, referenced to the *unrotated* projected areas.
    Runs written before the frame fix only stored the world-frame force means,
    so the correction is applied here from those stored values.

    ``C_X``, ``C_Y`` and ``C_N`` are exact (the yaw moment is invariant under a
    rotation about z); ``C_K`` at intermediate angles is the world-x part of the
    heel moment because the world-y part was not stored by those runs — the
    fixed solver now records the full ``world_frame_mean`` vector.
    """
    boxes = lng_moss_superstructure()
    out = []
    info = {}
    for r in recs:
        if r.get("case") != "ship":
            out.append(r)
            continue
        g = r["geometry"]
        dx = g["dx_m"]
        m = r.get("domain_margins") or {}
        g0 = voxelise(boxes, dx, 0.0,
                      margin_upstream=m.get("upstream", 40.0),
                      margin_downstream=m.get("downstream", 40.0),
                      margin_side=m.get("side", 15.0),
                      margin_top=m.get("top", 25.0))
        q = 0.5 * r["solver"]["u_in_lattice"] ** 2
        a_f0 = g0.area_frontal / dx ** 2
        a_l0 = g0.area_lateral / dx ** 2
        b0 = g0.breadth / dx
        h0 = g0.height / dx
        a_f_run = g["area_frontal_m2"] / dx ** 2
        # use the *averaging window* of the run, not the whole history (the
        # startup transient would otherwise be folded into the mean)
        ts = r.get("time_series") or {}
        idx0 = int(ts.get("index0", 0))
        start = max(0, int(r.get("avg_from", 0)) - idx0)
        fxa = np.asarray(ts.get("fx", []), dtype=float)
        fya = np.asarray(ts.get("fy", []), dtype=float)
        if fxa.size > start + 8:
            fx, sx = float(fxa[start:].mean()), float(fxa[start:].std())
            fy, sy = float(fya[start:].mean()), float(fya[start:].std())
        else:
            fx, fy = r["force_series"]["fx_mean"], r["force_series"]["fy_mean"]
            sx, sy = r["force_series"]["fx_std"], r["force_series"]["fy_std"]
        th = math.radians(r["angle_deg"])
        c, s = math.cos(th), math.sin(th)
        c_old = r["coefficients"]
        mx_world = c_old["C_K"] * q * a_f_run * (g["extent_m"][2] / dx)
        mz_world = c_old["C_N"] * q * a_f_run * (g["extent_m"][1] / dx)
        r2 = json.loads(json.dumps(r))
        cf = r2["coefficients"]
        cf["C_X_world"] = cf["C_X"]
        cf["C_Y_world"] = cf["C_Y"]
        cf["C_X"] = (c * fx + s * fy) / (q * a_f0)
        cf["C_Y"] = (-s * fx + c * fy) / (q * a_l0)
        cf["C_X_std"] = math.sqrt((c * sx) ** 2 + (s * sy) ** 2) / (q * a_f0)
        cf["C_Y_std"] = math.sqrt((s * sx) ** 2 + (c * sy) ** 2) / (q * a_l0)
        # pressure / wall parts are rotated with the same cosine weighting
        cf["C_X_pressure"] = (c * c_old["C_X_pressure"] * a_f_run
                              + s * c_old["C_Y_pressure"] * a_l0) / a_f0
        cf["C_X_wall"] = c * c_old["C_X_wall"] * a_f_run / a_f0
        cf["C_K"] = (c * mx_world + s * 0.0) / (q * a_l0 * h0)
        cf["C_N"] = mz_world / (q * a_l0 * b0)
        cf["C_K_note"] = ("world-x heel-moment component (world-y part not "
                          "stored by this run)" if abs(r["angle_deg"]) % 180 else
                          "exact (theta = 0/180)")
        cf["reference"] = {"A_F_m2": g0.area_frontal, "A_L_m2": g0.area_lateral,
                           "B_m": g0.breadth, "H_m": g0.height}
        info[r["label"]] = {"angle_deg": r["angle_deg"],
                            "A_F_run_m2": g["area_frontal_m2"],
                            "A_F_ref_m2": g0.area_frontal,
                            "A_L_ref_m2": g0.area_lateral}
        out.append(r2)
    return out, info


def blockage_ratio(rec: dict) -> float:
    g = rec["geometry"]
    dx = g["dx_m"]
    return g["area_frontal_m2"] / ((g["grid_xyz"][1] * dx) * (g["grid_xyz"][2] * dx))


def blockage_correct(recs: list[dict]) -> tuple[list[dict], dict]:
    """Correct the near-field runs for test-section blockage.

    The near-field family (Δx = 1.0/2.0/0.5 m) confines the superstructure in a
    ~2H x 2H box (blockage ~21 %), which raises the dynamic pressure over the
    body; the far-field family (Δx = 1.5 m, blockage ~7 %) is the free-stream
    reference.  The ratio of the two families at the two angles where both
    exist (0 deg and 90 deg) is measured directly and applied to the remaining
    angles as a documented, first-order blockage correction.
    """
    far = [r for r in recs if r.get("tag") == "farfield" and r.get("case") == "ship"]
    ship_only = [r for r in recs if r.get("case") == "ship"]
    if not far:
        return recs, {"applied": False, "reason": "no far-field family"}
    f0 = next((r for r in far if abs(r["angle_deg"]) < 1e-9), None)
    f90 = next((r for r in far if abs(r["angle_deg"] - 90.0) < 1e-9), None)
    def _near(angle):
        cand = [r for r in recs if r.get("case") == "ship"
                and not r.get("tag") and abs(r["angle_deg"] - angle) < 1e-9]
        return min(cand, key=lambda r: r["geometry"]["dx_m"]) if cand else None

    n0 = _near(0.0)
    n90 = _near(90.0)
    if not (f0 and n0):
        return recs, {"applied": False, "reason": "missing 0 deg pair"}
    cx_far = f0["coefficients"]["C_X"]
    cx_near = n0["coefficients"]["C_X"]
    fx = cx_far / cx_near if cx_near else 1.0
    fl = 1.0
    if f90 and n90 and n90["coefficients"]["C_Y"]:
        fl = (f90["coefficients"]["C_Y"] / n90["coefficients"]["C_Y"]) or 1.0
    out = []
    for r in recs:
        r2 = json.loads(json.dumps(r))
        if r.get("case") == "ship" and r.get("tag") != "farfield":
            c = r2["coefficients"]
            for k, f in (("C_X", fx), ("C_X_std", fx), ("C_X_pressure", fx),
                         ("C_X_wall", fx), ("C_Y", fl), ("C_Y_std", fl),
                         ("C_Y_pressure", fl), ("C_Y_wall", fl),
                         ("C_N", fl), ("C_K", fl)):
                if k in c:
                    c[k] = c[k] * f
            r2["blockage_correction"] = {"factor_x": fx, "factor_lateral": fl}
        out.append(r2)
    info = {
        "applied": True,
        "factor_x": fx,
        "factor_lateral": fl,
        "blockage_near": blockage_ratio(n0),
        "blockage_far": blockage_ratio(f0),
        "C_X_near_0": cx_near, "C_X_far_0": cx_far,
        "note": "near-field coefficients multiplied by the measured far/near "
                "ratio at 0 deg (x-force) and 90 deg (lateral force/moments); "
                "far-field records are used unscaled",
    }
    return out, info


def load_runs() -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        if "smoke" in p:
            continue
        with open(p) as fh:
            recs.append(json.load(fh))
    return recs


_UNION_CACHE: dict[float, float] = {}


def projected_union_area(boxes, angle_deg: float, res: float = 0.1) -> float:
    """Area of the body projected on the plane normal to the wind.

    Each box projects to a rectangle ``[y_min, y_max] x [z0, z1]`` in the flow
    frame (the y-extent of the rotated footprint spans every y it covers), so
    the union is a cheap 2-D rectangle-union raster.  Bilinear in memory, no
    three-dimensional voxelisation needed.
    """
    a = math.radians(angle_deg)
    c, s_ = math.cos(a), math.sin(a)
    xs = [b.x0 for b in boxes] + [b.x1 for b in boxes]
    ys = [b.y0 for b in boxes] + [b.y1 for b in boxes]
    xc, yc = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    rects = []
    for b in boxes:
        yy = []
        zz = []
        for xb in (b.x0, b.x1):
            for yb in (b.y0, b.y1):
                dx_, dy_ = xb - xc, yb - yc
                yy.append(s_ * dx_ + c * dy_)
        zz = [b.z0, b.z1]
        rects.append((min(yy), max(yy), min(zz), max(zz)))
    ylo = min(r[0] for r in rects) - 1.0
    yhi = max(r[1] for r in rects) + 1.0
    zlo = 0.0
    zhi = max(r[3] for r in rects) + 1.0
    ny = int(np.ceil((yhi - ylo) / res))
    nz = int(np.ceil((zhi - zlo) / res))
    grid = np.zeros((nz, ny), dtype=bool)
    ycents = ylo + (np.arange(ny) + 0.5) * res
    zcents = zlo + (np.arange(nz) + 0.5) * res
    Yg, Zg = np.meshgrid(ycents, zcents)
    for y0, y1, z0, z1 in rects:
        grid |= (Yg >= y0) & (Yg < y1) & (Zg >= z0) & (Zg < z1)
    return float(grid.sum()) * res * res


def component_buildup(angle_deg: float) -> dict:
    """Analytic drag build-up on the stepped geometry (engineering reference)."""
    boxes = lng_moss_superstructure()
    total = 0.0
    per = []
    ca, sa = abs(math.cos(math.radians(angle_deg))), abs(math.sin(math.radians(angle_deg)))
    for b in boxes:
        # projected area of one box normal to the flow (rotated footprint)
        w = ca * (b.y1 - b.y0) + sa * (b.x1 - b.x0)
        area = w * (b.z1 - b.z0)
        cd = CD_BUILDUP.get(b.label, 1.15)
        contrib = cd * area
        per.append((b.label, area, cd, contrib))
        total += contrib
    # reference area is the *union* projection (not the sum of the components,
    # which double-counts the sheltered area of the packed superstructure)
    a_union = projected_union_area(boxes, angle_deg)
    return {"C_X_buildup": INTERFERENCE * total / a_union,
            "area_union_m2": a_union, "sum_cd_a": total,
            "components": per}


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    raw_recs = load_runs()
    if not raw_recs:
        raise SystemExit("no run records found")
    recs, frame_info = ship_frame_correct(raw_recs)
    recs, blockage_info = blockage_correct(recs)

    # --- derived quantities per record -----------------------------------
    for r in recs:
        ts = r.get("time_series", {})
        dt = ts.get("dt_s")
        fy = np.array(ts.get("fy", []), dtype=float)
        fx = np.array(ts.get("fx", []), dtype=float)
        h = r["geometry"]["extent_m"][2]
        u_phys = r["units"]["u_phys_ms"]
        if fy.size > 32 and dt:
            st = strouhal_from_series(fy, dt, h, u_phys)
            r["strouhal"] = {"St": st["St"], "f_peak_hz": st["f_peak_hz"],
                             "peak_energy_fraction": st["peak_energy_fraction"],
                             "spectrum": st["spectrum"], "freqs_hz": st["freqs_hz"],
                             "n_samples": st["n_samples"]}
        ref = float(np.mean(fx)) if fx.size else 0.0
        r["signal_ratio"] = (float(np.std(fy) / abs(np.mean(fx)))
                             if fy.size and ref else None)

    # --- grid convergence for the 0 deg drag ------------------------------
    ship0 = sorted([r for r in recs if r["case"] == "ship"
                    and abs(r["angle_deg"]) < 1e-9 and not r.get("tag")],
                   key=lambda r: -r["geometry"]["dx_m"])
    # one record per grid level only
    seen = set()
    ship0 = [r for r in ship0
             if not (r["geometry"]["dx_m"] in seen or seen.add(r["geometry"]["dx_m"]))]
    gci = None
    if len(ship0) >= 3:
        f = [r["coefficients"]["C_X"] for r in ship0[:3]]      # coarse -> fine
        gci = gci_study(f[0], f[1], f[2], r=2.0)
        gci["dx_m"] = [r["geometry"]["dx_m"] for r in ship0[:3]]
        gci["C_X_by_level"] = f
    elif len(ship0) == 2:
        gci = {"status": "two_levels_only", "dx_m": [r["geometry"]["dx_m"] for r in ship0],
               "C_X_by_level": [r["coefficients"]["C_X"] for r in ship0]}

    # --- cube validation --------------------------------------------------
    cube = [r for r in recs if r["case"] == "cube"]
    cube_res = None
    if cube:
        c = cube[0]
        cube_res = {
            "re_D": c["meta"]["re_D"],
            "cells_per_side": c["meta"]["cells_per_side"],
            "C_D": c["coefficients"]["C_X"],
            "C_D_reference": 1.05,          # sharp-edged cube, Re > 1e4
            "rel_error": relative_error(c["coefficients"]["C_X"], 1.05),
            "St": c.get("strouhal", {}).get("St"),
            "St_reference": 0.10,           # cube lateral-force Strouhal number
            "rel_error_St": relative_error(c.get("strouhal", {}).get("St") or 0.0, 0.10),
            "mass_budget_rel_error": c.get("mass_budget", {}).get("mean_rel_error"),
        }

    # --- platform baseline (empirical model of the ship platform) ---------
    plat_path = os.path.join(ART, "platform_baseline.json")
    platform = None
    if os.path.exists(plat_path):
        with open(plat_path) as fh:
            pb = json.load(fh)
        a_f, a_l = pb["areas"]["A_fw"], pb["areas"]["A_lw"]
        q = 0.5 * 1.225 * pb["wind_speed"] ** 2
        rows = []
        for row in pb["responses"]:
            d = row["response"]
            rows.append({
                "angle_deg": row["angle_deg"],
                "F_x_N": d.get("F_wind_x", 0.0), "F_y_N": d.get("F_wind_y", 0.0),
                "N_Nm": d.get("N_wind", 0.0),
                "C_X": abs(d.get("F_wind_x", 0.0)) / (q * a_f),
                "C_Y": abs(d.get("F_wind_y", 0.0)) / (q * a_l),
                "C_N": abs(d.get("N_wind", 0.0)) / (q * a_l * pb["reference_breadth_m"]),
            })
        platform = {"endpoint": pb["endpoint"], "areas": pb["areas"], "rows": rows,
                    "note": pb.get("note", "")}

    # --- coefficient table ------------------------------------------------
    # the coefficient table prefers the far-field domain (blockage ~7 %) and,
    # within a domain family, the finest grid (see build_table ordering)
    recs_for_table = sorted(
        recs, key=lambda r: (0 if r.get("tag") == "farfield" else 1,
                             r["geometry"]["dx_m"]))
    recs_for_table = [r for r in recs_for_table
                      if r.get("case") == "ship" and r.get("tag") != "masscontrol"]
    table = build_table(
        recs_for_table, geometry_name="LNG-Moss-type-superstructure-50x30x25",
        provenance={
            "code": {"module": "tensorlbm.wind_load_lng",
                     "solver": "D3Q19 BGK + Smagorinsky LES + log-law wall function",
                     "driver": "scripts/run_lng_wind_load.py"},
            "physical_model": {
                "cs_smag": recs[0]["solver"]["cs_smag"],
                "tau_min": recs[0]["solver"].get("tau_min", 0.51),
                "wall_model": recs[0]["solver"]["wall_model"],
                "wall_feedback": recs[0]["solver"]["wall_feedback"],
                "u_in_lattice": recs[0]["solver"]["u_in_lattice"],
                "normalise_outflow": "density-pinned zero-gradient",
            },
            "grid_cells": {r["label"]: r["geometry"]["grid_xyz"] for r in recs},
            "wall_time_s": {r["label"]: r["wall_time_s"] for r in recs},
            "steps": {r["label"]: r["steps"] for r in recs},
            "reynolds": {r["label"]: r["meta"] for r in recs},
            "blockage_correction": blockage_info,
        "ship_frame_reference": frame_info,
            "ship_frame_reference": frame_info,
            "raw_coefficients_before_blockage_correction": {
                r["label"]: {k: r["coefficients"][k] for k in ("C_X", "C_Y")}
                for r in raw_recs},
        },
        validation={"grid_convergence": gci, "cube_benchmark": cube_res,
                    "platform_baseline": platform},
    )

    # --- criteria ---------------------------------------------------------
    cx0 = interpolate(table, 0.0)["C_X"]
    cx90 = interpolate(table, 90.0)["C_X"]
    build0 = component_buildup(0.0)["C_X_buildup"]
    build90 = component_buildup(90.0)["C_X_buildup"]
    # reference: mean of the analytic build-up and the platform empirical model
    ref0 = 0.5 * (build0 + (platform["rows"][0]["C_X"] if platform else build0))
    # C_X(90 deg) is the fore-aft *side* force of a beam wind: a component
    # build-up cannot produce it (it only predicts the drag along the flow), so
    # the independent platform empirical model is the only usable reference.
    ref90 = (platform["rows"][1]["C_X"] if platform else build90)
    st_lbm = None
    for r in recs:
        if r["case"] == "ship" and abs(r["angle_deg"] - 90.0) < 1e-9 and r.get("strouhal"):
            st_lbm = r["strouhal"]["St"]
    if st_lbm is None:
        cands = [r["strouhal"]["St"] for r in recs
                 if r["case"] == "ship" and r.get("strouhal")]
        st_lbm = float(np.median(cands)) if cands else 0.0
    mass_err = max((r.get("mass_budget", {}).get("mean_rel_error") or 0.0)
                   for r in recs if r["case"] == "ship")
    ugci = gci.get("U_GCI") if (gci and "U_GCI" in gci) else 1.0

    pytest_passed = 0
    pj = os.path.join(RUNS, "logs", "pytest.log")
    if os.path.exists(pj):
        txt = open(pj).read()
        for line in txt.splitlines():
            if " passed" in line:
                for tok in line.replace("=", " ").split():
                    if tok.isdigit():
                        pytest_passed = max(pytest_passed, int(tok))

    table_path = "artifacts/p16n7/wind_load_table.json"
    crit = acceptance_criteria(
        C_X_0=cx0, C_X_90=cx90, ref_C_X_0=ref0, ref_C_X_90=ref90,
        U_GCI=float(ugci), St=float(st_lbm), St_ref=0.10,
        mass_error=float(mass_err), pytest_passed=pytest_passed,
        table_path=table_path,
        reference_class=(
            "no digitised wind-tunnel dataset for this geometry was available "
            "offline; reference = mean of (i) engineering drag build-up on the "
            "stepped geometry and (ii) the ship platform's empirical wind model "
            "(/api/v1/calculations/environmental-forces). Strouhal reference = "
            "sharp-edged bluff-body literature (cube/prism St ~ 0.10)."),
    )
    table["validation"]["criteria"] = crit
    table["validation"]["buildup"] = {"C_X_0": build0, "C_X_90": build90,
                                      "C_Y_90": component_buildup(90.0)["C_X_buildup"],
                                      "interference_factor": INTERFERENCE,
                                      "note": "component drag build-up referenced to "
                                              "the union projected area; a build-up "
                                              "cannot predict the beam-wind side force "
                                              "C_X(90 deg), which is compared with the "
                                              "platform empirical model instead"}
    dumps(table, os.path.join(HERE, table_path))

    summary = {
        "schema": SCHEMA,
        "cases": [{"label": r["label"], "angle_deg": r["angle_deg"],
                   "dx_m": r["geometry"]["dx_m"], "cells": r["geometry"]["grid_xyz"],
                   "C_X": r["coefficients"]["C_X"], "C_Y": r["coefficients"]["C_Y"],
                   "C_N": r["coefficients"]["C_N"], "C_K": r["coefficients"]["C_K"],
                   "C_X_std": r["coefficients"]["C_X_std"],
                   "blockage_ratio": (r["geometry"]["area_frontal_m2"]
                                      / ((r["geometry"]["grid_xyz"][1]
                                          * r["geometry"]["dx_m"])
                                         * (r["geometry"]["grid_xyz"][2]
                                            * r["geometry"]["dx_m"]))),
                   "tag": r.get("tag", ""),
                   "ms_per_step": r["ms_per_step"],
                   "wall_time_s": r["wall_time_s"],
                   "mass_budget": r.get("mass_budget"),
                   "strouhal": r.get("strouhal"),
                   "re": r["meta"]} for r in recs],
        "grid_convergence": gci,
        "cube_benchmark": cube_res,
        "platform_baseline": platform,
        "criteria": crit,
        "blockage_correction": blockage_info,
        "ship_frame_reference": frame_info,
        "table": table_path,
    }
    with open(os.path.join(ART, "campaign_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    make_figures(recs, table, gci, cube_res, platform, summary)
    print(json.dumps({"all_pass": crit["all_pass"],
                      "C_X_0": cx0, "C_X_90": cx90,
                      "U_GCI": ugci, "St": st_lbm,
                      "mass_err": mass_err, "pytest": pytest_passed},
                     indent=1))
    return 0


def make_figures(recs, table, gci, cube_res, platform, summary) -> None:
    ship = [r for r in recs if r["case"] == "ship"]
    finest = {}
    for r in ship:
        a = float(r["angle_deg"])
        if a not in finest or r["geometry"]["dx_m"] < finest[a]["geometry"]["dx_m"]:
            finest[a] = r
    angles = sorted(finest)
    cx = [finest[a]["coefficients"]["C_X"] for a in angles]
    cy = [finest[a]["coefficients"]["C_Y"] for a in angles]
    cn = [finest[a]["coefficients"]["C_N"] for a in angles]
    ck = [finest[a]["coefficients"]["C_K"] for a in angles]

    # ---- fig 1: C_X vs angle
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(angles, cx, "o-", color="#14417a", label="LBM (wall-modelled LES)",
            markersize=6)
    if platform:
        pa = [row["angle_deg"] for row in platform["rows"]]
        pc = [row["C_X"] for row in platform["rows"]]
        order = np.argsort(pa)
        ax.plot(np.array(pa)[order], np.array(pc)[order], "s--", color="#c0392b",
                label="ship-platform empirical model", markersize=5)
    allang = np.linspace(0, 180, 37)
    bu = [component_buildup(a)["C_X_buildup"] for a in allang]
    ax.plot(allang, bu, ":", color="#7f8c8d", label="engineering drag build-up")
    ax.set_xlabel(r"wind angle $\theta$ [deg]  (0 = head wind, 90 = beam)")
    ax.set_ylabel(r"$C_X$  (ref. $q\,A_F$)")
    ax.set_title("LNG Moss superstructure: longitudinal wind coefficient")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_CX_vs_angle.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: C_Y, C_N, C_K
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(angles, cy, "o-", label=r"$C_Y$ (side force, ref $qA_L$)")
    ax.plot(angles, cn, "s-", label=r"$C_N$ (yaw moment, ref $qA_LB$)")
    ax.plot(angles, ck, "^-", label=r"$C_K$ (heel moment, ref $qA_LH$)")
    if platform:
        pa = [row["angle_deg"] for row in platform["rows"]]
        pc = [row["C_Y"] for row in platform["rows"]]
        order = np.argsort(pa)
        ax.plot(np.array(pa)[order], np.array(pc)[order], "x--", color="#c0392b",
                label=r"platform empirical $C_Y$")
    ax.set_xlabel(r"wind angle $\theta$ [deg]")
    ax.set_ylabel("coefficient [-]")
    ax.set_title("Transverse force, yaw and heel moment coefficients")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_CY_CN_CK_vs_angle.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: force time series + spectrum (beam wind)
    r90 = finest.get(90.0) or ship[0]
    ts = r90.get("time_series", {})
    fy = np.array(ts.get("fy", []))
    dt = ts.get("dt_s", 1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    if fy.size:
        q = 0.5 * r90["solver"]["u_in_lattice"] ** 2
        a_l = r90["geometry"]["area_lateral_m2"] / r90["geometry"]["dx_m"] ** 2
        tt = np.arange(fy.size) * dt
        ax1.plot(tt, fy / (q * a_l), lw=0.8, color="#14417a")
        ax1.set_xlabel("time [s]"); ax1.set_ylabel(r"$C_Y(t)$")
        ax1.set_title(f"beam wind (theta=90 deg), dx={r90['geometry']['dx_m']} m")
        ax1.grid(alpha=0.3)
    st = r90.get("strouhal")
    if st:
        freqs = np.array(st["freqs_hz"]); spec = np.array(st["spectrum"])
        ax2.semilogy(freqs[1:], spec[1:], color="#c0392b")
        ax2.axvline(st["f_peak_hz"], ls=":", color="k",
                    label=f"St = {st['St']:.3f}")
        ax2.set_xlim(0, min(5.0, freqs[-1]))
        ax2.set_xlabel("frequency [Hz]"); ax2.set_ylabel("normalised PSD")
        ax2.set_title("lateral-force spectrum")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_force_spectrum.png"), dpi=150)
    plt.close(fig)

    # ---- fig 4: grid convergence
    if gci and gci.get("C_X_by_level"):
        dxs = gci["dx_m"]; cxs = gci["C_X_by_level"]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(dxs, cxs, "o-", label="LBM (theta = 0 deg)")
        if "f_extrapolated" in gci:
            ax.axhline(gci["f_extrapolated"], ls="--", color="#c0392b",
                       label=f"Richardson extrapolation (p={gci['p_used']:.2f})")
            ax.fill_between([min(dxs) * 0.9, max(dxs) * 1.1],
                            gci["f_extrapolated"] * (1 - gci["U_GCI"]),
                            gci["f_extrapolated"] * (1 + gci["U_GCI"]),
                            color="#c0392b", alpha=0.12,
                            label=f"GCI band U={100*gci['U_GCI']:.1f}%")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel(r"cell size $\Delta x$ [m]"); ax.set_ylabel(r"$C_X(0°)$")
        ax.set_title("Grid convergence of the head-wind drag coefficient")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig4_grid_convergence.png"), dpi=150)
        plt.close(fig)

    # ---- fig 5: wake field (velocity magnitude / streamwise velocity)
    field_files = [p for p in glob.glob(os.path.join(RUNS, "*_field.npz"))
                   if "theta090" in p] or glob.glob(os.path.join(RUNS, "*_field.npz"))
    if field_files:
        d = np.load(sorted(field_files, key=lambda p: -os.path.getsize(p))[0])
        u = d["u"]; rho = d["rho"]; dx = float(d["dx"])
        k = u.shape[1] // 3                     # z index ~ 1/3 of the height
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
        im = ax1.imshow(u[0, k], origin="lower", cmap="RdBu_r")
        ax1.set_title(f"streamwise velocity $u_x$ (z index {k}, dx={dx} m)")
        ax1.set_xlabel("x index"); ax1.set_ylabel("y index")
        fig.colorbar(im, ax=ax1, shrink=0.85)
        j = u.shape[2] * 2 // 3
        im2 = ax2.imshow(u[0, :, j], origin="lower", cmap="RdBu_r",
                         aspect="auto")
        ax2.set_title("streamwise velocity, vertical plane in the wake")
        ax2.set_xlabel("y index"); ax2.set_ylabel("z index")
        fig.colorbar(im2, ax=ax2, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig5_wake_field.png"), dpi=150)
        plt.close(fig)

    # ---- fig 6: cube validation
    if cube_res:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.bar(["LBM cube", "reference"], [cube_res["C_D"], cube_res["C_D_reference"]],
               color=["#14417a", "#7f8c8d"])
        ax.set_ylabel(r"$C_D$ (ref. frontal area)")
        ax.set_title(f"cube Re_D={cube_res['re_D']:.0f}, "
                     f"{cube_res['cells_per_side']} cells/side\n"
                     f"error {100*cube_res['rel_error']:.1f}%")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig6_cube_validation.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
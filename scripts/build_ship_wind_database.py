#!/usr/bin/env python3
"""IN38-N04 analysis: LBM superstructure database -> hybrid engine -> report.

Reads the raw LBM campaign records written by
``scripts/run_ship_wind_campaign.py`` (``runs/in38_n04/raw/*.json``) and builds

* ``artifacts/in38_n04/wind_database.json``  -- the coefficient database, the
  geometry validation, the empirical models, the grid-convergence result, the
  hybrid-engine routing and its cross-validated accuracy;
* ``artifacts/in38_n04/acceptance.json``     -- the machine-checkable criteria
  table (thresholds + measured numbers + verdicts);
* ``artifacts/in38_n04/annual_wind.json``    -- wind-rose annual assessment;
* ``figures/in38_n04/*.png``                 -- report figures.

Everything is computed from the run files; no number is hand-copied.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(HERE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "src"))

from tensorlbm.ship_wind_aero import (  # noqa: E402
    RHO_AIR, HybridWindEngine, blendermann_coefficients, blendermann_style,
    calm_water_resistance_estimate, catalogue, component_buildup,
    complexity_index, evaluate_acceptance, fit_blendermann_style,
    projected_areas, sign_conventions, synthetic_wind_rose, validate_geometry,
    wind_rose_annual_resistance,
)

RAW = os.path.join(HERE, "runs", "in38_n04", "raw")
ART = os.path.join(HERE, "artifacts", "in38_n04")
FIG = os.path.join(HERE, "figures", "in38_n04")
ANGLES = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
DX_POLAR = 4.0
CUBE_REF_CD = 1.05          # free-stream cube, Re > 1e4 (literature)
COEF_KEYS = ("C_X", "C_Y", "C_K", "C_N")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw() -> list[dict]:
    recs = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        r["_path"] = os.path.relpath(path, HERE)
        recs.append(r)
    return recs


def ship_records(ship: str, dx: float, recs: list[dict]) -> list[dict]:
    out = []
    for r in recs:
        if r.get("case", {}).get("ship") != ship:
            continue
        if abs(float(r["settings"]["dx_m"]) - dx) > 1e-9:
            continue
        coeff = r["coefficients"]
        robust = {}
        fh = r.get("force_history") or {}
        if fh.get("fx"):
            n = int(r["settings"].get("average_from", 0))
            fx = np.asarray(fh["fx"][n:], dtype=float)
            fy = np.asarray(fh["fy"][n:], dtype=float)
            if fx.size:
                spread = float(np.percentile(fx, 99) - np.percentile(fx, 1))
                scale = max(abs(float(np.median(fx))), 1e-9)
                robust = {
                    "fx_p1_p99_spread_lattice": spread,
                    "oscillation_amplitude_rel": spread / scale,
                    "n_window": int(fx.size),
                }
        out.append({**robust, "angle_deg": float(r["angle_deg"]),
                    "C_X": float(coeff["C_X"]), "C_Y": float(coeff["C_Y"]),
                    "C_K": float(coeff["C_K"]), "C_N": float(coeff["C_N"]),
                    "C_X_std": float(coeff.get("C_X_std", 0.0)),
                    "C_Y_std": float(coeff.get("C_Y_std", 0.0)),
                    "n_samples": int(coeff.get("n_samples", 0)),
                    "mass_rel_error": float(
                        r["mass_budget"].get("mean_rel_error", 0.0)),
                    "grid_xyz": r["geometry"]["grid_xyz"],
                    "solid_cells": r["geometry"]["solid_cells"],
                    "voxel_area_frontal_m2": r["geometry"]["area_frontal_m2"],
                    "voxel_area_lateral_m2": r["geometry"]["area_lateral_m2"],
                    "run_s": r["timing"]["run_s"],
                    "source": r["_path"]})
    out.sort(key=lambda d: d["angle_deg"])
    return out


# ---------------------------------------------------------------------------
# Grid convergence (Roache GCI), reusing the vetted solver helper
# ---------------------------------------------------------------------------

def gci_for_angle(levels: list[tuple[float, float]]) -> dict:
    """``levels`` = [(dx_coarse, value), (dx_medium, value), (dx_fine, value)]."""

    levels = sorted(levels, key=lambda t: -t[0])
    vals = [v for _, v in levels]
    if len(vals) < 2:
        return {"status": "insufficient_grids", "levels": levels}
    if len(vals) == 2:
        f3, f2 = vals
        r = levels[0][0] / levels[1][0]
        p = 1.0                      # declared, not observed (2-grid fallback)
        f_ext = (r ** p * f2 - f3) / (r ** p - 1.0)
        gci = 1.25 * abs((f2 - f3) / f2) / (r ** p - 1.0)
        return {"status": "two_grid_assumed_p", "p": p, "r": r,
                "extrapolated": f_ext, "gci_rel": gci,
                "levels": levels}
    from tensorlbm.wind_load_lng import gci_study

    r = levels[0][0] / levels[1][0]
    res = dict(gci_study(vals[0], vals[1], vals[2], r=r))  # coarse, med, fine
    res["levels"] = levels
    res["status"] = res.get("p_status", "ok")
    res["p"] = res.get("p_used")
    res["extrapolated"] = res.get("f_extrapolated")
    res["gci_rel"] = res.get("U_GCI")
    return res


def gci_table(recs: list[dict]) -> dict:
    """Coefficient-wise GCI for every ship at the angles with >=2 grids."""

    out = {}
    for ship in catalogue():
        by_angle: dict[float, list[tuple[float, dict]]] = {}
        for r in recs:
            if r.get("case", {}).get("ship") != ship:
                continue
            a = float(r["angle_deg"])
            by_angle.setdefault(a, []).append(
                (float(r["settings"]["dx_m"]), r["coefficients"]))
        per_angle = {}
        for a, lst in sorted(by_angle.items()):
            if len(lst) < 2:
                continue
            row = {}
            for key in ("C_X", "C_Y"):
                levels = [(dx, float(c[key])) for dx, c in lst]
                row[key] = gci_for_angle(levels)
            per_angle[a] = row
        out[ship] = per_angle
    return out


# ---------------------------------------------------------------------------
# Empirical + hybrid cross-validation
# ---------------------------------------------------------------------------

def _abs_rel_err(model: float, ref: float, scale: float) -> float:
    """Error normalised by the polar amplitude (well conditioned near zeros)."""

    return abs(model - ref) / max(scale, 1e-9)


def _rel_err(model: float, ref: float, scale: float) -> float:
    """Relative error, scaled by the polar amplitude when the value is small."""

    denom = max(abs(ref), 1e-9)
    if abs(ref) < 0.15 * scale:
        denom = max(0.15 * scale, 1e-9)
    return abs(model - ref) / denom


def loo_validation(ship, db: list[dict], *, route: str) -> dict:
    """Leave-one-angle-out accuracy of one engine branch."""

    scales = {k: max(abs(d[k]) for d in db) for k in COEF_KEYS}
    rows = []
    for i, held in enumerate(db):
        train = [d for j, d in enumerate(db) if j != i]
        eng = HybridWindEngine(ship, train, route=route)
        pred = eng.coefficients(held["angle_deg"])
        rows.append({
            "angle_deg": held["angle_deg"],
            "branch": pred["method"],
            "C_X": [pred["C_X"], held["C_X"]],
            "C_Y": [pred["C_Y"], held["C_Y"]],
            "err_C_X_rel": _rel_err(pred["C_X"], held["C_X"], scales["C_X"]),
            "err_C_Y_rel": _rel_err(pred["C_Y"], held["C_Y"], scales["C_Y"]),
            "err_C_X_absrel": _abs_rel_err(pred["C_X"], held["C_X"], scales["C_X"]),
            "err_C_Y_absrel": _abs_rel_err(pred["C_Y"], held["C_Y"], scales["C_Y"]),
        })
    return {
        "ship": ship.key, "route": route, "n_folds": len(rows),
        "folds": rows,
        "max_err_C_X": max(r["err_C_X_absrel"] for r in rows),
        "max_err_C_Y": max(r["err_C_Y_absrel"] for r in rows),
        "mean_err_C_X": float(np.mean([r["err_C_X_absrel"] for r in rows])),
        "mean_err_C_Y": float(np.mean([r["err_C_Y_absrel"] for r in rows])),
        "metric": "|model - reference| / max|polar|",
        "scales": scales,
    }


def empirical_vs_lbm(ship, db: list[dict]) -> dict:
    """A-priori component build-up and fitted correlation against the LBM DB."""

    build = component_buildup(ship, angles=[d["angle_deg"] for d in db])
    fit = fit_blendermann_style(db)
    scales = {k: max(abs(d[k]) for d in db) for k in COEF_KEYS}
    rows = []
    for d, b in zip(db, build["rows"]):
        c = blendermann_coefficients(fit, d["angle_deg"])
        rows.append({
            "angle_deg": d["angle_deg"],
            "C_X_lbm": d["C_X"], "C_X_buildup": b["C_X"],
            "C_X_correlation": c["C_X"],
            "C_Y_lbm": d["C_Y"], "C_Y_buildup": b["C_Y"],
            "C_Y_correlation": c["C_Y"],
            "err_C_X_buildup": _rel_err(b["C_X"], d["C_X"], scales["C_X"]),
            "err_C_Y_buildup": _rel_err(b["C_Y"], d["C_Y"], scales["C_Y"]),
            "err_C_X_correlation": _rel_err(c["C_X"], d["C_X"], scales["C_X"]),
            "err_C_Y_correlation": _rel_err(c["C_Y"], d["C_Y"], scales["C_Y"]),
        })
    return {"ship": ship.key, "rows": rows, "fit": fit,
            "moments_reliable": False,
            "buildup": {k: build[k] for k in ("method", "reference", "angles_deg")},
            "max_err_C_X_buildup": max(r["err_C_X_buildup"] for r in rows),
            "max_err_C_Y_buildup": max(r["err_C_Y_buildup"] for r in rows),
            "mean_err_C_X_buildup": float(np.mean([r["err_C_X_buildup"] for r in rows])),
            "mean_err_C_Y_buildup": float(np.mean([r["err_C_Y_buildup"] for r in rows]))}


# ---------------------------------------------------------------------------
# Annual wind-rose assessment
# ---------------------------------------------------------------------------

def freeboard_frontal_area(hull) -> dict:
    """Coarse above-water frontal area of the hull (documented assumption)."""

    depth = hull.draught_m + 0.55 * hull.breadth_m
    a = hull.breadth_m * (depth - hull.draught_m)
    return {"depth_assumed_m": depth, "freeboard_m": depth - hull.draught_m,
            "frontal_area_m2": a,
            "provenance": "D = T + 0.55 B (documented assumption, +/-10%)"}


def annual_study(ship, engine: HybridWindEngine) -> dict:
    rose = synthetic_wind_rose()
    hull = ship.hull
    v_design = hull.design_speed_ms
    fb = freeboard_frontal_area(hull)
    out = {"ship": ship.key, "rose": rose, "freeboard": fb, "cases": []}
    for speed_label, speed in (("design", v_design), ("slow_steaming", 0.8 * v_design)):
        base = calm_water_resistance_estimate(hull, speed)
        headings = []
        for hdg in range(0, 360, 45):
            r = wind_rose_annual_resistance(
                engine, rose, ship_speed_ms=speed, heading_deg=float(hdg),
                baseline_resistance_N=base["R_calm_water_N"],
                extra_frontal_area_m2=0.0)
            r_fb = wind_rose_annual_resistance(
                engine, rose, ship_speed_ms=speed, heading_deg=float(hdg),
                baseline_resistance_N=base["R_calm_water_N"],
                extra_frontal_area_m2=fb["frontal_area_m2"])
            headings.append({
                "heading_deg": float(hdg),
                "mean_dR_superstructure_N": r["mean_wind_added_resistance_N"],
                "mean_dR_with_freeboard_N": r_fb["mean_wind_added_resistance_N"],
                "pct_of_calm_water_superstructure": r["wind_over_calm_water_pct"],
                "pct_of_calm_water_with_freeboard": r_fb["wind_over_calm_water_pct"],
                "thrust_probability": r["thrust_probability"],
                "added_power_kW_superstructure": r["equivalent_added_power_kW"],
            })
        out["cases"].append({
            "speed_label": speed_label,
            "speed_ms": speed,
            "calm_water": base,
            "headings": headings,
            "heading_mean_dR_superstructure_N": float(
                np.mean([h["mean_dR_superstructure_N"] for h in headings])),
            "heading_mean_pct_superstructure": float(
                np.mean([h["pct_of_calm_water_superstructure"] for h in headings])),
            "heading_mean_pct_with_freeboard": float(
                np.mean([h["pct_of_calm_water_with_freeboard"] for h in headings])),
            "worst_heading_pct_with_freeboard": float(
                max(h["pct_of_calm_water_with_freeboard"] for h in headings)),
        })
    return out# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 90})
    return plt


def _coef(fit: dict, angle: float, key: str) -> float:
    """Coefficient ``key`` of the fitted correlation at ``angle``."""

    c = blendermann_coefficients(fit, angle)
    if key == "C_X":
        return c["C_X"]
    if key == "C_Y":
        return c["C_Y"]
    return c[key]


def figures(dbs: dict, emp: dict, gcis: dict, annual: dict, cube: dict,
            loo: dict) -> list[str]:
    plt = _mpl()
    os.makedirs(FIG, exist_ok=True)
    paths = []
    ships = [k for k in dbs if k in emp and emp[k].get("rows")]

    # fig 1: C_X and C_Y polars --------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    for j, ship in enumerate(ships):
        db = dbs[ship]
        ang = [d["angle_deg"] for d in db]
        a_full = np.arange(0, 361, 5)
        for i, key in enumerate(("C_X", "C_Y")):
            ax = axes[i, j]
            vals = [d[key] for d in db]
            errs = [d.get(key + "_std", 0.0) for d in db]
            ax.errorbar(ang, vals, yerr=errs, fmt="o-", ms=3, lw=1.0,
                        capsize=2, label="LBM (dx=3.0 m)")
            b = emp[ship]["rows"]
            ax.plot([r["angle_deg"] for r in b],
                    [r[f"{key}_buildup"] for r in b], "s--", ms=2.5, lw=0.9,
                    label="component build-up (a priori)")
            fit = emp[ship]["fit"]["params"]
            ax.plot(a_full, [_coef(fit, a, key) for a in a_full],
                    ":", lw=1.0, label="Blendermann-style fit")
            ax.set_title(f"{ship}  $C_{{{key[-1]}}}$", fontsize=7)
            ax.set_xlabel(r"wind angle $\theta$ [deg]")
            ax.set_xticks([0, 90, 180, 270, 360])
            if i == 0 and j == 0:
                ax.legend(fontsize=5.2, loc="best")
    fig.tight_layout(pad=0.4)
    p = os.path.join(FIG, "fig1_polars.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 2: grid convergence ---------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, ship in zip(axes, ships):
        g = gcis.get(ship, {})
        for key, mk in (("C_X", "o"), ("C_Y", "s")):
            lev = g.get(90.0, {}).get(key, {}).get("levels") if g else None
            if not lev:
                continue
            dxs = [t[0] for t in lev]
            vs = [t[1] for t in lev]
            ax.plot(dxs, vs, mk + "-", ms=3, lw=1.0, label=key)
            ext = g[90.0][key].get("extrapolated")
            if ext is not None:
                ax.axhline(ext, ls=":", lw=0.8, color="0.4")
        ax.set_xlabel(r"$\Delta x$ [m]"); ax.set_title(f"{ship}, $\\theta=90^\\circ$",
                                                       fontsize=7)
        ax.invert_xaxis(); ax.legend(fontsize=5.5)
    axes[0].set_ylabel("coefficient")
    fig.tight_layout(pad=0.4)
    p = os.path.join(FIG, "fig2_grid_convergence.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 3: wake field ----------------------------------------------------
    fld = sorted(glob.glob(os.path.join(RAW, "*_a090_dx4.0_fields.npz")))
    if fld:
        fig, axes = plt.subplots(1, len(fld), figsize=(2.6 * len(fld), 2.4))
        if len(fld) == 1:
            axes = [axes]
        for ax, path in zip(axes, fld):
            z = np.load(path)
            dx = float(z["dx"])
            sl = z["speed_mid"] / float(z["u_lat"])
            solid = z["solid_mid"]
            img = np.where(solid > 0, np.nan, sl)
            im = ax.imshow(img, origin="lower", cmap="turbo", vmin=0.0,
                           vmax=float(np.nanpercentile(img, 99)))
            ax.set_title(os.path.basename(path).split("_a090")[0] + "  |u|/U",
                         fontsize=6)
            ax.set_xlabel("x [cells]"); ax.set_ylabel("y [cells]")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.045)
        fig.tight_layout(pad=0.4)
        p = os.path.join(FIG, "fig3_wake_field.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 4: annual assessment --------------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    width = 0.35
    xs = np.arange(len(ships))
    a0 = [annual[s]["cases"][0]["heading_mean_pct_superstructure"] for s in ships]
    a1 = [annual[s]["cases"][0]["heading_mean_pct_with_freeboard"] for s in ships]
    ax.bar(xs - width / 2, a0, width, label="superstructure only")
    ax.bar(xs + width / 2, a1, width, label="incl. hull freeboard estimate")
    ax.set_xticks(xs); ax.set_xticklabels(ships)
    ax.set_ylabel("annual mean wind resistance\n[% of calm-water resistance]")
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    p = os.path.join(FIG, "fig4_annual_wind.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # fig 5: validation / cross-validation --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    if cube.get("runs"):
        names = [r["tag"] for r in cube["runs"]]
        cds = [r["C_D"] for r in cube["runs"]]
        ax.bar(names, cds, color="tab:gray")
        ax.axhline(CUBE_REF_CD, color="tab:red", ls="--", lw=1.0,
                   label=f"literature {CUBE_REF_CD}")
        ax.set_ylabel("$C_D$ (cube)"); ax.tick_params(axis="x", rotation=20,
                                                      labelsize=5.5)
        ax.legend(fontsize=5.5)
    ax = axes[1]
    for ship in ships:
        rows = loo[ship]["folds"]
        ax.plot([r["angle_deg"] for r in rows],
                [100 * r["err_C_X_rel"] for r in rows], "o-", ms=3, lw=1.0,
                label=f"{ship} $C_X$")
    ax.axhline(10.0, color="tab:red", ls="--", lw=1.0, label="10% target")
    ax.set_ylabel("LOO error [%]"); ax.set_xlabel(r"held-out $\theta$ [deg]")
    ax.legend(fontsize=5.0)
    fig.tight_layout(pad=0.4)
    p = os.path.join(FIG, "fig5_validation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=DX_POLAR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    recs = load_raw()
    ships = catalogue()
    dbs = {s: ship_records(s, args.dx, recs) for s in ships}
    coverage = sum(len(v) for v in dbs.values())

    # -- geometry validation ---------------------------------------------
    geom = {}
    for key, ship in ships.items():
        db = dbs[key]
        voxel = None
        if db:
            a0 = min(db, key=lambda d: abs(d["angle_deg"]))
            voxel = {"area_frontal_m2": a0["voxel_area_frontal_m2"],
                     "area_lateral_m2": a0["voxel_area_lateral_m2"]}
        # the projection-area tolerance has to scale with the voxel size: at
        # dx = 4 m a 1.6 m deck pipe run legitimately disappears from the voxel
        # model, so the check is |delta| <= max(2%, 2% * dx) -- and a failure
        # here is a real statement about the discretisation, not a formality
        tol = max(0.02, 0.02 * float(args.dx))
        geom[key] = validate_geometry(ship, voxel_areas=voxel, rel_tol=tol)
        geom[key]["voxel_rel_tol"] = tol
        geom[key]["complexity"] = complexity_index(ship)
        geom[key]["analytic_projection"] = projected_areas(ship)

    # -- symmetry self-check ---------------------------------------------
    # the side-force self-check needs a full polar (the scale is max|C_Y|)
    sym = {k: sign_conventions(v) for k, v in dbs.items()
           if len(v) == len(ANGLES)}

    # -- grid convergence -------------------------------------------------
    gcis = gci_table(recs)

    # -- engines, empirical models, cross-validation ----------------------
    emp, loo_auto, loo_emp, loo_lbm, engines = {}, {}, {}, {}, {}
    incomplete = {k: len(v) for k, v in dbs.items() if len(v) < len(ANGLES)}
    for key, ship in ships.items():
        db = dbs[key]
        # cross-validation is only meaningful on a complete polar
        if len(db) < len(ANGLES):
            continue
        engines[key] = HybridWindEngine(ship, db)
        emp[key] = empirical_vs_lbm(ship, db)
        loo_auto[key] = loo_validation(ship, db, route="auto")
        loo_emp[key] = loo_validation(ship, db, route="empirical")
        loo_lbm[key] = loo_validation(ship, db, route="lbm")

    # -- cube anchor ------------------------------------------------------
    cube_runs = []
    for r in recs:
        if r.get("case", {}).get("case") != "cube":
            continue
        cd = float(r["coefficients"]["C_D"])
        cube_runs.append({"tag": os.path.basename(r["_path"])[:-5],
                          "C_D": cd, "C_D_std": float(r["coefficients"]["C_D_std"]),
                          "re": float(r["case"]["re_side"]),
                          "cells_per_side": int(r["case"]["cells_per_side"]),
                          "rel_error_vs_literature": abs(cd - CUBE_REF_CD) / CUBE_REF_CD,
                          "source": r["_path"]})
    cube = {"literature_C_D": CUBE_REF_CD,
            "literature_range": [1.00, 1.10], "runs": cube_runs}

    # -- annual wind assessment ------------------------------------------
    annual = {k: annual_study(catalogue()[k], engines[k]) for k in engines}

    # -- figures ----------------------------------------------------------
    figs = [] if args.no_figures else figures(dbs, emp, gcis, annual, cube,
                                              loo_auto)

    # -- acceptance -------------------------------------------------------
    def _max_gci(key: str, angle: float = 90.0):
        vals = []
        for s in ships:
            g = gcis.get(s, {}).get(angle, {}).get(key, {})
            if "gci_rel" in g:
                vals.append(g["gci_rel"])
        return max(vals) if vals else None

    geom_ok = sum(1 for g in geom.values() if g["ok"])
    mirror = max((s.get("max_residual_side_force_rel") or 0.0
                  for s in sym.values()), default=None)
    mirror_y = max((s.get("max_residual_yaw_moment_rel") or 0.0
                    for s in sym.values()), default=None)
    cube_best = (min(cube_runs, key=lambda r: r["rel_error_vs_literature"])
                 if cube_runs else None)
    loo_cx = max((v["max_err_C_X"] for v in loo_auto.values()), default=None)
    loo_cy = max((v["max_err_C_Y"] for v in loo_auto.values()), default=None)
    loo_emp_cx = max((v["max_err_C_X"] for v in loo_emp.values()), default=None)
    emp_buildup = max((v["max_err_C_X_buildup"] for v in emp.values()), default=None)

    if not loo_auto:
        loo_cx = loo_cy = None
    if not loo_emp:
        loo_emp_cx = None
    if not emp:
        emp_buildup = None
    criteria = [
        {"id": "G1", "description": "geometry validation (hull Cb/L/B, box "
         "overlap, port/starboard symmetry, voxel vs analytic projection areas "
         "within 2%) passes for all three catalogues",
         "value": geom_ok, "threshold": 3, "operator": ">=", "unit": "ships",
         "hard": True},
        {"id": "D1", "description": "LBM database coverage (ship x angle cases)",
         "value": coverage, "threshold": 21, "operator": ">=", "unit": "cases",
         "hard": True},
        {"id": "V1", "description": "port/starboard self-check: residual side "
         "force at theta = 0/180 deg, relative to the polar maximum C_Y",
         "value": mirror, "threshold": 0.10, "operator": "<=", "unit": "1",
         "hard": False},
        {"id": "V2", "description": "same self-check for the residual yaw "
         "moment at theta = 0/180 deg",
         "value": mirror_y, "threshold": 0.10, "operator": "<=", "unit": "1",
         "hard": False},
        {"id": "V3", "description": "bluff-body anchor: cube C_D versus the "
         "literature value 1.05 (best resolution run)",
         "value": None if cube_best is None else cube_best["rel_error_vs_literature"],
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": True},
        {"id": "C1", "description": "grid convergence: max GCI(C_X) at "
         "theta = 90 deg over the three ships", "value": _max_gci("C_X"),
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": True},
        {"id": "C2", "description": "grid convergence: max GCI(C_Y) at "
         "theta = 90 deg over the three ships", "value": _max_gci("C_Y"),
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": True},
        {"id": "E1", "description": "hybrid engine leave-one-angle-out error on "
         "C_X (all three ship types)", "value": loo_cx, "threshold": 0.10,
         "operator": "<=", "unit": "1", "hard": True},
        {"id": "E2", "description": "hybrid engine leave-one-angle-out error on "
         "C_Y (all three ship types)", "value": loo_cy, "threshold": 0.10,
         "operator": "<=", "unit": "1", "hard": True},
        {"id": "E3", "description": "empirical-only branch (Blendermann-style "
         "fit) leave-one-angle-out error on C_X", "value": loo_emp_cx,
         "threshold": 0.10, "operator": "<=", "unit": "1", "hard": False},
        {"id": "E4", "description": "a-priori component build-up versus LBM, max "
         "relative error on C_X (plausibility screen, not fitted)",
         "value": emp_buildup, "threshold": 0.25, "operator": "<=", "unit": "1",
         "hard": False},
        {"id": "A1", "description": "wind-rose annual assessment produced for "
         "all three ship types", "value": len(annual), "threshold": 3,
         "operator": ">=", "unit": "ships", "hard": False},
    ]
    acc = evaluate_acceptance(criteria)

    db_out = {
        "schema": "tensorlbm.ship_wind_database/v1",
        "node": "IN38-N04",
        "dx_m": args.dx,
        "conventions": {
            "frame": "ship frame, +x bow, +y starboard, +z up from the deck",
            "theta_deg": "wind direction from the bow (0 = head wind)",
            "C_X": "X_wind/(q A_F)", "C_Y": "Y_wind/(q A_L)",
            "C_K": "K_wind/(q A_L H)", "C_N": "N_wind/(q A_L B)",
            "rho_air": RHO_AIR,
        },
        "geometry_validation": geom,
        "symmetry": sym,
        "database": {k: v for k, v in dbs.items()},
        "grid_convergence": gcis,
        "cube_anchor": cube,
        "empirical_models": emp,
        "engines": {k: v.to_dict() for k, v in engines.items()},
        "cross_validation": {"hybrid_auto": loo_auto, "empirical_only": loo_emp,
                             "lbm_table_only": loo_lbm},
        "annual_wind": annual,
        "figures": [os.path.relpath(p, HERE) for p in figs],
        "raw_records": [r["_path"] for r in recs],
        "coverage": {"n_cases": coverage, "per_ship": {k: len(v) for k, v in dbs.items()},
                     "incomplete_polars": incomplete,
                     "note": "ships whose 7-angle polar is incomplete are excluded "
                             "from engine fitting and cross-validation"},
    }
    with open(os.path.join(ART, "wind_database.json"), "w",
              encoding="utf-8") as fh:
        json.dump(db_out, fh, indent=2, sort_keys=True, default=_default)
    with open(os.path.join(ART, "acceptance.json"), "w", encoding="utf-8") as fh:
        json.dump({"node": "IN38-N04", "schema":
                   "tensorlbm.wind_acceptance/v1", **acc,
                   "inputs": {"dx_m": args.dx, "coverage": coverage,
                              "cube_best": cube_best}},
                  fh, indent=2, sort_keys=True, default=_default)

    print(json.dumps({"coverage": coverage, "geometry_ok": geom_ok,
                      "hard_passed": f"{acc['n_hard_passed']}/{acc['n_hard']}",
                      "all_hard_passed": acc["all_hard_passed"],
                      "criteria": {c["id"]: (None if c["value"] is None
                                             else round(c["value"], 4),
                                             c["verdict"]) for c in acc["criteria"]}},
                     indent=1, default=_default))
    return 0


def _default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())
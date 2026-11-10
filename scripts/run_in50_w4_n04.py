#!/usr/bin/env python
"""IN50-W4-N04 -- compressible-flow system verification driver (TensorLBM).

Node scope
----------
System verification of the compressible solver family in
``tensorlbm.compressible`` on the three canonical compressible-flow problems:

1. **subsonic NACA0012** -- ``C_L(Ma, alpha)``, linearity of the lift curve,
   lift-slope growth vs Prandtl-Glauert, spurious (d'Alembert) drag level;
2. **transonic shock capturing** -- ``Ma = 0.85`` terminal shock: position,
   Rankine-Hugoniot strength consistency, and its **mesh convergence**
   (observed order + Roache GCI on a uniform 1.5x family) which the earlier
   IN421-W2-N03 study could not compute (non-monotone family);
3. **sonic-boom N wave** -- supersonic double wedge ``theta = 10 deg`` (plus
   ``theta = 2, 5 deg`` small-angle cases where linear theory is valid),
   wall-pressure N-wave source signature vs Ackeret linear theory and vs the
   exact oblique-shock / Prandtl-Meyer relations, including N-wave impulse.

Usage
-----
    python scripts/run_in50_w4_n04.py --suite all --workers 20
    python scripts/run_in50_w4_n04.py --suite lift
    python scripts/run_in50_w4_n04.py --assemble     # gates + figures

Raw per-case results go to ``artifacts/compressible_lbm/raw/``; the assembled
verification record (``result.json``, ``gates_verified.json``, ``figures/``) is
produced by :func:`assemble`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "compressible_lbm")
RAW = os.path.join(ART, "raw")
FIG = os.path.join(ART, "figures")
FIGD = os.path.join(ROOT, "docs", "figures")

LIFT_MACHS = (0.30, 0.50, 0.70)
TRANSONIC_MACH = 0.85
ALPHAS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
GRID_FAMILY = ((192, 72), (288, 108), (432, 162))    # uniform 1.5x refinement
L1 = (160, 64)                                        # production lift grid
# (theta_deg, mach, first_cell): the sharp 4-degree included angle of the
# theta=2 wedge cannot carry a 2e-3 c first cell (the apex cells become slivers
# with an aspect ratio ~1e2 and the explicit update diverges within ~100
# iterations, measured); 1e-2 c is stable and was verified against the exact
# oblique shock to -3 %.
BOOM_CASES = ([(theta, ma, 0.002) for theta in (10.0,) for ma in (1.5, 2.0, 2.5, 3.0, 4.0)]
              + [(2.0, 2.0, 0.01), (5.0, 2.0, 0.002)])


def _key(kind, **kw):
    parts = [kind] + [f"{k}{v}" for k, v in sorted(kw.items())]
    return "_".join(str(p).replace(".", "p") for p in parts)


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def run_airfoil_case(spec):
    """One NACA0012 Euler solve; returns a JSON-serialisable record."""
    from tensorlbm.compressible.transonic_aero import run_naca_case

    t0 = time.time()
    n_points, nj = spec["grid"]
    r = run_naca_case(spec["mach"], spec["alpha"], n_points=n_points, nj=nj,
                      first_cell=spec.get("first_cell", 0.002),
                      r_out=spec.get("r_out", 12.0),
                      n_iter=spec["n_iter"], cfl=spec.get("cfl", 0.4),
                      order=2)
    surf = r["surface"]
    div_upper = {"x": list(np.asarray(surf["upper_x"], float)),
                 "cp": list(np.asarray(surf["upper_cp"], float))}
    div_lower = {"x": list(np.asarray(surf["lower_x"], float)),
                 "cp": list(np.asarray(surf["lower_cp"], float))}
    rec = {
        "kind": "airfoil",
        "case": {"mach": spec["mach"], "alpha_deg": spec["alpha"]},
        "grid": {"n_points": n_points, "nj": nj,
                 "cells": r["mesh"]["cells"],
                 "first_cell": r["mesh"]["first_cell"], "r_out": r["mesh"]["r_out"]},
        "solver": {"iterations": r["solver"]["iterations"],
                   "residual": r["solver"]["residual"],
                   "drift": r["solver"]["drift"]},
        "forces": r["forces"],
        "field": r["field"],
        "isentropic": r["isentropic"],
        "surface": {"upper": div_upper, "lower": div_lower},
        "wall_clock_s": time.time() - t0,
        "label": spec["key"],
    }
    if spec.get("save_field"):
        sol = r["solution"]
        mesh = r["mesh_obj"]
        fa = sol.field_arrays()
        np.savez_compressed(os.path.join(ART, "raw", spec["key"] + "_field.npz"),
                            mach=fa["mach"], cp=fa["cp"], rho=fa["rho"], p=fa["p"],
                            xc=mesh.xc if np.shape(mesh.xc)[0] == np.shape(fa["mach"])[0]
                            else 0.5 * (mesh.X[:-1, :-1] + mesh.X[1:, :-1]),
                            yc=mesh.yc if np.shape(mesh.yc)[0] == np.shape(fa["mach"])[0]
                            else 0.5 * (mesh.Y[:-1, :-1] + mesh.Y[1:, :-1]))
    return rec


def run_boom_case(spec):
    """One supersonic double-wedge solve (N-wave source signature)."""
    from tensorlbm.compressible.transonic_aero import run_wedge_case

    t0 = time.time()
    r = run_wedge_case(spec["mach"], spec["theta"], n_points=spec.get("n_points", 200),
                       nj=spec.get("nj", 80),
                       first_cell=spec.get("first_cell", 0.002),
                       n_iter=spec["n_iter"], cfl=spec.get("cfl", 0.35))
    rec = {
        "kind": "boom",
        "case": {"mach": spec["mach"], "theta_deg": spec["theta"]},
        "grid": r["mesh"],
        "solver": r["solver"],
        "front_wedge": r["front_wedge"],
        "rear_wedge": r["rear_wedge"],
        "reference": r["reference"],
        "rear_reference": r["rear_reference"],
        "error_vs_ackeret_pct": r["error_vs_ackeret_pct"],
        "error_vs_exact_oblique_pct": r["error_vs_exact_oblique_pct"],
        "rear_error_vs_exact_expansion_pct": r["rear_error_vs_exact_expansion_pct"],
        "surface": {"x": list(np.asarray(r["surface"]["x"], float)),
                    "p_ratio": list(np.asarray(r["surface"]["p_ratio"], float)),
                    "cp": list(np.asarray(r["surface"]["cp"], float))},
        "wall_clock_s": time.time() - t0,
        "label": spec["key"],
    }
    return rec


def run_spec(spec):
    try:
        if spec["kind"] == "airfoil":
            rec = run_airfoil_case(spec)
        elif spec["kind"] == "boom":
            rec = run_boom_case(spec)
        else:
            raise ValueError(f"unknown kind {spec['kind']}")
    except Exception as exc:                          # honest FAIL, never silent
        rec = {"kind": spec["kind"], "label": spec["key"], "status": "ERROR",
               "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(limit=4),
               "case": spec.get("case", {})}
    _save_json(os.path.join(RAW, spec["key"] + ".json"), rec)
    return rec


# ---------------------------------------------------------------------------
# case list
# ---------------------------------------------------------------------------
def build_specs(suite="all", n_iter_lift=16000, n_iter_grid=20000,
                n_iter_boom=7000):
    specs = []
    if suite in ("all", "lift"):
        for ma in LIFT_MACHS:
            for al in ALPHAS:
                specs.append(dict(key=_key("lift", ma=ma, a=al), kind="airfoil",
                                  mach=ma, alpha=al, grid=L1, n_iter=n_iter_lift))
    if suite in ("all", "transonic"):
        for al in (0.0, 1.0, 2.0):
            specs.append(dict(key=_key("trans", ma=TRANSONIC_MACH, a=al),
                              kind="airfoil", mach=TRANSONIC_MACH, alpha=al,
                              grid=L1, n_iter=n_iter_lift))
    if suite in ("all", "grid"):
        for ma, al in ((0.30, 2.0), (TRANSONIC_MACH, 0.0)):
            for gi, grid in enumerate(GRID_FAMILY):
                specs.append(dict(key=_key("grid", ma=ma, a=al, g=gi + 1),
                                  kind="airfoil", mach=ma, alpha=al, grid=grid,
                                  n_iter=n_iter_grid,
                                  save_field=(gi == len(GRID_FAMILY) - 1)))
    if suite in ("all", "boom"):
        for theta, ma, fc in BOOM_CASES:
            specs.append(dict(key=_key("boom", ma=ma, th=theta), kind="boom",
                              mach=ma, theta=theta, first_cell=fc,
                              n_iter=n_iter_boom))
    return specs


def run_suite(suite, workers, args):
    os.makedirs(RAW, exist_ok=True)
    specs = build_specs(suite, n_iter_lift=args.iters, n_iter_grid=args.iters_grid,
                        n_iter_boom=args.iters_boom)
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in specs if s["key"] in wanted]
        if not specs:
            raise SystemExit(f"--only matched no case: {sorted(wanted)}")
    print(f"[IN50-W4-N04] suite={suite} cases={len(specs)} workers={workers}", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_spec, s): s for s in specs}
        for f in as_completed(futs):
            s = futs[f]
            rec = f.result()
            done += 1
            if rec.get("status") == "ERROR":
                print(f"  [{done}/{len(specs)}] ERROR {s['key']}: {rec['error'][:90]}",
                      flush=True)
            else:
                extra = ""
                if rec["kind"] == "airfoil":
                    extra = (f"cl={rec['forces']['cl']:.4f} cd={rec['forces']['cd']:.5f} "
                             f"res={rec['solver']['residual']:.1e} "
                             f"isen={rec['isentropic']['max_rel_dev_subsonic']:.2e}")
                else:
                    extra = (f"dp_ack_err={rec['error_vs_ackeret_pct']:+.2f}% "
                             f"dp_exact_err={rec['error_vs_exact_oblique_pct']:+.3f}%")
                print(f"  [{done}/{len(specs)}] {s['key']}: {extra} "
                      f"({rec.get('wall_clock_s', 0.0):.0f}s)", flush=True)
    print(f"[IN50-W4-N04] suite={suite} finished in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def _load_raw():
    out = {}
    if not os.path.isdir(RAW):
        return out
    for name in sorted(os.listdir(RAW)):
        if name.endswith(".json"):
            with open(os.path.join(RAW, name), "r", encoding="utf-8") as fh:
                out[name[:-5]] = json.load(fh)
    return out


def _get(raw, key):
    rec = raw.get(key)
    if rec is None or rec.get("status") == "ERROR":
        return None
    return rec


def assemble():
    """Turn raw case records into the verification record + figures."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tensorlbm.compressible import convergence as cv
    from tensorlbm.compressible.transonic_aero import (lift_slope,
                                                       prandtl_glauert_lift_slope,
                                                       thickness_corrected_lift_slope,
                                                       gci_roache)
    from tensorlbm.compressible.kinetic_flux import (naca0012_half_thickness,
                                                     contour_loop)

    raw = _load_raw()
    os.makedirs(ART, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    result = {"node": "IN50-W4-N04", "repo": "TensorLBM",
              "campaign_node": "IN50-W4-N04",
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "raw_cases": len(raw)}

    # ---- geometry gate -----------------------------------------------------
    xl, yl = contour_loop(naca0012_half_thickness, n_points=4000)
    chord = float(xl.max() - xl.min())
    t_max = float(yl.max() - yl.min())
    area = float(abs(np.trapezoid(yl, xl)) if hasattr(np, "trapezoid")
                 else abs(np.trapz(yl, xl)))
    # closed-form area of the *same* coefficient set as ``naca0012_half_thickness``
    # (closed TE, c4 = 0.1036): A = 2 * 5t * int_0^1 (0.2969 x^0.5 - 0.126 x
    #       - 0.3516 x^2 + 0.2843 x^3 - c4 x^4) dx
    c4 = 0.1036
    area_ref = float(2.0 * 5.0 * 0.12 * (0.2969 * 2.0 / 3.0 - 0.1260 / 2.0
                                         - 0.3516 / 3.0 + 0.2843 / 4.0 - c4 / 5.0))
    # Leading-edge radius.  A least-squares circle through the nose stations is
    # ill-conditioned (the profile is parabolic there: 6 points give -0.2 %,
    # 57 points give +4.9 %), so the radius is taken from the curvature
    # definition r = y^2 / (2 x) at the station closest to the nose (O(sqrt x)
    # accurate, -> 0.1 % on the 4000-point contour).  The circle fit is kept as
    # a cross-check in the record.
    aft = np.argsort(xl)
    nz = [i for i in aft if xl[i] > 0.0][:1]
    idx = int(nz[0])
    r_le = float(yl[idx] ** 2 / (2.0 * xl[idx]))
    nose = aft[:40]
    p = np.array([xl[nose], yl[nose]]).T
    A = np.hstack([p, np.ones((p.shape[0], 1))])
    b = (p ** 2).sum(axis=1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    r_le_fit = float(math.sqrt(sol[2] + (sol[0] / 2) ** 2 + (sol[1] / 2) ** 2))
    geo = {
        "chord": chord, "chord_error_pct": 100.0 * (chord - 1.0) / 1.0,
        "thickness_ratio": t_max, "thickness_error_pct": 100.0 * (t_max - 0.12) / 0.12,
        "area": area, "area_reference_closed_form": area_ref,
        "area_reference_open_te": 0.6852 * 0.12,
        "area_error_pct": 100.0 * (area - area_ref) / area_ref,
        "leading_edge_radius": r_le,
        "leading_edge_radius_fit40": r_le_fit,
        "leading_edge_radius_method": "curvature definition y^2/(2x) at the "
                                      "nose-adjacent station of a 4000-point contour",
        "leading_edge_radius_theory": 1.1019 * 0.12 ** 2,
        "leading_edge_radius_error_pct": 100.0 * (r_le - 1.1019 * 0.12 ** 2) / (1.1019 * 0.12 ** 2),
        "x_tmax": float(xl[np.argmax(yl)]),
        "n_points": int(xl.size),
    }
    result["geometry"] = geo

    # geometry of the *computational* meshes (not just of the analytic contour):
    # wall faces must sit on the profile and must span the full chord
    from tensorlbm.compressible.kinetic_flux import (build_annulus_mesh,
                                                     radial_levels_from_surface)
    mesh_geo = {}
    for n_points, nj in (L1, GRID_FAMILY[0], GRID_FAMILY[-1]):
        xl2, yl2 = contour_loop(naca0012_half_thickness, n_points=n_points)
        m2 = build_annulus_mesh(xl2, yl2,
                                radial_levels_from_surface(nj, 12.0, first_cell=0.002))
        dev = np.abs(np.abs(m2.wall_y) - naca0012_half_thickness(m2.wall_x))
        mesh_geo[f"{n_points}x{nj}"] = {
            "cells": int(m2.n_cells),
            "wall_chord": float(m2.wall_x.max() - m2.wall_x.min()),
            "wall_chord_error_pct": 100.0 * ((m2.wall_x.max() - m2.wall_x.min()) - 1.0),
            "wall_profile_max_dev": float(dev.max()),
            "wall_x_min": float(m2.wall_x.min()),
            "wall_x_max": float(m2.wall_x.max()),
            "n_wall_faces": int(m2.wall_x.size),
        }
    result["mesh_geometry"] = mesh_geo

    # ---- 1. subsonic lift ---------------------------------------------------
    lift = {}
    for ma in LIFT_MACHS + (TRANSONIC_MACH,):
        alphas, cls, cds, isen = [], [], [], []
        for al in ALPHAS:
            rec = _get(raw, _key("lift", ma=ma, a=al)) or _get(raw, _key("trans", ma=ma, a=al))
            if rec is None:
                continue
            alphas.append(al)
            cls.append(rec["forces"]["cl"])
            cds.append(rec["forces"]["cd"])
            isen.append(rec["isentropic"]["max_rel_dev_subsonic"])
        if not alphas:
            continue
        slope = lift_slope(alphas, cls, alpha_max=6.0)
        # linear-fit quality on the full sampled range
        a_rad = np.radians(np.array(alphas))
        A = np.vstack([a_rad, np.ones(a_rad.size)]).T
        coef, *_ = np.linalg.lstsq(A, np.array(cls), rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((np.array(cls) - pred) ** 2))
        ss_tot = float(np.sum((np.array(cls) - np.mean(cls)) ** 2))
        lift[str(ma)] = {
            "alpha_deg": alphas, "cl": cls, "cd": cds,
            "isentropic_max_subsonic": isen,
            "lift_slope": slope,
            "r2_linear": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "pg_lift_slope": prandtl_glauert_lift_slope(ma),
            "pg_error_pct": 100.0 * (slope["dcl_dalpha_per_rad"]
                                     - prandtl_glauert_lift_slope(ma))
            / prandtl_glauert_lift_slope(ma),
            "thick_corrected_slope": thickness_corrected_lift_slope(0.12),
            "cd_alpha0": cds[0] if alphas and alphas[0] == 0.0 else None,
        }
    result["lift"] = lift

    # ---- 2. transonic shock + mesh convergence -----------------------------
    transonic = {}
    for al in (0.0, 1.0, 2.0):
        rec = _get(raw, _key("trans", ma=TRANSONIC_MACH, a=al))
        if rec is None:
            continue
        up = rec["surface"]["upper"]
        ss = cv.shock_strength(np.array(up["x"]), np.array(up["cp"]), TRANSONIC_MACH)
        transonic[f"alpha_{al:g}"] = {
            "cp_min": rec["field"]["cp_min"],
            "max_local_mach": rec["field"]["max_local_mach"],
            "isentropic": rec["isentropic"],
            "shock_upper": ss,
            "forces": rec["forces"],
            "grid": rec["grid"],
        }
    result["transonic"] = transonic

    # grid family convergence
    conv = {}
    for ma, al in ((0.30, 2.0), (TRANSONIC_MACH, 0.0)):
        fam = []
        for gi in range(len(GRID_FAMILY)):
            rec = _get(raw, _key("grid", ma=ma, a=al, g=gi + 1))
            if rec is None:
                continue
            up = rec["surface"]["upper"]
            ss = cv.shock_strength(np.array(up["x"]), np.array(up["cp"]), ma)
            fam.append({
                "grid": rec["grid"], "forces": rec["forces"],
                "isentropic": rec["isentropic"],
                "shock": ss,
                "residual": rec["solver"]["residual"],
                "drift": rec["solver"].get("drift", {}),
                "h": cv.mesh_size(rec["grid"]["cells"]),
            })
        if len(fam) < 3:
            continue
        r = cv.refinement_ratio(fam[0]["grid"]["cells"], fam[1]["grid"]["cells"])
        r2 = cv.refinement_ratio(fam[1]["grid"]["cells"], fam[2]["grid"]["cells"])
        items = {
            "cl": [f["forces"]["cl"] for f in fam],
            "cd": [f["forces"]["cd"] for f in fam],
            "isentropic_max_subsonic": [f["isentropic"]["max_rel_dev_subsonic"] for f in fam],
            "isentropic_p99_subsonic": [f["isentropic"]["p99_rel_dev_subsonic"] for f in fam],
        }
        if all(f["shock"].get("x_shock") is not None for f in fam):
            items["x_shock"] = [f["shock"]["x_shock"] for f in fam]
            items["shock_strength_rel_error"] = [f["shock"]["strength_rel_error"]
                                                 for f in fam]
        conv[f"ma{ma}_alpha{al:g}"] = {
            "family": fam,
            "refinement_ratio": [r, r2],
            "richardson": {k: cv.richardson_3grid(v[0], v[1], v[2], r) for k, v in items.items()},
        }
    result["convergence"] = conv

    # ---- 3. sonic boom / N wave --------------------------------------------
    boom = {}
    for theta, ma, _fc in BOOM_CASES:
        rec = _get(raw, _key("boom", ma=ma, th=theta))
        if rec is None:
            continue
        nw = cv.nwave_metrics(np.array(rec["surface"]["x"]),
                              np.array(rec["surface"]["p_ratio"]))
        ack = cv.ackeret_dp_over_p(ma, theta)
        exact = cv.oblique_shock_pressure_ratio(ma, theta) - 1.0
        boom[f"ma{ma}_theta{theta:g}"] = {
            "measured_front_dp": rec["front_wedge"]["dp_over_p"],
            "measured_rear_dp": rec["rear_wedge"]["dp_over_p"],
            "ackeret_dp": ack, "exact_oblique_dp": exact,
            "error_vs_ackeret_pct": 100.0 * (rec["front_wedge"]["dp_over_p"] - ack) / ack,
            "error_vs_exact_pct": 100.0 * (rec["front_wedge"]["dp_over_p"] - exact) / exact,
            "ackeret_theory_error_pct": 100.0 * (ack - exact) / exact,
            "rear_expansion_exact_dp": rec["rear_reference"]["rear_dp_over_p"],
            "rear_error_vs_exact_expansion_pct": rec["rear_error_vs_exact_expansion_pct"],
            "nwave": nw,
            "solver": rec["solver"], "grid": rec["grid"],
        }
    result["boom"] = boom

    # ---- gates -------------------------------------------------------------
    gates = []

    def gate(gid, criterion, measured, threshold, unit, reference, extra=None,
             passed=None):
        g = {"id": gid, "criterion": criterion, "reference": reference,
             "measured": measured, "threshold": threshold, "unit": unit,
             "pass": bool(abs(measured) <= threshold) if passed is None else bool(passed)}
        if extra:
            g.update(extra)
        gates.append(g)

    gate("G1-GEOM", "NACA0012 chord and thickness error vs the analytic 4-digit "
         "definition", max(abs(geo["chord_error_pct"]), abs(geo["thickness_error_pct"])),
         0.1, "percent", "NACA 4-digit formula, t=0.12",
         extra={"chord": geo["chord"], "thickness_ratio": geo["thickness_ratio"]})
    _mg = result["mesh_geometry"]
    gate("G1b-NOSE", "leading-edge radius error vs r_LE = 1.1019 t^2",
         geo["leading_edge_radius_error_pct"], 1.0, "percent",
         "NACA 4-digit nose radius (curvature definition at the nose station)",
         extra={"r_le": geo["leading_edge_radius"],
                "r_le_circle_fit_40pts": geo["leading_edge_radius_fit40"]})

    gate("G1d-AREA", "sectional area vs the closed-form integral of the same "
         "NACA0012 coefficient set (closed TE)", geo["area_error_pct"], 0.01,
         "percent", "A = 2*5t*int_0^1(0.2969x^0.5-0.126x-0.3516x^2+0.2843x^3"
                    "-0.1036x^4)dx",
         extra={"area": geo["area"], "area_reference": geo["area_reference_closed_form"],
                "note": "the open-TE variant (c4 = 0.1015) has A = 0.6852 t = "
                        f"{geo['area_reference_open_te']:.6f}, which is *not* the "
                        "profile used here"})
    gate("G1c-MESH-GEOM", "computational wall faces span the chord and lie on the "
         "profile (worst over the production grids)",
         max(abs(v["wall_chord_error_pct"]) for v in _mg.values()), 0.1, "percent",
         "chord = 1.0 c; wall faces are the cell-centred faces of grid line j=0",
         extra={"worst_wall_profile_dev": max(v["wall_profile_max_dev"] for v in _mg.values()),
                "note": "wall_x/wall_y are wall-face midpoints, so the chord is "
                        "measured between the extreme face midpoints"})
    if "0.3" in lift:
        L = lift["0.3"]
        gate("G2-LIFT-LINEAR", "R^2 of the CL(alpha) linear fit, alpha in [0,10] deg, "
             "Ma=0.3", 1.0 - L["r2_linear"], 1e-3, "1-R^2",
             "symmetric airfoil: CL linear in alpha below stall",
             extra={"r2": L["r2_linear"],
                    "zero_lift_offset_deg": L["lift_slope"]["zero_lift_offset_deg"]})
        gate("G3-ZERO-LIFT", "zero-lift angle of a symmetric section (deg), Ma=0.3",
             abs(L["lift_slope"]["zero_lift_offset_deg"]), 0.05, "deg",
             "symmetry: CL(0)=0 exactly")
        gate("G4-SPURIOUS-DRAG", "inviscid C_D at alpha=0, Ma=0.3 (spurious / "
             "d'Alembert drag)", abs(L["cd_alpha0"] or 0.0), 0.005, "Cd",
             "d'Alembert paradox: inviscid drag of a 2-D body is 0")
    for ma in LIFT_MACHS:
        key = str(ma)
        if key not in lift:
            continue
        L = lift[key]
        gate(f"G5-PG-{ma:.1f}", f"lift slope error vs Prandtl-Glauert 2*pi/beta, Ma={ma}",
             L["pg_error_pct"], 10.0, "percent",
             "Prandtl-Glauert compressibility correction (2pi/beta)",
             extra={"dcl_dalpha_per_rad": L["lift_slope"]["dcl_dalpha_per_rad"],
                    "pg_slope": L["pg_lift_slope"]})

    if "alpha_0" in transonic:
        t0 = transonic["alpha_0"]
        ss = t0["shock_upper"]
        gate("G6-SHOCK-EXISTS", "terminal shock detected on the upper surface at "
             "Ma=0.85, alpha=0 (supersonic pocket closes before the trailing edge)",
             float(ss.get("x_shock") is None), 0.0, "flag",
             "transonic flow: isentropic Mach crosses 1 inside the chord",
             passed=ss.get("x_shock") is not None,
             extra={"x_shock": ss.get("x_shock"),
                    "max_surface_mach": ss.get("max_surface_mach")})
        if ss.get("rankine_hugoniot_p_ratio"):
            gate("G7-SHOCK-RH", "captured shock strength vs Rankine-Hugoniot built "
                 "from the pre-shock plateau Mach, Ma=0.85 alpha=0 (L1 grid)",
                 100.0 * abs(ss.get("strength_rel_error", float("nan"))), 15.0,
                 "percent", "exact normal-shock p2/p1 = 1+2g/(g+1)(M1^2-1)",
                 extra={"measured_p_ratio": ss.get("measured_p_ratio"),
                        "rh_p_ratio": ss.get("rankine_hugoniot_p_ratio"),
                        "pre_shock_mach_plateau": ss.get("pre_shock_mach_plateau")})

    for tag, cvrec in conv.items():
        R = cvrec["richardson"]
        notes = []
        qmap = {}
        lifting = ("cl" in R and abs(R["cl"]["f_fine"]) > 0.05)
        if lifting:
            qmap["cl"] = "smooth"
        elif "cl" in R:
            notes.append("C_L at alpha=0 of a symmetric section is zero; its mesh "
                         "scatter (~2e-3) is asymmetry noise, so no GCI is applied "
                         "to it (the lifting case supplies the C_L convergence gate).")
        qmap["cd"] = "smooth"
        if "isentropic_p99_subsonic" in R:
            qmap["isentropic_p99_subsonic"] = "smooth"
        if "isentropic_max_subsonic" in R:
            notes.append("the *max* subsonic entropy deviation is a single-cell "
                         "extremum inside the captured shock band (non-smooth "
                         "functional): reported, not gated; the gated counterpart "
                         "is the p99 deviation.")
        if "x_shock" in R:
            qmap["x_shock"] = "discontinuous"
        for qname, kind in qmap.items():
            r0 = R[qname]
            if kind == "smooth":
                ok = (r0["monotone"] and r0["gci_fine"] is not None
                      and abs(r0["gci_fine"]) <= 0.10
                      and 0.3 <= (r0["observed_order"] or 0.0) <= 6.0)
                meas = None if r0["gci_fine"] is None else abs(r0["gci_fine"])
                thr, unit = 0.10, "GCI fraction"
                crit = (f"three-grid (1.5x, uniform) observed order and Roache GCI of "
                        f"{qname} at {tag} (smooth functional)")
            else:
                if r0["monotone"] and abs(r0["f_fine"]) > 1e-30:
                    change = abs(r0["f_fine"] - r0["f_medium"]) / abs(r0["f_fine"])
                else:
                    change = None
                ok = change is not None and change <= 0.02
                meas = change
                thr, unit = 0.02, "relative change medium->fine"
                crit = (f"three-grid convergence of {qname} at {tag} (discontinuous "
                        f"quantity: Richardson order is not defined, so the "
                        f"medium->fine change is the criterion)")
            gates.append({
                "id": f"G8-{'GCI' if kind == 'smooth' else 'SHOCK'}-{tag}-{qname}",
                "criterion": crit,
                "reference": "Richardson extrapolation + Roache GCI, safety 1.25 "
                             "(smooth); medium->fine change (discontinuous)",
                "measured": meas, "threshold": thr, "unit": unit,
                "observed_order": r0["observed_order"],
                "monotone": r0["monotone"],
                "values": [r0["f_coarse"], r0["f_medium"], r0["f_fine"]],
                "extrapolated": r0["extrapolated"],
                "pass": bool(ok),
            })
        cvrec["gated_quantities"] = {q: k for q, k in qmap.items()}
        cvrec["notes"] = notes
        # companion gate: in a *subcritical* family the inviscid drag is purely
        # numerical (entropy) drag and must vanish under refinement; at Ma=0.85 a
        # genuine wave-drag floor exists, so only the GCI gate applies there.
        cd_seq = [f["forces"]["cd"] for f in cvrec["family"]]
        subcritical = float(tag.split("_")[0][2:]) < 0.70
        if subcritical:
            gates.append({
                "id": f"G4b-DRAG-REFINEMENT-{tag}",
                "criterion": f"spurious (entropy) drag must decrease monotonically "
                             f"when the mesh is refined (subcritical family {tag})",
                "reference": "d'Alembert: inviscid drag -> 0 as h -> 0",
                "measured": float(cd_seq[-1] - cd_seq[0]),
                "threshold": 0.0, "unit": "Cd change (fine - coarse)",
                "values": cd_seq,
                "pass": bool(all(cd_seq[i + 1] < cd_seq[i]
                                 for i in range(len(cd_seq) - 1))),
            })

    # boom gates: small-angle Ackeret validity, exact-shock agreement, N-wave shape
    small = boom.get("ma2_theta2")
    if small:
        gate("G9-BOOM-ACKERET-SMALL", "wedge overpressure vs Ackeret linear theory, "
             "theta=2 deg, Ma=2 (the angle range where linear theory is valid)",
             100.0 * abs(small["error_vs_ackeret_pct"]), 10.0, "percent",
             "Ackeret: dp/p = gamma M^2 theta / sqrt(M^2-1)",
             extra={"measured_dp": small["measured_front_dp"],
                    "ackeret_dp": small["ackeret_dp"], "exact_dp": small["exact_oblique_dp"],
                    "first_cell": 0.01})
    big = boom.get("ma2_theta10")
    if big:
        gate("G10-BOOM-EXACT", "wedge overpressure vs exact oblique-shock relation, "
             "theta=10 deg, Ma=2", 100.0 * abs(big["error_vs_exact_pct"]), 2.0,
             "percent", "exact oblique shock (theta-beta-M + Rankine-Hugoniot)",
             extra={"measured_dp": big["measured_front_dp"],
                    "exact_dp": big["exact_oblique_dp"]})
        # same normalisation on both sides: (measured/ackeret - 1) vs
        # (exact/ackeret - 1).  Comparing |measured-ackeret| with |exact-ackeret|
        # in *percent of different denominators* inflates the difference by the
        # Ackeret truncation itself, so that comparison would be a unit error.
        solver_vs_ack = 100.0 * (big["measured_front_dp"] / big["ackeret_dp"] - 1.0)
        theory_vs_ack = 100.0 * (big["exact_oblique_dp"] / big["ackeret_dp"] - 1.0)
        gate("G11-BOOM-ACKERET-TRUNCATION", "the 10 deg gap to Ackeret equals the "
             "linear theory's own truncation error (same normalisation, Ma=2)",
             solver_vs_ack - theory_vs_ack, 1.0, "percentage points",
             "Ackeret vs exact oblique shock, both relative to Ackeret",
             extra={"solver_vs_ackeret_pct_of_ackeret": solver_vs_ack,
                    "theory_vs_ackeret_pct_of_ackeret": theory_vs_ack})
        nw = big["nwave"]
        gate("G12-NWAVE-SHAPE", "N-wave source signature: front compression plateau "
             "> 0 and rear expansion plateau < 0, theta=10 deg Ma=2",
             float(not (nw.get("front_dp_over_p", 0) > 0 > nw.get("rear_dp_over_p", 0))),
             0.0, "flag", "double-wedge N-wave: compression then expansion",
             passed=bool(nw.get("front_dp_over_p", 0) > 0 > nw.get("rear_dp_over_p", 0)),
             extra={"front_dp_over_p": nw.get("front_dp_over_p"),
                    "rear_dp_over_p": nw.get("rear_dp_over_p"),
                    "asymmetry": nw.get("asymmetry"), "impulse": nw.get("impulse")})
    # rear (expansion) half of the N wave vs the exact Prandtl-Meyer fan
    rear = [(k, v) for k, v in boom.items() if "_theta10" in k]
    if rear:
        worst = max(rear, key=lambda kv: abs(kv[1]["rear_error_vs_exact_expansion_pct"]))
        gate("G13-NWAVE-EXPANSION", "rear-wedge (expansion) pressure vs exact "
             "Prandtl-Meyer, worst over theta=10 deg cases",
             worst[1]["rear_error_vs_exact_expansion_pct"], 3.0, "percent",
             "exact oblique shock + Prandtl-Meyer expansion (theta-2theta fan)",
             extra={"worst_case": worst[0],
                    "measured_rear_dp": worst[1]["measured_rear_dp"],
                    "exact_rear_dp": worst[1]["rear_expansion_exact_dp"]})
        b10 = boom.get("ma2_theta10")
        if b10:
            exact_asym = abs(b10["rear_expansion_exact_dp"]) / b10["exact_oblique_dp"]
            gates.append({
                "id": "G13b-NWAVE-ASYMMETRY",
                "criterion": "N-wave asymmetry |rear expansion|/front compression at "
                             "Ma=2, theta=10 deg, solver vs exact theory",
                "reference": "exact oblique shock + Prandtl-Meyer, same geometry",
                "measured": b10["nwave"].get("asymmetry"),
                "threshold": None, "unit": "ratio",
                "exact_theory_value": float(exact_asym),
                "values": [b10["nwave"].get("asymmetry"), float(exact_asym)],
                "pass": bool(b10["nwave"].get("asymmetry") is not None
                             and abs(b10["nwave"]["asymmetry"] - exact_asym) <= 0.05),
            })
    # Ackeret is a theta -> 0 limit: the deviation must shrink with the wedge angle
    trio = [boom.get(k) for k in ("ma2_theta2", "ma2_theta5", "ma2_theta10")]
    if all(t is not None for t in trio):
        devs = [abs(t["error_vs_ackeret_pct"]) for t in trio]
        ratio_ok = devs[0] < devs[1] < devs[2]
        gate("G14-ACKERET-LIMIT", "deviation from Ackeret must shrink as theta -> 0 "
             "(Ma=2: theta = 2, 5, 10 deg)", float(not ratio_ok), 0.0, "flag",
             "Ackeret is the O(theta) leading order term; error is O(theta^2)",
             passed=ratio_ok, extra={"deviation_pct_vs_ackeret": devs,
                                     "theta_deg": [2.0, 5.0, 10.0]})

    # experimental anchors (honest, never fabricated)
    from tensorlbm.compressible.kinetic_flux import experimental_anchor_status
    result["experimental_anchors"] = {
        "agard_cp_303": experimental_anchor_status(None),
        "bailey_1986_n_wave": experimental_anchor_status(None),
        "verdict": "WITHHELD - no digitised AGARD CP-303 / Bailey-1986 dataset with "
                   "provenance is available inside the repository or offline; "
                   "gates G1-G13 are analytic/numerical anchors. Experimental "
                   "agreement is NOT claimed.",
    }

    # honest explanations for every failing gate (numbers injected, no prose-only)
    def _fmt(v, nd=5):
        try:
            return f"{float(v):.{nd}g}"
        except (TypeError, ValueError):
            return str(v)

    expl = {}
    g4b = [x for x in gates if x["id"].startswith("G4b-DRAG-REFINEMENT")]
    for gg in gates:
        if gg["pass"]:
            continue
        gid = gg["id"]
        if gid == "G4-SPURIOUS-DRAG":
            tail = (f"; the subcritical family G4b gate shows C_D = {_fmt(g4b[0]['values'])} "
                    f"under refinement" if g4b else "")
            expl[gid] = (f"inviscid C_D(alpha=0, Ma=0.3) on the 160x64 grid is "
                         f"{_fmt(gg['measured'])} (threshold 0.005): the whole drag is "
                         f"numerical entropy drag, i.e. a mesh property, not physics"
                         f"{tail}.")
        elif gid.endswith("-cd"):
            vals = gg.get("values")
            change = None
            if isinstance(vals, list) and len(vals) > 2 and abs(vals[2]) > 1e-30:
                change = abs(vals[2] - vals[1]) / abs(vals[2])
            expl[gid] = (f"C_D on the three grids is {_fmt(vals)}; the finest pair "
                         f"differs by {_fmt(change, 3)} and the sequence is "
                         f"{'monotone' if gg.get('monotone') else 'non-monotone'}. "
                         f"C_D is a small difference of large pressure integrals, so it "
                         f"converges much more slowly than C_L; at Ma=0.85 the captured "
                         f"shock adds a physical wave-drag floor plus shock-capturing "
                         f"noise.  Reported as FAIL instead of relaxing the threshold.")
        else:
            expl[gid] = ("measured value exceeds the threshold "
                         f"({_fmt(gg['measured'])} > {_fmt(gg['threshold'])} "
                         f"{gg.get('unit', '')}); reported as FAIL.")
    result["failed_gate_explanations"] = expl

    n_pass = sum(1 for g in gates if g["pass"])
    result["gates"] = gates
    result["gate_summary"] = {
        "n_gates": len(gates), "n_pass": n_pass,
        "all_pass": n_pass == len(gates),
        "failed": [g["id"] for g in gates if not g["pass"]],
    }
    result["artifacts"] = {
        "result_json": "artifacts/compressible_lbm/result.json",
        "gates_json": "artifacts/compressible_lbm/gates_verified.json",
        "figures_dir": "artifacts/compressible_lbm/figures",
        "raw_dir": "artifacts/compressible_lbm/raw",
    }
    _save_json(os.path.join(ART, "result.json"), result)
    _save_json(os.path.join(ART, "gates_verified.json"), {
        "node": "IN50-W4-N04", "repo": "TensorLBM",
        "all_pass": result["gate_summary"]["all_pass"],
        "n_gates": len(gates), "n_pass": n_pass, "gates": gates,
        "generated_utc": result["generated_utc"],
    })
    make_figures(result, raw)
    # figures are produced after the first write, so persist the final record
    _save_json(os.path.join(ART, "result.json"), result)
    return result


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(result, raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorlbm.compressible import convergence as cv

    made = []
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

    # fig1: CL vs alpha
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for ma, L in sorted(result["lift"].items(), key=lambda kv: float(kv[0])):
        ax[0].plot(L["alpha_deg"], L["cl"], "o-", label=f"Ma={ma}")
        ax[1].plot(L["alpha_deg"], L["cd"], "o-", label=f"Ma={ma}")
    ax[0].set_xlabel(r"$\alpha$ (deg)"); ax[0].set_ylabel(r"$C_L$")
    ax[0].set_title("NACA0012 inviscid Euler: lift"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\alpha$ (deg)"); ax[1].set_ylabel(r"$C_D$")
    ax[1].set_title("drag (inviscid: wave + numerical)"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "fig1_cl_cd_vs_alpha.png"); fig.savefig(p); plt.close(fig)
    made.append(p)

    # fig2: lift slope vs Ma, Prandtl-Glauert
    if result["lift"]:
        mas = np.array(sorted(float(k) for k in result["lift"]))
        slopes = np.array([result["lift"][f"{m:g}"]["lift_slope"]["dcl_dalpha_per_rad"]
                           for m in mas])
        pg = np.array([result["lift"][f"{m:g}"]["pg_lift_slope"] for m in mas])
        fig, ax = plt.subplots(figsize=(5.2, 4))
        ax.plot(mas, slopes, "o-", label="solver (alpha<=6 deg fit)")
        ax.plot(mas, pg, "s--", label=r"Prandtl-Glauert $2\pi/\beta$")
        ax.axhline(2 * math.pi, color="k", lw=0.8, ls=":", label=r"thin airfoil $2\pi$")
        ax.set_xlabel("Ma"); ax.set_ylabel(r"$dC_L/d\alpha$ (1/rad)")
        ax.set_title("lift slope vs Mach"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); p = os.path.join(FIG, "fig2_lift_slope_vs_ma.png"); fig.savefig(p); plt.close(fig)
        made.append(p)

    # fig3: Cp distributions at Ma=0.85 (all three grids) with shock markers
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for gi in (1, 2, 3):
        rec = _get(raw, _key("grid", ma=TRANSONIC_MACH, a=0.0, g=gi))
        if rec is None:
            continue
        up = rec["surface"]["upper"]
        ax.plot(up["x"], up["cp"], "-", lw=1.2,
                label=f"grid {rec['grid']['n_points']}x{rec['grid']['nj']}")
        ss = cv.shock_strength(np.array(up["x"]), np.array(up["cp"]), TRANSONIC_MACH)
        if ss.get("x_shock") is not None:
            ax.plot([ss["x_shock"]], [ss["post_plateau_cp"]], "kv", ms=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.invert_yaxis(); ax.set_xlabel("x/c"); ax.set_ylabel(r"$C_p$ (upper surface)")
    ax.set_title("Ma=0.85, alpha=0: terminal shock and mesh family (triangles = shock)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "fig3_cp_transonic_grids.png"); fig.savefig(p); plt.close(fig)
    made.append(p)

    # fig4: grid convergence
    keys = list(result["convergence"].keys())
    if keys:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for tag in keys:
            fam = result["convergence"][tag]["family"]
            h = [f["h"] for f in fam]
            lab = tag
            axes[0].plot(h, [f["forces"]["cl"] for f in fam], "o-", label=lab)
            axes[1].plot(h, [f["isentropic"]["max_rel_dev_subsonic"] for f in fam], "o-", label=lab)
            xs = [f["shock"].get("x_shock") for f in fam]
            if all(v is not None for v in xs):
                axes[2].plot(h, xs, "o-", label=lab)
        for a, t, yl in ((axes[0], r"$C_L$", "CL"), (axes[1], "max isentropic deviation",
                                                    "rel"), (axes[2], "shock x/c", "x/c")):
            a.set_xlabel("cell size h (c)"); a.set_ylabel(yl); a.grid(alpha=.3); a.legend()
            a.set_title("mesh convergence")
        fig.tight_layout(); p = os.path.join(FIG, "fig4_grid_convergence.png"); fig.savefig(p); plt.close(fig)
        made.append(p)

    # fig5: N-wave wall signature
    fig, ax = plt.subplots(figsize=(6, 4))
    for theta in (2.0, 5.0, 10.0):
        rec = _get(raw, _key("boom", ma=2.0, th=theta))
        if rec is None:
            continue
        ax.plot(rec["surface"]["x"], rec["surface"]["p_ratio"], "o-", ms=3,
                label=f"theta={theta:g} deg")
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("x/c"); ax.set_ylabel(r"$p/p_\infty$")
    ax.set_title("N-wave source signature (Ma=2 double wedge)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "fig5_nwave_signature.png"); fig.savefig(p); plt.close(fig)
    made.append(p)

    # fig6: boom overpressure vs Ma, Ackeret and exact
    bo10 = [(float(k.split("_")[0][2:]), v) for k, v in result["boom"].items() if "_theta10" in k]
    if bo10:
        bo10.sort()
        mas = np.array([b[0] for b in bo10])
        meas = np.array([b[1]["measured_front_dp"] for b in bo10])
        ack = np.array([b[1]["ackeret_dp"] for b in bo10])
        ex = np.array([b[1]["exact_oblique_dp"] for b in bo10])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(mas, meas, "o-", label="solver")
        ax[0].plot(mas, ex, "s--", label="exact oblique shock")
        ax[0].plot(mas, ack, "^:", label="Ackeret (linear)")
        ax[0].set_xlabel("Ma"); ax[0].set_ylabel(r"$\Delta p/p_\infty$")
        ax[0].set_title(r"wedge $\theta=10^\circ$: overpressure"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(mas, 100 * (meas - ack) / ack, "o-", label="solver vs Ackeret")
        ax[1].plot(mas, 100 * (ex - ack) / ack, "s--", label="exact vs Ackeret (theory error)")
        ax[1].set_xlabel("Ma"); ax[1].set_ylabel("relative difference (%)")
        ax[1].set_title("the gap is linear-theory truncation, not solver error")
        ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout(); p = os.path.join(FIG, "fig6_boom_vs_ackeret.png"); fig.savefig(p); plt.close(fig)
        made.append(p)

    # fig7: Mach field at Ma=0.85 (finest grid used in the convergence family)
    fpath = os.path.join(ART, "raw", _key("grid", ma=TRANSONIC_MACH, a=0.0,
                                          g=len(GRID_FAMILY)) + "_field.npz")
    if os.path.exists(fpath):
        d = np.load(fpath)
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        xx = d["xc"]; yy = d["yc"]
        # close the O-grid for display (periodic in i)
        X = np.vstack([xx, xx[:1]]); Y = np.vstack([yy, yy[:1]])
        M = np.vstack([d["mach"], d["mach"][:1]])
        cs = ax.pcolormesh(X, Y, M, shading="auto", cmap="turbo")
        fig.colorbar(cs, ax=ax, label="local Mach")
        ax.set_aspect("equal"); ax.set_title("Ma=0.85, alpha=0 (finest grid): supersonic pocket + terminal shock")
        fig.tight_layout(); p = os.path.join(FIG, "fig7_mach_field_transonic.png"); fig.savefig(p); plt.close(fig)
        made.append(p)

    result["figures"] = [os.path.relpath(p, ROOT) for p in made]
    return made


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all",
                    choices=["all", "lift", "transonic", "grid", "boom"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=16000)
    ap.add_argument("--iters-grid", type=int, default=20000)
    ap.add_argument("--iters-boom", type=int, default=7000)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated case keys to run")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    if args.assemble:
        res = assemble()
        print(json.dumps(res["gate_summary"], indent=1))
        return 0
    run_suite(args.suite, args.workers, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
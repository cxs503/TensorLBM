#!/usr/bin/env python3
"""IN41-N05 compressible-shock family verification (Mach 2-10): runner.

A *new* verification surface on top of the ``tensorlbm.compressible_shock``
module (built by W18-N07).  The acceptance criteria here are stricter than the
W18-N07 node and use different headline quantities:

``normal_shock``  Mach 2/3/5/8/10 x 3 grids (n = 400/800/1600).
    * core 15 cases: variant ``jump`` (exact Rankine-Hugoniot jump prescribed,
      shock-preservation test) -- the post-shock rho / p / T plateau is
      measured from the field and compared with the exact gamma=1.4 relations;
    * 15 extra cases: variant ``riemann`` (driver state inverted from the
      shock-tube relations, so the shock Mach number is a solver *output*,
      non-circular).
    Criterion: post-shock rho / p / T ratios within **1%** of the exact
    relations for every core case.

``dmr``  double Mach reflection (Woodward-Colella problem 2), Mach-10 oblique
    shock at 30 deg to the wall, t = 0.2, grids 480x120 and 960x240.
    Criterion: the Mach-stem (triple-point) height on the 480x120 grid within
    **5%** of the grid-converged (Richardson, order 1) value; the density-field
    topology (three-shock structure, slip line, roll-up) is compared with the
    Woodward-Colella reference figure.

``ffs``  forward-facing step (Woodward-Colella problem 1), Mach 3, step height
    h = H/5 = 0.2, run to the steady state, grids 300x100 and 600x200.
    Criterion: the bow-shock detachment angle at the corner on the 300x100
    grid within **2 deg** of the grid-converged value (and reported against the
    90 deg normal-shock limit).

Usage
-----
    python run.py --suite all
    python run.py --suite dmr --case 480x120
    python run.py --suite ffs --case 600x200 --reuse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm.compressible_shock import analysis, cases  # noqa: E402
from tensorlbm.compressible_shock.flux import GAMMA  # noqa: E402
from tensorlbm.compressible_shock.rankine_hugoniot import (  # noqa: E402
    normal_shock,
    sound_speed,
)

MACH_LIST = (2.0, 3.0, 5.0, 8.0, 10.0)
NS_GRIDS = (400, 800, 1600)
DMR_GRIDS = ((480, 120), (960, 240))
FFS_GRIDS = ((300, 100), (600, 200))
NS_TOL = 0.01   # 1 %
DMR_TOL = 0.05  # 5 %
FFS_TOL_DEG = 2.0
_REUSE = False


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _dump(outdir: str, name: str, obj):
    with open(os.path.join(outdir, name), "w") as fh:
        json.dump(_jsonable(obj), fh, indent=1)


# --------------------------------------------------------------------- 1-D scan
def _normal_shock_one(variant: str, mach: float, n: int, outdir: str) -> dict:
    solver, meta = cases.normal_shock_case(mach, n=n, variant=variant)
    ref = meta["reference"]
    rho1, p1 = meta["pre_state"]["rho"], meta["pre_state"]["p"]
    t0 = time.time()
    # front tracking for the measured shock speed (three snapshots)
    times, fronts = [], []
    t_end = meta["t_end"]
    for frac in (0.4, 0.7, 1.0):
        solver.advance(frac * t_end)
        rho = solver.primitives()[0]
        thr = 0.5 * (rho1 + ref["rho2"])
        times.append(solver.t)
        fronts.append(analysis.front_x(rho, solver.xc, thr, x_min=meta["x_diaphragm"] * 0.5))
    fit = analysis.fit_speed(times, fronts)
    rho = solver.primitives()[0]
    x_front = fronts[-1]
    lo = x_front - 20 * solver.dx
    hi = x_front - 3 * solver.dx
    st = analysis.sample_state(solver.U, solver.xc, lo, hi)
    jump = analysis.jump_ratios(
        {"rho": rho1, "p": p1, "T": p1 / rho1, "u": 0.0, "a": sound_speed(rho1, p1)}, st
    )
    err = {
        "rho_ratio": analysis.relative_error(jump["rho_ratio"], ref["rho_ratio"]),
        "p_ratio": analysis.relative_error(jump["p_ratio"], ref["p_ratio"]),
        "T_ratio": analysis.relative_error(jump["T_ratio"], ref["T_ratio"]),
        "shock_speed": analysis.relative_error(fit["speed"], ref["W"]),
    }
    rec = {
        "variant": variant,
        "mach": mach,
        "n": n,
        "dx": meta["dx"],
        "t_end": t_end,
        "measured": {
            "shock_speed": fit["speed"],
            "fit_r2": fit["r2"],
            "shock_x_final": x_front,
            "rho": st["rho"],
            "p": st["p"],
            "T": st["T"],
            "u": st["u"],
            "rho_ratio": jump["rho_ratio"],
            "p_ratio": jump["p_ratio"],
            "T_ratio": jump["T_ratio"],
            "M2": jump["M2"],
        },
        "exact": {
            "shock_speed": ref["W"],
            "rho_ratio": ref["rho_ratio"],
            "p_ratio": ref["p_ratio"],
            "T_ratio": ref["T_ratio"],
            "M2": ref["M2_lab"],
        },
        "error": err,
        "max_error": max(err.values()),
        "pass_1pct": bool(max(err.values()) < NS_TOL),
        "wall_time_s": time.time() - t0,
    }
    print(
        f"[ns:{variant}] M={mach:>4} n={n:>5} "
        f"err rho={err['rho_ratio'] * 100:5.2f}% p={err['p_ratio'] * 100:5.2f}% "
        f"T={err['T_ratio'] * 100:5.2f}% W={err['shock_speed'] * 100:5.2f}% "
        f"({'PASS' if rec['pass_1pct'] else 'FAIL'} 1%)",
        flush=True,
    )
    return rec


def run_normal_shock(outdir: str, machs=MACH_LIST, grids=NS_GRIDS,
                     variants=("jump", "riemann"), only=None):
    records = []
    for variant in variants:
        for mach in machs:
            for n in grids:
                if only is not None and not (variant == only[0] and mach == only[1] and n == only[2]):
                    continue
                records.append(_normal_shock_one(variant, mach, n, outdir))
    core = [r for r in records if r["variant"] == "jump"]
    summary = {
        "suite": "normal_shock",
        "machs": list(machs),
        "grids": list(grids),
        "gas": {"gamma": GAMMA},
        "criterion": "post-shock rho/p/T ratios within 1% of exact R-H (core = jump variant)",
        "records": records,
        "n_core": len(core),
        "n_cases": len(records),
        "max_error_core": max((r["max_error"] for r in core), default=float("nan")),
        "pass_all_core_1pct": all(r["pass_1pct"] for r in core) if core else None,
        "max_error_overall": max((r["max_error"] for r in records), default=float("nan")),
    }
    _dump(outdir, "normal_shock.json", summary)
    return summary


# --------------------------------------------------------------------- 2-D DMR
def _dmr_one(nx: int, ny: int, outdir: str, t_end: float = 0.2) -> dict:
    solver, meta = cases.double_mach_reflection_case(nx=nx, ny=ny, t_end=t_end)
    thr = 0.5 * (meta["pre_shock"]["rho"] + meta["post_shock"]["rho"])
    field_path = os.path.join(outdir, f"dmr_field_{nx}x{ny}.npz")
    t0 = time.time()
    if _REUSE and os.path.exists(field_path):
        d = np.load(field_path)
        rho, u, v, p = d["rho"], d["u"], d["v"], d["p"]
        solver.t = float(d["t"])
        solver.U = np.stack(
            [rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)],
            axis=-1,
        )
        wall = 0.0
    else:
        solver.advance(t_end)
        wall = time.time() - t0
        rho, u, v, p = solver.primitives()
        np.savez_compressed(
            field_path, rho=rho, u=u, v=v, p=p, xc=solver.xc, yc=solver.yc, t=solver.t
        )
    prof = analysis.front_profile(rho, solver.xc, solver.yc, threshold=thr, x_min=meta["x_wall"])
    stem = analysis.mach_stem_height(prof, solver.yc)
    # incident branch: linear fit of x(y) in the upper part of the channel
    upper = np.isfinite(prof) & (solver.yc > 0.6 * solver.ly)
    b, a = np.polyfit(solver.yc[upper], prof[upper], 1)
    angle_meas = float(np.degrees(np.arctan2(1.0, b)))
    x_foot_exact = meta["x_wall"] + meta["foot_speed_along_wall"] * solver.t
    # post-shock plateau: between the shock and the reflected-shock system
    j = int(0.75 * ny)
    i_lo = int((prof[j] - 0.50) / solver.dx)
    i_hi = int((prof[j] - 0.20) / solver.dx)
    st = analysis.sample_state(solver.U[j], solver.xc, solver.xc[i_lo], solver.xc[i_hi])
    ref = meta["post_shock"]
    err_plateau = {
        "rho": analysis.relative_error(st["rho"], ref["rho"]),
        "p": analysis.relative_error(st["p"], ref["p"]),
        "u": analysis.relative_error(st["u"], ref["u"]),
        "v": analysis.relative_error(st["v"], ref["v"]),
    }
    # slip line: strongest density jump at (numerically) constant pressure
    drho = np.abs(np.diff(rho, axis=1))
    dp = np.abs(np.diff(p, axis=1))
    Y = solver.yc[:, None]
    X = solver.xc[None, :-1]
    region = (X > 1.2 * meta["x_wall"]) & (Y > 0.15) & (Y < 0.65)
    p_ref = np.maximum(p[:, :-1], 1e-12)
    contact = region & (dp / p_ref < 0.01) & (drho / rho[:, :-1] > 0.02)
    slip = {"found": False}
    if np.any(contact):
        score = np.where(contact, drho / rho[:, :-1], -1.0)
        jj, ii = np.unravel_index(int(np.argmax(score)), score.shape)
        slip = {
            "found": True,
            "x": float(solver.xc[ii]),
            "y": float(solver.yc[jj]),
            "drho_rel": float(abs(rho[jj, ii + 1] - rho[jj, ii]) / rho[jj, ii]),
            "dp_rel": float(abs(p[jj, ii + 1] - p[jj, ii]) / p[jj, ii]),
            "du_abs": float(abs(u[jj, ii + 1] - u[jj, ii])),
        }
    # roll-up evidence: non-monotonic rho along the wall behind the stem
    foot = float(prof[0])
    seg = rho[0][(solver.xc > 1.5 * meta["x_wall"]) & (solver.xc < foot - 0.02)]
    d = np.diff(seg)
    sign = np.sign(d)
    reversals = 0
    for k in range(1, len(sign)):
        if sign[k] != sign[k - 1] and sign[k] != 0 and sign[k - 1] != 0:
            reversals += 1
    rec = {
        "nx": nx,
        "ny": ny,
        "dx": solver.dx,
        "dy": solver.dy,
        "t_end": solver.t,
        "wall_time_s": wall,
        "n_admissibility_fixes": solver.n_admissibility_fixes,
        "stem_height": stem["stem_height"],
        "triple_point_y": stem["triple_point_y"],
        "incident_angle_deg": angle_meas,
        "incident_angle_exact_deg": meta["incident_angle_to_x_deg"],
        "incident_angle_err_pct": analysis.relative_error(angle_meas, meta["incident_angle_to_x_deg"]) * 100,
        "foot_x_measured": float(prof[0]),
        "foot_x_exact": float(x_foot_exact),
        "foot_err_pct": analysis.relative_error(prof[0], x_foot_exact) * 100,
        "plateau_err": err_plateau,
        "slip_line": slip,
        "rollup_reversals": int(reversals),
        "fields": {"rho_min": float(rho.min()), "rho_max": float(rho.max())},
    }
    print(
        f"[dmr] {nx}x{ny} stem_h={stem['stem_height']:.4f} "
        f"angle={angle_meas:.2f}deg slip={'yes' if slip['found'] else 'no'} "
        f"reversals={reversals} ({wall:.0f}s)",
        flush=True,
    )
    return rec


def run_dmr(outdir: str, grids=DMR_GRIDS, t_end: float = 0.2, only=None):
    if only is not None:
        grids = [only]
    records = []
    for nx, ny in grids:
        if only is not None and (nx, ny) != only:
            continue
        records.append(_dmr_one(nx, ny, outdir, t_end))
    summary = {"suite": "dmr", "t_end": t_end, "records": records}
    if len(records) >= 2:
        c, f = records[0]["stem_height"], records[1]["stem_height"]
        conv = analysis.gci(c, f, ratio=2.0, order=1.0)
        summary["stem_convergence"] = conv
        for r in records:
            r["stem_height_err_vs_converged"] = analysis.relative_error(
                r["stem_height"], conv["richardson_estimate"]
            )
        summary["pass_stem_5pct"] = bool(
            records[0]["stem_height_err_vs_converged"] < DMR_TOL
        )
    _dump(outdir, "dmr.json", summary)
    return summary


# --------------------------------------------------------------------- 2-D step
def _ffs_one(nx: int, ny: int, outdir: str, t_end: float = 4.0) -> dict:
    solver, meta = cases.forward_facing_step_case(nx=nx, ny=ny, t_end=t_end)
    jump = normal_shock(meta["mach"])
    field_path = os.path.join(outdir, f"ffs_field_{nx}x{ny}.npz")
    t0 = time.time()
    if _REUSE and os.path.exists(field_path):
        d = np.load(field_path)
        rho, u, v, p = d["rho"], d["u"], d["v"], d["p"]
        solver.t = float(d["t"])
        solver.U = np.stack(
            [rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)],
            axis=-1,
        )
        wall = 0.0
    else:
        solver.advance(t_end)
        wall = time.time() - t0
        rho, u, v, p = solver.primitives()
        np.savez_compressed(
            field_path, rho=rho, u=u, v=v, p=p, xc=solver.xc, yc=solver.yc,
            t=solver.t, solid=solver.solid,
        )
    rho_in, p_in = meta["inflow"]["rho"], meta["inflow"]["p"]
    thr = 0.5 * (rho_in + rho_in * jump["rho_ratio"])
    # stand-off: row through the middle of the step face
    j = int(0.5 * meta["step_h"] / solver.dy)
    x_hi = meta["step_x"] - 3.0 * solver.dx
    stand = analysis.standoff_distance(rho[j], solver.xc, x_hi, thr, search_lo=0.2)
    stand["standoff"] = float(meta["step_x"] - stand["front_x"])
    # detachment angle at the corner (new IN41-N05 metric)
    det = analysis.shock_detachment_angle(
        rho, solver.xc, solver.yc, meta["step_x"], meta["step_h"], thr
    )
    # near-normal jump across the bow shock on the mid-face row
    i_f = int(np.argmin(np.abs(solver.xc - stand["front_x"])))
    n_s = max(2, int(round(3.0e-2 / solver.dx)))
    up = analysis.sample_state(
        solver.U[j], solver.xc, solver.xc[max(0, i_f - 2 * n_s)], solver.xc[max(0, i_f - n_s)]
    )
    dn = analysis.sample_state(
        solver.U[j], solver.xc, solver.xc[i_f + n_s],
        solver.xc[min(solver.nx - 1, i_f + 2 * n_s)],
    )
    jr = analysis.jump_ratios(up, dn)
    err = {
        "rho_ratio": analysis.relative_error(jr["rho_ratio"], jump["rho_ratio"]),
        "p_ratio": analysis.relative_error(jr["p_ratio"], jump["p_ratio"]),
        "T_ratio": analysis.relative_error(jr["T_ratio"], jump["T_ratio"]),
    }
    rec = {
        "nx": nx,
        "ny": ny,
        "dx": solver.dx,
        "dy": solver.dy,
        "t_end": solver.t,
        "wall_time_s": wall,
        "n_admissibility_fixes": solver.n_admissibility_fixes,
        "standoff": stand,
        "standoff_over_step_h": stand["standoff"] / meta["step_h"],
        "detachment_angle_deg": det["angle_deg"],
        "detachment_slope": det["slope"],
        "detachment_n_rows": det["n_rows"],
        "jump_measured": jr,
        "jump_exact": {
            "rho_ratio": jump["rho_ratio"],
            "p_ratio": jump["p_ratio"],
            "T_ratio": jump["T_ratio"],
        },
        "jump_error": err,
        "max_jump_error": max(err.values()),
        "fields": {"rho_min": float(rho.min()), "rho_max": float(rho.max())},
    }
    print(
        f"[ffs] {nx}x{ny} standoff={stand['standoff']:.4f} "
        f"detach={det['angle_deg']:.2f}deg jump_err={max(err.values()) * 100:.2f}% "
        f"({wall:.0f}s)",
        flush=True,
    )
    return rec


def run_ffs(outdir: str, grids=FFS_GRIDS, t_end: float = 4.0, only=None):
    if only is not None:
        grids = [only]
    records = []
    for nx, ny in grids:
        if only is not None and (nx, ny) != only:
            continue
        records.append(_ffs_one(nx, ny, outdir, t_end))
    summary = {"suite": "forward_facing_step", "mach": 3.0, "t_end": t_end, "records": records}
    if len(records) >= 2:
        a0, a1 = records[0]["detachment_angle_deg"], records[1]["detachment_angle_deg"]
        summary["detachment_angle_grid_diff_deg"] = abs(a1 - a0)
        summary["pass_detach_2deg"] = bool(abs(a1 - a0) < FFS_TOL_DEG)
        # deviation of the converged (finest) angle from the normal-shock limit
        summary["finest_detach_dev_from_90_deg"] = abs(a1 - 90.0)
    _dump(outdir, "ffs.json", summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["normal_shock", "dmr", "ffs", "all"])
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "out"),
    )
    ap.add_argument("--case", default=None, help="e.g. 480x120 or jump,10.0,800")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args(argv)
    global _REUSE
    _REUSE = bool(args.reuse)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    only = None
    if args.case:
        if "x" in args.case and "," not in args.case:
            a, b = args.case.split("x")
            only = (int(a), int(b))
        else:
            parts = args.case.split(",")
            only = (parts[0], float(parts[1]), int(parts[2]))
    if args.suite in ("normal_shock", "all"):
        run_normal_shock(outdir, only=only if args.suite == "normal_shock" else None)
    if args.suite in ("dmr", "all"):
        run_dmr(outdir, only=only)
    if args.suite in ("ffs", "all"):
        run_ffs(outdir, only=only)
    print("done ->", outdir, flush=True)


if __name__ == "__main__":
    main()

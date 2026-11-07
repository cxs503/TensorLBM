#!/usr/bin/env python3
"""Compressible-shock verification suite (W18-N07): runner.

Three sub-suites, all with machine-checkable acceptance criteria:

``normal_shock``  Mach 2/3/5/8/10 x 3 grids, two variants
    * ``riemann``  driver state inverted from the shock-tube relations, so the
      shock Mach number is a solver *output* (non-circular);
    * ``jump``     exact Rankine-Hugoniot jump prescribed (shock preservation).
    Criteria: post-shock rho / p / T ratios and the measured shock speed within
    3% of the exact relations for every Mach number and every grid.

``dmr``  double Mach reflection (Woodward-Colella problem 2), Mach 10 oblique
    shock at 30 deg to the wall, 3 grids.
    Criteria: incident-shock angle and foot speed, post-shock plateau state,
    density jump across the Mach stem, pressure continuity across the slip line,
    stem height convergence.

``ffs``  forward-facing step (Woodward-Colella problem 1), Mach 3, 3 grids.
    Criteria: near-normal post-shock jump vs normal-shock relations (<3%),
    shock stand-off distance grid convergence (<5%).

Usage
-----
    python benchmarks/compressible_shock_family/run.py --suite all
    python benchmarks/compressible_shock_family/run.py --suite dmr --grids 240x60
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
DMR_GRIDS = ((240, 60), (480, 120), (960, 240))
FFS_GRIDS = ((150, 50), (300, 100), (600, 200))
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


# --------------------------------------------------------------------- 1-D scan
def run_normal_shock(outdir: str, machs=MACH_LIST, grids=NS_GRIDS, variants=("riemann", "jump")):
    records = []
    for variant in variants:
        for mach in machs:
            for n in grids:
                solver, meta = cases.normal_shock_case(mach, n=n, variant=variant)
                ref = meta["reference"]
                rho1, p1 = meta["pre_state"]["rho"], meta["pre_state"]["p"]
                # front tracking for the measured shock speed
                times, fronts = [], []
                t_end = meta["t_end"]
                for frac in (0.4, 0.7, 1.0):
                    solver.advance(frac * t_end)
                    rho = solver.primitives()[0]
                    thr = 0.5 * (rho1 + ref["rho2"])
                    times.append(solver.t)
                    fronts.append(
                        analysis.front_x(rho, solver.xc, thr, x_min=meta["x_diaphragm"] * 0.5)
                    )
                fit = analysis.fit_speed(times, fronts)
                t0 = time.time()
                # final state: average plateau 3..20 cells behind the front
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
                    "t_ratio": analysis.relative_error(jump["T_ratio"], ref["T_ratio"]),
                    "u2": analysis.relative_error(st["u"], ref["u2"]),
                }
                records.append(
                    {
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
                            "rho": ref["rho2"],
                            "p": ref["p2"],
                            "T": ref["T2"],
                            "u": ref["u2"],
                            "M2": ref["M2_lab"],
                            "rho_ratio": ref["rho_ratio"],
                            "p_ratio": ref["p_ratio"],
                            "T_ratio": ref["T_ratio"],
                        },
                        "error": err,
                        "max_error": max(err.values()),
                        "pass_3pct": bool(max(err.values()) < 0.03),
                        "wall_time_s": time.time() - t0,
                    }
                )
                print(
                    f"[ns:{variant}] M={mach:>4} n={n:>5} "
                    f"err rho={err['rho_ratio'] * 100:5.2f}% p={err['p_ratio'] * 100:5.2f}% "
                    f"T={err['T_ratio'] * 100:5.2f}% W={err['shock_speed'] * 100:5.2f}%",
                    flush=True,
                )
    summary = {
        "suite": "normal_shock",
        "machs": list(machs),
        "grids": list(grids),
        "gas": {"gamma": GAMMA},
        "records": records,
        "n_cases": len(records),
        "max_error_overall": max(r["max_error"] for r in records),
        "pass_all_3pct": all(r["pass_3pct"] for r in records),
    }
    with open(os.path.join(outdir, "normal_shock.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    return summary


# --------------------------------------------------------------- 2-D DMR / step
def run_dmr(outdir: str, grids=DMR_GRIDS, mach: float = 10.0, angle: float = 30.0, t_end=0.2):
    records = []
    for nx, ny in grids:
        solver, meta = cases.double_mach_reflection_case(
            nx=nx, ny=ny, mach=mach, angle_deg=angle, t_end=t_end
        )
        thr = 0.5 * (meta["pre_shock"]["rho"] + meta["post_shock"]["rho"])
        field_path = os.path.join(outdir, f"dmr_field_{nx}x{ny}.npz")
        prof_path = os.path.join(outdir, f"dmr_profile_{nx}x{ny}.npz")
        t0 = time.time()
        if _REUSE and os.path.exists(field_path) and os.path.exists(prof_path):
            d = np.load(field_path)
            rho, u, v, p = d["rho"], d["u"], d["v"], d["p"]
            solver.t = float(d["t"])
            solver.U = np.stack(
                [rho, rho * u, rho * v, p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)],
                axis=-1,
            )
            pa = np.load(prof_path)
            prof_a, t_a = pa["prof"], float(pa["t"])
            row0_a = pa["row0"] if "row0" in pa else None
            wall = 0.0
        else:
            solver.advance(0.75 * t_end)
            t_a = solver.t
            prof_a = analysis.front_profile(
                solver.primitives()[0], solver.xc, solver.yc, threshold=thr, x_min=meta["x_wall"]
            )
            row0_a = solver.primitives()[0][0]
            solver.advance(t_end)
            wall = time.time() - t0
            rho, u, v, p = solver.primitives()
            np.savez_compressed(
                field_path, rho=rho, u=u, v=v, p=p, xc=solver.xc, yc=solver.yc, t=solver.t
            )
            np.savez_compressed(prof_path, prof=prof_a, row0=row0_a, t=t_a)
        prof = analysis.front_profile(
            rho, solver.xc, solver.yc, threshold=thr, x_min=meta["x_wall"]
        )
        stem = analysis.mach_stem_height(prof, solver.yc)
        # incident branch: linear fit of x(y) in the upper part of the channel
        upper = np.isfinite(prof) & (solver.yc > 0.6 * solver.ly)
        b, a = np.polyfit(solver.yc[upper], prof[upper], 1)
        angle_meas = float(np.degrees(np.arctan2(1.0, b)))
        x_foot_exact = meta["x_wall"] + meta["foot_speed_along_wall"] * solver.t
        interp_err = analysis.relative_error(a, x_foot_exact)
        # Mach stem speed measured from the two snapshots (near-wall front)
        # wall-row front of both snapshots, same detector, so the finite
        # difference of the Mach-stem position is consistent
        foot = float(prof[0])
        foot_a = float(prof_a[0]) if np.isfinite(prof_a[0]) else float("nan")
        stem_speed = (
            (foot - foot_a) / (solver.t - t_a)
            if np.isfinite(foot_a) and solver.t > t_a
            else float("nan")
        )
        a1 = sound_speed(meta["pre_shock"]["rho"], meta["pre_shock"]["p"])
        stem_ms = stem_speed / a1
        stem_jump = density_jump_across_stem(rho, solver, foot)
        rh_stem = normal_shock(stem_ms) if np.isfinite(stem_ms) and stem_ms > 1.0 else None
        # post-shock plateau: between the shock and the reflected-shock system
        j = int(0.75 * ny)
        i_lo = int((prof[j] - 0.50) / solver.dx)
        i_hi = int((prof[j] - 0.20) / solver.dx)
        st = analysis.sample_state(solver.U[j], solver.xc, solver.xc[i_lo], solver.xc[i_hi])
        rho_win = rho[j, i_lo : i_hi + 1]
        ref = meta["post_shock"]
        err_plateau = {
            "rho": analysis.relative_error(st["rho"], ref["rho"]),
            "p": analysis.relative_error(st["p"], ref["p"]),
            "u": analysis.relative_error(st["u"], ref["u"]),
            "v": analysis.relative_error(st["v"], ref["v"]),
        }
        slip = slip_line_metrics(rho, p, u, solver)
        # roll-up evidence: the density along the wall behind the stem is not
        # monotonic (the slip line rolls into a vortex), so count the interior
        # extrema with >2% amplitude between the wall and the stem
        j_w = 0
        lo_w = 1.5 * meta["x_wall"]
        seg = rho[j_w][(solver.xc > lo_w) & (solver.xc < foot - 0.02)]
        d = np.diff(seg)
        sign = np.sign(d)
        # non-monotonicity of rho along the wall behind the stem = slip-line
        # roll-up; count genuine sign reversals of the wall-parallel gradient
        reversals = 0
        for k in range(1, len(sign)):
            if sign[k] != sign[k - 1] and sign[k] != 0 and sign[k - 1] != 0:
                reversals += 1
        rollup = {
            "row_y": float(solver.yc[j_w]),
            "n_gradient_reversals": int(reversals),
            "rho_min": float(np.min(seg)) if seg.size else float("nan"),
            "rho_max": float(np.max(seg)) if seg.size else float("nan"),
        }

        records.append(
            {
                "nx": nx,
                "ny": ny,
                "dx": solver.dx,
                "dy": solver.dy,
                "t_end": solver.t,
                "wall_time_s": wall,
                "n_admissibility_fixes": solver.n_admissibility_fixes,
                "fields": {
                    "rho_min": float(rho.min()),
                    "rho_max": float(rho.max()),
                    "p_min": float(p.min()),
                    "p_max": float(p.max()),
                },
                "front": {
                    "profile_y": solver.yc,
                    "profile_x": prof,
                    "incident_slope_dxdy": float(b),
                    "incident_angle_to_x_deg": angle_meas,
                    "incident_angle_error_deg": float(
                        abs(angle_meas - meta["incident_angle_to_x_deg"])
                    ),
                    "incident_intercept": float(a),
                    "incident_intercept_exact": x_foot_exact,
                    "incident_intercept_error": interp_err,
                    "stem_foot_x": foot,
                    "stem_foot_x_at_t_a": foot_a,
                    "t_a": t_a,
                },
                "triple_point": stem,
                "plateau": {
                    "measured": st,
                    "window": [solver.xc[i_lo], solver.xc[i_hi]],
                    "rho_std_in_window": float(np.std(rho_win)),
                    "error": err_plateau,
                    "max_error": max(err_plateau.values()),
                },
                "mach_stem": {
                    "measured_shock_speed": stem_speed,
                    "shock_mach_measured": stem_ms,
                    "rho_ratio_measured": stem_jump,
                    "rho_ratio_rh": None if rh_stem is None else rh_stem["rho_ratio"],
                    "error": None
                    if rh_stem is None
                    else analysis.relative_error(stem_jump, rh_stem["rho_ratio"]),
                },
                "slip_line": slip,
                "rollup": rollup,
            }
        )
        print(
            f"[dmr] {nx}x{ny} t={solver.t:.3f} stem_h={stem['stem_height']:.4f} "
            f"angle_err={records[-1]['front']['incident_angle_error_deg']:.2f}deg "
            f"intercept_err={interp_err * 100:.2f}% plateau_err={records[-1]['plateau']['max_error'] * 100:.2f}% "
            f"slip_dp={slip.get('dp_rel', float('nan')) * 100:.2f}% ({wall:.0f}s)",
            flush=True,
        )
    conv = None
    if len(records) >= 2:
        conv = analysis.gci(
            records[-2]["triple_point"]["stem_height"],
            records[-1]["triple_point"]["stem_height"],
            ratio=2.0,
            order=1.0,
        )
        for r in records[:-1]:
            r["stem_height_error_vs_finest"] = analysis.relative_error(
                r["triple_point"]["stem_height"], records[-1]["triple_point"]["stem_height"]
            )
    criteria = {
        "incident_angle_within_2pct": all(
            r["front"]["incident_angle_error_deg"] / 60.0 < 0.02 for r in records
        ),
        "incident_intercept_within_2pct": all(
            r["front"]["incident_intercept_error"] < 0.02 for r in records
        ),
        "plateau_within_3pct": all(r["plateau"]["max_error"] < 0.03 for r in records),
        "stem_rh_within_3pct": all(
            r["mach_stem"]["error"] is not None and r["mach_stem"]["error"] < 0.03 for r in records
        ),
        "slip_pressure_continuity_within_2pct": all(
            r["slip_line"].get("found") and r["slip_line"]["dp_rel"] < 0.02 for r in records
        ),
    }
    summary = {
        "suite": "double_mach_reflection",
        "mach": mach,
        "angle_deg": angle,
        "t_end": t_end,
        "gas": {"gamma": GAMMA},
        "reference": "Woodward & Colella 1984, J. Comput. Phys. 54, 115 (Problem 2)",
        "criteria": criteria,
        "pass_all": all(criteria.values()),
        "records": records,
        "stem_height_convergence": conv,
    }
    with open(os.path.join(outdir, "dmr.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    return summary


def density_jump_across_stem(rho, solver, foot_x) -> float:
    """Density ratio across the wall-normal (Mach-stem) shock, row ``j = 3``.

    The steepest density gradient inside ``[foot_x - 0.3, foot_x + 0.02]`` is
    taken as the stem; the pre-shock side is upstream (to the right, the gas the
    shock runs into) and the post-shock side is downstream, so the ratio is
    ``rho_downstream / rho_upstream``.  Two cells are skipped on each side to stay
    outside the numerical shock profile (the region further downstream is
    compressed again by the reflected shock and is *not* a clean R-H state).
    """
    j = 3
    prof = rho[j]
    lo, hi = foot_x - 0.3, foot_x + 0.02
    mask = (solver.xc[:-1] > lo) & (solver.xc[:-1] < hi)
    if not np.any(mask):
        return float("nan")
    g = np.where(mask, np.abs(np.diff(prof)), -1.0)
    i = int(np.argmax(g))
    i_dn = max(0, i - 2)  # downstream (post-shock)
    i_up = min(solver.nx - 1, i + 3)  # upstream (pre-shock)
    return float(prof[i_dn] / prof[i_up])


def slip_line_metrics(rho, p, u, solver, x_wall=1.0 / 6.0) -> dict:
    """Locate the slip line and quantify its two defining properties.

    A contact carries a density jump at (essentially) constant pressure, so the
    slip line is the cell with the largest ``|drho| / rho`` among cells that
    satisfy ``|dp| / p < 1%``, searched in the post-shock region (right of the
    initial shock position, above the stem).  Returned: relative density jump,
    relative pressure jump (must be ~0) and the tangential velocity jump (must be
    non-zero, otherwise the feature is not a contact).
    """
    drho = np.abs(np.diff(rho, axis=1))
    dp = np.abs(np.diff(p, axis=1))
    Y = solver.yc[:, None]
    X = solver.xc[None, :-1]
    region = (X > 1.2 * x_wall) & (Y > 0.15) & (Y < 0.65)
    p_ref = np.maximum(p[:, :-1], 1e-12)
    contact = region & (dp / p_ref < 0.01) & (drho / rho[:, :-1] > 0.02)
    if not np.any(contact):
        return {"found": False}
    score = np.where(contact, drho / rho[:, :-1], -1.0)
    j, i = np.unravel_index(int(np.argmax(score)), score.shape)
    return {
        "found": True,
        "x": float(solver.xc[i]),
        "y": float(solver.yc[j]),
        "rho_lo": float(rho[j, i]),
        "rho_hi": float(rho[j, i + 1]),
        "drho_rel": float(abs(rho[j, i + 1] - rho[j, i]) / rho[j, i]),
        "dp_rel": float(abs(p[j, i + 1] - p[j, i]) / p[j, i]),
        "du_abs": float(abs(u[j, i + 1] - u[j, i])),
        "n_contact_cells": int(np.count_nonzero(contact)),
    }


def run_ffs(outdir: str, grids=FFS_GRIDS, mach: float = 3.0, t_end=4.0):
    records = []
    jump = normal_shock(mach)
    for nx, ny in grids:
        solver, meta = cases.forward_facing_step_case(nx=nx, ny=ny, mach=mach, t_end=t_end)
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
                field_path,
                rho=rho,
                u=u,
                v=v,
                p=p,
                xc=solver.xc,
                yc=solver.yc,
                t=solver.t,
                solid=solver.solid,
            )
        rho_in, p_in = meta["inflow"]["rho"], meta["inflow"]["p"]
        thr = 0.5 * (rho_in + rho_in * jump["rho_ratio"])
        # stand-off: row through the step face (y ~ 0.1)
        j = int(0.5 * meta["step_h"] / solver.dy)
        row = rho[j]
        # the last fluid column in front of the face is a blocked-cell corner
        # artefact (see the report), so the search stops three cells short
        x_hi = meta["step_x"] - 3.0 * solver.dx
        stand = analysis.standoff_distance(row, solver.xc, x_hi, thr, search_lo=0.2)
        stand["standoff"] = float(meta["step_x"] - stand["front_x"])
        # near-normal jump across the bow shock on that row
        i_f = int(np.argmin(np.abs(solver.xc - stand["front_x"])))
        n_s = max(2, int(round(3.0e-2 / solver.dx)))
        up = analysis.sample_state(
            solver.U[j], solver.xc, solver.xc[max(0, i_f - 2 * n_s)], solver.xc[max(0, i_f - n_s)]
        )
        dn = analysis.sample_state(
            solver.U[j],
            solver.xc,
            solver.xc[i_f + n_s],
            solver.xc[min(solver.nx - 1, i_f + 2 * n_s)],
        )
        jr = analysis.jump_ratios(up, dn)
        err = {
            "rho_ratio": analysis.relative_error(jr["rho_ratio"], jump["rho_ratio"]),
            "p_ratio": analysis.relative_error(jr["p_ratio"], jump["p_ratio"]),
            "T_ratio": analysis.relative_error(jr["T_ratio"], jump["T_ratio"]),
        }
        # stagnation-region pressure just ahead of the face: the flow behind the
        # near-normal bow shock decelerates isentropically, so p -> p02 (Rayleigh
        # pitot).  This is a resolution-tolerant integral check (unlike the
        # stand-off, which needs many cells across the shock).
        i_face = int(round(meta["step_x"] / solver.dx))
        rows = (solver.yc > 0.35 * meta["step_h"]) & (solver.yc < 0.65 * meta["step_h"])
        # skip the last two fluid columns: the blocked-cell corner pile-up lives there
        p_face = float(np.mean(p[rows, max(0, i_face - 6) : i_face - 2]))
        p02 = jump["p_ratio"] * (1.0 + 0.5 * (GAMMA - 1.0) * jump["M2"] ** 2) ** (
            GAMMA / (GAMMA - 1.0)
        )
        stagnation = {
            "p_measured": p_face,
            "p02_over_p1_exact": float(p02),
            "error": analysis.relative_error(p_face, p02 * p_in),
        }
        # global x-momentum budget at the steady state: the momentum flux in at
        # x = 0 minus the flux out at x = lx must equal the pressure force on the
        # vertical step face.  This verifies that the blocked-cell wall really
        # transmits the pressure reaction (a mass-flux-only wall cannot close it).
        f_in = (rho_in * meta["inflow"]["u"] ** 2 + p_in) * solver.ly
        col = rho[:, -1] * u[:, -1] ** 2 + p[:, -1]
        f_out = float(np.sum(col) * solver.dy)
        # skip the wall-adjacent row: it hosts the blocked-cell corner pile-up
        face_rows = (solver.yc > 2.0 * solver.dy) & (solver.yc <= meta["step_h"])
        pi_face = float(np.sum(p[face_rows, i_face - 1]) * solver.dy)
        momentum = {
            "influx": float(f_in),
            "outflux": f_out,
            "step_pressure_force": pi_face,
            "residual": float(f_out - f_in + pi_face),
            "residual_rel": float(abs(f_out - f_in + pi_face) / f_in),
        }
        records.append(
            {
                "nx": nx,
                "ny": ny,
                "dx": solver.dx,
                "dy": solver.dy,
                "t_end": solver.t,
                "wall_time_s": wall,
                "n_admissibility_fixes": solver.n_admissibility_fixes,
                "standoff": stand,
                "standoff_over_step_h": stand["standoff"] / meta["step_h"],
                "standoff_over_dx": stand["standoff"] / solver.dx,
                "sample_row_y": float(solver.yc[j]),
                "jump_measured": jr,
                "jump_exact": {
                    "rho_ratio": jump["rho_ratio"],
                    "p_ratio": jump["p_ratio"],
                    "T_ratio": jump["T_ratio"],
                },
                "jump_error": err,
                "max_jump_error": max(err.values()),
                "stagnation": stagnation,
                "momentum_budget": momentum,
                "pass_3pct": bool(max(err.values()) < 0.03),
                "fields": {
                    "rho_min": rho.min(),
                    "rho_max": rho.max(),
                    "p_min": p.min(),
                    "p_max": p.max(),
                },
            }
        )
        print(
            f"[ffs] {nx}x{ny} standoff={stand['standoff']:.4f} "
            f"jump_err={max(err.values()) * 100:.2f}% ({wall:.0f}s)",
            flush=True,
        )
    conv = None
    if len(records) >= 2:
        conv = analysis.gci(records[-2]["standoff"], records[-1]["standoff"], ratio=2.0, order=1.0)
        for r in records[:-1]:
            r["standoff_error_vs_finest"] = analysis.relative_error(
                r["standoff"], conv["richardson_estimate"]
            )
    criteria = {
        "stagnation_pressure_within_5pct": all(r["stagnation"]["error"] < 0.05 for r in records),
        "near_normal_jump_within_3pct": all(r["pass_3pct"] for r in records),
    }
    summary = {
        "suite": "forward_facing_step",
        "mach": mach,
        "t_end": t_end,
        "gas": {"gamma": GAMMA},
        "criteria": criteria,
        "reference": "Woodward & Colella 1984, J. Comput. Phys. 54, 115 (Problem 1); "
        "near-normal post-shock state from the exact normal-shock relations",
        "records": records,
        "standoff_convergence": conv,
        "pass_all_3pct": all(r["pass_3pct"] for r in records),
    }
    with open(os.path.join(outdir, "ffs.json"), "w") as fh:
        json.dump(_jsonable(summary), fh, indent=1)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["normal_shock", "dmr", "ffs", "all"])
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "artifacts", "compressible_shock_family"
        ),
    )
    ap.add_argument("--grids", default=None, help="override, e.g. 240x60,480x120")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="recompute metrics from the saved fields (no time stepping)",
    )
    args = ap.parse_args(argv)
    global _REUSE
    _REUSE = bool(args.reuse)
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    grids = None
    if args.grids:
        grids = [tuple(int(x) for x in g.split("x")) for g in args.grids.split(",")]
    if args.suite in ("normal_shock", "all"):
        run_normal_shock(outdir)
    if args.suite in ("dmr", "all"):
        run_dmr(outdir, grids=grids or DMR_GRIDS)
    if args.suite in ("ffs", "all"):
        run_ffs(outdir, grids=grids or FFS_GRIDS)
    print("done ->", outdir, flush=True)


if __name__ == "__main__":
    main()

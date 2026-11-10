#!/usr/bin/env python3
"""AMR free-surface verification face: Dam break with indicator-driven refinement.

Campaign node ``P190-N02``: does adaptive refinement near the free surface keep
the interface accuracy, the mass ledger and the convergence order of a uniform
grid at the *same* finest spacing, while using far fewer cells and less time?

Routes measured against the *same* metrics
------------------------------------------
``amr-L3 / L2 / L1``
    :class:`tensorlbm.amr_free_surface.AMRFreeSurface` with 3 / 2 / 1 refinement
    levels above the same level-0 grid: finest spacing ``h/8 / h/4 / h/2``.
    Refinement is driven by ``|∇ε| > threshold`` (the free-surface form of
    ``|∇ρ|``) plus every Körner ``INTERFACE`` cell.
``uniform-fine``
    The identical Körner kernel on a single grid at the AMR's finest spacing
    (``levels = 0``), i.e. the fixed grid the AMR must beat.
``swe``
    Second-order finite-volume shallow water (``tensorlbm.shallow_water2d``) on
    the *same* tank geometry: the reference for the free-surface position.
``ritter``
    The Ritter (1892) self-similar solution.  Reported for context only: it
    assumes an infinitely long reservoir, so at ``t* = 3.1`` (``t = 1 s`` with a
    1 m column) the finite tank has already drained and it is *not* a valid
    anchor — the deviation is measured, not tuned.

Non-dimensionalisation
----------------------
Lengths in initial water depth ``H``, time in ``sqrt(H/g)``: ``t* = t sqrt(g/H)``.
A level-0 step advances ``dt* = c0 / H_lat`` (``c0 = sqrt(g_lat H_lat)``).

Usage::

    PYTHONPATH=src python benchmarks/amr_free_surface/run_study.py --out runs/amr_fs/study.json
    PYTHONPATH=src python benchmarks/amr_free_surface/run_study.py --fast
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from tensorlbm.amr_free_surface import AMRFreeSurface, AMRFreeSurfaceConfig  # noqa: E402
from tensorlbm.shallow_water2d import (  # noqa: E402
    dam_break_initial_state,
    depth_profile_x,
    simulate,
)

# ---------------------------------------------------------------------------
# geometry (dimensionless: H = 1)
# ---------------------------------------------------------------------------

H_LAT = 6
"""Initial depth in level-0 cells."""

NX0 = 48
"""Tank length in level-0 cells (8 H: no front reflection up to t* = 3.7)."""

NY0 = 10
"""Tank height in level-0 cells (1.67 H)."""

COLUMN_WIDTH = 3
"""Initial column width in level-0 cells (0.5 H, the classic 0.5 m x 1 m column)."""

C0 = 0.12
"""Gravity-wave celerity in lattice units (front Mach 0.21)."""

REYNOLDS = 60.0
"""Grid-independent dam-break Reynolds number."""

T_STAR_END = 3.13
"""``t sqrt(g/H)`` at ``t = 1 s`` for a 1 m column on Earth."""

THRESHOLD_H = 0.01
"""A column counts as wet above ``0.01 H`` — one finest cell at ``H = 48`` cells."""


def ritter_front(t_star: float) -> float:
    """Ritter (1892) front position in units of ``H`` from the dam (0.5 H)."""
    return 0.5 + 2.0 * t_star


def wet_front(
    depth: np.ndarray, dx: float, threshold: float = THRESHOLD_H, x_start: float = 0.0
) -> float:
    """Rightmost wet abscissa (in units of ``H``) for a depth profile."""
    wet = np.nonzero(depth > threshold)[0]
    if wet.size == 0:
        return x_start
    return x_start + (wet[-1] + 1) * dx


def swe_reference(t_star_end: float, nx: int = 4096, ny: int = 3) -> dict:
    """Planar dam break in the same closed tank, second-order shallow water."""
    dx = (NX0 / H_LAT) / nx
    state = dam_break_initial_state(
        ny,
        nx,
        depth=1.0,
        column_width=COLUMN_WIDTH / H_LAT,
        dx=dx,
        dy=dx,
        g=1.0,
    )
    state = simulate(state, t_end=t_star_end, cfl=0.35)
    depth = depth_profile_x(state)
    x = (np.arange(nx) + 0.5) * dx
    return {
        "t_star": float(state.t),
        "x": x.tolist(),
        "depth": depth.tolist(),
        "front": wet_front(depth, dx),
        "mass_drift": float(np.max(np.abs(np.asarray(state.history["mass"]) - state.history["mass"][0]))
                            / state.history["mass"][0]),
        "dx": dx,
    }


def profile_l1(depth: np.ndarray, x: np.ndarray, ref: dict, x_max: float) -> float:
    """Relative L1 deviation of the depth profile from *ref* over the wet region."""
    rx = np.asarray(ref["x"])
    rd = np.asarray(ref["depth"])
    sel = x <= x_max
    ref_sel = rx <= x_max
    interp = np.interp(x[sel], rx, rd)
    sim = depth[sel]
    denom = max(float(np.trapezoid(np.abs(rd[ref_sel]), rx[ref_sel])), 1e-12)
    return float(np.trapezoid(np.abs(sim - interp), x[sel]) / denom)


def run_case(levels: int, n_steps: int, adapt_interval: int, grid_scale: int = 1,
             refine_liquid: bool = False) -> dict:
    """Integrate one AMR (or uniform, ``levels = 0``) configuration."""
    if grid_scale == 1:
        cfg = AMRFreeSurfaceConfig(
            ny0=NY0,
            nx0=NX0,
            levels=levels,
            column_width=COLUMN_WIDTH,
            column_height=H_LAT,
            c0=C0,
            reynolds=REYNOLDS,
            adapt_interval=adapt_interval,
            refine_liquid=refine_liquid,
        )
        h_finest = H_LAT * 2**levels
    else:  # uniform grid at the finest spacing of the AMR ladder
        cfg = AMRFreeSurfaceConfig(
            ny0=NY0 * grid_scale,
            nx0=NX0 * grid_scale,
            levels=0,
            column_width=COLUMN_WIDTH * grid_scale,
            column_height=H_LAT * grid_scale,
            c0=C0,
            # same physical viscosity as the AMR's deepest level: a single-level
            # grid at that spacing takes tau = tau_finest directly.
            reynolds=REYNOLDS,
            adapt_interval=0,
        )
        h_finest = H_LAT * grid_scale
    solver = AMRFreeSurface(cfg)
    t0 = time.perf_counter()
    solver.run(n_steps)
    wall = time.perf_counter() - t0
    # depth_profile() sums fill over the column in *cells*; report depth in H
    depth = solver.depth_profile().numpy() / h_finest
    dx_star = 1.0 / h_finest
    x = (np.arange(depth.size) + 0.5) * dx_star
    cells = solver.cell_count()
    final_mass = solver.composite_mass()
    # ledger at the 1000-finest-step mark
    fine_per_coarse = 2**levels
    if fine_per_coarse > 0 and len(solver.drift_history) >= max(1, 1000 // fine_per_coarse):
        k = max(1, 1000 // fine_per_coarse)
        drift_1000 = float(np.sum(solver.drift_history[:k]))
    else:
        drift_1000 = float(np.sum(solver.drift_history))
    return {
        "levels": levels,
        "grid_scale": grid_scale,
        "refine_liquid": refine_liquid,
        "n_steps": n_steps,
        "h_finest_over_H": 1.0 / h_finest,
        "t_star": n_steps * cfg.time_scale(),
        "tau": [cfg.tau(l) for l in range(levels + 1)],
        "gravity": [cfg.gravity(l) for l in range(levels + 1)],
        "wall_seconds": wall,
        "cells": cells,
        "mass0": solver.mass0,
        "mass_final": final_mass,
        "mass_rel_error_closed": abs(final_mass - solver.mass0) / solver.mass0,
        "mass_rel_error_raw": abs(solver.raw_drift) / solver.mass0,
        "mass_rel_drift_raw_1000_fine_steps": abs(drift_1000) / solver.mass0,
        "regrids": solver.regrid_count,
        "front": wet_front(depth, dx_star),
        "depth_profile": depth.tolist(),
        "x": x.tolist(),
        "boxes": {
            f"level_{l}": [tuple(b) for b in
                           [(bx.y0, bx.y1, bx.x0, bx.x1) for bx in solver.levels[l].boxes]]
            for l in range(min(levels, len(solver.levels) - 1) + 1)
        },
        "steps_per_second": n_steps / wall,
    }


def convergence_order(errors: dict[int, float], spacings: dict[int, float]) -> float | None:
    """Least-squares order ``p`` in ``err = C h^p`` over the given levels."""
    ks = sorted(errors)
    if len(ks) < 2:
        return None
    hs = np.log(np.array([spacings[k] for k in ks]))
    es = np.log(np.array([max(errors[k], 1e-12) for k in ks]))
    p, _ = np.polyfit(hs, es, 1)
    return float(p)


def _case_spec(name: str) -> dict:
    """Which configuration a case name maps to."""
    if name.startswith("amr_L"):
        lv = int(name.split("amr_L")[1])
        return dict(levels=lv, n_steps=_N_STEPS, adapt_interval=8)
    if name == "uniform_fine":
        lv = max(int(k.split("amr_L")[1]) for k in _LADDER)
        return dict(levels=lv, n_steps=_N_STEPS * 2**lv, adapt_interval=0, grid_scale=2**lv)
    if name == "control_refine_liquid":
        lv = max(int(k.split("amr_L")[1]) for k in _LADDER)
        return dict(levels=lv, n_steps=_N_STEPS, adapt_interval=8, refine_liquid=True)
    raise SystemExit(f"unknown case {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_HERE / "study.json")
    ap.add_argument("--case-dir", type=Path, default=_HERE / "cases")
    ap.add_argument("--only", default="", help="comma separated case names to run")
    ap.add_argument("--combine", action="store_true", help="assemble study.json from case files")
    ap.add_argument("--horizon", type=float, default=T_STAR_END)
    args = ap.parse_args()
    global _N_STEPS, _LADDER
    torch.set_num_threads(int(__import__("os").environ.get("TL_THREADS", "4")))
    _N_STEPS = max(4, int(round(args.horizon / (C0 / H_LAT))))
    _LADDER = ["amr_L1", "amr_L2", "amr_L3"]
    args.case_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        for name in [x for x in args.only.split(",") if x]:
            spec = _case_spec(name)
            t0 = time.perf_counter()
            res = run_case(**spec)
            res["case"] = name
            res["seconds_total"] = time.perf_counter() - t0
            (args.case_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
            print(f"[{name}] wall={res['wall_seconds']:.1f}s front={res['front']:.2f}H "
                  f"cells={res['cells']['ratio_to_uniform_finest']:.4f} "
                  f"mass_raw={res['mass_rel_error_raw']:.2e}", flush=True)
        return 0

    if not args.combine:
        ap.error("either --only or --combine is required")

    # ---- combine ------------------------------------------------------
    report: dict = {
        "benchmark": "amr_free_surface_dam_break",
        "node": "P190-N02",
        "nondimensionalisation": "H = initial depth; t* = t sqrt(g/H); dt* per level-0 step = c0/H_lat",
        "geometry": {
            "H_lat_level0": H_LAT,
            "tank_L_over_H": NX0 / H_LAT,
            "tank_h_over_H": NY0 / H_LAT,
            "column_width_over_H": COLUMN_WIDTH / H_LAT,
            "c0": C0,
            "reynolds": REYNOLDS,
            "t_star_target": args.horizon,
            "n_level0_steps": _N_STEPS,
        },
        "runs": {},
    }
    ref = swe_reference(args.horizon)
    report["swe_reference"] = {
        "t_star": ref["t_star"], "front": ref["front"], "mass_drift": ref["mass_drift"],
        "dx": ref["dx"], "x": ref["x"][::8], "depth": ref["depth"][::8],
    }
    report["ritter_front"] = ritter_front(args.horizon)
    runs = {}
    for name in _LADDER + ["uniform_fine", "control_refine_liquid"]:
        path = args.case_dir / f"{name}.json"
        if not path.exists():
            continue
        runs[name] = json.loads(path.read_text())
    report["runs"] = runs
    for name, res in runs.items():
        if name in _LADDER or name == "control_refine_liquid":
            res["front_error_vs_swe"] = abs(res["front"] - ref["front"]) / max(
                ref["front"] - COLUMN_WIDTH / H_LAT, 1e-9)
            res["front_error_vs_ritter"] = abs(res["front"] - ritter_front(args.horizon)) / max(
                ritter_front(args.horizon) - COLUMN_WIDTH / H_LAT, 1e-9)
        res["profile_l1_vs_swe"] = profile_l1(
            np.asarray(res["depth_profile"]), np.asarray(res["x"]), ref, x_max=NX0 / H_LAT)
    if "uniform_fine" in runs and "amr_L3" in runs:
        uf, best = runs["uniform_fine"], runs["amr_L3"]
        uf["profile_l1_vs_amr"] = profile_l1(
            np.asarray(uf["depth_profile"]), np.asarray(uf["x"]),
            {"x": best["x"], "depth": best["depth_profile"]}, x_max=NX0 / H_LAT)
        uf["front_diff_vs_amr"] = abs(uf["front"] - best["front"]) / max(
            best["front"] - COLUMN_WIDTH / H_LAT, 1e-9)
        report["efficiency"] = {
            "amr_cells_over_uniform": best["cells"]["ratio_to_uniform_finest"],
            "amr_seconds_over_uniform": best["wall_seconds"] / uf["wall_seconds"],
            "uniform_seconds": uf["wall_seconds"],
            "amr_seconds": best["wall_seconds"],
        }
    if "control_refine_liquid" in runs and "uniform_fine" in runs:
        ctrl, uf = runs["control_refine_liquid"], runs["uniform_fine"]
        ctrl["profile_l1_vs_uniform"] = profile_l1(
            np.asarray(ctrl["depth_profile"]), np.asarray(ctrl["x"]),
            {"x": uf["x"], "depth": uf["depth_profile"]}, x_max=NX0 / H_LAT)
        report["control"] = {
            "note": "refinement covers the whole liquid body instead of only the "
                    "interface ribbon: separates bulk error from interface error",
            "profile_l1_vs_uniform_interface_only": uf.get("profile_l1_vs_amr"),
            "profile_l1_vs_uniform_full_liquid": ctrl["profile_l1_vs_uniform"],
            "cells_interface_only": runs["amr_L3"]["cells"]["ratio_to_uniform_finest"],
            "cells_full_liquid": ctrl["cells"]["ratio_to_uniform_finest"],
        }
    ladder = [n for n in _LADDER if n in runs]
    spacings = {int(n.split("amr_L")[1]): 1.0 / (H_LAT * 2 ** int(n.split("amr_L")[1])) for n in ladder}
    errs_front = {int(n.split("amr_L")[1]): runs[n]["front_error_vs_swe"] for n in ladder}
    errs_prof = {int(n.split("amr_L")[1]): runs[n]["profile_l1_vs_swe"] for n in ladder}
    report["convergence"] = {
        "front_errors": errs_front, "profile_errors": errs_prof,
        "order_front": convergence_order(errs_front, spacings),
        "order_profile": convergence_order(errs_prof, spacings),
    }
    report["verdict"] = evaluate(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdict"], indent=1))
    return 0


def evaluate(report: dict) -> dict:
    """Machine-checkable acceptance criteria of the campaign node."""
    runs = report["runs"]
    best = runs.get("amr_L3") or runs[max(runs, key=lambda k: runs[k]["levels"])]
    out = {
        "a_front_error_vs_swe_lt_5pct": {
            "value": best.get("front_error_vs_swe"),
            "limit": 0.05,
            "pass": bool(best.get("front_error_vs_swe") is not None
                         and best["front_error_vs_swe"] < 0.05),
            "metric": "|x_front(AMR) - x_front(SWE same tank)| / front advance",
        },
        "b_mass_error_lt_1e-6_1000_steps": {
            "value": best.get("mass_rel_error_closed"),
            "raw_value": best.get("mass_rel_error_raw"),
            "raw_1000_fine": best.get("mass_rel_drift_raw_1000_fine_steps"),
            "limit": 1e-6,
            "pass": bool(best["mass_rel_error_closed"] < 1e-6),
            "metric": "relative composite Körner inventory drift, reflux closed",
        },
        "c_amr_cells_lt_30pct_uniform": {
            "value": best["cells"]["ratio_to_uniform_finest"],
            "limit": 0.30,
            "pass": bool(best["cells"]["ratio_to_uniform_finest"] < 0.30),
            "metric": "AMR cells / uniform finest-grid cells",
        },
        "e_order_front_ge_2": {
            "value": report["convergence"]["order_front"],
            "limit": 2.0,
            "pass": bool(report["convergence"]["order_front"] is not None
                         and report["convergence"]["order_front"] >= 2.0),
            "metric": "fitted order p of the front error over h/2, h/4, h/8",
        },
        "e_order_profile_ge_2": {
            "value": report["convergence"]["order_profile"],
            "limit": 2.0,
            "pass": bool(report["convergence"]["order_profile"] is not None
                         and report["convergence"]["order_profile"] >= 2.0),
            "metric": "fitted order p of the depth-profile L1 error",
        },
    }
    if "efficiency" in report:
        out["d_amr_time_lt_50pct_uniform"] = {
            "value": report["efficiency"]["amr_seconds_over_uniform"],
            "limit": 0.50,
            "pass": bool(report["efficiency"]["amr_seconds_over_uniform"] < 0.50),
            "metric": "AMR wall time / uniform finest-grid wall time",
        }
    if "uniform_fine" in runs:
        uf = runs["uniform_fine"]
        out["f_amr_vs_uniform_profile_l1_lt_5pct"] = {
            "value": uf.get("profile_l1_vs_amr"),
            "front_diff": uf.get("front_diff_vs_amr"),
            "limit": 0.05,
            "pass": bool(uf.get("profile_l1_vs_amr") is not None
                         and uf["profile_l1_vs_amr"] < 0.05),
            "metric": "relative L1 depth-profile deviation between AMR(L3) and the "
                      "uniform grid at the same finest spacing (AMR verification)",
        }
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
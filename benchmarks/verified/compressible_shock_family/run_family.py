#!/usr/bin/env python3
"""Compressible-shock family verification (campaign P19-N04): production runner.

Three sub-suites on the Mach 2-10 shock surface:

``ns``    normal shock, Mach 2/3/5/8/10 x 3 grids (400/800/1600 cells), driver
          state inverted from the shock-tube relations so the shock Mach number
          is a solver *output* (non-circular), plus a prescribed-jump variant.
``dmr``   double Mach reflection (Woodward & Colella 1984, problem 2), Mach 10,
          30 deg incidence, three grids.  Two snapshots (t1, t2) are stored so
          the triple-point / Mach-stem kinematics can be measured.
``ffs``   forward-facing step (Woodward & Colella 1984, problem 1), Mach 3,
          h = H/5, three grids, run to a steady state (with a steadiness probe).

The heavy 2-D cases are split into one process per grid so they can be run in
parallel; :func:`main` with ``--assemble`` then turns the raw fields into the
metrics, the gate report (``result.json`` / ``gates_verified.json``) and the
figures.  Nothing here contains a tuned constant: every acceptance number lives
in ``gates.py`` next to its physical justification.

Usage
-----
    python run_family.py --suite ns
    python run_family.py --case dmr --grid 480x120
    python run_family.py --case ffs --grid 300x100
    python run_family.py --assemble
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.compressible_shock import analysis, cases  # noqa: E402
from tensorlbm.compressible_shock.flux import GAMMA  # noqa: E402
from tensorlbm.compressible_shock.rankine_hugoniot import sound_speed  # noqa: E402

MACH_LIST = (2.0, 3.0, 5.0, 8.0, 10.0)
NS_GRIDS = (400, 800, 1600)
DMR_GRIDS = ((240, 60), (360, 90), (480, 120))
FFS_GRIDS = ((150, 50), (225, 75), (300, 100))
DMR_T1, DMR_T2 = 0.15, 0.20
FFS_T, FFS_T_PROBE = 4.0, 3.8


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


def out_dir() -> str:
    d = os.path.join(HERE, "out")
    os.makedirs(os.path.join(d, "fields"), exist_ok=True)
    return d


def _dump(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=1)
    print(f"[wrote] {path}", flush=True)


# ------------------------------------------------------------------ 1-D Mach scan
def run_normal_shock() -> dict:
    records = []
    for variant in ("riemann", "jump"):
        for mach in MACH_LIST:
            for n in NS_GRIDS:
                t0 = time.time()
                solver, meta = cases.normal_shock_case(mach, n=n, variant=variant)
                ref = meta["reference"]
                rho1, p1 = meta["pre_state"]["rho"], meta["pre_state"]["p"]
                thr = 0.5 * (rho1 + ref["rho2"])
                times, fronts = [], []
                t_end = meta["t_end"]
                for frac in (0.4, 0.7, 1.0):
                    solver.advance(frac * t_end)
                    rho = solver.primitives()[0]
                    times.append(float(solver.t))
                    fronts.append(
                        analysis.front_x(
                            rho, solver.xc, thr, x_min=meta["x_diaphragm"] * 0.5
                        )
                    )
                fit = analysis.fit_speed(times, fronts)
                x_front = fronts[-1]
                st = analysis.sample_state(
                    solver.U,
                    solver.xc,
                    x_front - 20 * solver.dx,
                    x_front - 3 * solver.dx,
                )
                jump = analysis.jump_ratios(
                    {"rho": rho1, "p": p1, "T": p1 / rho1}, st
                )
                err = {
                    "rho_ratio": analysis.relative_error(jump["rho_ratio"], ref["rho_ratio"]),
                    "p_ratio": analysis.relative_error(jump["p_ratio"], ref["p_ratio"]),
                    "T_ratio": analysis.relative_error(jump["T_ratio"], ref["T_ratio"]),
                }
                # non-circular shock speed: the measured front speed is an output
                err["shock_speed"] = analysis.relative_error(fit["speed"], ref["W"])
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
                            "rho": st["rho"],
                            "p": st["p"],
                            "T": st["T"],
                            "u": st["u"],
                            "rho_ratio": jump["rho_ratio"],
                            "p_ratio": jump["p_ratio"],
                            "T_ratio": jump["T_ratio"],
                        },
                        "exact": {
                            "shock_speed": ref["W"],
                            "rho": ref["rho2"],
                            "p": ref["p2"],
                            "T": ref["T2"],
                            "u": ref["u2"],
                            "rho_ratio": ref["rho_ratio"],
                            "p_ratio": ref["p_ratio"],
                            "T_ratio": ref["T_ratio"],
                        },
                        "error": err,
                        "max_ratio_error": max(err[k] for k in ("rho_ratio", "p_ratio", "T_ratio")),
                        "wall_time_s": time.time() - t0,
                    }
                )
                print(
                    f"[ns:{variant}] M={mach:>4} n={n:>5} "
                    f"rho={err['rho_ratio'] * 100:5.2f}% p={err['p_ratio'] * 100:5.2f}% "
                    f"T={err['T_ratio'] * 100:5.2f}% W={err['shock_speed'] * 100:5.2f}%",
                    flush=True,
                )
    summary = {
        "suite": "normal_shock",
        "machs": list(MACH_LIST),
        "grids": list(NS_GRIDS),
        "gamma": GAMMA,
        "records": records,
        "n_cases": len(records),
        "max_ratio_error": max(r["max_ratio_error"] for r in records),
        "max_all_error": max(max(r["error"].values()) for r in records),
    }
    _dump(os.path.join(out_dir(), "normal_shock.json"), summary)
    return summary


# ------------------------------------------------------------------- 2-D DMR
def run_dmr(nx: int, ny: int) -> dict:
    t0 = time.time()
    solver, meta = cases.double_mach_reflection_case(nx=nx, ny=ny)
    thr = 0.5 * (meta["pre_shock"]["rho"] + meta["post_shock"]["rho"])
    solver.advance(DMR_T1)
    rho = solver.primitives()[0]
    prof1 = analysis.front_profile(rho, solver.xc, solver.yc, threshold=thr, x_min=meta["x_wall"])
    t1 = float(solver.t)
    solver.advance(DMR_T2)
    rho, u, v, p = solver.primitives()
    prof2 = analysis.front_profile(rho, solver.xc, solver.yc, threshold=thr, x_min=meta["x_wall"])
    path = os.path.join(out_dir(), "fields", f"dmr_{nx}x{ny}.npz")
    np.savez_compressed(
        path,
        rho=rho.astype(np.float32),
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        p=p.astype(np.float32),
        xc=solver.xc.astype(np.float32),
        yc=solver.yc.astype(np.float32),
        t=float(solver.t),
        t1=t1,
        prof1=prof1,
        prof2=prof2,
        meta=json.dumps(_jsonable(meta)),
        n_admissibility_fixes=solver.n_admissibility_fixes,
        wall_time_s=time.time() - t0,
    )
    print(f"[dmr] {nx}x{ny} done in {time.time() - t0:.0f}s -> {path}", flush=True)
    return {"nx": nx, "ny": ny, "path": path}


# ------------------------------------------------------------------- 2-D FFS
def run_ffs(nx: int, ny: int) -> dict:
    t0 = time.time()
    solver, meta = cases.forward_facing_step_case(nx=nx, ny=ny, t_end=FFS_T)
    solver.advance(FFS_T_PROBE)
    rho1 = solver.primitives()[0].copy()
    solver.advance(FFS_T)
    rho, u, v, p = solver.primitives()
    path = os.path.join(out_dir(), "fields", f"ffs_{nx}x{ny}.npz")
    np.savez_compressed(
        path,
        rho=rho.astype(np.float32),
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        p=p.astype(np.float32),
        rho_probe=rho1.astype(np.float32),
        xc=solver.xc.astype(np.float32),
        yc=solver.yc.astype(np.float32),
        t=float(solver.t),
        t_probe=float(solver.t),
        meta=json.dumps(_jsonable(meta)),
        n_admissibility_fixes=solver.n_admissibility_fixes,
        wall_time_s=time.time() - t0,
    )
    print(f"[ffs] {nx}x{ny} done in {time.time() - t0:.0f}s -> {path}", flush=True)
    return {"nx": nx, "ny": ny, "path": path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None, choices=("ns",))
    ap.add_argument("--case", default=None, choices=("dmr", "ffs"))
    ap.add_argument("--grid", default=None, help="NXxNY, e.g. 480x120")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--all-2d", action="store_true", help="run every 2-D grid in-process")
    args = ap.parse_args(argv)

    if args.suite == "ns":
        run_normal_shock()
        return 0
    if args.case is not None:
        nx, ny = (int(s) for s in args.grid.lower().split("x"))
        (run_dmr if args.case == "dmr" else run_ffs)(nx, ny)
        return 0
    if args.all_2d:
        for nx, ny in DMR_GRIDS:
            run_dmr(nx, ny)
        for nx, ny in FFS_GRIDS:
            run_ffs(nx, ny)
        return 0
    if args.assemble:
        from analyze_family import assemble

        rep = assemble()
        print(json.dumps({k: v for k, v in rep.items() if k in ("all_gates_pass",)}, indent=1))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
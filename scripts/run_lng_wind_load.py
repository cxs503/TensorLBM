#!/usr/bin/env python3
"""Driver: LNG Moss-type superstructure wind-load campaign (P16-N7).

One invocation = one simulation case, writing a JSON record (geometry,
lattice/unit mapping, force coefficients, mass balance, timing, force time
series statistics) plus a restartable ``.npz`` state file.

Examples
--------
    # canonical cube validation (Re = 3e3, 8 cells per side)
    python scripts/run_lng_wind_load.py --case cube --re 3000 --cells 8 \
        --steps 900 --avg-from 500 --u-lat 0.08 --threads 10 \
        --out runs/p16n7/cube_re3000.json

    # ship, beam wind, medium grid
    python scripts/run_lng_wind_load.py --case ship --angle 90 --dx 1.0 \
        --steps 1500 --avg-from 600 --u-lat 0.15 --threads 10 \
        --out runs/p16n7/ship_theta090_dx1.0.json --state runs/p16n7/ship_theta090_dx1.0.npz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(HERE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "src"))

import torch  # noqa: E402

from tensorlbm.wind_load_lng import (  # noqa: E402
    LatticeUnits,
    WindLoadSolver,
    cube_case,
    lng_moss_superstructure,
    voxelise,
)


def build_solver(args) -> tuple[WindLoadSolver, dict]:
    if args.case == "cube":
        geom = cube_case(side=1.0, cells_per_side=args.cells)
        units = LatticeUnits.from_reynolds(
            dx=geom.dx, re_target=args.re, length_m=1.0, u_lat=args.u_lat
        )
        meta = {"re_D": args.re, "cells_per_side": args.cells}
    else:
        geom = voxelise(lng_moss_superstructure(), args.dx, args.angle,
                        margin_upstream=args.margin_upstream,
                        margin_downstream=args.margin_downstream,
                        margin_side=args.margin_side,
                        margin_top=args.margin_top)
        units = LatticeUnits.from_velocity(
            dx=geom.dx, u_phys=args.u_phys, u_lat=args.u_lat
        )
        meta = {"re_H": units.re_phys(geom.height),
                "re_L": units.re_phys(50.0)}
    kw = {}
    if args.case == "ship":
        g0 = voxelise(lng_moss_superstructure(), args.dx, 0.0,
                      margin_upstream=args.margin_upstream,
                      margin_downstream=args.margin_downstream,
                      margin_side=args.margin_side,
                      margin_top=args.margin_top)
        kw = {"ref_areas": (g0.area_frontal, g0.area_lateral),
              "ref_breadth_height": (g0.breadth, g0.height)}
    solver = WindLoadSolver(
        geom, units, **kw,
        cs_smag=args.cs,
        wall_model=args.wall_model,
        wall_feedback=args.wall_feedback,
        normalise_outflow=args.normalise_outflow,
        device="cpu",
    )
    return solver, meta


def prolongate(solver: WindLoadSolver, src: dict) -> None:
    """Initialise ``solver`` from a coarser converged state (nearest neighbour)."""
    rho_c = np.asarray(src["rho"])
    u_c = np.asarray(src["u"])
    dx_c = float(src["dx"])
    org_c = np.asarray(src["origin"])
    g = solver.geom
    # physical coordinates of the fine cell centres
    zz = org_c[2] + 0.0  # only used for clarity
    del zz
    iz = np.clip(((np.arange(g.nz) + 0.5) * g.dx + g.origin[2] - org_c[2]
                  ) / dx_c, 0, rho_c.shape[0] - 1e-6).astype(int)
    iy = np.clip(((np.arange(g.ny) + 0.5) * g.dx + g.origin[1] - org_c[1]
                  ) / dx_c, 0, rho_c.shape[1] - 1e-6).astype(int)
    ix = np.clip(((np.arange(g.nx) + 0.5) * g.dx + g.origin[0] - org_c[0]
                  ) / dx_c, 0, rho_c.shape[2] - 1e-6).astype(int)
    rho_f = rho_c[np.ix_(iz, iy, ix)]
    u_f = u_c[:, iz][:, :, iy][:, :, :, ix]
    rho_t = torch.as_tensor(rho_f, dtype=solver.dtype)
    u_t = torch.as_tensor(u_f, dtype=solver.dtype)
    solver.f = solver.equilibrium(rho_t, u_t)
    solver.f[:, solver.solid] = 0.0
    solver.apply_boundaries(solver.f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ship", "cube"], default="ship")
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--cells", type=int, default=8, help="cube cells per side")
    ap.add_argument("--re", type=float, default=3000.0, help="cube Re (side)")
    ap.add_argument("--u-lat", type=float, default=0.15)
    ap.add_argument("--u-phys", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--avg-from", type=int, default=500)
    ap.add_argument("--cs", type=float, default=0.16)
    ap.add_argument("--wall-model", action="store_true", default=True)
    ap.add_argument("--no-wall-model", dest="wall_model", action="store_false")
    ap.add_argument("--wall-feedback", action="store_true", default=True)
    ap.add_argument("--no-wall-feedback", dest="wall_feedback", action="store_false")
    ap.add_argument("--normalise-outflow", action="store_true", default=True)
    ap.add_argument("--no-normalise-outflow", dest="normalise_outflow",
                    action="store_false")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--margin-upstream", type=float, default=40.0)
    ap.add_argument("--margin-downstream", type=float, default=40.0)
    ap.add_argument("--margin-side", type=float, default=15.0)
    ap.add_argument("--margin-top", type=float, default=25.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--init-from", default=None,
                    help="npz with coarse (rho, u, dx, origin) for prolongation")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    solver, meta = build_solver(args)
    if args.init_from:
        prolongate(solver, dict(np.load(args.init_from)))

    g = solver.geom
    t0 = time.time()
    summary = solver.run(args.steps, average_from=args.avg_from)
    wall = time.time() - t0

    # time series diagnostics (last 60 % of the window for spectral analysis)
    hist = solver.force_history
    fx = np.array([s.fx for s in hist])
    fy = np.array([s.fy for s in hist])
    rec = {
        "case": args.case,
        "angle_deg": args.angle,
        "label": os.path.basename(args.out).replace(".json", ""),
        "tag": args.tag,
        "domain_margins": {"upstream": args.margin_upstream,
                           "downstream": args.margin_downstream,
                           "side": args.margin_side, "top": args.margin_top},
        "geometry": g.to_dict(),
        "units": solver.units.to_dict(),
        "solver": {
            "cs_smag": solver.cs_smag, "wall_model": solver.use_wall_model,
            "wall_feedback": solver.wall_feedback,
            "normalise_outflow": solver.normalise_outflow,
            "u_in_lattice": solver.u_in, "nu0_lattice": solver.nu0,
            "tau0": solver.tau0, "u_clip": solver.u_clip,
            "n_wall_cells": solver.n_wall_cells,
            "n_links": int(sum(len(i) for i in solver.link_idx if i is not None)),
        },
        "meta": meta,
        "steps": args.steps,
        "steps_total": solver.step_count,
        "avg_from": args.avg_from,
        "wall_time_s": wall,
        "ms_per_step": 1000.0 * wall / max(1, args.steps),
        "coefficients": {k: float(v) for k, v in summary.items()},
        "mass_flux_error": solver.mass_flux_error(),
        "mass_budget": solver.mass_budget(args.avg_from),
        "force_series": {"fx_mean": float(fx.mean()), "fx_std": float(fx.std()),
                         "fy_mean": float(fy.mean()), "fy_std": float(fy.std())},
        "time_series": {
            "dt_s": solver.units.dt,
            "fx": fx[-2000:].tolist(), "fy": fy[-2000:].tolist(),
            "index0": max(0, len(hist) - 2000),
        },
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform(), "threads": args.threads},
    }
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=1)

    if args.state:
        solver.save_state(args.state)
        np.savez_compressed(
            args.state.replace(".npz", "_field.npz"),
            rho=solver.macroscopic(solver.f)[0].cpu().numpy(),
            u=solver.macroscopic(solver.f)[1].cpu().numpy(),
            dx=g.dx, origin=np.array(g.origin), angle=g.angle_deg,
        )
    print(json.dumps({k: rec[k] for k in ("label", "ms_per_step",
                                          "mass_flux_error")}))
    print("C_X=%.4f C_Y=%.4f C_N=%.4f C_K=%.4f" % (
        summary["C_X"], summary["C_Y"], summary["C_N"], summary["C_K"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
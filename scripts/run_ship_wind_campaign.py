#!/usr/bin/env python3
"""Driver: ship superstructure wind-load campaign (node IN38-N04).

One invocation = one (ship type, wind angle, grid) LBM case.  Writes a JSON
record with the geometry, lattice-unit mapping, force coefficients, mass
balance and timing, plus an optional restartable ``.npz`` state file.

Examples
--------
    python scripts/run_ship_wind_campaign.py --ship container --angle 90 \
        --dx 2.5 --steps 1600 --avg-from 800 --out runs/in38_n04/container_090.json

    # canonical bluff-body anchor (cube, Re based on the side length)
    python scripts/run_ship_wind_campaign.py --case cube --re 20000 --cells 12 \
        --steps 800 --avg-from 400 --out runs/in38_n04/cube_re20000.json
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

from tensorlbm.ship_wind_aero import catalogue  # noqa: E402
from tensorlbm.wind_load_lng import (  # noqa: E402
    LatticeUnits,
    WindLoadSolver,
    cube_case,
    voxelise,
)


def build(args) -> tuple[WindLoadSolver, dict, dict]:
    if args.case == "cube":
        geom = cube_case(side=1.0, cells_per_side=args.cells,
                         domain=tuple(float(v) for v in args.domain.split(",")))
        units = LatticeUnits.from_reynolds(
            dx=geom.dx, re_target=args.re, length_m=1.0, u_lat=args.u_lat)
        solver = WindLoadSolver(geom, units, cs_smag=args.cs,
                                wall_model=args.wall_model,
                                wall_feedback=args.wall_feedback,
                                normalise_outflow=args.normalise_outflow,
                                wall_law=args.wall_law,
                                u_clip=args.u_clip)
        meta = {"case": "cube", "re_side": args.re,
                "cells_per_side": args.cells,
                "reference_area_m2": 1.0}
        return solver, meta, {"C_D_literature": 1.05}

    ships = catalogue()
    ship = ships[args.ship]
    boxes = ship.to_lng_boxes()
    g0 = voxelise(boxes, args.dx, 0.0, margin_upstream=args.margin_upstream,
                  margin_downstream=args.margin_downstream,
                  margin_side=args.margin_side, margin_top=args.margin_top)
    geom = g0 if args.angle == 0.0 else voxelise(
        boxes, args.dx, args.angle, margin_upstream=args.margin_upstream,
        margin_downstream=args.margin_downstream, margin_side=args.margin_side,
        margin_top=args.margin_top)
    units = LatticeUnits.from_velocity(dx=geom.dx, u_phys=args.u_phys,
                                       u_lat=args.u_lat)
    solver = WindLoadSolver(
        geom, units,
        ref_areas=(g0.area_frontal, g0.area_lateral),
        ref_breadth_height=(g0.breadth, g0.height),
        cs_smag=args.cs, wall_model=args.wall_model,
        wall_feedback=args.wall_feedback, normalise_outflow=args.normalise_outflow,
        wall_law=args.wall_law, u_clip=args.u_clip)
    meta = {
        "case": "ship", "ship": ship.key,
        "re_H": units.re_phys(g0.height),
        "re_L_super": units.re_phys(ship.extent[0]),
        "reference_area_frontal_m2": g0.area_frontal,
        "reference_area_lateral_m2": g0.area_lateral,
        "reference_breadth_m": g0.breadth,
        "reference_height_m": g0.height,
        "analytic_reference": ship.reference,
    }
    return solver, meta, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["ship", "cube"], default="ship")
    ap.add_argument("--ship", choices=sorted(catalogue()), default="container")
    ap.add_argument("--angle", type=float, default=0.0)
    ap.add_argument("--dx", type=float, default=2.5)
    ap.add_argument("--cells", type=int, default=12, help="cube cells per side")
    ap.add_argument("--re", type=float, default=20000.0)
    ap.add_argument("--domain", default="10.0,4.0,4.0",
                    help="cube domain in cube sides (L_up+body, W, H)")
    ap.add_argument("--u-lat", type=float, default=0.15)
    ap.add_argument("--u-phys", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--avg-from", type=int, default=800)
    ap.add_argument("--cs", type=float, default=0.16)
    ap.add_argument("--u-clip", type=float, default=0.25,
                    help="velocity clip used to keep the LBM stable")
    ap.add_argument("--wall-model", action="store_true", default=True)
    ap.add_argument("--no-wall-model", dest="wall_model", action="store_false")
    ap.add_argument("--wall-law", choices=["log", "spalding", "reichardt"],
                    default="log",
                    help="wall-adjacent shear law (log = legacy Kader law)")
    ap.add_argument("--wall-feedback", action="store_true", default=True)
    ap.add_argument("--no-wall-feedback", dest="wall_feedback",
                    action="store_false")
    ap.add_argument("--normalise-outflow", action="store_true", default=True)
    ap.add_argument("--no-normalise-outflow", dest="normalise_outflow",
                    action="store_false")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--margin-upstream", type=float, default=40.0)
    ap.add_argument("--margin-downstream", type=float, default=40.0)
    ap.add_argument("--margin-side", type=float, default=18.0)
    ap.add_argument("--margin-top", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--fields", default=None,
                    help="npz path for the mid-height velocity slice (figure)")
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    t0 = time.time()
    solver, meta, extra = build(args)
    setup_s = time.time() - t0

    if args.init_from:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
        from run_lng_wind_load import prolongate  # type: ignore

        prolongate(solver, dict(np.load(args.init_from, allow_pickle=False)))

    t1 = time.time()
    coeff = solver.run(args.steps, average_from=args.avg_from)
    run_s = time.time() - t1
    summary = solver.coefficient_summary(from_index=args.avg_from)

    record = {
        "schema": "tensorlbm.ship_wind_campaign/v1",
        "node": "IN38-N04",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag,
        "angle_deg": float(args.angle),
        "case": meta,
        "geometry": solver.geom.to_dict(),
        "units": solver.units.to_dict(),
        "settings": {
            "dx_m": args.dx, "steps": args.steps, "average_from": args.avg_from,
            "u_lat": args.u_lat, "u_phys_ms": args.u_phys, "cs_smag": args.cs,
            "wall_model": args.wall_model, "wall_feedback": args.wall_feedback,
            "u_clip": args.u_clip,
            "normalise_outflow": args.normalise_outflow,
            "margins_m": [args.margin_upstream, args.margin_downstream,
                          args.margin_side, args.margin_top],
            "collision": "BGK + Smagorinsky",
            "velocity_set": "D3Q19",
        },
        "coefficients": coeff,
        "mass_budget": solver.mass_budget(from_index=args.avg_from),
        "timing": {"setup_s": setup_s, "run_s": run_s,
                   "steps_per_s": (args.steps / run_s) if run_s > 0 else None},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "threads": args.threads},
        **extra,
    }
    if meta.get("case") == "cube":
        A_F = float(meta["reference_area_m2"])
        q = 0.5 * solver.units.u_lat ** 2
        a_lat = A_F / solver.geom.dx ** 2
        record["coefficients"] = {"C_D": coeff["C_X"],
                                  "C_D_std": coeff["C_X_std"],
                                  "q_lattice": q, "area_cells": a_lat}

    if args.fields:
        s = solver
        k = max(1, int(0.30 * s.shape[0]))
        u = np.asarray(s.macroscopic(s.f)[1].cpu() if hasattr(s.f, "cpu")
                       else s.macroscopic(s.f)[1])
        speed = np.sqrt((u ** 2).sum(axis=0))
        np.savez_compressed(args.fields,
                            speed_mid= speed[k].astype(np.float32),
                            speed_max=speed.max(axis=0).astype(np.float32),
                            ux_mid=u[0][k].astype(np.float32),
                            solid_mid=np.asarray(
                                s.solid.cpu() if hasattr(s.solid, "cpu")
                                else s.solid)[k].astype(np.uint8),
                            dx=s.geom.dx, origin=np.asarray(s.geom.origin),
                            angle_deg=s.geom.angle_deg,
                            u_lat=s.units.u_lat,
                            nz=s.shape[0], mid_k=k)
        record["fields"] = args.fields
    record["force_history"] = {
        "fx": [round(s.fx, 6) for s in solver.force_history],
        "fy": [round(s.fy, 6) for s in solver.force_history]}

    if args.state:
        solver.save_state(args.state)
        record["state"] = args.state

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    print(json.dumps({"out": args.out, "ship": meta.get("ship", "cube"),
                      "angle": args.angle, "dx": args.dx,
                      "C_X": record["coefficients"].get("C_X"),
                      "C_Y": record["coefficients"].get("C_Y"),
                      "run_s": round(run_s, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
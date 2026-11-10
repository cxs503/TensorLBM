#!/usr/bin/env python
"""Parallel point runner for campaign ``p233w1-N07`` (local, time-boxed).

The control-plane box is shared, so each call must finish inside the tool's
command budget; every point is therefore an independent process and the runner
exits once its budget is spent, leaving the finished points on disk.  Points
already on disk are skipped, so the same command can simply be re-issued until
nothing is left.

Examples
--------
    python scripts/p233w1_n07_run_points.py --kind openwater \
        --points 12:0.2 12:0.4 12:0.6 12:0.8 12:1.0 --workers 5 --budget 100
    python scripts/p233w1_n07_run_points.py --kind cylinder \
        --re 20 40 60 --workers 3 --budget 100
    python scripts/p233w1_n07_run_points.py --kind aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts/p233w1_n07_ibm_openwater"
POINTS = OUT / "points"


def _openwater_worker(job: dict) -> dict:
    import torch

    torch.set_num_threads(job["threads"])
    from tensorlbm.ibm_openwater import OpenWaterConfig, PropellerGeometry, run_open_water_point

    geom = PropellerGeometry(diameter=job["diameter"], radial_stations=job["radial_stations"])
    cfg = OpenWaterConfig(
        advance_ratios=(job["J"],),
        tip_speed=job["tip_speed"],
        tau=job["tau"],
        steps_per_rev=job["steps_per_rev"],
        warmup_revs=job["warmup"],
        measure_revs=job["measure"],
        lateral_factor=job["lateral"],
        upstream_factor=job["upstream"],
        downstream_factor=job["downstream"],
        forcing=job["forcing"],
        collision=job["collision"],
        dtype=job["dtype"],
        max_steps=job["max_steps"],
    )
    t0 = time.time()
    res = run_open_water_point(geom, cfg, job["J"])
    res["wall_seconds"] = time.time() - t0
    res["configuration"] = {
        "diameter": job["diameter"],
        "advance_ratios": [job["J"]],
        "tip_speed": job["tip_speed"],
        "tau": job["tau"],
        "nu": cfg.nu,
        "steps_per_rev": job["steps_per_rev"],
        "warmup_revs": job["warmup"],
        "measure_revs": job["measure"],
        "lateral_factor": job["lateral"],
        "upstream_factor": job["upstream"],
        "downstream_factor": job["downstream"],
        "forcing": job["forcing"],
        "collision": job["collision"],
        "radial_stations": job["radial_stations"],
        "n_blades": geom.n_blades,
        "pitch_ratio_07": geom.pitch_ratio_07,
        "blade_area_ratio": geom.blade_area_ratio,
    }
    res["criterion"] = "|eta_0 - eta_ref| / eta_ref <= 0.15"
    res["passed"] = bool(abs(res["error"]["eta_rel"]) <= 0.15 and not res["numerics"]["diverged"])
    path = POINTS / f"openwater_d{job['diameter']:g}_j{job['J']:g}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))
    return {"point": str(path), "KT": res["KT"], "eta": res["eta_0"], "sec": res["wall_seconds"]}


def _cylinder_worker(job: dict) -> dict:
    import torch

    torch.set_num_threads(job["threads"])
    from tensorlbm.ibm_openwater import CylinderConfig, cylinder_drag_2d

    cfg = CylinderConfig(
        diameter=job["diameter"],
        reynolds=(job["Re"],),
        inflow=job["inflow"],
        nx_per_d=job["nx_per_d"],
        ny_per_d=job["ny_per_d"],
        wall_offset=job["wall_offset"],
        n_steps=job["steps"],
    )
    t0 = time.time()
    res = cylinder_drag_2d(job["Re"], cfg)
    res["wall_seconds"] = time.time() - t0
    path = POINTS / f"cylinder_re{int(job['Re'])}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))
    return {"point": str(path), "Cd": res["Cd"], "sec": res["wall_seconds"]}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def aggregate() -> dict:
    """Merge per-point JSON files into the sweep artefacts."""
    out = {}

    # --- open-water curves per mesh ---------------------------------------
    for mesh_dir in sorted(POINTS.glob("openwater_d*_j*.json")):
        pass
    by_mesh: dict[float, list[dict]] = {}
    for p in POINTS.glob("openwater_d*_j*.json"):
        d = json.loads(p.read_text())
        by_mesh.setdefault(float(d["numerics"]["diameter"]), []).append(d)
    sweeps = []
    for diameter, pts in sorted(by_mesh.items()):
        pts.sort(key=lambda r: r["J"])
        eta_err = [abs(r["error"]["eta_rel"]) for r in pts if math.isfinite(r["eta_0"])]
        kt_err = [abs(r["error"]["KT_rel"]) for r in pts if math.isfinite(r["KT"])]
        summary = {
            "diameter": diameter,
            "advance_ratios": [r["J"] for r in pts],
            "max_abs_eta_error": max(eta_err) if eta_err else float("inf"),
            "rms_abs_eta_error": (sum(e * e for e in eta_err) / len(eta_err)) ** 0.5
            if eta_err
            else float("inf"),
            "max_abs_KT_error": max(kt_err) if kt_err else float("inf"),
            "points": pts,
            "diverged": any(r["numerics"]["diverged"] for r in pts),
            "partial": len(pts) < 5,
            "criterion": "max |eta_0 - eta_ref| / eta_ref <= 0.15 for J in 0.2..1.0",
        }
        summary["passed"] = bool(
            summary["max_abs_eta_error"] <= 0.15 and not summary["diverged"]
        )
        sweeps.append(summary)
        if len(pts) >= 5:
            (OUT / f"openwater_d{diameter:g}.json").write_text(
                json.dumps(summary, indent=1, default=str)
            )
        out[f"d{diameter:g}"] = {
            "n_points": len(pts),
            "max_abs_eta_error": summary["max_abs_eta_error"],
        }

    # --- grid convergence --------------------------------------------------
    if len(sweeps) >= 3:
        sweeps.sort(key=lambda s: s["diameter"])
        # the three meshes are compared on the advance ratios they have in
        # common (J = 0.5 here), otherwise the meshes would be mixed up with
        # the J sweep.
        common = set.intersection(*[{p["J"] for p in s["points"]} for s in sweeps])
        per_mesh = []
        for s in sweeps:
            sel = [p for p in s["points"] if p["J"] in common]
            per_mesh.append(max(abs(p["error"]["eta_rel"]) for p in sel))
        errors = per_mesh
        from tensorlbm.ibm_openwater import eta_error_is_monotone

        payload = {
            "meshes": sweeps,
            "eta_errors_coarse_to_fine": errors,
            "common_advance_ratios": sorted(common),
            "monotone": eta_error_is_monotone(errors),
            "criterion": "max |eta error| non-increasing over 3 refined meshes",
            "diverged": any(s["diverged"] for s in sweeps),
            "partial": any(s["partial"] for s in sweeps),
        }
        payload["passed"] = bool(payload["monotone"] and not payload["diverged"])
        (OUT / "grid_convergence.json").write_text(json.dumps(payload, indent=1, default=str))
        out["grid_convergence"] = {
            "errors": errors,
            "monotone": payload["monotone"],
            "common_J": sorted(common),
        }

    # --- cylinder ----------------------------------------------------------
    cyl = [json.loads(p.read_text()) for p in POINTS.glob("cylinder_re*.json")]
    if cyl:
        cyl.sort(key=lambda r: r["Re"])
        errs = [abs(c["Cd_rel_error"]) for c in cyl if math.isfinite(c["Cd"])]
        payload = {
            "cases": cyl,
            "max_abs_rel_error": max(errs) if errs else float("inf"),
            "mean_abs_rel_error": sum(errs) / len(errs) if errs else float("inf"),
            "diverged": any(c["diverged"] for c in cyl),
            "criterion": "max |Cd - Cd_ref| / Cd_ref <= 0.10 for Re in 20..200",
            "n_cases": len(cyl),
        }
        payload["passed"] = bool(payload["max_abs_rel_error"] <= 0.10 and not payload["diverged"])
        (OUT / "cylinder.json").write_text(json.dumps(payload, indent=1, default=str))
        out["cylinder"] = {
            "n": len(cyl),
            "max_abs_rel_error": payload["max_abs_rel_error"],
        }
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    import torch as _torch

    global torch_dtype
    torch_dtype = _torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["openwater", "cylinder", "aggregate"])
    ap.add_argument("--points", nargs="+", default=[])
    ap.add_argument("--re", nargs="+", type=float, default=[])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--tau", type=float, default=0.5005)
    ap.add_argument("--tip-speed", type=float, default=0.12)
    ap.add_argument("--steps-per-rev", type=int, default=180)
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--measure", type=float, default=1.5)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--lateral", type=float, default=1.2)
    ap.add_argument("--upstream", type=float, default=1.2)
    ap.add_argument("--downstream", type=float, default=2.2)
    ap.add_argument("--radial-stations", type=int, default=10)
    ap.add_argument("--forcing", default="slip_friction")
    ap.add_argument("--collision", default="mrt")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--diameter-cyl", type=float, default=10.0)
    ap.add_argument("--inflow", type=float, default=0.05)
    ap.add_argument("--steps-cyl", type=int, default=8000)
    ap.add_argument("--nx-per-d", type=float, default=20.0)
    ap.add_argument("--ny-per-d", type=float, default=12.0)
    ap.add_argument("--wall-offset", type=float, default=0.5)
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.kind == "aggregate":
        print(json.dumps(aggregate(), indent=1))
        return 0

    jobs = []
    if args.kind == "openwater":
        for spec in args.points:
            d, j = spec.split(":")
            path = POINTS / f"openwater_d{float(d):g}_j{float(j):g}.json"
            if path.exists():
                continue
            jobs.append(
                {
                    "diameter": float(d),
                    "J": float(j),
                    "tau": args.tau,
                    "tip_speed": args.tip_speed,
                    "steps_per_rev": args.steps_per_rev,
                    "warmup": args.warmup,
                    "measure": args.measure,
                    "max_steps": args.max_steps,
                    "lateral": args.lateral,
                    "upstream": args.upstream,
                    "downstream": args.downstream,
                    "radial_stations": args.radial_stations,
                    "forcing": args.forcing,
                    "collision": args.collision,
                    "threads": args.threads,
                    "dtype": getattr(torch_dtype, args.dtype),
                }
            )
        worker = _openwater_worker
    else:
        for re in args.re:
            path = POINTS / f"cylinder_re{int(re)}.json"
            if path.exists():
                continue
            jobs.append(
                {
                    "Re": float(re),
                    "diameter": args.diameter_cyl,
                    "inflow": args.inflow,
                    "steps": args.steps_cyl,
                    "nx_per_d": args.nx_per_d,
                    "ny_per_d": args.ny_per_d,
                    "wall_offset": args.wall_offset,
                    "threads": args.threads,
                    "dtype": getattr(torch_dtype, args.dtype),
                }
            )
        worker = _cylinder_worker

    if not jobs:
        print(json.dumps({"pending": 0, "aggregate": aggregate()}, indent=1))
        return 0

    print(f"{len(jobs)} point(s) pending, workers={args.workers}, budget={args.budget}s", flush=True)
    results = []
    with mp.get_context("spawn").Pool(processes=min(args.workers, len(jobs))) as pool:
        async_res = pool.map_async(worker, jobs)
        try:
            for r in async_res.get(timeout=max(1.0, args.budget)):
                results.append(r)
                print(json.dumps(r, default=str), flush=True)
        except mp.TimeoutError:
            print(f"[budget] {args.budget}s reached, in-flight points will be redone", flush=True)
            pool.terminate()
    print(f"done {len(results)}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
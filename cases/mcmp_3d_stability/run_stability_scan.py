"""Acceptance run for GAP-PP-1-3DMCMP-STABILITY: 3D MCMP stability window + loss-of-stability map.

Protocol (predeclared in :mod:`tensorlbm.mcmp3d_stability`; nothing here is
re-tuned after looking at results):

Stage A  coupling scan, family ``coexist`` (equal mixture density, maximal
         component contrast -- the 3D counterpart of the 2D stratified-slab
         initialisation of ``cases/mcmp_stratified_poiseuille``), 8 G_12 values.
Stage B  the same scan again from an independently seeded initial state
         (2% density noise, seed 7) -- the reproducibility criterion.
Stage C  resolution check: the two G_12 values bracketing the stage-A boundary
         re-run on a (64, 16, 16) box.
Stage D  anchor: the largest stable |G_12| of each family run for 5000 steps
         (the "no divergence in 5000 steps" criterion) plus the density-contrast
         family ``water_air`` (rho1: 1.0/0.06, rho2: 0.01/0.20) to record the
         density ratio the stable window actually delivers.

Outputs (all numbers are written by this script, never hand-typed):
    artifacts/mcmp_3d_stability/<stage>/<case>.json   per-cell record
    artifacts/mcmp_3d_stability/summary.json          boundaries + criteria verdicts

Usage:
    PYTHONPATH=src python cases/mcmp_3d_stability/run_stability_scan.py --stage all --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from tensorlbm.mcmp3d_stability import (  # noqa: E402
    DEFAULT_CRITERIA,
    PROTOCOL_BOX,
    run_case,
    scan,
    stability_boundary,
)

OUT = "artifacts/mcmp_3d_stability"

FAMILIES = {
    "coexist": dict(rho1_in=0.92, rho1_out=0.14, rho2_in=0.08, rho2_out=0.86),
    "water_air": dict(rho1_in=1.00, rho1_out=0.06, rho2_in=0.01, rho2_out=0.20),
}
G_GRID = [-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5, -2.75]
SCAN_STEPS = 2000
ANCHOR_STEPS = 5000
RES_BOX = (64, 16, 16)


def case_name(fam: str, G: float, box: tuple[int, int, int]) -> str:
    return f"{fam}_G{G:+.2f}_{box[0]}x{box[1]}x{box[2]}"


def specs_for(fam: str, grid, box, steps, noise=0.0, seed=0) -> list[dict]:
    out = []
    for G in grid:
        spec = dict(FAMILIES[fam])
        spec.update(
            name=case_name(fam, G, box),
            G_12=G,
            steps=steps,
            nz=box[0],
            ny=box[1],
            nx=box[2],
            noise=noise,
            seed=seed,
            guo=True,
            psi1="linear",
            psi2="linear",
        )
        out.append(spec)
    return out


def write_records(stage: str, records: list[dict]) -> None:
    d = os.path.join(OUT, stage)
    os.makedirs(d, exist_ok=True)
    for r in records:
        with open(os.path.join(d, r["name"] + ".json"), "w") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)
    print(f"[{stage}] wrote {len(records)} records to {d}", flush=True)


def load_stage(stage: str) -> list[dict]:
    d = os.path.join(OUT, stage)
    if not os.path.isdir(d):
        return []
    return [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]


def stage_a(workers: int) -> dict:
    specs = specs_for("coexist", G_GRID, PROTOCOL_BOX, SCAN_STEPS)
    recs = scan(specs, workers=workers)
    write_records("scan_coexist", recs)
    return stability_boundary(recs)


def stage_b(workers: int) -> dict:
    specs = specs_for("coexist", G_GRID, PROTOCOL_BOX, SCAN_STEPS, noise=0.02, seed=7)
    recs = scan(specs, workers=workers)
    write_records("scan_coexist_repro", recs)
    return stability_boundary(recs)


def stage_a2(workers: int) -> dict:
    specs = specs_for("water_air", G_GRID, PROTOCOL_BOX, SCAN_STEPS)
    recs = scan(specs, workers=workers)
    write_records("scan_water_air", recs)
    return stability_boundary(recs)


def stage_c(boundary_a: dict, workers: int) -> dict:
    grid = [g for g in (boundary_a["last_stable"], boundary_a["first_unstable"]) if g is not None]
    if not grid:
        grid = G_GRID[:2]
    specs = specs_for("coexist", grid, RES_BOX, SCAN_STEPS)
    recs = scan(specs, workers=max(1, min(workers, len(specs))))
    write_records("resolution_coexist", recs)
    return stability_boundary(recs)


def stage_d(boundary_a: dict, boundary_b: dict, workers: int) -> dict:
    anchors = []
    for fam, bnd in (("coexist", boundary_a), ("water_air", boundary_b)):
        g = bnd["last_stable"]
        if g is None:
            continue
        spec = dict(FAMILIES[fam])
        spec.update(
            name=f"anchor_{case_name(fam, g, PROTOCOL_BOX)}",
            G_12=g,
            steps=ANCHOR_STEPS,
            nz=PROTOCOL_BOX[0],
            ny=PROTOCOL_BOX[1],
            nx=PROTOCOL_BOX[2],
            guo=True,
            sample_every=100,
        )
        anchors.append(spec)
    recs = scan(anchors, workers=max(1, len(anchors)))
    write_records("anchor", recs)
    return {"anchors": [{k: r[k] for k in ("name", "G_12", "steps_run", "diverged", "u_max_final", "bulk", "classification") if k in r} for r in recs]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "a", "b", "a2", "c", "d"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    summary: dict = {"protocol": {"criteria": DEFAULT_CRITERIA, "box": list(PROTOCOL_BOX),
                                 "scan_steps": SCAN_STEPS, "anchor_steps": ANCHOR_STEPS,
                                 "resolution_box": list(RES_BOX), "G_grid": G_GRID}}

    b_a = stage_a(args.workers) if args.stage in ("all", "a") else stability_boundary(load_stage("scan_coexist"))
    b_b = stage_b(args.workers) if args.stage in ("all", "b") else stability_boundary(load_stage("scan_coexist_repro"))
    b_a2 = stage_a2(args.workers) if args.stage in ("all", "a2") else stability_boundary(load_stage("scan_water_air"))
    b_c = stage_c(b_a, args.workers) if args.stage in ("all", "c") else stability_boundary(load_stage("resolution_coexist"))
    d = stage_d(b_a, b_a2, args.workers) if args.stage in ("all", "d") else {}

    summary["boundary_coexist"] = b_a
    summary["boundary_coexist_repro"] = b_b
    summary["boundary_water_air"] = b_a2
    summary["boundary_resolution"] = b_c
    summary["anchors"] = d.get("anchors", [])
    summary["criteria_verdict"] = {
        "boundary_reproducible": (b_a["boundary_estimate"] is not None
                                  and b_a["boundary_estimate"] == b_b["boundary_estimate"]),
        "boundary_repro_delta": (None if b_a["boundary_estimate"] is None or b_b["boundary_estimate"] is None
                                 else abs(b_a["boundary_estimate"] - b_b["boundary_estimate"])),
        "delta_rho_ge_100_in_stable_window": any(
            (a.get("bulk") or {}).get("delta_rho", 0.0) >= 100.0 for a in summary["anchors"]),
        "anchor_5000_steps_ok": bool(summary["anchors"]) and all(
            (not a["diverged"]) and a["steps_run"] >= ANCHOR_STEPS for a in summary["anchors"]),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(json.dumps(summary["criteria_verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
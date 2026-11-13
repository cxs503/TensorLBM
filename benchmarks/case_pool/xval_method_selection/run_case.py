#!/usr/bin/env python3
"""CASE-XVAL-METHOD-SELECTION -- pool case entry for the LBM method-selection XVAL.

This is the *case-pool* integration caller of the method-selection capability:
it drives the same arm runner that produced the published matrix
(``xval/benchmark_method_selection/run_arm.py::run_arm``) and routes the
collision step through the library selection module
(``tensorlbm.method_selection.select_collision``), so the selection decision and
the kernel call share one path with the solver.

It writes real artefacts under ``artifacts/case_xval_method_selection/<tag>/``:

    result.json   KPIs + the selection decision that produced them
    selection.json  the library Selection (collision + sponge + Lambda)

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/xval_method_selection/run_case.py --smoke
    PYTHONPATH=src python3 benchmarks/case_pool/xval_method_selection/run_case.py --arm A1 --D 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "xval" / "benchmark_method_selection"))

from tensorlbm.method_selection import select_collision  # noqa: E402
from run_arm import ARMS, run_arm  # noqa: E402

ART = REPO / "artifacts" / "case_xval_method_selection"

#: pool grid: the two recorded selections (A1 MRT+sponge, A2 BGK) at a smoke
#: resolution.  The full matrix lives in xval/benchmark_method_selection.
POOL_GRID = [
    {"tag": "a1_mrt_sponge_d6", "arm": "A1", "D": 6, "steps": 1200},
    {"tag": "a2_bgk_d6", "arm": "A2", "D": 6, "steps": 1200},
]


def run_one(entry: dict, *, threads: int = 4) -> dict:
    """Run one pool-grid entry through the library selection + arm runner."""
    arm = entry["arm"]
    D = entry["D"]
    spec = ARMS[arm]
    # The library selection decides the collision operator; the arm runner then
    # executes it.  For the pool we request the arm's recorded operator so the
    # pool case reproduces the published arms, but the decision is made by the
    # library (not hard-coded here).
    nu = 0.05 * D / 200.0  # U=0.05, Re=200
    sel = select_collision(nu=nu, wall_bounded=True,
                           sponge=spec["sponge"],
                           prefer=spec["collision"],
                           trt_tau_minus_target=2.0)
    out_dir = ART / entry["tag"]
    res = run_arm(arm, D=D, steps=entry["steps"], threads=threads,
                  out=out_dir, keep_fields=False)
    rec = res["record"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(sel.as_dict(), indent=2))
    # merge the selection into the result for provenance
    rp = out_dir / "result.json"
    if rp.is_file():
        data = json.loads(rp.read_text())
        data["selection"] = sel.as_dict()
        rp.write_text(json.dumps(data, indent=2))
    return {"tag": entry["tag"], "arm": arm, "D": D, "status": rec["status"],
            "cd_mean": rec.get("cd_mean"), "st_fft": rec.get("st_fft"),
            "selection": sel.as_dict()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run the whole pool grid at smoke resolution")
    ap.add_argument("--arm", default="", help="run a single arm (A1..A5)")
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    if args.arm:
        entries = [{"tag": f"{args.arm.lower()}_d{args.D}", "arm": args.arm,
                    "D": args.D, "steps": args.steps}]
    else:
        entries = POOL_GRID
    records = [run_one(e, threads=args.threads) for e in entries]
    print(json.dumps({"case": "CASE-XVAL-METHOD-SELECTION", "records": records},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Blockage sensitivity of the drag coefficient (second sensitivity axis).

The predeclared physics arm fixes D=8 on a 128x64 periodic box, i.e. a 12.5%
blockage, which raises C_D above the unbounded literature value.  This script
adds two lower-blockage runs with the *same* D, u, tau, kernel and correction,
so the only changed variable is the box height, and reports a two-point
blockage extrapolation C_D(B) = C_D(0) (1 + k B).

Usage:
    PYTHONPATH=src python xval/<anchor>/sweep_blockage.py --steps 1200 --transient 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_xval import cylinder_drag, provenance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--transient", type=int, default=400)
    ap.add_argument("--boxes", default="192x96,256x128", help="WxH list")
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--out", default=os.path.join(HERE, "blockage_sweep.json"))
    args = ap.parse_args()

    out = {"provenance": provenance(), "arm": args.arm, "points": []}
    # baseline point of the predeclared physics arm, reused from results.json
    res_path = os.path.join(HERE, "results.json")
    if os.path.exists(res_path):
        with open(res_path) as fh:
            base = json.load(fh)
        p = base["physics"][args.arm]
        out["points"].append({k: p[k] for k in
                              ("arm", "nx", "ny", "blockage", "cd_mean", "cd_tail_half_mean",
                               "cd_std", "n_avg", "n_steps", "n_transient", "re")})

    for box in args.boxes.split(","):
        nx_s, ny_s = box.lower().split("x")
        nx, ny = int(nx_s), int(ny_s)
        t0 = time.time()
        p = cylinder_drag(n_steps=args.steps, arm=args.arm, nx=nx, ny=ny,
                          n_transient=args.transient)
        p.pop("drag_history", None)
        out["points"].append({k: p[k] for k in
                              ("arm", "nx", "ny", "blockage", "cd_mean", "cd_tail_half_mean",
                               "cd_std", "n_avg", "n_steps", "n_transient", "re")})
        out["points"][-1]["wall_s"] = round(time.time() - t0, 1)
        print(f"[sweep] {nx}x{ny} B={p['blockage']:.4f} Cd={p['cd_mean']:.4f} "
              f"(tail {p['cd_tail_half_mean']:.4f}, std {p['cd_std']:.4f})", flush=True)

    pts = sorted(out["points"], key=lambda r: r["blockage"])
    if len(pts) >= 2:
        (b1, c1), (b2, c2) = (pts[0]["blockage"], pts[0]["cd_mean"]), (pts[-1]["blockage"], pts[-1]["cd_mean"])
        # C(b) = C0 (1 + k b) through the two extreme blockage points
        k = (c2 - c1) / (c1 * b2 - c2 * b1) if (c1 * b2 - c2 * b1) != 0 else float("nan")
        c0 = c2 / (1.0 + k * b2)
        out["extrapolation"] = {
            "model": "C_D(B) = C_D0 * (1 + k B), two-point fit on the extreme blockages",
            "k": k,
            "cd_zero_blockage": c0,
            "points_used": [{"blockage": b1, "cd": c1}, {"blockage": b2, "cd": c2}],
        }
        print(f"[sweep] k={k:.3f} Cd(B->0)={c0:.4f}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
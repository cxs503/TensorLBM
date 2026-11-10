#!/usr/bin/env python3
"""Run one P40-N02 AMR flux-conservation case and dump JSON.

    python scripts/run_p40n02_amr_case.py --case tg --out artifacts/p40n02/tg.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.amr_flux_verification import (  # noqa: E402
    CavityConfig,
    CylinderConfig,
    cavity_case,
    cylinder_case,
    reflux_conservation_case,
    taylor_green_case,
)

CASES = ("tg", "reflux", "cavity_amr", "cavity_uniform", "cylinder_amr",
         "cylinder_uniform")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--fluid", type=int, default=64)
    ap.add_argument("--strip", type=int, default=16)
    args = ap.parse_args()

    if args.case == "tg":
        steps = args.steps or 600
        res = taylor_green_case(n=64, re=100.0, u0=0.05, steps=steps, record=10,
                                patch=(16, 16, 32, 32), reflux=False)
    elif args.case == "reflux":
        steps = args.steps or 200
        res = reflux_conservation_case(n=48, re=100.0, u0=0.05, steps=steps,
                                       patch=(16, 16, 16, 16))
    elif args.case.startswith("cavity"):
        cfg = CavityConfig(fluid=args.fluid, strip=args.strip,
                           max_steps=args.max_steps or 60000)
        kind = "amr" if args.case.endswith("amr") else "uniform"
        res = cavity_case(cfg, kind=kind)
        res["config"] = {"fluid": cfg.fluid, "re": cfg.re, "u_lid": cfg.u_lid,
                         "strip": cfg.strip, "tol": cfg.tol,
                         "max_steps": cfg.max_steps}
    else:
        cfg = CylinderConfig()
        if args.steps:
            cfg.steps = args.steps
            cfg.transient = max(1, args.steps // 2)
        kind = "amr" if args.case.endswith("amr") else "uniform"
        res = cylinder_case(cfg, kind=kind)
        res["config"] = {k: getattr(cfg, k) for k in
                         ("d_coarse", "lx_over_d", "ly_over_d", "re", "u_in",
                          "xc_over_d", "patch_pad", "steps", "transient")}

    with open(args.out, "w") as fh:
        json.dump(res, fh)
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

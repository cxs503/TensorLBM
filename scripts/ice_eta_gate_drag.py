"""Stokes drag gate in the shear cell (uniform stream past a fixed sphere).

three resolutions at fixed Re_p = 0.1 -> drag vs Stokes + grid-convergence index

    python scripts/ice_eta_gate_drag.py --out runs/g2 --jobs 3 --threads 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/g2")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--grids", default="2.5:20:27,3:24:33,4:32:43")
    args = ap.parse_args(argv)
    from tensorlbm.ice_suspension_rheology.calibrate import _dispatch

    os.makedirs(args.out, exist_ok=True)
    specs = []
    for g in args.grids.split(","):
        a, box, nz = g.split(":")
        specs.append({"radius": float(a), "box": int(box), "nz": int(nz),
                      "re_p": 0.1, "nu": 0.1, "n_steps": args.steps})
    print(f"[gate-drag] {len(specs)} resolutions", flush=True)
    cases = _dispatch(specs, mode="drag", jobs=args.jobs, threads=args.threads,
                      workdir=os.path.join(args.out, "cases"))
    cases.sort(key=lambda c: c.get("radius", 0))
    gci = None
    ok = [c for c in cases if "drag" in c and c["drag"]]
    if len(ok) >= 2:
        r = ok[-1]["radius"] / ok[-2]["radius"]
        eps = abs(ok[-1]["drag"] - ok[-2]["drag"]) / abs(ok[-2]["drag"])
        gci = float(1.25 * eps / (r**2.0 - 1.0))
    out = {
        "cases": cases,
        "re_p": 0.1,
        "gci": gci,
        "gci_grids": [ok[-2]["radius"], ok[-1]["radius"]] if len(ok) >= 2 else [],
        "gci_formula": "GCI = 1.25 |eps| / (r^p - 1), p = 2, r = a_fine/a_coarse, eps on the drag force",
        "max_err_pct": max((c.get("err_pct", float("nan")) for c in ok), default=float("nan")),
        "max_err_pct_corrected": max((c.get("err_pct_corrected", float("nan")) for c in ok),
                                     default=float("nan")),
        "pass_3pct": bool(ok and max(c["err_pct"] for c in ok) <= 3.0),
        "pass_3pct_corrected": bool(ok and max(c["err_pct_corrected"] for c in ok) <= 3.0),
        "pass_gci_5pct": bool(gci is not None and gci <= 0.05),
        "note": ("reference = Stokes 6 pi mu a_eff U with a_eff from the discretised sphere; "
                 "the channel is tight (beta = a_eff/R ~ 0.14-0.23) so the raw error contains "
                 "the confinement effect, quantified by the Haberman-Sayre centreline factor"),
    }
    json.dump(out, open(os.path.join(args.out, "drag_gate.json"), "w"), indent=2, default=float)
    for c in cases:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()
               if k in ("radius", "a_eff", "beta", "u_stream", "drag", "drag_stokes",
                        "err_pct", "err_pct_corrected", "error")})
    print("gci", gci, "max_err_pct", out["max_err_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Single-sphere drag pre-gate (Stokes + GCI) for the ice-viscosity campaign.

    python scripts/ice_eta_gate.py --out runs/ice_eta_campaign --jobs 3 --threads 4
"""
from __future__ import annotations

import argparse
import json
import os

from tensorlbm.ice_suspension_rheology.calibrate import _stokes_case, _dispatch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ice_eta_campaign")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2200)
    ap.add_argument("--diameters", default="8,12,16")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    ds = [int(x) for x in args.diameters.split(",")]
    specs = [{"d": d, "re_p": 0.1, "steps": args.steps, "domain_factor": 4.0} for d in ds]
    cases = _dispatch(specs, mode="stokes", jobs=args.jobs, threads=args.threads,
                      workdir=os.path.join(args.out, "gate_cases"))
    for c, d in zip(cases, ds):
        c.setdefault("D", int(d))
    cases.sort(key=lambda c: c["D"])
    gci = None
    if len(cases) >= 2 and cases[-2].get("drag"):
        r = cases[-1]["D"] / cases[-2]["D"]
        eps = abs(cases[-1]["drag"] - cases[-2]["drag"]) / abs(cases[-2]["drag"])
        gci = float(1.25 * eps / (r**2.0 - 1.0))
    out = {
        "cases": cases,
        "re_p": 0.1,
        "steps": args.steps,
        "domain_factor": 4.0,
        "gci": gci,
        "gci_grids": [cases[-2]["D"], cases[-1]["D"]] if len(cases) >= 2 else [],
        "gci_formula": "GCI = 1.25 |eps| / (r^p - 1), p = 2, r = D_fine/D_coarse, eps on the drag force",
        "max_err_pct": max(c["err_pct"] for c in cases),
        "pass_3pct": bool(max(c["err_pct"] for c in cases) <= 3.0),
        "pass_gci_5pct": bool(gci is not None and gci <= 0.05),
    }
    json.dump(out, open(os.path.join(args.out, "drag_gate.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

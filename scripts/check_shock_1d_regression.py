"""A5 no-regression gate: the 30-case 1-D normal-shock scan.

The compressible-shock family verification (``benchmarks/verified/
compressible_shock_family``) certifies the 1-D normal-shock relations on
2 variants x 5 shock Mach numbers x 3 grids = 30 cases, with a worst-case error
of 1.53 % (riemann, Ms=10, n=400; see
``docs/COMPRESSIBLE_SHOCK_FAMILY_W18_N07.md``).  Any change to the resolvability
module must leave that number untouched: this script re-runs the *same* 30 cases
through the public case builder and fails if the worst error grows.

Usage::

    python scripts/check_shock_1d_regression.py --out runs/resolvability/a5_shock_1d_scan.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.compressible_shock import analysis, cases  # noqa: E402

MACH_LIST = (2.0, 3.0, 5.0, 8.0, 10.0)
NS_GRIDS = (400, 800, 1600)
VARIANTS = ("riemann", "jump")
#: worst error accepted by A5 (the pre-existing certified number, docs/COMPRESSIBLE_SHOCK_FAMILY_W18_N07.md)
A5_WORST = 0.0153


def one_case(mach: float, n: int, variant: str) -> dict:
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
        fronts.append(analysis.front_x(rho, solver.xc, thr,
                                       x_min=meta["x_diaphragm"] * 0.5))
    fit = analysis.fit_speed(times, fronts)
    st = analysis.sample_state(solver.U, solver.xc,
                               fronts[-1] - 20 * solver.dx,
                               fronts[-1] - 3 * solver.dx)
    jump = analysis.jump_ratios({"rho": rho1, "p": p1, "T": p1 / rho1}, st)
    err = {
        "rho_ratio": analysis.relative_error(jump["rho_ratio"], ref["rho_ratio"]),
        "p_ratio": analysis.relative_error(jump["p_ratio"], ref["p_ratio"]),
        "T_ratio": analysis.relative_error(jump["T_ratio"], ref["T_ratio"]),
        "shock_speed": analysis.relative_error(fit["speed"], ref["W"]),
    }
    return {"variant": variant, "mach": mach, "n": n, "dx": meta["dx"],
            "measured": {"shock_speed": fit["speed"],
                         "rho_ratio": jump["rho_ratio"],
                         "p_ratio": jump["p_ratio"],
                         "T_ratio": jump["T_ratio"]},
            "exact": {"shock_speed": ref["W"], "rho_ratio": ref["rho_ratio"],
                      "p_ratio": ref["p_ratio"], "T_ratio": ref["T_ratio"]},
            "error": err,
            "max_error": float(max(err.values()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/resolvability/a5_shock_1d_scan.json")
    ap.add_argument("--worst-accepted", type=float, default=A5_WORST)
    args = ap.parse_args(argv)

    recs = [one_case(m, n, v) for v in VARIANTS for m in MACH_LIST for n in NS_GRIDS]
    worst = max(recs, key=lambda r: r["max_error"])
    ratios = [max(r["error"][k] for k in ("rho_ratio", "p_ratio", "T_ratio"))
              for r in recs]
    out = {
        "criterion": "worst error over the 30-case 1-D normal-shock scan (A5)",
        "n_cases": len(recs),
        "worst_max_error": worst["max_error"],
        "worst_case": {k: worst[k] for k in ("variant", "mach", "n")},
        "worst_ratios_only": float(max(ratios)),
        "worst_accepted": args.worst_accepted,
        "pass": bool(worst["max_error"] <= args.worst_accepted),
        "records": recs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("n_cases", "worst_max_error", "worst_case", "pass")}, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
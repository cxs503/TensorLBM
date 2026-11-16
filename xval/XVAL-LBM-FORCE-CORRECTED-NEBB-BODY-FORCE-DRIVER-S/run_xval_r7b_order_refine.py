"""Round 7b: order of accuracy of the remaining error, after the drive fix.

Anchor: ``XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S``.

Round 7 measured the achieved centre-line velocity against the *declared*
target and found it 3.1 % high at ny=10 even with the closure-aware
normalisation -- i.e. the normalisation is now consistent, and what is left is
discretisation error of the D3Q19 BGK + Guo closure.  Two predeclared ids
(``predeclared_r7b.json``) separate the two questions:

* **P6** -- the differential factor ``D = ratio(legacy)/ratio(closure)`` must
  equal ``((ny-1)/(ny-2))**2``: the normalisation defect is measured without
  the truncation error that made Round-7's absolute 1 % band fail.
* **P7** -- the absolute error must *converge* at first order in the wall-normal
  spacing ``H = ny-1`` at fixed ``tau`` and fixed lattice Mach number, which is
  what identifies it as truncation error rather than a second definition defect.

Usage
-----
    python run_xval_r7b_order_refine.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_xval_r6_driver_consistency import rel_err_pointwise, run_core_path  # noqa: E402
from nebb_bodyforce_evidence_r7 import ref_profile  # noqa: E402  (benchmarks/xval)

U_MAX = 0.05
TAU = 1.1
NY = (10, 26, 42)
MAX_STEPS = 400_000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "artifacts", "r7"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # ---- P7: refinement study of the residual error ----------------------
    rows = []
    for ny in NY:
        row = run_core_path(ny, "nebb", u_max=U_MAX, tau=TAU, max_steps=MAX_STEPS)
        phys = np.asarray(row["profile_phys"], dtype=np.float64)
        ref = ref_profile("nebb", ny, float(row["nu"]), float(row["body_force"]))
        row.update({
            "H": float(ny - 1),
            "err_pointwise_closed_form": rel_err_pointwise(phys, ref),
            "err_closed_form_stored": float(row["err_phys_pointwise"]),
            "converged": bool(row["steady_residual"] < 1e-12),
        })
        rows.append(row)
        print(f"[refine ny={ny:3d}] H={row['H']:5.1f} steps={row['steps_used']:6d} "
              f"res={row['steady_residual']:.1e} err_pt={row['err_pointwise_closed_form']:.4e} "
              f"ratio_vs_declared={row['ratio_phys_to_target']:.6f}", flush=True)

    rates = []
    for a, b in zip(rows, rows[1:]):
        e0 = a["err_pointwise_closed_form"]
        e1 = b["err_pointwise_closed_form"]
        p = (math.log(e0 / e1) / math.log(b["H"] / a["H"])) if e0 > 0 and e1 > 0 else float("nan")
        rates.append({"ny_from": a["ny"], "ny_to": b["ny"],
                      "H_from": a["H"], "H_to": b["H"],
                      "err_from": e0, "err_to": e1, "rate": float(p)})
        print(f"[rate {a['ny']}->{b['ny']}] err {e0:.3e} -> {e1:.3e}  p={p:.3f}", flush=True)

    p7_ok = all(0.7 <= r["rate"] <= 1.4 for r in rates)

    # ---- P6: differential normalisation factor from the Round-7 arms -----
    r7_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "r7", "drive_width_ab.json")
    p6 = []
    if os.path.exists(r7_path):
        r7 = json.loads(open(r7_path).read())
        arms = {(x["wall_bc"], x["ny"], x["drive"]): x for x in r7["arms"]}
        for ny in (10, 18):
            if ("nebb", ny, "closure_width") not in arms:
                continue
            rc = arms[("nebb", ny, "closure_width")]["ratio_vs_declared"]
            rl = arms[("nebb", ny, "legacy_halfway_width")]["ratio_vs_declared"]
            pred = ((ny - 1) / (ny - 2)) ** 2
            d = rl / rc
            ok = abs(d - pred) < 0.01 * pred
            p6.append({"ny": ny, "D": d, "predicted": pred,
                       "abs_dev": abs(d - pred), "pass": bool(ok)})
            print(f"[P6 ny={ny:3d}] D={d:.6f} pred={pred:.6f} "
                  f"dev={abs(d - pred):.2e} pass={ok}", flush=True)
    p6_ok = bool(p6) and all(x["pass"] for x in p6)

    payload = {
        "anchor": "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S",
        "round": "7b",
        "protocol": {"ny": list(NY), "tau": TAU, "u_max_target": U_MAX,
                     "wall_bc": "nebb", "drive": "closure_width",
                     "steady_tol": 1e-12, "check_every": 200},
        "predeclared": "predeclared_r7b.json",
        "rows": rows,
        "rates": rates,
        "P7_order_pass": bool(p7_ok),
        "P6_differential": p6,
        "P6_pass": bool(p6_ok),
        "ok": bool(p7_ok and p6_ok),
    }
    path = os.path.join(args.out, "order_refine.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"P6={p6_ok} P7={p7_ok} -> {path}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
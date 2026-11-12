"""Diagnostic: DHC Nu convergence in steps (de Vahl Davis Ra=1e4, Pr=0.71).

Root-cause question for audit finding D12: is the DHC Nu failure
(Nu=1.584 vs 2.243, -29.4%) an under-converged run or a physics defect?
"""
from __future__ import annotations

import argparse
import json
import time

from tensorlbm.thermal_verification import DE_VAHL_DAVIS, simulate_dhc_2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--ra", type=float, default=1e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--t-hot", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--report-every", type=int, default=5000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    res = simulate_dhc_2d(
        nx=args.nx, ra=args.ra, pr=args.pr, tau=args.tau,
        t_hot=args.t_hot, t_cold=0.0, steps=args.steps,
        report_every=args.report_every, device="cpu",
    )
    el = time.time() - t0
    H = res.H
    dT = args.t_hot
    ra_eff = res.g_beta * dT * H**3 / (res.nu_lat * res.alpha_lat)
    ref = DE_VAHL_DAVIS.get(args.ra)
    payload = {
        "nx": args.nx, "ra_requested": args.ra, "t_hot": args.t_hot,
        "ra_effective": ra_eff, "g_beta": res.g_beta, "tau_T": res.tau_T,
        "nu_lat": res.nu_lat, "alpha_lat": res.alpha_lat, "H": H,
        "steps": args.steps, "elapsed_s": el,
        "nu_grad2": res.nu_grad2, "nu_grad1": res.nu_grad1,
        "nu_halfway": res.nu_halfway,
        "nu_left_grad2": res.nu_left_grad2, "nu_right_grad2": res.nu_right_grad2,
        "u_max": res.u_max, "T_min": res.T_min, "T_max": res.T_max,
        "last_resid_u": res.last_resid_u, "last_resid_T": res.last_resid_T,
        "ref_devahldavis": ref,
        "err_grad2": (abs(res.nu_grad2 - ref) / ref) if ref else None,
        "history": res.nu_history,
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "history"}, indent=2))
    for h in res.nu_history:
        print(h)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
"""Diagnostic: is the RB Nu failure a spin-up (under-converged) effect or a bug?

Prints Nu_bot/Nu_top, u_max, and the effective Rayleigh number
Ra_eff = g_beta * dT * H^3 / (nu * alpha) every `report_every` steps.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from tensorlbm.thermal_verification import simulate_rayleigh_benard_2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--ra", type=float, default=1e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--t-hot", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--seed-mode", default="random")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    res = simulate_rayleigh_benard_2d(
        nx=args.nx, ny=args.ny, ra=args.ra, pr=args.pr, tau=args.tau,
        t_hot=args.t_hot, t_cold=0.0, steps=args.steps,
        report_every=args.report_every, device="cpu", seed_mode=args.seed_mode,
    )
    el = time.time() - t0
    H = float(args.ny - 1)
    dT = args.t_hot
    ra_eff = res.g_beta * dT * H**3 / (res.nu_lat * res.alpha_lat)
    payload = {
        "nx": args.nx, "ny": args.ny, "ra_requested": args.ra, "t_hot": args.t_hot,
        "dT": dT, "ra_effective": ra_eff, "g_beta": res.g_beta,
        "nu_lat": res.nu_lat, "alpha_lat": res.alpha_lat, "tau_T": res.tau_T,
        "steps": args.steps, "seed_mode": args.seed_mode, "elapsed_s": el,
        "history": res.nu_history,
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "history"}, indent=2))
    print("step      nu_bot    nu_top    nu        u_max      energy_imb")
    for h in res.nu_history:
        print(f"{h['step']:>7d}  {h['nu_bot']:8.4f}  {h['nu_top']:8.4f}  "
              f"{h['nu']:8.4f}  {h['u_max']:.3e}  {h['energy_imbalance']:.4f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Resumable capillary-rise runner (checkpointed chunks).

Why: a 20 000-step CAC capillary-rise run costs ~5 min at (120x64); the agent
shell and some CI hosts cap a single process at ~2 min, and the cluster queue is
shared.  This runner advances the same physical setup as
``benchmarks/case_pool/capillary_rise/jurin_cac.py`` (identical geometry and
measurement chain, geometry copied from
``tensorlbm.wetting.ladder.jurin``) in chunks, checkpointing ``f``/``g`` to a
``.pt`` file so a run can be continued by any number of processes.

    PYTHONPATH=src python3 bench.../chunk_run.py --theta 60 --total-steps 20000 \
        --budget 95 --state /tmp/s60.pt --traj /tmp/t60.jsonl

Prints one JSON line per invocation with the current state.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm.wetting import metrics as M

BASE = dict(rho_l=1.0, rho_g=0.1, mu_l=0.05, mu_g=0.005, sigma=0.03, width=4.0,
            mobility=0.05, mu_law="harmonic", p0=0.01)


def build(theta_deg: float, gap: int, baffle: int, pool: int, shape: tuple[int, int]):
    ny, nx = shape
    mask = cac.solid_plate(shape, 2, "bottom") | cac.solid_plate(shape, 2, "top")
    x_c = nx // 2
    x_left = x_c - gap // 2 - baffle
    x_right = x_c + (gap - gap // 2)
    mask[pool : ny - 2, x_left : x_left + baffle] = True
    mask[pool : ny - 2, x_right : x_right + baffle] = True
    return mask, x_left, x_right


def make_sim(theta_deg: float, gravity: float, shape, mask):
    return cac.CAC2D(
        shape, rho_l=BASE["rho_l"], rho_g=BASE["rho_g"], mu_l=BASE["mu_l"], mu_g=BASE["mu_g"],
        sigma=BASE["sigma"], width=BASE["width"], mobility=BASE["mobility"],
        mu_law=BASE["mu_law"], p0=BASE["p0"], gravity=(0.0, -gravity), gravity_mode="rho",
        axes="wp", wall_mask=mask, wall_phi_policy="contact", contact_angle_deg=theta_deg,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def measure(sim, ch_lo, ch_hi, res_lo, res_hi):
    phi = cac.macro(sim)["phi"].numpy()
    h_ch, h_ch_std = M.meniscus_height(phi, ch_lo, ch_hi)
    h_res, h_res_std = M.meniscus_height(phi, res_lo, res_hi)
    u = cac.macro(sim)["u"]
    speed = float((u[0] ** 2 + u[1] ** 2).sqrt().max().item())
    return {"step": int(sim.step_count), "y_channel": float(h_ch), "y_channel_std": float(h_ch_std),
            "y_reservoir": float(h_res), "y_reservoir_std": float(h_res_std),
            "h": float(h_ch - h_res), "u_max": speed,
            "phi_integral": float(numpy_phi_sum(sim))}


def numpy_phi_sum(sim):
    """Total conserved phase over the fluid nodes (walls carry ghost values)."""
    return float(sim.f.sum(dim=0)[sim.fluid_mask].sum().item())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=60.0)
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--baffle", type=int, default=2)
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--gravity", type=float, default=1.0e-4)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--shape", type=str, default="120,64")
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--budget", type=float, default=95.0, help="wall seconds for this invocation")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    BASE["sigma"] = args.sigma
    ny, nx = (int(v) for v in args.shape.split(","))
    shape = (ny, nx)
    mask, x_left, x_right = build(args.theta, args.gap, args.baffle, args.pool, shape)
    sim = make_sim(args.theta, args.gravity, shape, mask)
    ch_lo, ch_hi = x_left + args.baffle + 1, x_right - 1
    res_lo, res_hi = 4, x_left - 2

    traj = []
    if args.state.exists() and not args.fresh:
        st = torch.load(args.state, map_location="cpu", weights_only=False)
        sim.f, sim.g, sim.step_count = st["f"], st["g"], st["step_count"]
        sim.phi_u_prev = tuple(st["phi_u_prev"])
    else:
        sim.initialize(cac.init_phi_layers(shape, BASE["width"], y0=float(args.pool)))
        sim.step_count = 0
        traj.append(measure(sim, ch_lo, ch_hi, res_lo, res_hi))
    if args.traj.exists() and not args.fresh:
        traj = [json.loads(l) for l in args.traj.read_text().splitlines() if l.strip()]

    t0 = time.time()
    while sim.step_count < args.total_steps:
        n = min(args.sample_every, args.total_steps - sim.step_count)
        cac.run(sim, n, sample_every=0)
        traj.append(measure(sim, ch_lo, ch_hi, res_lo, res_hi))
        if time.time() - t0 > args.budget:
            break
    torch.save({"f": sim.f, "g": sim.g, "step_count": sim.step_count,
                "phi_u_prev": sim.phi_u_prev}, args.state)
    args.traj.write_text("\n".join(json.dumps(t) for t in traj))
    print(json.dumps({"step": sim.step_count, "done": sim.step_count >= args.total_steps,
                      "last": traj[-1], "n_samples": len(traj),
                      "elapsed_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
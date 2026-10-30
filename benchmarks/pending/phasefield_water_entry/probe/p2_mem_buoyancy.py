"""P2 probe — momentum-exchange (MEM) force gate: Archimedes on a static body.

Why this probe exists
---------------------
``tensorlbm.cac_moving_body`` (new) measures the force on a solid with Ladd's
momentum-exchange method and applies moving-wall bounce-back.  Before any
water-entry arm is run, the force *reader* itself must be shown to return a
known answer.  The cheapest independent answer is **Archimedes**: a body at
rest, fully submerged in fluid at rest, must feel exactly

    F_y = rho_l * g * V_body        (V_body in lattice cells, dx = 1)

This is a strict test of (i) the link bookkeeping, (ii) the density used in
the Ladd term, (iii) the pressure coupling of the CAC hydrodynamic
distribution, and (iv) the static-wall bounce next to the body.  It says
nothing about entry dynamics.

Arms
----
* ``submerged_two_phase``  : interface far above the disc, rho ratio 100
* ``submerged_uniform``    : rho_l = rho_g (no interface at all) -> removes any
                             phase-field contribution from the force
* ``no_gravity``           : same geometry, g = 0 -> force must be ~0
                             (residual = spurious currents only)

Not a benchmark; no external reference data are compared beyond F = rho g V.

Run
---
    PYTHONPATH=src python benchmarks/pending/phasefield_water_entry/probe/p2_mem_buoyancy.py \
        --out /tmp/p2_mem_buoyancy.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm import cac_lbm as m
from tensorlbm.cac_moving_body import MovingBodyCAC, disc_mask, wall_box


def run_arm(
    *,
    label: str,
    shape: tuple[int, int, int],
    radius: float,
    centre: tuple[float, float],
    y_interface: float,
    rho_g: float,
    g: float,
    n_steps: int,
    sample_every: int,
    nu: float = 0.1,
    sigma: float = 0.01,
    width: float = 4.0,
    mobility: float = 0.05,
) -> dict:
    wall = wall_box(shape, z_periodic=True)
    sim = m.CACSim(
        shape,
        rho_l=1.0,
        rho_g=rho_g,
        mu_l=nu * 1.0,
        mu_g=nu * rho_g,
        sigma=sigma,
        width=width,
        mobility=mobility,
        p0=0.01,
        gravity=(0.0, -g, 0.0),
        gravity_mode="rho",
        axes="pwp",
        wall_mask=wall,
        wall_phi_policy="freeze",
    )
    sim.initialize(m.init_phi_layers(shape, width=width, y0=y_interface))
    body = disc_mask(shape, centre_xy=centre, radius=radius)
    driver = MovingBodyCAC(sim, static_wall=wall)

    volume = float(body.sum().item())
    expected = 1.0 * g * volume  # rho_l = 1
    t0 = time.time()
    series = []
    for step in range(1, n_steps + 1):
        driver.step(body, (0.0, 0.0, 0.0))
        if step % sample_every == 0 or step == n_steps:
            series.append(
                {
                    "step": step,
                    "fy": driver.force[1],
                    "fx": driver.force[0],
                    "fy_phase_part": driver.force_phase[1],
                    "u_max": m.diagnostics(sim)["u_max_fluid"],
                }
            )
    tail = series[len(series) // 2 :]
    fy_mean = sum(s["fy"] for s in tail) / len(tail)
    return {
        "label": label,
        "shape": list(shape),
        "radius": radius,
        "body_cells": int(volume),
        "rho_g": rho_g,
        "g": g,
        "n_steps": n_steps,
        "expected_fy": expected,
        "fy_mean_second_half": fy_mean,
        "fy_over_expected": (fy_mean / expected) if expected else None,
        "fy_first_sample": series[0]["fy"],
        "fy_last_sample": series[-1]["fy"],
        "fx_max_abs": max(abs(s["fx"]) for s in tail),
        "u_max_last": series[-1]["u_max"],
        "wall_time_s": time.time() - t0,
        "series": series,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/p2_mem_buoyancy.json"))
    ap.add_argument("--n-steps", type=int, default=6000)
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    shape = (4, 96, 64)
    centre = (32.0, 32.0)
    common = dict(
        shape=shape,
        radius=8.0,
        centre=centre,
        n_steps=args.n_steps,
        sample_every=args.sample_every,
    )
    arms = [
        dict(label="submerged_two_phase_g1e-4", y_interface=84.0, rho_g=0.01, g=1e-4),
        dict(label="submerged_uniform_g1e-4", y_interface=84.0, rho_g=1.0, g=1e-4),
        dict(label="submerged_two_phase_g0", y_interface=84.0, rho_g=0.01, g=0.0),
    ]
    out = {"probe": "p2_mem_buoyancy", "model": "cac_moving_body.MovingBodyCAC", "arms": []}
    for arm in arms:
        print(f"[p2] {arm['label']} ...", flush=True)
        res = run_arm(**common, **arm)
        print(
            f"[p2]   Fy={res['fy_mean_second_half']:+.5f} expected={res['expected_fy']:+.5f} "
            f"ratio={res['fy_over_expected'] if res['fy_over_expected'] is None else round(res['fy_over_expected'], 4)} "
            f"| max|u| {res['u_max_last']:.2e} | {res['wall_time_s']:.0f}s",
            flush=True,
        )
        out["arms"].append(res)

    out["notes"] = (
        "Prerequisite gate for the force reader only: static fully submerged body must feel "
        "rho_l g V (Archimedes). It does not validate entry dynamics, added mass, or the "
        "moving-wall bounce-back under motion."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[p2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

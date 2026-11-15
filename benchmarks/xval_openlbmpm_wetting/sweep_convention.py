#!/usr/bin/env python3
"""Sessile-drop sweep: convention x beta x contact angle (cluster entry).

The convention fix (:mod:`tensorlbm.cg_wetting_convention`) changes which angle
the wall boundary condition enforces; the remaining error sources are dynamical
(spurious currents, and a spurious condensation of the wetting phase on the far
wall).  This sweep therefore crosses the convention with the recoloring
amplitude ``beta`` (the anti-diffusive parameter that sharpens the interface and
nucleates spurious droplets from diffuse tails) and reports, per arm:

* ``theta_cap`` / ``theta_circle`` at the last sample (drop component only, see
  ``wall_drop_measure``) and their tail means, with the % error against the
  prescribed angle;
* the wall-blob area (a *diagnostic*, never silently removed);
* ``|u|max`` and the interface-velocity ``dphi`` as convergence indicators.

Sharding: ``--shard i --nshards n`` runs config ``i, i+n, i+2n, ...`` of the
deterministic config list, so a scan can be spread over cluster slots.

Usage
-----
python benchmarks/xval_openlbmpm_wetting/sweep_convention.py \
       --thetas 60,90,120 --betas 0.7,0.3 --conventions physical,reference \
       --steps 8000 --ny 128 --nx 160 --radius 24 --top open --out sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.cg_wetting_geometric import sessile_drop_initial  # noqa: E402
from tensorlbm.color_gradient2d import color_gradient_two_tau_step  # noqa: E402
from tensorlbm.solver import OPPOSITE_2D, stream  # noqa: E402
from wall_drop_measure import measure_wall_drop  # noqa: E402

OPP = tuple(int(v) for v in OPPOSITE_2D)  # canonical map; see R3 doc
TOL_PCT = 15.0


def run_arm(theta: float, convention: str, beta: float, *, steps: int, ny: int, nx: int,
            radius: float, xi: float, A: float, tau: float, top: str,
            sample_every: int, top_theta: float | None = 90.0) -> dict:
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    if top == "wall":
        solid[ny - 1, :] = True
    solid3 = solid.unsqueeze(0)
    solid_np = solid.numpy()
    f_r, f_b = sessile_drop_initial(ny, nx, radius, theta, xi=xi, dtype=torch.float64)
    t0 = time.time()
    hist = []
    prev_phi = None
    for it in range(1, steps + 1):
        kw = dict(tau_r=tau, tau_b=tau, A=A, beta=beta, solid_mask=solid, visc_mode="mixture",
                  wall_contact_angle=float(theta), contact_angle_convention=convention)
        if top_theta is not None and top == "wall":
            kw["wall_contact_angle_by_direction"] = {(0, 1): float(top_theta)}
        f_r, f_b = color_gradient_two_tau_step(f_r, f_b, **kw)
        f_r, f_b = stream(f_r), stream(f_b)
        f_r = torch.where(solid3, f_r[list(OPP)], f_r)
        f_b = torch.where(solid3, f_b[list(OPP)], f_b)
        if it % sample_every == 0:
            rr, rb = f_r.sum(dim=0), f_b.sum(dim=0)
            rho = rr + rb
            phi = ((rr - rb) / rho.clamp(min=1e-12)).detach().cpu().numpy()
            jx = ((f_r + f_b)[1] - (f_r + f_b)[3]).sum(dim=0).detach().cpu().numpy()
            jy = ((f_r + f_b)[2] - (f_r + f_b)[4]).sum(dim=0).detach().cpu().numpy()
            u = np.sqrt((jx / rho.detach().numpy()) ** 2 + (jy / rho.detach().numpy()) ** 2)
            g = measure_wall_drop(phi, solid_np)
            g.update(step=it, u_max=float(u[~solid_np].max()),
                     dphi_max=float(np.abs(phi - prev_phi).max()) if prev_phi is not None else float("nan"),
                     volume=float(phi.sum()))
            prev_phi = phi
            hist.append(g)
    last = hist[-1]
    tail = [h for h in hist if h["step"] > 0.7 * steps] or [last]
    cap_tail = [h["theta_cap_deg"] for h in tail if math.isfinite(h["theta_cap_deg"])]
    circ_tail = [h["theta_circle_deg"] for h in tail if math.isfinite(h["theta_circle_deg"])]
    cap = float(np.mean(cap_tail)) if cap_tail else float("nan")
    circ = float(np.mean(circ_tail)) if circ_tail else float("nan")
    return {
        "theta_cmd": float(theta), "convention": convention, "beta": beta,
        "params": {"steps": steps, "ny": ny, "nx": nx, "radius": radius, "xi": xi,
                   "A": A, "tau": tau, "top": top},
        "theta_cap_tail_deg": cap, "theta_circle_tail_deg": circ,
        "err_pct_cap": 100.0 * (cap - theta) / theta if math.isfinite(cap) else float("nan"),
        "err_pct_circle": 100.0 * (circ - theta) / theta if math.isfinite(circ) else float("nan"),
        "pass15_cap": bool(math.isfinite(cap) and abs(100.0 * (cap - theta) / theta) <= TOL_PCT),
        "pass15_circle": bool(math.isfinite(circ) and abs(100.0 * (circ - theta) / theta) <= TOL_PCT),
        "blob_area_final": last["blob_area"], "drop_area_final": last["drop_area"],
        "u_max_final": last["u_max"], "dphi_max_final": last["dphi_max"],
        "W_final": last["W"], "H_final": last["H"],
        "volume_initial": hist[0]["volume"], "volume_final": last["volume"],
        "wall_time_s": time.time() - t0,
        "history": hist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", default="60,90,120")
    ap.add_argument("--betas", default="0.7,0.3")
    ap.add_argument("--conventions", default="physical,reference")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nx", type=int, default=160)
    ap.add_argument("--radius", type=float, default=24.0)
    ap.add_argument("--xi", type=float, default=2.0)
    ap.add_argument("--A", type=float, default=0.04)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--top", choices=["wall", "open"], default="open")
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="sweep")
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",") if t.strip()]
    betas = [float(b) for b in args.betas.split(",") if b.strip()]
    conventions = [c.strip() for c in args.conventions.split(",") if c.strip()]
    configs = [(th, conv, be) for th in thetas for conv in conventions for be in betas]
    mine = [c for k, c in enumerate(configs) if k % args.nshards == args.shard]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    for th, conv, be in mine:
        r = run_arm(th, conv, be, steps=args.steps, ny=args.ny, nx=args.nx, radius=args.radius,
                    xi=args.xi, A=args.A, tau=args.tau, top=args.top,
                    sample_every=args.sample_every)
        results.append(r)
        print(f"[sweep] {conv:9s} beta={be:.1f} theta={th:5.1f} "
              f"cap={r['theta_cap_tail_deg']:7.2f}({r['err_pct_cap']:+7.2f}%) "
              f"circ={r['theta_circle_tail_deg']:7.2f}({r['err_pct_circle']:+7.2f}%) "
              f"pass15={r['pass15_cap']!s:5s} blob={r['blob_area_final']:5.0f} "
              f"W={r['W_final']:6.2f} H={r['H_final']:6.2f} "
              f"|u|max={r['u_max_final']:.2e} dphi={r['dphi_max_final']:.2e} "
              f"({r['wall_time_s']:.0f}s)", flush=True)

    stamp = f"shard{args.shard}of{args.nshards}"
    path = outdir / f"sweep_{stamp}.json"
    path.write_text(json.dumps({"params": vars(args), "results": results}, indent=2))
    print(f"[sweep] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
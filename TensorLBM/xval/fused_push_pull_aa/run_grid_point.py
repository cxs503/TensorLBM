"""Run one Poiseuille grid point of the fused push-pull XVAL and cache JSON.

Split out of ``run_xval.py`` so a single grid point stays a short (< 2 min)
single-core job, both for local smoke runs and for cluster shards.

    python run_grid_point.py --n-fluid 32 --steps 15000 --out /tmp/pt32.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

import tensorlbm.aa_fused_push_pull as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fluid", type=int, required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--g", type=float, default=1e-6)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    steps = args.steps or int(round(7.0 * args.n_fluid**2 / (9.8696 * args.nu) / 1000.0)) * 1000
    n, nx, g, nu = args.n_fluid, args.nx, args.g, args.nu
    tau = 3.0 * nu + 0.5
    ny = n + 2
    solid = M.channel_masks(nx, ny, 1)
    fluid = ~solid
    force = torch.zeros(2, ny, nx, dtype=torch.float64)
    force[0] = g

    rho0 = torch.ones(ny, nx, dtype=torch.float64)
    z = torch.zeros_like(rho0)
    f = M.equilibrium(rho0, z, z).clone()

    prev = 0.0
    t0 = time.perf_counter()
    for k in range(steps):
        f = M.fused_push_pull_step(f, tau, solid=solid, force=force)
        if k == steps - 1000:
            prev = float(M.macroscopic(f, force)[1].max())
    elapsed = time.perf_counter() - t0

    _, ux, uy = M.macroscopic(f, force)
    prof = ux[fluid].reshape(n, nx).mean(dim=1)
    analytic, u_max = M.analytic_poiseuille(n, g, nu)
    err = (prof - analytic).abs()
    rec = {
        "label": f"n_fluid={n}",
        "grid_fluid_nodes": n,
        "cells_per_step": nx * ny,
        "g": g,
        "nu": nu,
        "tau": tau,
        "steps": steps,
        "u_max_numeric": float(prof.max()),
        "u_max_analytic": float(u_max),
        "u_max_rel_err": float(abs(prof.max() - u_max) / u_max),
        "profile_max_err_over_umax": float(err.max() / u_max),
        "profile_l2_rel_err": float(err.norm() / analytic.norm()),
        "steady_residual_over_umax": float(abs(float(prof.max()) - prev) / u_max),
        "max_uy_over_umax": float(uy[fluid].abs().max() / u_max),
        "elapsed_s": elapsed,
        "mlups": nx * ny * steps / elapsed / 1e6,
        "pass": bool(float(err.norm() / analytic.norm()) < 0.03),
        "profile": prof.tolist(),
        "profile_analytic": analytic.tolist(),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if not k.startswith("profile")}, indent=1))


if __name__ == "__main__":
    main()
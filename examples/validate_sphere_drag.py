"""Sphere drag validation driver -- compares LBM (or DG-band) Cd against
theoretical + experimental benchmarks (NOT against another LBM run).

Usage
-----
    PYTHONPATH=src python examples/validate_sphere_drag.py --re 50 --D 30 --steps 4000
    PYTHONPATH=src python examples/validate_sphere_drag.py --re 100 --D 40 --steps 8000 --gpu

The driver auto-sizes a low-blockage duct (sphere centred in the cross-section,
at x = 0.25*Lx) and reports the time-averaged drag coefficient Cd together with
the reference Cd and the relative error.  A first-order blockage correction is
printed but the *raw* Cd is the primary quantity compared to the infinite-medium
reference.

Reference curve: sphere_drag_reference.cd_reference (Stokes / Schiller-Naumann /
Achenbach).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import os
import torch

from tensorlbm.boundaries3d import (
    apply_simple_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.obstacles import compute_obstacle_forces_3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sphere_drag_reference as REF  # noqa: E402


def run(
    re: float,
    D: int = 30,
    steps: int = 4000,
    u_in: float = 0.1,
    blockage: float = 0.01,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    device: str = "cuda",
    dtype=torch.float32,
):
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable, falling back to cpu")
        device = "cpu"
    dev = torch.device(device)

    R = D / 2.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    # keep BGK stable
    if tau < 0.55:
        u_in = (0.55 - 0.5) / 3.0 / D * re
        nu = u_in * D / re
        tau = 3.0 * nu + 0.5
        print(f"[adjust] tau too low -> u_in reduced to {u_in:.4f} (tau={tau:.3f})")

    # domain sizing for target blockage beta = pi R^2 / (ny*nz)
    if ny is None or nz is None:
        side = int(math.ceil(R * math.sqrt(math.pi / blockage)))
        ny = nz = max(side, D + 4)
    if nx is None:
        nx = max(int(25.0 * D), 4 * D)

    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device=dev)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=dev)

    rho0 = torch.ones(nz, ny, nx, device=dev, dtype=dtype)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev, dtype=dtype)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0)).to(dtype)

    area = math.pi * R * R
    dyn = 0.5 * u_in * u_in * area
    beta = REF.blockage_ratio(D, ny)

    ref_cd = REF.cd_reference(re)
    print(f"[setup] Re={re:.3f} D={D} grid={nx}x{ny}x{nz} tau={tau:.3f} "
          f"u_in={u_in:.4f} blockage(beta)={beta*100:.3f}%  ref Cd={ref_cd:.4f}")

    cd_hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, solid)
        f = apply_simple_channel_boundaries_3d(
            f, u_in=u_in, wall_mask=wall_mask, obstacle_mask=solid
        )
        if step > steps // 2:
            cd_hist.append(abs(float(fx)) / dyn)
        if step % max(1, steps // 10) == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            um = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            print(f"  step {step:5d}  max|u|={um:.4f}  Cd(avg)={avg:.4f}  "
                  f"t={time.time()-t0:.1f}s")

    cd_mean = sum(cd_hist) / max(len(cd_hist), 1)
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist), 1)) ** 0.5
    cd_corr = REF.blockage_correction(cd_mean, beta)
    err_raw = abs(cd_mean - ref_cd) / ref_cd * 100
    err_corr = abs(cd_corr - ref_cd) / ref_cd * 100
    print(f"[RESULT] Re={re:.3f}  Cd(raw)={cd_mean:.4f}+/-{cd_std:.4f}  "
          f"Cd(corr)={cd_corr:.4f}  ref={ref_cd:.4f}  "
          f"err_raw={err_raw:.2f}%  err_corr={err_corr:.2f}%  "
          f"PASS={'YES' if err_corr < 1.0 else 'NO'}")
    return {"re": re, "cd": cd_mean, "cd_corr": cd_corr, "ref": ref_cd,
            "err_raw": err_raw, "err_corr": err_corr, "pass": err_corr < 1.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, required=True)
    ap.add_argument("--D", type=int, default=30)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--u_in", type=float, default=0.1)
    ap.add_argument("--blockage", type=float, default=0.01)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()
    run(re=a.re, D=a.D, steps=a.steps, u_in=a.u_in, blockage=a.blockage,
        nx=a.nx, ny=a.ny, nz=a.nz, device=a.device)

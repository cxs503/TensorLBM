#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""3D sphere drag — grid convergence at Re=100, then high-Re push.

Establishes the 3D baseline (history: Re=100 Cd=1.0651, 2.3% error at
80x40x40 BGK) and pushes to Re=500/1000 with MRT+Smag + far-field BC.
Uniform grid (no AMR) to isolate the drag physics from AMR errors.
"""

import json
import math
import sys
import time

sys.path.insert(0, "/DATA/cxs_host/TensorLBM/src")

import torch

from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


def schiller_naumann_cd(re):
    return 24.0 / re * (1.0 + 0.15 * re**0.687)


def clift_gauvin_cd(re):
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def run_sphere(
    re,
    nx,
    ny,
    nz,
    radius,
    u_in,
    n_steps,
    device="cuda:0",
    collision="smag",
    cs=0.12,
    n_warmup_frac=0.3,
):
    """3D sphere, uniform grid, MEM drag via compute_obstacle_forces_3d.

    IMPORTANT (from calibration): the sphere MUST be centred (cx=0.5*nx)
    and the solid cells MUST NOT be frozen (Ladd MEM needs the normal
    collision on solid cells; freezing corrupts the momentum exchange).
    """
    dev = torch.device(device)
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nx, device=dev),
        indexing="ij",
    )
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2).sqrt() < radius

    A = math.pi * radius**2
    dyn_p = 0.5 * u_in**2 * A
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=dev)
    initial_mass = float(f.sum().item())
    cd_ref = clift_gauvin_cd(re) if re < 2e5 else 0.2

    cd_list = []
    t0 = time.time()
    warmup = int(n_steps * n_warmup_frac)

    for step in range(1, n_steps + 1):
        if collision == "smag":
            col = collide_smagorinsky_mrt3d(f, tau, C_s=cs)
        else:
            from tensorlbm.solver3d import collide_bgk3d

            col = collide_bgk3d(f, tau)
        f = col
        f = stream3d(f)
        # Ladd MEM: after stream, before BB (compute_obstacle_forces_3d)
        fx, _, _ = compute_obstacle_forces_3d(f, solid)
        f = far_field_bc_3d(f, u_in, obstacle_mask=solid)  # includes BB
        if step % 1000 == 0:
            f = f * (initial_mass / f.sum().item())
        if step > warmup:
            cd = float(fx.item()) / dyn_p
            cd_list.append(cd)
        if step % 1000 == 0 or step == n_steps:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            el = time.time() - t0
            print(
                f"  step {step:5d}: Cd={cd_avg:.4f} (ref {cd_ref:.4f}, "
                f"err {abs(cd_avg - cd_ref) / cd_ref * 100:.1f}%), {el:.0f}s"
            )

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    err = abs(cd_avg - cd_ref) / cd_ref * 100
    print(f"\n  FINAL: Cd={cd_avg:.4f} ref={cd_ref:.4f} err={err:.1f}% tau={tau:.5f}")
    return {
        "re": re,
        "grid": [nx, ny, nz],
        "radius": radius,
        "tau": tau,
        "cd": cd_avg,
        "cd_ref": cd_ref,
        "err_pct": err,
        "steps": n_steps,
    }


if __name__ == "__main__":
    device = "cuda:0"
    results = []

    print("=" * 70)
    print("3D sphere drag — baseline + high-Re (uniform grid)")
    print("=" * 70)

    # Baseline: Re=100 centred (calibrated 4.5% at 160x120x120)
    print("\n[Re=100] baseline (centred, no-freeze):")
    r = run_sphere(100, 160, 120, 120, 12.0, 0.06, 3000, device)
    results.append(r)

    # High-Re push with MRT+Smag (LES handles the sub-grid scales)
    print("\n[Re=500]:")
    r = run_sphere(500, 160, 120, 120, 12.0, 0.05, 4000, device)
    results.append(r)

    print("\n[Re=1000]:")
    r = run_sphere(1000, 192, 144, 144, 14.0, 0.05, 5000, device)
    results.append(r)

    print("\n[Re=5000]:")
    r = run_sphere(5000, 224, 168, 168, 16.0, 0.04, 6000, device)
    results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY (3D sphere, uniform grid, MRT+Smag)")
    print("=" * 70)
    for r in results:
        print(
            f"  Re={r['re']:>5}: Cd={r['cd']:.4f} ref={r['cd_ref']:.4f} "
            f"err={r['err_pct']:.1f}%  grid={r['grid']} r={r['radius']} tau={r['tau']:.5f}"
        )

    with open("/tmp/sphere3d_accurate.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nsaved /tmp/sphere3d_accurate.json")

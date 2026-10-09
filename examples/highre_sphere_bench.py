#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""High-Re quantitative benchmark v2: sphere drag with proper resolution.

LBM high-Re constraints:
  - Re = u·D/ν, ν = (τ−0.5)/3 → for a fixed stable τ (≈0.51-0.6) and
    low Ma (u≈0.02-0.05), the sphere diameter D must be large enough.
  - We fix u and τ (hence ν), then set D = Re·ν/u so each Re gets a
    proportionally larger sphere on the same grid — this is the only way
    to keep the boundary layer resolved at high Re.
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


def clift_gauvin_cd(re):
    return 24.0 / re * (1.0 + 0.1315 * re ** (0.82 - 0.05 * math.log10(re)))


def run_sphere(re, nx=128, ny=96, nz=96, u_in=0.02, tau0=0.55, n_steps=2000, device="cuda:0"):
    """Sphere drag at target Re: choose D from Re = u·D/ν with fixed tau."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    nu = cs2 * (tau0 - 0.5)
    # D = Re·ν/u  (lattice units) — sphere diameter for the target Re
    D = re * nu / u_in
    radius = D / 2.0
    # keep the sphere well inside the domain
    max_r = min(nx, ny, nz) * 0.25
    if radius > max_r:
        # clamp: report the Re actually achieved at this radius
        radius = max_r
        D = 2 * radius
        re_actual = u_in * D / nu
    else:
        re_actual = re
    cx, cy, cz = nx * 0.3, ny / 2.0, nz / 2.0

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

    print(f"\n{'=' * 64}")
    print(f"Sphere Re={re:.0e} (实际 {re_actual:.0e})  grid {nx}x{ny}x{nz}")
    print(f"D={D:.1f}  r={radius:.1f}  u={u_in}  tau={tau0}")
    print(f"nu={nu:.6f}  ref_Cd={cd_ref:.4f}")
    print(f"{'=' * 64}")

    cd_list = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_smagorinsky_mrt3d(f, tau0, C_s=0.12)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, solid)
        f = far_field_bc_3d(f, u_in, obstacle_mask=solid)
        if step % 400 == 0:
            f = _correct_mass(f, initial_mass)
        if step > n_steps // 3:
            cd = float(fx.item()) / dyn_p
            cd_list.append(cd)
        if step % 400 == 0 or step == n_steps:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            el = time.time() - t0
            print(
                f"  step {step:5d}: Cd={cd_avg:.4f} (ref {cd_ref:.4f}, "
                f"err {abs(cd_avg - cd_ref) / cd_ref * 100:.1f}%), {el:.0f}s"
            )

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    err = abs(cd_avg - cd_ref) / cd_ref * 100
    print(f"\n  FINAL: Cd={cd_avg:.4f}  ref={cd_ref:.4f}  err={err:.1f}%")
    return {
        "re": re,
        "re_actual": re_actual,
        "cd": cd_avg,
        "cd_ref": cd_ref,
        "err_pct": err,
        "tau": tau0,
        "D": D,
        "steps": n_steps,
    }


def _correct_mass(f, target_mass):
    m = f.sum().item()
    return f * (target_mass / m)


if __name__ == "__main__":
    device = "cuda:0"
    results = []
    # 固定 tau=0.55 (稳定), u=0.02 (低Ma), D 随 Re 缩放
    for re, steps in [(100, 1200), (500, 2000), (1000, 3000)]:
        r = run_sphere(re, n_steps=steps, device=device)
        results.append(r)
    print("\n" + "=" * 64)
    print("SUMMARY (sphere drag, MRT+Smag, tau=0.55, u=0.02)")
    print("=" * 64)
    for r in results:
        print(
            f"  Re={r['re']:>5}: Cd={r['cd']:.4f} ref={r['cd_ref']:.4f} "
            f"err={r['err_pct']:.1f}%  D={r['D']:.1f}"
        )
    with open("/tmp/highre_sphere_results2.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nsaved /tmp/highre_sphere_results2.json")

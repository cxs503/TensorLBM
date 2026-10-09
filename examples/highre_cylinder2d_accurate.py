#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""2D cylinder drag at Re=100/1000 — grid convergence + long-run statistics.

Goal: establish an ACCURATE drag baseline for the high-Re fixes.
  Re=100:  Cd≈1.35, St≈0.165 (literature, e.g. Tritton 1959)
  Re=1000: 2D Cd≈1.5-1.8, St≈0.21 (literature 2D; 3D ≈1.0)
Uses the 2D-validated flow order: collide→freeze→stream→[MEM force]→BB.
"""

import json
import math
import sys
import time

sys.path.insert(0, "/DATA/cxs_host/TensorLBM/src")

import torch

from tensorlbm.boundaries import far_field_bc_2d
from tensorlbm.d2q9 import C as C2D
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import stream
from tensorlbm.turbulence import collide_smagorinsky_mrt


def cylinder_mask(ny, nx, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    return ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt() < radius


def run_cylinder(re, nx, ny, D, u_in, n_steps, device="cuda:0", cs=0.12, n_warmup_frac=0.3):
    """2D cylinder, MRT+Smag, far-field BC, MEM drag + St from Cl zeros."""
    dev = torch.device(device)
    radius = D / 2.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    cx, cy = nx * 0.25, ny * 0.5
    solid = cylinder_mask(ny, nx, cx, cy, radius, dev)

    dyn_p = 0.5 * u_in**2 * D
    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)
    initial_mass = float(f.sum().item())

    cd_list, cl_list, t_shed = [], [], []
    cl_prev = 0.0
    cdev = C2D.to(dev).float()
    t0 = time.time()
    warmup = int(n_steps * n_warmup_frac)

    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_smagorinsky_mrt(f, tau, C_s=cs)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        # MEM force (post-stream, pre-BB) — Ladd 2·Σ c·f on all solid
        fx = 2.0 * (cdev[:, 0].view(9, 1, 1) * f * solid.unsqueeze(0)).sum().item()
        fy = 2.0 * (cdev[:, 1].view(9, 1, 1) * f * solid.unsqueeze(0)).sum().item()
        f = far_field_bc_2d(f, u_in, obstacle_mask=solid)
        if step % 2000 == 0:
            f = f * (initial_mass / f.sum().item())
        if step > warmup:
            cd = fx / dyn_p
            cl = -fy / dyn_p
            cd_list.append(cd)
            cl_list.append(cl)
            if cl_prev * cl < 0:
                t_shed.append(step)
            cl_prev = cl
        if step % 10000 == 0 or step == n_steps:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            el = time.time() - t0
            print(f"  step {step:6d}: Cd={cd_avg:.4f} ({el:.0f}s)")

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    cd_rms = math.sqrt(sum((c - cd_avg) ** 2 for c in cd_list) / max(len(cd_list), 1))
    st = float("nan")
    if len(t_shed) > 3:
        periods = [t_shed[i + 1] - t_shed[i] for i in range(len(t_shed) - 1)]
        T = sum(periods) / len(periods)
        st = D / (T * u_in)
    return {
        "re": re,
        "nx": nx,
        "ny": ny,
        "D": D,
        "tau": tau,
        "cd": cd_avg,
        "cd_rms": cd_rms,
        "st": st,
        "steps": n_steps,
    }


if __name__ == "__main__":
    device = "cuda:0"
    results = []

    print("=" * 70)
    print("2D cylinder drag — grid convergence + long-run statistics")
    print("=" * 70)

    # Re=100: grid convergence (literature Cd≈1.35, St≈0.165)
    print("\n[Re=100] grid convergence:")
    for nx, ny, D in [(100, 50, 16), (160, 80, 16), (240, 120, 16), (320, 160, 16)]:
        r = run_cylinder(100, nx, ny, D, 0.06, 20000, device)
        results.append(r)
        print(f"  grid {nx}x{ny} D={D}: Cd={r['cd']:.4f} (ref 1.35) St={r['st']:.4f} (ref 0.165)")

    # Re=1000: long-run (literature 2D Cd≈1.5-1.8, St≈0.21)
    print("\n[Re=1000] long-run:")
    r = run_cylinder(1000, 320, 160, 16, 0.08, 40000, device)
    results.append(r)
    print(f"  grid 320x160 D=16: Cd={r['cd']:.4f} (ref ~1.5-1.8) St={r['st']:.4f} (ref ~0.21)")

    with open("/tmp/cyl2d_accurate.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nsaved /tmp/cyl2d_accurate.json")

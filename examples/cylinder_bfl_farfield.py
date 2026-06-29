"""2D cylinder with BFL smooth wall + far-field BC — Cd vs Williamson.

The smooth Bouzidi-Firdaouss-Lallemand interpolated bounce-back replaces the
staircased boolean mask; combined with far-field lateral BC (no blockage),
this should bring Cd close to the Williamson reference (1.33 at Re=100).

    PYTHONPATH=src python examples/cylinder_bfl_farfield.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d2q9 import C as C2D, equilibrium, macroscopic
from tensorlbm.boundaries import far_field_bc_2d
from tensorlbm.interpolated_bc import bouzidi_bounce_back, compute_q_circle
from tensorlbm.solver import collide_bgk, stream, correct_mass

REF_CD = 1.33   # Henderson (1995) / Williamson extrapolation, Re=100


def run(nx=320, ny=200, radius=12.0, u_in=0.08, re=100, n_steps=6000, warmup=2000, device="cuda"):
    nu = u_in * 2 * radius / re
    tau = 3 * nu + 0.5
    cx, cy = nx * 0.25, ny * 0.5
    fluid_bc, q_field = compute_q_circle(nx, ny, cx, cy, radius, torch.device(device))
    c_dev = C2D.to(device).float()
    # Boolean mask for solid interior (reset to equilibrium each step)
    yy, xx = torch.meshgrid(torch.arange(ny, device=device, dtype=torch.float32),
                            torch.arange(nx, device=device, dtype=torch.float32), indexing="ij")
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    f_eq_solid = equilibrium(torch.ones(ny, nx, device=device),
                             torch.zeros(ny, nx, device=device),
                             torch.zeros(ny, nx, device=device))

    rho0 = torch.ones(ny, nx, device=device)
    ux0 = torch.full((ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium(rho0, ux0, torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())
    dyn_p = 0.5 * u_in ** 2 * 2 * radius

    cd_samples = []
    print(f"Cylinder BFL+far-field: Re={re} tau={tau:.4f} r={radius:.0f} grid={nx}x{ny}")
    print(f"Reference Cd (Williamson/Henderson) = {REF_CD}\n")

    for step in range(1, n_steps + 1):
        f[:, solid] = f_eq_solid[:, solid]
        f = collide_bgk(f, tau)
        f_prev = f.clone()
        f = stream(f)
        f = far_field_bc_2d(f, u_in=u_in)           # far-field (no blockage, no obstacle BB)
        f_pre = f.clone()
        for d in range(1, 9):
            if bool(fluid_bc[d].any()):
                f = bouzidi_bounce_back(f, f_prev, fluid_bc[d], q_field[d], d)
        if step % 200 == 0:
            f = correct_mass(f, initial_mass)
        if step > warmup:
            fx_b = 0.0
            for d in range(1, 9):
                if bool(fluid_bc[d].any()):
                    delta = f[d][fluid_bc[d]] - f_pre[d][fluid_bc[d]]
                    fx_b -= float((delta * c_dev[d, 0]).sum().item())
            cd_samples.append(fx_b / dyn_p)
        if step % 1000 == 0 or step == n_steps:
            cd = sum(cd_samples) / max(len(cd_samples), 1)
            print(f"  step {step:5d}: Cd={cd:.4f} (ref {REF_CD}, err {abs(cd-REF_CD)/REF_CD*100:.1f}%)")
    cd = sum(cd_samples) / max(len(cd_samples), 1)
    print(f"\nFinal Cd = {cd:.4f}  vs {REF_CD}  (err {abs(cd-REF_CD)/REF_CD*100:.1f}%)")


if __name__ == "__main__":
    run()

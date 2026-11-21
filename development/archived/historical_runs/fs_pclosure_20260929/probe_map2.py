#!/usr/bin/env python3
"""Map the reservoir top layers (y=14..17) to locate H retention."""
from __future__ import annotations
import argparse, math, os
import torch
from tensorlbm import (equilibrium3d, free_surface_step, init_flags_from_fill, init_mass_from_fill)

LIQUID, INTERFACE, GAS = 1, 2, 0


def build_domain(a, device):
    nx, ny, nz = 5 * a, 2 * a + a // 2, 2 * a
    ch = 2 * a
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    fill[1 : a + 1, :ch, 1 : a + 1] = 1.0
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[0, :, :] = True; solid[-1, :, :] = True
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)
    active = (flags == LIQUID) | (flags == INTERFACE)
    z = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(torch.where(active, torch.ones_like(z), z), z, z, z)
    return f, fill, flags, mass, solid, ch


def run(a, g, steps, tau, rho_gas, device):
    dev = torch.device(device)
    f, fill, flags, mass, solid, ch = build_domain(a, dev)
    m0 = float(mass.sum().item())
    ts = math.sqrt(2.0 * g / a)
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g, rho_liquid=1.0,
            rho_gas=rho_gas, surface_tension=0.0, paired_liquid_interface_debit=True)
        if step % 40 == 0 or step == steps:
            wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
            band = wet[:, :, 1 : a // 2 + 1]
            ys = band.any(dim=2).any(dim=0).nonzero(as_tuple=False)
            h = int(ys[-1, 0].item()) if ys.numel() else 0
            print(f"st={step} T={step*ts:.3f} H={h/(2*a):.3f} drift={(float(mass.sum())-m0)/m0:+.2e}", flush=True)
    # map y=13..17 for x=1..a+1 at z=mid
    zmid = ch // 2
    print("  y\\x " + " ".join(f"{x:6d}" for x in range(0, a + 3)))
    for y in range(11, 18):
        row = []
        for x in range(0, a + 3):
            row.append(f"{float(mass[zmid,y,x]):6.2f}")
        print(f"  {y:3d}  " + " ".join(row))
    # across z at x=2, y=15
    print("  z\\ (x=2,y=15):")
    for z in range(0, ch // 2 + 1):
        print(f"    z={z:2d} m={float(mass[z,15,2]):.3f} f={int(flags[z,15,2])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print("env:", {k: v for k, v in os.environ.items() if k.startswith("TL_FS_")})
    run(args.a, args.g, args.steps, args.tau, args.rho_gas, args.device)
#!/usr/bin/env python3
"""A/B the initial mass=0 INTERFACE shell density: rho_liquid vs rho_gas."""
from __future__ import annotations
import argparse, math, os
import torch
from tensorlbm import (equilibrium3d, free_surface_step, init_flags_from_fill,
                       init_mass_from_fill, macroscopic3d)

LIQUID, INTERFACE, GAS, SOLID = 1, 2, 0, 3


def build_domain(a, device, rho_gas, shell):
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
    z = torch.zeros((nz, ny, nx), device=device)
    liq = flags == LIQUID
    ifc = flags == INTERFACE
    rho = torch.zeros((nz, ny, nx), device=device)
    if shell == "liq":
        rho[liq | ifc] = 1.0          # legacy: shell at rho=1
    elif shell == "gas":
        rho[liq] = 1.0
        rho[ifc] = rho_gas            # shell at rho_gas
    elif shell == "fill":
        rho[liq] = 1.0
        rho[ifc] = rho_gas + (1.0 - rho_gas) * fill[ifc]
    f = equilibrium3d(rho, z, z, z)
    return f, fill, flags, mass, solid, ch


def run(a, g, steps, interval, tau, rho_gas, device, shell):
    dev = torch.device(device)
    f, fill, flags, mass, solid, ch = build_domain(a, dev, rho_gas, shell)
    m0 = float(mass.sum().item())
    ts = math.sqrt(2.0 * g / a)
    print(f"[shell={shell}] env:", {k: v for k, v in os.environ.items() if k.startswith("TL_FS_")}, flush=True)
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g, rho_liquid=1.0,
            rho_gas=rho_gas, surface_tension=0.0, paired_liquid_interface_debit=True)
        if step % interval == 0 or step == steps:
            wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
            colsall = wet.any(dim=0).any(dim=0)
            idx = colsall.nonzero(as_tuple=False)
            front = int(idx[-1, 0].item()) if idx.numel() else 0
            band = wet[:, :, 1 : a // 2 + 1]
            ys = band.any(dim=2).any(dim=0).nonzero(as_tuple=False)
            h = int(ys[-1, 0].item()) if ys.numel() else 0
            drift = (float(mass.sum().item()) - m0) / m0
            fin = bool(torch.isfinite(f).all())
            print(f"  st={step:4d} T={step*ts:.3f} X={front/a:.3f} H={h/(2*a):.3f} "
                  f"drift={drift:+.2e} iface={int((flags==INTERFACE).sum())} fin={fin}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--interval", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shell", default="liq")
    args = ap.parse_args()
    run(args.a, args.g, args.steps, args.interval, args.tau, args.rho_gas, args.device, args.shell)
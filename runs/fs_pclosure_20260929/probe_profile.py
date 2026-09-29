#!/usr/bin/env python3
"""Print vertical (y) and horizontal (x/z) profiles of rho/uy/mass/flags."""
from __future__ import annotations
import argparse, math, os
import torch
from tensorlbm import (equilibrium3d, free_surface_step, init_flags_from_fill,
                       init_mass_from_fill, macroscopic3d)

LIQUID, INTERFACE, GAS, SOLID = 1, 2, 0, 3


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
    ts = math.sqrt(2.0 * g / a)
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g, rho_liquid=1.0,
            rho_gas=rho_gas, surface_tension=0.0, paired_liquid_interface_debit=True)
    rho, ux, uy, uz = macroscopic3d(f)
    zc = ch // 4
    print(f"T={steps*ts:.3f}  (after {steps} steps)")
    print("--- vertical profile at x=4,z=4 (interior) and x=1,z=4 (back wall) ---")
    print(" y | x4z4: f    m     rho    uy   | x1z4: f    m     rho    uy")
    for y in range(ch + 1, -1, -1):
        def cell(x, z):
            return (int(flags[z,y,x]), float(mass[z,y,x]), float(rho[z,y,x]), float(uy[z,y,x]))
        A = cell(4, zc); B = cell(1, zc)
        print(f"{y:3d} |        {A[0]} {A[1]:+5.3f} {A[2]:6.3f} {A[3]:+.4f} |        {B[0]} {B[1]:+5.3f} {B[2]:6.3f} {B[3]:+.4f}")
    print("--- horizontal profile at y=12, z=4 ---")
    print(" x | f m rho uy")
    for x in range(0, a + 3):
        print(f"{x:3d} | {int(flags[zc,12,x])} {float(mass[zc,12,x]):+5.3f} {float(rho[zc,12,x]):6.3f} {float(uy[zc,12,x]):+.4f}")
    print("--- horizontal profile at y=15, z=4 ---")
    for x in range(0, a + 3):
        print(f"{x:3d} | {int(flags[zc,15,x])} {float(mass[zc,15,x]):+5.3f} {float(rho[zc,15,x]):6.3f} {float(uy[zc,15,x]):+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.a, args.g, args.steps, args.tau, args.rho_gas, args.device)
#!/usr/bin/env python3
"""Dump an ASCII map of the mid-z fill field at the end of a short run."""
from __future__ import annotations
import argparse, math, os
import torch
from tensorlbm import equilibrium3d, free_surface_step, init_flags_from_fill, init_mass_from_fill

LIQUID, INTERFACE, GAS = 1, 2, 0


def build(a, dev):
    nx, ny, nz = 5 * a, 2 * a + a // 2, 2 * a
    ch = 2 * a
    fill = torch.zeros((nz, ny, nx), device=dev)
    fill[1:a+1, :ch, 1:a+1] = 1.0
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    solid[0, :, :] = True; solid[-1, :, :] = True
    solid[:, 0, :] = True; solid[:, -1, :] = True
    solid[:, :, 0] = True; solid[:, :, -1] = True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, 1.0)
    active = (flags == LIQUID) | (flags == INTERFACE)
    z = torch.zeros((nz, ny, nx), device=dev)
    f = equilibrium3d(torch.where(active, torch.ones_like(z), z), z, z, z)
    return f, fill, flags, mass, solid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--steps2text", default="0,100,200,400")
    ap.add_argument("--rho_gas", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.8)
    args = ap.parse_args()
    dev = torch.device("cpu")
    f, fill, flags, mass, solid = build(args.a, dev)
    marks = {int(x) for x in args.steps2text.split(",")}
    for step in range(0, args.steps + 1):
        if step in marks:
            nz = fill.shape[0]
            ymid = fill.shape[1] // 2
            xmid = fill.shape[2] // 2
            sl = fill[nz // 2]  # (ny, nx)
            fl = flags[nz // 2]
            ms = mass[nz // 2]
            print(f"--- step {step}  (rows y={sl.shape[0]-1}..0 top->bottom) ---", flush=True)
            for y in range(sl.shape[0] - 1, -1, -1):
                row = ""
                for x in range(sl.shape[1]):
                    v = float(sl[y, x])
                    fg = int(fl[y, x])
                    if fg == 1:
                        row += "L" if v >= 0.999 else "l"
                    elif fg == 2:
                        row += "#" if v >= 0.999 else ("+" if v >= 0.5 else ("." if v > 0.01 else ","))
                    else:
                        row += " "
                print(row, flush=True)
            print("  mass vert x=%d (y=1..%d):" % (xmid, sl.shape[0] - 1), flush=True)
            print("   ", [round(float(ms[y, xmid]), 3) for y in range(1, sl.shape[0])], flush=True)
            print("  mass horiz y=2 (x=1..%d):" % (sl.shape[1] - 1), flush=True)
            print("   ", [round(float(ms[2, x]), 3) for x in range(1, sl.shape[1])], flush=True)
        if step == args.steps:
            break
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=args.tau, gy=-args.g,
            rho_liquid=1.0, rho_gas=args.rho_gas, surface_tension=0.0,
            paired_liquid_interface_debit=True,
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Per-step mass-exchange decomposition at the wall-adjacent top cells."""
from __future__ import annotations
import argparse, math, os
import torch
os.environ.setdefault("TL_FS_DIAG_FIELD", "1")
from tensorlbm import (equilibrium3d, free_surface_step, init_flags_from_fill,
                       init_mass_from_fill, macroscopic3d)
import tensorlbm.free_surface_lbm as fsl

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
    zc = ch // 4
    cells = [(1, 15, zc, "x1y15"), (1, 14, zc, "x1y14"), (2, 15, zc, "x2y15"),
             (2, 14, zc, "x2y14"), (4, 15, zc, "x4y15"), (4, 14, zc, "x4y14")]
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g, rho_liquid=1.0,
            rho_gas=rho_gas, surface_tension=0.0, paired_liquid_interface_debit=True)
        if step % 20 == 0 or step == steps:
            D = fsl.__dict__.get("_FS_DIAG")
            _, _, uy, _ = macroscopic3d(f)
            print(f"=== step {step} ===", flush=True)
            for (x, y, z, nm) in cells:
                md = float(D["mass_delta"][z, y, x])
                mdl = float(D["mass_delta_liquid"][z, y, x])
                # count live links
                fl = D["flags"]; fex = D["f_exchange"]; fp = D["f_post"]
                nl = int(D["from_liq"][:, z, y, x].sum())
                ni = int(D["from_iface"][:, z, y, x].sum())
                print(f"  {nm}: f={int(fl[z,y,x])} m={float(D['mass_after_exchange'][z,y,x]):+.3f} "
                      f"md={md:+.5f} md_liq={mdl:+.5f} n_liq={nl} n_if={ni} uy={float(uy[z,y,x]):+.4f}",
                      flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=0.95)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.a, args.g, args.steps, args.tau, args.rho_gas, args.device)
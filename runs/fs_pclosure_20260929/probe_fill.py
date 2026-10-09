#!/usr/bin/env python3
"""H-freeze forensics: vertical mass/fill/flag profile in the reservoir band."""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path
import torch
from tensorlbm import (equilibrium3d, free_surface_step, init_flags_from_fill, init_mass_from_fill)
from tensorlbm.free_surface_lbm import _C

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


def hu(fill, flags, mass, a):
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    band = wet[:, :, 1 : a // 2 + 1]
    ys = band.any(dim=2).any(dim=0).nonzero(as_tuple=False)
    h = int(ys[-1, 0].item()) if ys.numel() else 0
    # mass-based height: topmost y with mass>=0.5 in band
    mband = (mass[:, :, 1 : a // 2 + 1] >= 0.5)
    ys2 = mband.any(dim=2).any(dim=0).nonzero(as_tuple=False)
    hm = int(ys2[-1, 0].item()) if ys2.numel() else 0
    # topmost LIQUID-flagged row in band, and its mass
    lband = (flags[:, :, 1 : a // 2 + 1] == LIQUID)
    ys3 = lband.any(dim=2).any(dim=0).nonzero(as_tuple=False)
    hl = int(ys3[-1, 0].item()) if ys3.numel() else 0
    return h, hm, hl


def run(a, g, steps, interval, tau, rho_gas, device, tag, outdir):
    dev = torch.device(device)
    f, fill, flags, mass, solid, ch = build_domain(a, dev)
    m0 = float(mass.sum().item())
    ts = math.sqrt(2.0 * g / a)
    zmid = ch // 2
    xc = max(1, a // 2)
    print(f"[{tag}] env:", {k: v for k, v in os.environ.items() if k.startswith("TL_FS_")}, flush=True)
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid, mass=mass, tau=tau, gy=-g, rho_liquid=1.0,
            rho_gas=rho_gas, surface_tension=0.0, paired_liquid_interface_debit=True,
        )
        if step % interval == 0 or step == steps:
            h, hm, hl = hu(fill, flags, mass, a)
            drift = (float(mass.sum().item()) - m0) / m0
            fin = bool(torch.isfinite(f).all())
            # top 8 cells profile at xc, zmid: y descending from ch-1
            prof = []
            for y in range(ch, max(ch - 9, 0), -1):
                prof.append((y, int(flags[zmid, y, xc].item()),
                             round(float(mass[zmid, y, xc]), 4),
                             round(float(fill[zmid, y, xc]), 4)))
            print(f"  st={step:4d} T={step*ts:.3f} Hwet={h/(2*a):.3f} Hms={hm/(2*a):.3f} "
                  f"Hliq={hl/(2*a):.3f} iface={int((flags==INTERFACE).sum())} drift={drift:+.2e} fin={fin}",
                  flush=True)
            if step % (4 * interval) == 0 or step == steps:
                for (y, fl, m, fi) in prof:
                    print(f"      y={y:2d} flag={fl} mass={m:.4f} fill={fi:.4f}", flush=True)
            if not fin:
                print("  !! NON-FINITE", flush=True); break
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--interval", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="fill")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    run(args.a, args.g, args.steps, args.interval, args.tau, args.rho_gas,
        args.device, args.tag, Path(args.outdir) if args.outdir else None)
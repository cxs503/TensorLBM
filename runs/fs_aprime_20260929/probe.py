#!/usr/bin/env python3
"""A-prime free-surface closure probe harness (CPU).

Reproduces the dam_break_3d_mm domain and measures X(T), H(T), iface, drift.
Env knobs are forwarded to free_surface_step through TL_FS_* variables.

Usage:
  python probe.py --a 8 --g 0 --steps 200 --interval 25
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from tensorlbm import equilibrium3d, free_surface_step, init_flags_from_fill, init_mass_from_fill

LIQUID = 1
INTERFACE = 2


def build_domain(a: int, device):
    nx, ny, nz = 5 * a, 2 * a + a // 2, 2 * a
    ch = 2 * a
    fill = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    fill[1 : a + 1, :ch, 1 : a + 1] = 1.0
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags, rho_liquid=1.0)
    active = (flags == LIQUID) | (flags == INTERFACE)
    zero = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(
        torch.where(active, torch.ones((nz, ny, nx), device=device), zero),
        zero, zero, zero,
    )
    return f, fill, flags, mass, solid, (nx, ny, nz)


def measure(fill, flags, a: int):
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    cols = wet.any(dim=0).any(dim=0)
    idx = cols.nonzero(as_tuple=False)
    front = int(idx[-1, 0].item()) if idx.numel() else 0
    region = wet[:, :, 1 : a // 2 + 1]
    ys = region.any(dim=2).any(dim=0).nonzero(as_tuple=False)
    h = int(ys[-1, 0].item()) if ys.numel() else 0
    return front, h, int((flags == INTERFACE).sum().item())


def run(a, g, steps, interval, tau, rho_gas, device, outdir, run_name, closure, masked):
    t0 = time.time()
    dev = torch.device(device)
    f, fill, flags, mass, solid, (nx, ny, nz) = build_domain(a, dev)
    m0 = float(mass.sum().item())
    iface0 = int((flags == INTERFACE).sum().item())
    t_scale = math.sqrt(g / a) if g > 0 else 0.0
    series = []
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid,
            mass=mass, tau=tau, gy=-g, rho_liquid=1.0, rho_gas=rho_gas,
            surface_tension=0.0, paired_liquid_interface_debit=True,
            enable_i_to_g_ownership_closure=closure,
        )
        if step % interval == 0 or step == steps:
            front, h, iface = measure(fill, flags, a)
            drift = (float(mass.sum().item()) - m0) / m0
            print(f"  st={step:4d} T={step*t_scale:.3f} X={front/a:.3f} "
                  f"front={front} H={h/(2*a):.3f} iface={iface} drift={drift:+.6%}", flush=True)
            series.append({"step": step, "T": step*t_scale, "front": front, "X": front/a,
                           "h": h, "H": h/(2*a), "mass_drift_rel": drift, "iface_cells": iface})
    elapsed = time.time() - t0
    print(f"[{run_name}] elapsed={elapsed:.1f}s iface {iface0}->{series[-1]['iface_cells']} "
          f"drift={series[-1]['mass_drift_rel']:+.6%} X_end={series[-1]['X']:.3f} H_end={series[-1]['H']:.3f}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{run_name}.json").write_text(json.dumps(
        {"a": a, "g": g, "steps": steps, "rho_gas": rho_gas, "closure": closure,
         "iface_initial": iface0, "iface_growth": series[-1]["iface_cells"]/max(iface0,1),
         "drift_end": series[-1]["mass_drift_rel"], "series": series}, indent=2) + "\n")
    return series


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--interval", type=int, default=25)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--closure", action="store_true")
    ap.add_argument("--outdir", default="runs/fs_aprime_20260929")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    name = args.name or f"probe_a{args.a}_g{args.g:.0e}_rg{args.rho_gas}"
    run(args.a, args.g, args.steps, args.interval, args.tau, args.rho_gas,
        args.device, Path(args.outdir), name, args.closure, False)
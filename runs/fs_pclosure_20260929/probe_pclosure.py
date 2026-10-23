#!/usr/bin/env python3
"""Free-surface pressure-closure diagnostic probe.

Runs the a=8 dam-break column with a given env configuration and reports:
  - X(T), H(T), interface-cell count, mass drift
  - the vertical pressure (rho) profile inside the column at mid x,z
  - the horizontal pressure profile near the bottom
  - max |u| and whether f stays finite

Usage: PYTHONPATH=... python probe_pclosure.py --a 8 --steps 620 --tag NAME
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from tensorlbm import (
    equilibrium3d,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)
from tensorlbm.free_surface_lbm import _C

LIQUID = 1
INTERFACE = 2
GAS = 0


def build_domain(a: int, device: torch.device):
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
    return f, fill, flags, mass, solid, (nx, ny, nz), ch


def measure(fill, flags, a: int):
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    cols = wet.any(dim=0).any(dim=0)
    idx = cols.nonzero(as_tuple=False)
    front = int(idx[-1, 0].item()) if idx.numel() else 0
    region = wet[:, :, 1 : a // 2 + 1]
    ys = region.any(dim=2).any(dim=0).nonzero(as_tuple=False)
    h = int(ys[-1, 0].item()) if ys.numel() else 0
    return front, h, int((flags == INTERFACE).sum().item())


def run(a, g, steps, interval, tau, rho_gas, device, outdir, tag):
    t0 = time.time()
    dev = torch.device(device)
    f, fill, flags, mass, solid, (nx, ny, nz), ch = build_domain(a, dev)
    m0 = float(mass.sum().item())
    iface0 = int((flags == INTERFACE).sum().item())
    t_scale = math.sqrt(2.0 * g / a)
    series = []
    zmid = nz // 2
    xmid = max(1, a // 2)
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid,
            mass=mass, tau=tau, gy=-g, rho_liquid=1.0, rho_gas=rho_gas,
            surface_tension=0.0, paired_liquid_interface_debit=True,
        )
        if step % interval == 0 or step == steps:
            front, h, iface = measure(fill, flags, a)
            drift = (float(mass.sum().item()) - m0) / m0
            rho = f.sum(dim=0)
            finite = bool(torch.isfinite(f).all())
            # velocities (post-step) for drive diagnosis
            _cd = _C.to(dev).float()
            mom = torch.stack(
                [(f * _cd[:, i].view(19, 1, 1, 1)).sum(0) for i in range(3)]
            )  # (3, nz, ny, nx)
            ux = (mom[0] / rho.clamp(min=1e-6))
            uy = (mom[1] / rho.clamp(min=1e-6))
            uz = (mom[2] / rho.clamp(min=1e-6))
            wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
            umag = torch.sqrt(ux**2 + uy**2 + uz**2)
            umax = float(umag[wet].max()) if bool(wet.any()) else 0.0
            # vertical rho profile at column centre (x=xmid, z=zmid), y=1..ch
            vert = [round(float(rho[zmid, y, xmid]), 6) for y in range(1, ch + 1)]
            # horizontal rho profile near bottom (y=2, z=zmid), x=1..3a
            horiz = [round(float(rho[zmid, 2, x]), 6) for x in range(1, 3 * a + 1)]
            ux_horiz = [round(float(ux[zmid, 2, x]), 7) for x in range(1, 3 * a + 1)]
            uy_vert = [round(float(uy[zmid, y, xmid]), 7) for y in range(1, ch + 1)]
            rec = {
                "step": step, "T": step * t_scale, "X": front / a, "H": h / (2 * a),
                "front": front, "h": h, "iface": iface, "drift": drift,
                "finite": finite, "umax": umax,
                "ux_front": float(ux[zmid, 2, min(front, 3 * a)]),
                "rho_vert": vert, "rho_horiz": horiz,
                "ux_horiz": ux_horiz, "uy_vert": uy_vert,
            }
            series.append(rec)
            print(
                f"  st={step:4d} T={step*t_scale:.3f} X={front/a:.3f} H={h/(2*a):.3f} "
                f"iface={iface} drift={drift:+.3e} fin={finite}",
                flush=True,
            )
            if not finite:
                print("  !! NON-FINITE f -- aborting", flush=True)
                break
    elapsed = time.time() - t0
    out = {
        "tag": tag, "a": a, "g": g, "rho_gas": rho_gas, "steps": steps,
        "elapsed_s": elapsed,
        "env": {k: os.environ.get(k) for k in (
            "TL_FS_WALL_MODE", "TL_FS_TOGAS_EPS", "TL_FS_APRIME",
            "TL_FS_ABB_MODE", "TL_FS_HYDRO_COEF", "TL_FS_WALL_HYDRO_COEF", "TL_FS_RECV_GRADED",
        )},
        "series": series,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{tag}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[{tag}] elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=620)
    ap.add_argument("--interval", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="runs/fs_pclosure_20260929")
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()
    run(args.a, args.g, args.steps, args.interval, args.tau,
        args.rho_gas, args.device, Path(args.outdir), args.tag)
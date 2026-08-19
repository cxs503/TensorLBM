#!/usr/bin/env python3
"""3D dam-break benchmark — Martin & Moyce (1952), free-surface (VOF) common module.

Setup: water column a x a x 2a against the x=0 wall (corner), gravity -y,
solid walls on all six faces, rest is gas.  Dimensionless:
    T = t*sqrt(g/a),  X = x_front/a,  H = h_residual/(2a)  (starts at 1.0)
Reference (Martin & Moyce 1952): T=1 -> X~1.5, T=2 -> X~2.7; T=1 -> H~0.8.

Measurements:
  front x(t) = rightmost x-column holding LIQUID or INTERFACE with fill>=0.5;
  residual height h(t) = highest wet cell in original column region
  x in [1, a//2+1); mass drift and interface-cell count are tracked as
  quality-of-simulation diagnostics.

STATUS: NOT VERIFIED (2026-08-19).  The tensorlbm.free_surface_lbm common
module (free_surface_step) has systematic mass-conservation / interface-stability
defects: interface-cell counts explode ~100x and the front position is polluted
(rho_gas=0.1: X~4.75 at T=1.6 vs reference ~2.1; rho_gas=1.0: +48% mass drift).
See /tmp/dambreak_gap.md and benchmarks/pending/dam_break_3d_mm/README.md.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm import (
    equilibrium3d,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
)

LIQUID = 1
INTERFACE = 2

# Martin & Moyce (1952) dimensionless reference curves (a = column side)
MM_X = [(0.0, 1.0), (0.5, 1.2), (1.0, 1.5), (1.5, 2.0), (2.0, 2.7), (2.5, 3.2), (3.0, 3.7)]
MM_H = [(0.0, 1.0), (0.5, 0.92), (1.0, 0.78), (1.5, 0.65), (2.0, 0.55), (2.5, 0.45), (3.0, 0.37)]


def build_domain(a: int, g: float, device: torch.device):
    """Corner column a x a x 2a; walls on all six faces; rest is gas."""
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


def measure(f, fill, flags, mass, a: int):
    """Return (front_x, residual_height_y, interface_cells) in lattice units."""
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    cols = wet.any(dim=0).any(dim=0)
    idx = cols.nonzero(as_tuple=False)
    front = int(idx[-1, 0].item()) if idx.numel() else 0
    region = wet[:, :, 1 : a // 2 + 1]
    rows = region.any(dim=2).nonzero(as_tuple=False)
    h = int(rows[-1, 0].item()) if rows.numel() else 0
    return front, h, int((flags == INTERFACE).sum().item())


def run(a: int, g: float, steps: int, out_interval: int, tau: float,
        rho_gas: float, device: str, outdir: Path, run_name: str) -> dict:
    t0 = time.time()
    dev = torch.device(device)
    f, fill, flags, mass, solid, (nx, ny, nz) = build_domain(a, g, dev)
    m0 = float(mass.sum().item())
    iface0 = int((flags == INTERFACE).sum().item())
    t_scale = math.sqrt(g / a)
    series = []
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f, fill, flags, solid,
            mass=mass, tau=tau, gy=-g, rho_liquid=1.0, rho_gas=rho_gas,
            surface_tension=0.0, paired_liquid_interface_debit=True,
        )
        if step % out_interval == 0 or step == steps:
            front, h, iface = measure(f, fill, flags, mass, a)
            series.append({
                "step": step,
                "T": step * t_scale,
                "front": front,
                "X": front / a,
                "h": h,
                "H": h / (2 * a),
                "mass_drift_rel": (float(mass.sum().item()) - m0) / m0,
                "iface_cells": iface,
            })
    elapsed = time.time() - t0

    def interp(key, T_target):
        pts = [(s["T"], s[key]) for s in series]
        if T_target <= pts[0][0]:
            return pts[0][1]
        for (t1, v1), (t2, v2) in zip(pts, pts[1:]):
            if t1 <= T_target <= t2:
                return v1 + (v2 - v1) * (T_target - t1) / (t2 - t1)
        return pts[-1][1]

    checks = []
    for T_ref, X_ref in [(1.0, 1.5), (2.0, 2.7)]:
        X_sim = interp("X", T_ref)
        checks.append({"T": T_ref, "kind": "X", "ref": X_ref, "sim": X_sim,
                       "err_pct": abs(X_sim - X_ref) / X_ref * 100.0})
    H_sim = interp("H", 1.0)
    checks.append({"T": 1.0, "kind": "H", "ref": 0.8, "sim": H_sim,
                   "err_pct": abs(H_sim - 0.8) / 0.8 * 100.0})

    max_err = max(c["err_pct"] for c in checks)
    final = series[-1]
    result = {
        "case": "dam_break_3d_martin_moyce",
        "status": "NOT_VERIFIED",
        "reason": ("tensorlbm.free_surface_lbm mass-conservation/interface-stability "
                   "defects: interface-cell explosion + front pollution / mass drift; "
                   "see /tmp/dambreak_gap.md"),
        "module": "tensorlbm.free_surface_lbm (D3Q19, free_surface_step) — unmodified",
        "lattice": "D3Q19",
        "collision": "bgk",
        "reference": "Martin & Moyce (1952): T=1 X~1.5, T=2 X~2.7, H(T=1)~0.8",
        "config": {"a": a, "g": g, "tau": tau, "rho_gas": rho_gas, "steps": steps,
                   "domain": {"nx": nx, "ny": ny, "nz": nz, "cells": nx * ny * nz},
                   "T_max": steps * t_scale},
        "elapsed_s": elapsed,
        "checks": checks,
        "max_err_pct": max_err,
        "final": {"mass_drift_rel": final["mass_drift_rel"],
                  "iface_cells": final["iface_cells"],
                  "iface_initial": iface0,
                  "iface_growth": final["iface_cells"] / max(iface0, 1)},
        "series": series,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{run_name}_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[{run_name}] T_max={result['config']['T_max']:.2f} elapsed={elapsed:.1f}s "
          f"max_err={max_err:.1f}% drift={final['mass_drift_rel']:+.2%} "
          f"iface={iface0}->{final['iface_cells']} X(T=1)~{interp('X',1.0):.2f}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=32)
    ap.add_argument("--g", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--interval", type=int, default=25)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--rho_gas", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    run(args.a, args.g, args.steps, args.interval, args.tau, args.rho_gas,
        args.device, Path(args.outdir), f"a{args.a}_g{args.g:.0e}_rg{args.rho_gas}")

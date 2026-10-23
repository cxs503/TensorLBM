#!/usr/bin/env python3
"""3D dam-break benchmark — Martin & Moyce (1952), free-surface (VOF) common module.

Setup: water column a x a x 2a against the x=0 wall (corner), gravity -y,
solid walls on all six faces, rest is gas.  Dimensionless:
    T = t*sqrt(2*g/a),  X = x_front/a,  H = h_residual/(2a)  (starts at 1.0)

Reference convention (DECISIVE ruling, commit 857bbe8): Martin & Moyce (1952)
define the dimensionless time as

        T = t * sqrt(2 g / a)

(confirmed verbatim by Lethe's dam-break-2d.py post-processing
``time_list = [x * ((2 * g / L1) ** 0.5) for x in time_list]`` and by
FrankenSim).  The repo CORE already uses this (dam_break.py:299,
dam_break_3d.py:500); this benchmark historically used T = t*sqrt(g/a), i.e. a
factor sqrt(2) too small — FIXED here.

Reference table — FrankenSim square-column (n^2 = 2, i.e. the a x a x 2a
column solved here), in the SAME T = t*sqrt(2g/a) axis:

    T = [0.41,0.84,1.19,1.43,1.63,1.83,2.00,2.20,2.32,2.51,2.66,2.83,2.95]
    Z = [1.11,1.44,1.78,2.11,2.44,2.78,3.11,3.44,3.67,4.00,4.33,4.67,5.00]

The 3D square column spreads laterally too, so its front Z(T) runs ~15-37%
ahead of the 2D rectangular-column (Lethe/K&O) digitisation at the same T
(ratio 1.17 @T=0.84 -> 1.37 @T=2.95).  The 2D table (T=1->1.326, T=2->2.36,
T=2.95->3.654) is the WRONG reference for this a x a x 2a geometry; the
FrankenSim n^2=2 square-column table is used.  (If the 2D table were used
instead the front error is systematically understated at late T.)

Measurements:
  front x(t) = rightmost x-column holding LIQUID or INTERFACE with fill>=0.5;
  residual height h(t) = highest wet cell (y height) within the original
  column footprint x in [1, a//2+1).  NOTE (bug fixed 2026-09-29): the old
  code did region.any(dim=2).nonzero()[-1,0] which returned the max *z*
  index instead of the y height -> H was a bogus near-constant value.
  x in [1, a//2+1); mass drift and interface-cell count are tracked as
  quality-of-simulation diagnostics.

STATUS: re-run in progress on the corrected axis + FrankenSim square-column
table + the free-surface fix levers (TL_FS_WALL_MODE=halfway,
TL_FS_TOGAS_EPS=1e-6, TL_FS_APRIME=1).  Historical "+50-82%" errors were a
DOUBLE artefact (T-axis sqrt2 + wrong 2D table) — see commit 857bbe8 and
benchmarks/pending/dam_break_recompute_mm.json.
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
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    init_population_from_fill,
)

LIQUID = 1
INTERFACE = 2

# Martin & Moyce (1952) dimensionless reference curves.
#
# X reference — FrankenSim SQUARE-COLUMN digitisation (n^2 = 2, the a x a x 2a
# column solved here), on the correct T = t*sqrt(2g/a) axis.  This is the
# physically matching reference for a finite-width 3D column (the classic
# Lethe / K&O table is for the 2D rectangular column and sits 15-37% lower).
#   T=2.00 -> X=3.11, T=2.95 -> X=5.00;
#   T=1.00 -> X=1.595 (linear interpolation of (0.84,1.44)-(1.19,1.78)).
# (The previous table here was WRONG: the classic T=1->1.1, T=2->1.8,
#  T=3->2.7 digitisation was the 2D-rectangular M&M curve — wrong geometry
#  for this 3D square column — on a T-axis that was itself sqrt(2) too small.)
MM_X = [
    (0.00, 1.00),
    (0.41, 1.11),
    (0.84, 1.44),
    (1.19, 1.78),
    (1.43, 2.11),
    (1.63, 2.44),
    (1.83, 2.78),
    (2.00, 3.11),
    (2.20, 3.44),
    (2.32, 3.67),
    (2.51, 4.00),
    (2.66, 4.33),
    (2.83, 4.67),
    (2.95, 5.00),
]
# H reference — M&M residual-height curve (approximate digitisation, kept from
# the original benchmark; the pass/fail criterion is the X front error).
MM_H = [(0.0, 1.0), (0.5, 0.92), (1.0, 0.78), (1.5, 0.65), (2.0, 0.55), (2.5, 0.45), (3.0, 0.37)]


def _interp_table(tab, x):
    """Linear interpolation / edge-clamped evaluation of a (T, V) table."""
    if x <= tab[0][0]:
        return tab[0][1]
    for (t1, v1), (t2, v2) in zip(tab, tab[1:]):
        if t1 <= x <= t2:
            return v1 + (v2 - v1) * (x - t1) / (t2 - t1)
    return tab[-1][1]


def build_domain(a: int, g: float, device: torch.device, rho_gas: float = 1.0):
    """Corner column a x a x 2a; walls on all six faces; rest is gas.

    Populations are seeded by ``init_population_from_fill`` so the mass=0
    INTERFACE envelope is *gas-consistent* (rho_gas / Körner fill-weighted)
    instead of the historical rho_liquid.  ``TL_FS_SHELL_MODE=legacy`` restores
    the old rho_liquid envelope bit-for-bit.
    """
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
    # Gas-consistent envelope seeding (mass-free shell at rho_gas, partial
    # interface at the Körner fill-weighted density).  Legacy behaviour
    # (rho_liquid everywhere on the active set) is restored with
    # TL_FS_SHELL_MODE=legacy.
    f = init_population_from_fill(fill, flags, rho_liquid=1.0, rho_gas=rho_gas)
    return f, fill, flags, mass, solid, (nx, ny, nz)


def measure(f, fill, flags, mass, a: int):
    """Return (front_x, residual_height_y, interface_cells) in lattice units."""
    wet = (flags == LIQUID) | ((flags == INTERFACE) & (fill >= 0.5))
    cols = wet.any(dim=0).any(dim=0)
    idx = cols.nonzero(as_tuple=False)
    front = int(idx[-1, 0].item()) if idx.numel() else 0
    region = wet[:, :, 1 : a // 2 + 1]
    # Residual water height = highest *y* (height) index of the wet mask inside
    # the original column footprint.  region has shape (nz, ny, nx_region);
    # collapse z and x, then read the y coordinate.
    ys = region.any(dim=2).any(dim=0).nonzero(as_tuple=False)
    h = int(ys[-1, 0].item()) if ys.numel() else 0
    return front, h, int((flags == INTERFACE).sum().item())


def run(
    a: int,
    g: float,
    steps: int,
    out_interval: int,
    tau: float,
    rho_gas: float,
    device: str,
    outdir: Path,
    run_name: str,
) -> dict:
    t0 = time.time()
    dev = torch.device(device)
    f, fill, flags, mass, solid, (nx, ny, nz) = build_domain(a, g, dev)
    m0 = float(mass.sum().item())
    iface0 = int((flags == INTERFACE).sum().item())
    t_scale = math.sqrt(2.0 * g / a)  # T_MM = t * sqrt(2g/a)
    series = []
    for step in range(1, steps + 1):
        f, fill, flags, mass, df = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=tau,
            gy=-g,
            rho_liquid=1.0,
            rho_gas=rho_gas,
            surface_tension=0.0,
            paired_liquid_interface_debit=True,
        )
        if step % out_interval == 0 or step == steps:
            front, h, iface = measure(f, fill, flags, mass, a)
            drift = (float(mass.sum().item()) - m0) / m0
            print(
                f"  st={step:4d} T={step * t_scale:.3f} X={front / a:.3f} "
                f"H={h / (2 * a):.3f} iface={iface} drift={drift:+.4%}",
                flush=True,
            )
            series.append(
                {
                    "step": step,
                    "T": step * t_scale,
                    "front": front,
                    "X": front / a,
                    "h": h,
                    "H": h / (2 * a),
                    "mass_drift_rel": drift,
                    "iface_cells": iface,
                }
            )
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
    T_max_sim = series[-1]["T"]
    # M&M checkpoints on the CORRECT dimensionless-time axis T = t*sqrt(2g/a),
    # against the FrankenSim square-column (n^2=2) reference table.
    for T_ref in (1.0, 2.0, 2.95):
        X_ref = _interp_table(MM_X, T_ref)
        X_sim = interp("X", T_ref)
        extrap = T_ref > T_max_sim + 1e-9
        checks.append(
            {
                "T": T_ref,
                "kind": "X",
                "ref": X_ref,
                "sim": X_sim,
                "err_pct": abs(X_sim - X_ref) / X_ref * 100.0,
                "extrapolated": extrap,
            }
        )
    H_ref_T1 = _interp_table(MM_H, 1.0)
    H_sim = interp("H", 1.0)
    checks.append(
        {
            "T": 1.0,
            "kind": "H",
            "ref": H_ref_T1,
            "sim": H_sim,
            "err_pct": abs(H_sim - H_ref_T1) / H_ref_T1 * 100.0,
            "extrapolated": 1.0 > T_max_sim + 1e-9,
        }
    )

    # Only non-extrapolated checkpoints count towards the pass/fail error.
    valid_errs = [c["err_pct"] for c in checks if not c["extrapolated"]] or [
        max(c["err_pct"] for c in checks)
    ]
    max_err = max(valid_errs)
    final = series[-1]
    result = {
        "case": "dam_break_3d_martin_moyce",
        "status": "PENDING",
        "reason": (
            "T axis corrected to t*sqrt(2g/a); FrankenSim square-column "
            "(n^2=2) reference table; free-surface fix levers "
            "TL_FS_WALL_MODE=halfway + TL_FS_TOGAS_EPS=1e-6 on."
        ),
        "module": "tensorlbm.free_surface_lbm (D3Q19, free_surface_step) — env-gated fix levers",
        "lattice": "D3Q19",
        "collision": "bgk",
        "reference": (
            "Martin & Moyce (1952); T = t*sqrt(2g/a) (Lethe/FrankenSim verbatim); "
            "FrankenSim square-column n^2=2 table: T=1.00 X=1.595, T=2.00 X=3.11, "
            "T=2.95 X=5.00; H(T=1)~0.78"
        ),
        "config": {
            "a": a,
            "g": g,
            "tau": tau,
            "rho_gas": rho_gas,
            "steps": steps,
            "domain": {"nx": nx, "ny": ny, "nz": nz, "cells": nx * ny * nz},
            "T_max": steps * t_scale,
            "env": {
                "TL_FS_WALL_MODE": os.environ.get("TL_FS_WALL_MODE", "legacy"),
                "TL_FS_TOGAS_EPS": os.environ.get("TL_FS_TOGAS_EPS", "0.01"),
                "TL_FS_APRIME": os.environ.get("TL_FS_APRIME", "0"),
            },
        },
        "elapsed_s": elapsed,
        "checks": checks,
        "max_err_pct": max_err,
        "final": {
            "mass_drift_rel": final["mass_drift_rel"],
            "iface_cells": final["iface_cells"],
            "iface_initial": iface0,
            "iface_growth": final["iface_cells"] / max(iface0, 1),
        },
        "series": series,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{run_name}_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[{run_name}] T_max={result['config']['T_max']:.2f} elapsed={elapsed:.1f}s "
        f"max_err={max_err:.1f}% drift={final['mass_drift_rel']:+.2%} "
        f"iface={iface0}->{final['iface_cells']} X(T=1)~{interp('X', 1.0):.2f}"
    )
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
    tag = (
        f"wm{os.environ.get('TL_FS_WALL_MODE', 'legacy')}"
        f"_te{os.environ.get('TL_FS_TOGAS_EPS', '0.01')}"
        f"_ap{os.environ.get('TL_FS_APRIME', '0')}"
    )
    run(
        args.a,
        args.g,
        args.steps,
        args.interval,
        args.tau,
        args.rho_gas,
        args.device,
        Path(args.outdir),
        f"a{args.a}_g{args.g:.0e}_rg{args.rho_gas}_{tag}",
    )

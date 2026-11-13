#!/usr/bin/env python3
"""Cross-validation: anti-bounce-back (ABB) Inamuro wet-node for scalar Dirichlet.

XVAL anchor: XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE
OSS reference: lbm-principles-practice/code (Krüger textbook companion, MIT license)
  - chapter8/film_antibb.cpp:152-205  (anti-bounce-back)
  - chapter8/film_inamuro.cpp:157-215 (Inamuro wet-node)
  - chapter5/couette_wetnode.m:111-146 (wet-node Couette)

Physics: steady 2D channel flow with passive scalar transport.
  - Bottom wall: C = C_bottom (Dirichlet via ABB)
  - Top wall:    C = C_top    (Dirichlet via ABB)
  - Periodic in x, uniform velocity u = U0 in +x
  - Analytic solution: C(y) = C_bottom + (C_top - C_bottom) * y / H_eff

Two ABB variants compared:
  1. Standard ABB (half-way): g_i(x_f) = -g_opp(x_f) + 2*w_i*C_w
     Applied at fluid node adjacent to wall, using post-streaming populations.
  2. Inamuro wet-node: g_i(x_w) = -g_opp(x_w) + 2*w_i*C_w
     Applied at the wall node itself, using pre-streaming populations.
     This is the lbm-principles-practice/code approach.

Both use the same underlying formula (the ABB reconstruction), but differ in
*where* the populations are sampled and *when* (pre- vs post-streaming).

Usage:
    python run.py single H out.json [--tau T] [--u0 U] [--c-bottom CB] [--c-top CT]
    python run.py scan out_dir [--H ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# D2Q5 scalar lattice
# ---------------------------------------------------------------------------
CX_D2Q5 = np.array([0, 1, 0, -1, 0])
CY_D2Q5 = np.array([0, 0, 1, 0, -1])
W_D2Q5 = np.array([2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
OPP_D2Q5 = np.array([0, 3, 4, 1, 2])
CS2_D2Q5 = 1.0 / 3.0

DEVICE = torch.device("cpu")


def equilibrium_scalar_d2q5(c, ux, uy):
    """g_i^eq = w_i * C * (1 + 3 * (c_i . u))"""
    device, dtype = c.device, c.dtype
    cx = torch.tensor(CX_D2Q5, dtype=dtype, device=device).view(5, 1, 1)
    cy = torch.tensor(CY_D2Q5, dtype=dtype, device=device).view(5, 1, 1)
    w = torch.tensor(W_D2Q5, dtype=dtype, device=device).view(5, 1, 1)
    cu = cx * ux + cy * uy
    return w * c * (1.0 + cu / CS2_D2Q5)


def collide_scalar_bgk(g, c, ux, uy, tau):
    """BGK collision for D2Q5 scalar lattice."""
    geq = equilibrium_scalar_d2q5(c, ux, uy)
    return g - (g - geq) / tau


def stream_scalar(g):
    """Pull-streaming for D2Q5 (periodic in both directions)."""
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    cx = torch.tensor(CX_D2Q5, dtype=torch.long, device=device)
    cy = torch.tensor(CY_D2Q5, dtype=torch.long, device=device)
    x = torch.arange(nx, device=device)
    y = torch.arange(ny, device=device)
    sx = (x.view(1, nx) - cx.view(5, 1)) % nx
    sy = (y.view(1, ny) - cy.view(5, 1)) % ny
    q_idx = torch.arange(5, device=device).view(5, 1, 1).expand(5, ny, nx)
    y_idx = sy.unsqueeze(2).expand(5, ny, nx)
    x_idx = sx.unsqueeze(1).expand(5, ny, nx)
    return g[q_idx, y_idx, x_idx]


def scalar_field(g):
    """C = sum_i g_i"""
    return g.sum(dim=0)


# ---------------------------------------------------------------------------
# ABB boundary conditions — clean implementations
# ---------------------------------------------------------------------------

def apply_abb_halfway(g, wall_mask, c_wall):
    """Standard anti-bounce-back (half-way) for scalar Dirichlet.

    Formula: g_i(x_f) = -g_opp(x_f) + 2 * w_i * C_w

    Applied at fluid nodes adjacent to the wall, using post-streaming
    populations.  For each direction q, if the q-neighbour of a fluid
    node is a wall node, we replace g_q at the fluid node with the ABB
    reconstruction.

    This is the textbook ABB form.
    """
    g = g.clone()
    w = torch.tensor(W_D2Q5, dtype=g.dtype, device=g.device)

    for q in range(5):
        cqx, cqy = int(CX_D2Q5[q]), int(CY_D2Q5[q])
        # wall_shifted[x] = wall[x + c_q]: True where the q-neighbour is wall
        wall_shifted = torch.roll(wall_mask, (-cqy, -cqx), dims=(0, 1))
        # Fluid nodes whose q-neighbour is wall
        fluid_adjacent = (~wall_mask) & wall_shifted
        if not fluid_adjacent.any():
            continue
        # g_opp at the fluid node
        opp_q = OPP_D2Q5[q]
        g_opp = g[opp_q]
        # ABB reconstruction: g_q = -g_opp + 2 * w_q * C_w
        g_abb = -g_opp + 2.0 * w[q] * c_wall
        g[q] = torch.where(fluid_adjacent, g_abb, g[q])
    return g


def apply_abb_inamuro_wetnode(g_pre, g_post, wall_mask, c_wall):
    """Inamuro wet-node anti-bounce-back for scalar Dirichlet.

    Formula: g_i(x_w) = -g_opp(x_w) + 2 * w_i * C_w

    Applied at the wall node itself, using pre-streaming populations.
    This is the lbm-principles-practice/code approach (chapter8/film_inamuro.cpp).

    The wall node populations are set to the ABB reconstruction, then
    streamed into the fluid in the next step.
    """
    g = g_post.clone()
    w = torch.tensor(W_D2Q5, dtype=g.dtype, device=g.device)

    for q in range(5):
        opp_q = OPP_D2Q5[q]
        # g_opp at the wall node (from pre-streaming)
        g_opp_w = g_pre[opp_q]
        # ABB reconstruction at wall node
        g_abb = -g_opp_w + 2.0 * w[q] * c_wall
        g[q] = torch.where(wall_mask, g_abb, g[q])
    return g


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_case(H, tau, U0, c_bottom, c_top, abb_variant, max_steps, out_path):
    """Run one scalar Couette case."""
    ny = H + 2  # wall rows at y=0 and y=ny-1
    nx = max(H, 8)
    diffusivity = (tau - 0.5) / 3.0
    H_eff = float(H)
    Pe = U0 * H_eff / diffusivity

    # Wall masks
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True   # bottom wall
    wall[-1, :] = True  # top wall

    wall_bot = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall_bot[0, :] = True
    wall_top = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall_top[-1, :] = True

    # Velocity field (uniform flow, zero at walls)
    ux = torch.full((ny, nx), U0, device=DEVICE)
    uy = torch.zeros((ny, nx), device=DEVICE)
    ux[wall] = 0.0

    # Initial condition: linear profile
    yy = torch.arange(ny, device=DEVICE, dtype=torch.float32)
    c_init = (c_bottom + (c_top - c_bottom) * (yy - 0.5) / H_eff).view(ny, 1).expand(ny, nx).clone()
    c_init[0, :] = c_bottom
    c_init[-1, :] = c_top
    g = equilibrium_scalar_d2q5(c_init, ux, uy)

    # Time stepping
    t0 = time.time()
    for step in range(1, max_steps + 1):
        # Collision
        c = scalar_field(g)
        g = collide_scalar_bgk(g, c, ux, uy, tau)

        # Store pre-streaming for Inamuro wet-node
        g_pre = g.clone()

        # Streaming
        g = stream_scalar(g)

        # ABB boundary condition
        if abb_variant == 'standard':
            g = apply_abb_halfway(g, wall_bot, c_bottom)
            g = apply_abb_halfway(g, wall_top, c_top)
        elif abb_variant == 'inamuro_wetnode':
            g = apply_abb_inamuro_wetnode(g_pre, g, wall_bot, c_bottom)
            g = apply_abb_inamuro_wetnode(g_pre, g, wall_top, c_top)

    elapsed = time.time() - t0

    # Extract profile at mid-column
    col = nx // 2
    c = scalar_field(g)
    c_num = c[1:ny - 1, col].cpu().numpy()  # H fluid rows
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    c_ana = c_bottom + (c_top - c_bottom) * y_phys / H_eff

    # Errors
    l2_rel = float(np.linalg.norm(c_num - c_ana) / max(np.linalg.norm(c_ana), 1e-12))
    max_abs_err = float(np.max(np.abs(c_num - c_ana)))
    max_rel_err = float(np.max(np.abs(c_num - c_ana) / np.maximum(np.abs(c_ana), 1e-12))) * 100.0

    result = {
        "case": "scalar_couette_2d",
        "abb_variant": abb_variant,
        "lattice": "D2Q5",
        "H_eff": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "diffusivity_lb": diffusivity,
        "U0": U0,
        "Pe": Pe,
        "c_bottom": c_bottom,
        "c_top": c_top,
        "max_steps": max_steps,
        "n_steps": max_steps,
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_pct": max_rel_err,
        "c_profile_num": [round(float(v), 8) for v in c_num],
        "c_profile_ana": [round(float(v), 8) for v in c_ana],
        "elapsed_s": round(elapsed, 2),
        "oss_reference": {
            "repo": "lbm-principles-practice/code",
            "license": "MIT",
            "files": [
                "chapter8/film_antibb.cpp:152-205",
                "chapter8/film_inamuro.cpp:157-215",
                "chapter5/couette_wetnode.m:111-146"
            ],
            "formula": "g_i = -g_opp + 2 * w_i * C_w (ABB); applied at wall node (Inamuro wet-node)"
        }
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description="XVAL: ABB Inamuro wet-node for scalar Dirichlet")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("H", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--u0", type=float, default=0.05)
    p1.add_argument("--c-bottom", type=float, default=0.0)
    p1.add_argument("--c-top", type=float, default=1.0)
    p1.add_argument("--abb-variant", choices=["standard", "inamuro_wetnode"], default="inamuro_wetnode")
    p1.add_argument("--max-steps", type=int, default=5000)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--H", type=int, nargs="+", default=[20, 40, 80, 160])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--u0", type=float, default=0.05)
    p2.add_argument("--c-bottom", type=float, default=0.0)
    p2.add_argument("--c-top", type=float, default=1.0)
    p2.add_argument("--max-steps", type=int, default=5000)

    args = ap.parse_args()

    if args.mode == "single":
        r = run_case(
            args.H, args.tau, args.u0, args.c_bottom, args.c_top,
            args.abb_variant, args.max_steps, args.out_json
        )
        print(json.dumps(r, indent=2)[:2000])

    elif args.mode == "scan":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for variant in ["standard", "inamuro_wetnode"]:
            variant_results = []
            for H in args.H:
                p = out_dir / f"case_H{H}_{variant}.json"
                r = run_case(H, args.tau, args.u0, args.c_bottom, args.c_top,
                             variant, args.max_steps, str(p))
                variant_results.append(r)
                print(f"H={H:4d} variant={variant:20s} l2_err={r['l2_rel_err']:.2e} "
                      f"max_rel={r['max_rel_err_pct']:.4f}% Pe={r['Pe']:.1f}")
            results[variant] = variant_results

        summary = {
            "xval_anchor": "XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE",
            "oss_reference": "lbm-principles-practice/code (MIT)",
            "scan_results": results,
            "convergence": {
                variant: [
                    {"H": r["H_eff"], "l2_rel_err": r["l2_rel_err"],
                     "max_rel_err_pct": r["max_rel_err_pct"], "Pe": r["Pe"]}
                    for r in vresults
                ]
                for variant, vresults in results.items()
            }
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\nSummary written to", out_dir / "summary.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Coarse-band (n<=80) water-entry probe with the two-PDF partial-density
colour-gradient (Latva-Kokko / Rothman-Keller) model at a water/air-like
density ratio (1000).

This is a *feasibility* probe: it answers "does the model survive a 1000:1
density contrast in a dynamic (falling-drop) setting, and in which
direction does the penetration depth go relative to the DE ctrl baseline?"

The DE ctrl baseline (free-surface LBM, flag-based topology) is
``1.435220 D @ 11.0906 ms (abs) / 10.5433 (pc)``; the experimental anchor
is ``1.857434 D @ 10.394 ms (pc)``.  The window is ``24.005 ms``.

The probe measures the maximum penetration depth of the heavy phase
(below the initial pool surface) and the time of maximum depth, and
compares them with the ctrl baseline.

Additive: nothing in the existing modules is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorlbm.boundary.boundaries import bounce_back_cells
from tensorlbm.color_gradient_two_pdf import (
    colorgradient_two_pdf_step,
    two_pdf_macros,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import stream


def init_water_entry(
    ny: int,
    nx: int,
    n_cells: int,
    gap: float,
    rho_a0: float,
    rho_b0: float,
    width: float,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Initialise a 2-D water-entry geometry.

    Row 0 is the top, row ny-1 is the bottom.  The pool occupies the bottom
    ``pool_depth`` fraction of the domain; the drop is a circle of heavy
    fluid centred at ``cy`` (row index) above the pool surface ``y0``.

    Returns ``(f, g, y0, cy)``.
    """
    y = torch.arange(ny, dtype=torch.float64, device=dev).view(ny, 1)
    x = torch.arange(nx, dtype=torch.float64, device=dev).view(1, nx)
    y0 = int(round(0.7 * ny))  # pool surface at 70% of the domain height (from top)
    cy = y0 - int(round((gap + 0.5) * n_cells))  # drop centre (above the pool)
    r = n_cells / 2.0
    dy = (y - cy)
    dx = (x - nx / 2.0 + 0.5)
    dist = torch.sqrt(dy**2 + dx**2)
    drop = 0.5 * (1.0 - torch.tanh((dist - r) / max(width, 1e-6)))
    pool = 0.5 * (1.0 - torch.tanh((y0 - y) / max(width, 1e-6)))
    frac_a = torch.maximum(drop, pool)
    rho_a = frac_a * rho_a0
    rho_b = (1.0 - frac_a) * rho_b0
    z = torch.zeros((ny, nx), dtype=torch.float64, device=dev)
    f = equilibrium(rho_a, z, z, dev)
    g = equilibrium(rho_b, z, z, dev)
    return f, g, float(y0), float(cy)


def run_water_entry(
    n_cells: int,
    rho_a0: float,
    rho_b0: float,
    nu_a: float,
    nu_b: float,
    A: float,
    beta: float,
    gy: float,
    gap: float,
    steps: int,
    width: float = 4.0,
    dev: torch.device | None = None,
    sample_every: int = 10,
) -> dict:
    """Run a 2-D water-entry probe and measure the penetration depth."""
    if dev is None:
        dev = torch.device("cpu")
    ny = int(round(3.0 * n_cells))  # domain height = 3 * drop diameter
    nx = int(round(4.0 * n_cells))  # domain width = 4 * drop diameter
    f, g, y0, cy = init_water_entry(
        ny, nx, n_cells, gap, rho_a0, rho_b0, width, dev
    )
    m_a0, m_b0 = float(f.sum()), float(g.sum())

    # wall at the bottom row
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[-1, :] = True

    depth_hist = []
    u_max_hist = []
    nan_step = None
    max_depth = 0.0
    max_depth_step = 0
    for step in range(1, steps + 1):
        f, g = colorgradient_two_pdf_step(
            f, g,
            nu_a=nu_a, nu_b=nu_b, A=A, beta=beta,
            gx=0.0, gy=gy,
            solid_mask=wall,
        )
        f = bounce_back_cells(stream(f), wall)
        g = bounce_back_cells(stream(g), wall)
        if not bool(torch.isfinite(f).all() and torch.isfinite(g).all()):
            nan_step = step
            break
        if step % sample_every == 0:
            rho_a, rho_b, rho, phi, ux, uy = two_pdf_macros(f, g)
            # penetration depth: max row index of the heavy phase below y0
            heavy = rho_a > 0.5 * rho_a0
            below = heavy & (torch.arange(ny, device=dev).view(ny, 1) > y0)
            if bool(below.any()):
                max_row = int(torch.argmax(below.to(torch.int64)).item())
                depth = (max_row - y0) / n_cells  # in units of D
                depth_hist.append((step, depth))
                if depth > max_depth:
                    max_depth = depth
                    max_depth_step = step
            u_max_hist.append((step, spurious_current(f, g)))

    if nan_step is not None:
        return {
            "stable": False, "nan_step": nan_step,
            "n_cells": n_cells, "rho_a0": rho_a0, "rho_b0": rho_b0,
            "density_ratio": rho_a0 / rho_b0,
            "nu_a": nu_a, "nu_b": nu_b, "A": A, "beta": beta,
            "gy": gy, "gap": gap, "steps": steps, "width": width,
        }

    rho_a, rho_b, rho, phi, ux, uy = two_pdf_macros(f, g)
    return {
        "stable": True,
        "nan_step": None,
        "n_cells": n_cells,
        "rho_a0": rho_a0, "rho_b0": rho_b0,
        "density_ratio": rho_a0 / rho_b0,
        "nu_a": nu_a, "nu_b": nu_b, "A": A, "beta": beta,
        "gy": gy, "gap": gap, "steps": steps, "width": width,
        "ny": ny, "nx": nx,
        "y0": y0, "cy": cy,
        "max_depth_D": max_depth,
        "max_depth_step": max_depth_step,
        "u_max": spurious_current(f, g),
        "u_max_hist": u_max_hist,
        "depth_hist": depth_hist,
        "phi_range": [float(phi.min().item()), float(phi.max().item())],
        "min_rho_a": float(rho_a.min().item()),
        "min_rho_b": float(rho_b.min().item()),
        "mass_drift_a_per_step": abs(float(f.sum()) - m_a0) / steps,
        "mass_drift_b_per_step": abs(float(g.sum()) - m_b0) / steps,
    }


def spurious_current(f: torch.Tensor, g: torch.Tensor) -> float:
    _ra, _rb, _rho, _phi, ux, uy = two_pdf_macros(f, g)
    return float(torch.sqrt(ux**2 + uy**2).max().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=40)
    ap.add_argument("--rho-a0", type=float, default=1000.0)
    ap.add_argument("--rho-b0", type=float, default=1.0)
    ap.add_argument("--nu-a", type=float, default=5.0)
    ap.add_argument("--nu-b", type=float, default=5.0)
    ap.add_argument("--A", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gy", type=float, default=1e-4)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--width", type=float, default=4.0)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    dev = torch.device("cpu")
    t0 = time.perf_counter()
    r = run_water_entry(
        a.n_cells, a.rho_a0, a.rho_b0, a.nu_a, a.nu_b,
        a.A, a.beta, a.gy, a.gap, a.steps, a.width, dev,
    )
    r["wall_s"] = time.perf_counter() - t0
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=str))
        print(f"wrote {a.out}")
    print(json.dumps({k: v for k, v in r.items() if k not in ("depth_hist", "u_max_hist")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

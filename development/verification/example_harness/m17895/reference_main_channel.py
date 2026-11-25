#!/usr/bin/env python3
"""Independent control arm, written in the *example-per-main* style.

This file deliberately does NOT import :mod:`tensorlbm.example_harness`.  It is
a single-purpose script with its own lattice construction, its own BGK + Guo
source term, its own streaming (``torch.roll`` shifts instead of the library's
fancy-index gather) and its own bounce-back masks -- i.e. exactly the kind of
duplicated driver the anti-pattern produces.

Its only purpose is to be a *control*: if the consolidated runtime is a faithful
consolidation (and not a silent redefinition of the scheme), both must return
the same Poiseuille profile to round-off.

Usage::

    python reference_main_channel.py --ny 48 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from tensorlbm.d2q9 import C, OPPOSITE, W_EXACT

NY_DEFAULT = 48
NX_DEFAULT = 8
TAU_DEFAULT = 0.8
G_DEFAULT = 1e-5
STEPS_DEFAULT = 20000


def build_mask(ny: int, nx: int) -> torch.Tensor:
    """True where the population in direction i must be reflected.

    Bottom wall row 0 / top wall row ``ny-1``: with pull streaming, node y
    receives direction i from y - c_iy; if that source row is a wall, the
    population is replaced by the post-collision opposite one.
    """
    closed = torch.zeros((9, ny, nx), dtype=torch.bool)
    for i in range(9):
        cy = int(C[i, 1])
        if cy == 0:
            continue
        src = torch.arange(ny).view(ny, 1).expand(ny, nx) - cy
        closed[i] = ((src == 0) | (src == ny - 1)) & (src >= 0) & (src <= ny - 1)
    return closed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=NY_DEFAULT)
    ap.add_argument("--nx", type=int, default=NX_DEFAULT)
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ap.add_argument("--g", type=float, default=G_DEFAULT)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--out", default=None)
    ns = ap.parse_args(argv)

    ny, nx, tau, g = ns.ny, ns.nx, ns.tau, ns.g
    dt = torch.float64
    w = torch.tensor(W_EXACT, dtype=dt).view(9, 1, 1)
    c = C.to(dt)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    # -- initial state: equilibrium at rest ------------------------------
    rho = torch.ones((ny, nx), dtype=dt)
    ux = torch.zeros((ny, nx), dtype=dt)
    uy = torch.zeros((ny, nx), dtype=dt)
    cu = cx * ux + cy * uy
    f = w * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))

    closed = build_mask(ny, nx)

    for _ in range(ns.steps):
        r = f.sum(0)
        ux = (f * cx).sum(0) / r
        uy = (f * cy).sum(0) / r
        # physical (Guo second-order) velocity for the equilibrium
        ux_eq = ux + 0.5 * g / r
        uy_eq = uy
        cu = cx * ux_eq + cy * uy_eq
        feq = w * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux_eq * ux_eq + uy_eq * uy_eq)
        )
        src = (1.0 - 0.5 / tau) * (
            3.0 * w * ((cx - ux_eq) * g) + 9.0 * w * (cu * (cx * g))
        )
        fp = f - (f - feq) / tau + src
        # streaming by roll (pull), then half-way bounce-back
        new = torch.empty_like(fp)
        for i in range(9):
            new[i] = torch.roll(fp[i], shifts=(int(C[i, 1]), int(C[i, 0])), dims=(0, 1))
        for i in range(1, 9):
            if closed[i].any():
                new[i][closed[i]] = fp[OPPOSITE[i]][closed[i]]
        f = new

    r = f.sum(0)
    ux = (f * cx).sum(0) / r + 0.5 * g / r
    uy = (f * cy).sum(0) / r

    H = ny - 2
    nu = (tau - 0.5) / 3.0
    result = {
        "arm": "reference_main_channel",
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "body_force": g,
        "steps": ns.steps,
        "u_max": float(ux[ny // 2, 0].abs()),
        "u_max_analytic": g * H * H / (8.0 * nu),
        "profile": [float(v) for v in ux[1:-1, 0]],
        "rho_min": float(r.min()),
        "rho_max": float(r.max()),
        "mass_drift": float(r.mean() - 1.0),
        "streaming": "torch.roll pull + masked half-way bounce-back",
    }
    result["u_max_rel_err_vs_analytic"] = abs(
        result["u_max"] - result["u_max_analytic"]
    ) / result["u_max_analytic"]
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "profile"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
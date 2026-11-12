#!/usr/bin/env python3
"""3D elliptic-duct Poiseuille: half-way bounce-back vs SDF interpolated bounce-back.

Why this module exists
----------------------
``benchmarks/pending/poiseuille_3d_ellipse/README.md`` (m6422) recorded
``theta = 5.107 % / 2.432 %`` (a = 20 / 40) for the elliptic duct and attributed
the failure to the *staircase anisotropy* of the digital wall: the single
area-equivalent scale ``s^Q`` cannot absorb a wall displacement that is different
at the major-axis tip and on the flat minor-axis side.  The follow-up audit
``docs/POISEUILLE_3D_ELLIPSE_REFERENCE_GEOMETRY_AUDIT.md`` confirmed the
root cause (``delta_eff = +0.35 / +0.31`` cells) but explicitly left the
interpolated-bounce-back (Bouzidi) arm undone:

    文档第 4 节: "未做插值反弹（Bouzidi）对照臂；未做椭圆截面"

This script closes that gap.  It runs **two wall arms on the same domain, same
forcing, same measurement operator** — the production half-way operator
(:func:`tensorlbm.boundaries3d.bounce_back_cells_3d`, no-slip on the staircase
face) and the sub-grid SDF 2-point Bouzidi closure
(:func:`tensorlbm.boundary.interpolated_bounce_back.interpolated_bounce_back_3d`,
no-slip on the *continuous* ellipse).

Physics / analytic reference
----------------------------
Steady fully developed laminar flow in a straight duct of elliptic cross-section
(semi-axes ``a`` along y, ``b`` along z, aspect ratio 2:1), uniform body force
``g`` per unit mass along x, periodic in x (fully developed => x-independent;
``nx`` only has to exceed the 3-point D3Q19 stencil, it is *not* a resolution
knob).  With ``mu = rho nu`` and ``G = rho g`` the pressure gradient,

    u(y, z) = U_max (1 - (y-yc)^2/a^2 - (z-zc)^2/b^2),
    U_max   = g a^2 b^2 / (2 nu (a^2 + b^2)),
    Q       = pi a b U_max / 2                (White, Viscous Fluid Flow, 3rd ed., §3-3)

The digital wall is a staircase (half-way arm) or a set of cut links closed on
the continuous ellipse (IBB arm); both are compared with the *continuous*
reference above, so no reference is fitted to the simulation.

Measurement (identical for both arms)
-------------------------------------
* ``theta_nom``  — per-cell relative error of ``u`` against the *nominal*
  geometry (a, b as given) over ``u_ana > 0.2 U_max``; this is the
  "nominal geometry" honesty metric of the audit doc.
* ``theta_sQ``   — the m6422 judgment metric: one independent integral
  observable (the flux) defines ``s^Q = sqrt(2 Q / (pi a b U_max))``; profiles
  are binned by ``lam = (y/a)^2 + (z/b)^2`` and compared with
  ``U_max (1 - lam/(s^Q)^2)``.
* ``delta_eff``  — from the same flux: ``a_eff = a s^Q``, ``b_eff = b s^Q``
  (isotropic area-equivalent displacement, for comparison with
  ``verified/poiseuille_3d_pipe``: ``R_eff - R = +0.11`` cells).
* ``a_fit / b_fit`` — 2-parameter diagnostic fit ``u = c0 + c1 lam_y + c2 lam_z``
  (quadratic least squares), giving the *anisotropic* wall displacement
  ``(a_fit - a, b_fit - b)``.  Diagnostic only: never used as a reference.

Usage::

    PYTHONPATH=<repo>/src python run_ibb.py --a 20 40 --out ibb_result.json
    PYTHONPATH=<repo>/src python run_ibb.py --a 20 --walls halfway --max-steps 4000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundary.interpolated_bounce_back import interpolated_bounce_back_3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS2 = 1.0 / 3.0
NX = 6  # periodic x; only needs to resolve the D3Q19 stencil
TAU = 0.8


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def ellipse_setup(a: int, b: int, nx: int = NX):
    """Return (sdf, fluid, solid, lam, grid) for an elliptic duct along x.

    Grid (z, y, x) with 1-cell solid margin: ny = 2a+3, nz = 2b+3, centre
    (yc, zc) = (a+1, b+1) — the same lattice layout as the m6422 baseline
    (``run.py::ellipse_setup``), so the two arms and the historic record share
    one geometry convention.

    ``sdf`` is the first-order signed distance to the continuous ellipse,
    ``sdf = -h/|grad h|`` with ``h = ((y-yc)/a)^2 + ((z-zc)/b)^2 - 1``:
    positive in the fluid, negative in the solid, and within ``O(d^2)`` of the
    exact distance next to the wall (the only place the Bouzidi weight needs it).
    """
    ny = 2 * a + 3
    nz = 2 * b + 3
    yc = float(a + 1)
    zc = float(b + 1)
    z = torch.arange(nz, dtype=torch.float64).view(-1, 1)
    y = torch.arange(ny, dtype=torch.float64).view(1, -1)
    dy = (y - yc) / a
    dz = (z - zc) / b
    lam = dy**2 + dz**2
    grad = 2.0 * torch.sqrt((dy / a) ** 2 + (dz / b) ** 2).clamp_min(1e-9)
    sdf2d = -(lam - 1.0) / grad
    fluid2d = lam <= 1.0
    sdf2d = torch.where(fluid2d, sdf2d.clamp_min(1e-6), sdf2d.clamp_max(-1e-6))
    sdf = sdf2d.to(torch.float32).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    fluid = fluid2d.unsqueeze(-1).expand(nz, ny, nx).contiguous()
    solid = ~fluid
    grid = {"ny": ny, "nz": nz, "nx": nx, "yc": yc, "zc": zc, "a": a, "b": b}
    return sdf, fluid, solid, lam.to(torch.float32), grid


def ellipse_g(a: int, b: int, nu: float, u_max_target: float) -> float:
    """Body force that produces the target centre-line velocity ``u_max_target``."""
    return 2.0 * nu * (a * a + b * b) * u_max_target / (a * a * b * b)


def _force_source_3d(fx: torch.Tensor) -> torch.Tensor:
    """Momentum source s_i = 3 w_i (c_i . F); sum_i s_i c_i = F, sum_i s_i = 0."""
    from tensorlbm.d3q19 import C as C19
    from tensorlbm.d3q19 import W as W19

    out = torch.zeros((19,) + tuple(fx.shape), dtype=fx.dtype)
    for i in range(19):
        out[i] = 3.0 * W19[i] * (C19[i, 0] * fx)
    return out


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def measure(ux: np.ndarray, lam: np.ndarray, a: int, b: int, u_max_nom: float) -> dict:
    """All comparison metrics of one arm from the x-averaged ux cross-section."""
    fluid = lam <= 1.0
    U = float(u_max_nom)
    Q = float(ux[fluid].sum())  # flux per unit length (x is periodic, x-averaged)
    s_q = math.sqrt(2.0 * Q / (math.pi * a * b * U))

    u_ana_nom = U * (1.0 - lam)
    core_nom = fluid & (u_ana_nom > 0.2 * U)
    theta_nom = (
        float(np.max(np.abs(ux[core_nom] - u_ana_nom[core_nom]) / u_ana_nom[core_nom]) * 100.0)
        if core_nom.any()
        else float("nan")
    )
    l2_nom = (
        float(np.linalg.norm(ux[fluid] - u_ana_nom[fluid]) / np.linalg.norm(u_ana_nom[fluid]))
        if fluid.any()
        else float("nan")
    )

    # --- s^Q binned metric (m6422 judgment metric, unchanged) ---------------
    nb = int(b)
    bin_idx = np.clip(np.floor(lam * b).astype(int), 0, nb - 1)
    u_ana_cell = U * (1.0 - lam / s_q**2)
    rows = []
    for k in range(nb):
        m = fluid & (bin_idx == k)
        if not m.any():
            continue
        rows.append(
            {
                "bin": k,
                "lam": float(lam[m].mean()),
                "cells": int(m.sum()),
                "u_num": float(ux[m].mean()),
                "u_ana": float(u_ana_cell[m].mean()),
            }
        )
    ana = np.array([r["u_ana"] for r in rows])
    num = np.array([r["u_num"] for r in rows])
    central = ana > 0.2 * U
    theta_sq = (
        float(np.max(np.abs(num[central] - ana[central]) / ana[central]) * 100.0)
        if central.any()
        else float("nan")
    )
    l2_sq = float(np.linalg.norm(num - ana) / np.linalg.norm(ana))

    # --- 2-parameter diagnostic fit (anisotropic wall displacement) ----------
    nz_p, ny_p = lam.shape
    yy = np.abs(np.arange(ny_p) - (a + 1)) / a
    zz = np.abs(np.arange(nz_p) - (b + 1)) / b
    ly2 = np.broadcast_to((yy[np.newaxis, :] ** 2), lam.shape)
    lz2 = np.broadcast_to((zz[:, np.newaxis] ** 2), lam.shape)
    X = np.stack(
        [
            np.ones(len(rows)),
            np.array([ly2[fluid & (bin_idx == r["bin"])].mean() for r in rows]),
            np.array([lz2[fluid & (bin_idx == r["bin"])].mean() for r in rows]),
        ],
        axis=1,
    )
    w = np.sqrt(np.array([r["cells"] for r in rows], dtype=float))
    beta, *_ = np.linalg.lstsq(w[:, None] * X, w * num, rcond=None)
    c0, c1, c2 = (float(v) for v in beta)
    if c1 < 0 and c2 < 0 and c0 > 0:
        a_fit = a * math.sqrt(-c0 / c1)
        b_fit = b * math.sqrt(-c0 / c2)
    else:
        a_fit = b_fit = float("nan")

    return {
        "Q": Q,
        "s_Q": s_q,
        "a_eff": a * s_q,
        "b_eff": b * s_q,
        "delta_a_eff": a * s_q - a,
        "delta_b_eff": b * s_q - b,
        "theta_nom_pct": theta_nom,
        "theta_sQ_pct": theta_sq,
        "l2_nom": l2_nom,
        "l2_sQ": l2_sq,
        "a_fit": a_fit,
        "b_fit": b_fit,
        "delta_a_fit": a_fit - a,
        "delta_b_fit": b_fit - b,
        "u_axis": float(ux[fluid].max()),
        "u_axis_err_pct": (float(ux[fluid].max()) - U) / U * 100.0,
        "bins": rows,
    }


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
def run_arm(
    a: int,
    b: int,
    wall: str,
    *,
    tau: float = TAU,
    u_max_target: float = 0.02,
    nx: int = NX,
    max_steps: int = 12000,
    acc_steps: int = 400,
    tol: float = 1e-6,
    check_every: int = 500,
) -> dict:
    torch.manual_seed(0)
    nu = (tau - 0.5) / 3.0
    g = ellipse_g(a, b, nu, u_max_target)
    sdf, fluid, solid, lam, grid = ellipse_setup(a, b, nx)
    nz, ny = grid["nz"], grid["ny"]

    fx = torch.zeros((nz, ny, nx))
    fx[fluid] = g
    src = _force_source_3d(fx)

    rho0 = torch.ones((nz, ny, nx))
    zero = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, zero, zero, zero)

    def step(fn):
        f_pre = collide_bgk3d(fn, tau) + src
        f_st = stream3d(f_pre)
        if wall == "halfway":
            return bounce_back_cells_3d(f_st, solid)
        if wall == "ibb":
            return interpolated_bounce_back_3d(f_st, f_pre, sdf, solid)
        raise ValueError(f"unknown wall arm {wall!r}")

    t0 = time.time()
    prev = None
    steps = 0
    converged = False
    while steps < max_steps:
        for _ in range(check_every):
            f = step(f)
        steps += check_every
        _, ux, _, _ = macroscopic3d(f)
        u_now = ux.mean(dim=-1)
        if prev is not None:
            drift = float((u_now - prev).abs().max()) / u_max_target
            if drift < tol:
                converged = True
                break
        prev = u_now

    acc = torch.zeros((nz, ny, nx))
    for _ in range(acc_steps):
        _, ux, _, _ = macroscopic3d(f)
        acc += ux
        f = step(f)
    ux_mean = (acc / acc_steps).mean(dim=-1).numpy().astype(np.float64)
    lam_np = lam.numpy().astype(np.float64)

    u_max_nom = g * a * a * b * b / (2.0 * nu * (a * a + b * b))
    met = measure(ux_mean, lam_np, a, b, u_max_nom)
    met.update(
        {
            "case": "poiseuille_3d_ellipse",
            "wall": wall,
            "a": a,
            "b": b,
            "tau": tau,
            "nu": nu,
            "g": g,
            "u_max_nom": u_max_nom,
            "steps": steps,
            "converged": converged,
            "acc_steps": acc_steps,
            "wall_clock_s": time.time() - t0,
            "grid": grid,
            "sdf_note": "first-order signed distance to the continuous ellipse, -h/|grad h|",
        }
    )
    return met, ux_mean, lam_np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, nargs="+", default=[20, 40], help="semi-axes (b = a/2)")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--walls", nargs="+", default=["halfway", "ibb"])
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--u-max", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--acc-steps", type=int, default=400)
    ap.add_argument("--out", type=str, default=str(Path(__file__).with_name("ibb_result.json")))
    args = ap.parse_args()

    out = {
        "case": "poiseuille_3d_ellipse_ibb",
        "reference": "u=U_max(1-y^2/a^2-z^2/b^2), U_max=g a^2 b^2/(2 nu (a^2+b^2)), "
        "Q=pi a b U_max/2 (White, Viscous Fluid Flow §3-3)",
        "arms": [],
    }
    for a in args.a:
        b = int(round(a / args.ratio))
        for wall in args.walls:
            met, _, _ = run_arm(
                a,
                b,
                wall,
                tau=args.tau,
                u_max_target=args.u_max,
                max_steps=args.max_steps,
                acc_steps=args.acc_steps,
            )
            out["arms"].append(met)
            print(
                f"a={a} b={b} wall={wall:8s} steps={met['steps']:6d} "
                f"theta_nom={met['theta_nom_pct']:7.3f}% theta_sQ={met['theta_sQ_pct']:6.3f}% "
                f"delta_eff=({met['delta_a_eff']:+.3f},{met['delta_b_eff']:+.3f}) "
                f"fit=({met['delta_a_fit']:+.3f},{met['delta_b_fit']:+.3f})",
                flush=True,
            )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
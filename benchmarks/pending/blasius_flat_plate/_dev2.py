#!/usr/bin/env python3
"""Flexible Blasius flat-plate experiment harness (D2Q9).

Config knobs
  --wall bottom | mid        plate on bottom wall (solid row 0, x in [le,te))
                             or mid thin plate (solid row ny//2)
  --top  freestream|specular|neumann   far-field BC at y=ny-1
  --outlet zouhe|convective
  --q <Delta>                fractional wall distance from the first fluid
                             node to the wall (BFL interpolation).  q=0.5 is
                             standard half-way BB.

Wall BC is written pre-stream into the solid row (pull-stream convention:
f_2(row+1,x) = f_pre_2(row,x)), so the physical wall sits at q below the first
fluid node and matches the Blasius reference y_w = q (default 0.5).

Library primitives only (collide_bgk/stream/equilibrium/macroscopic,
zou_he_outlet_pressure); inlet/top/wall/symmetry built here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def blasius(eta_max=14.0, h=0.002):
    n = int(round(eta_max / h))
    etas = np.arange(n + 1) * h

    def rhs(s):
        return np.array([s[1], s[2], -0.5 * s[0] * s[2]])

    def shoot(a):
        s = np.array([0.0, 0.0, a])
        for _ in range(n):
            k1 = rhs(s)
            k2 = rhs(s + 0.5 * h * k1)
            k3 = rhs(s + 0.5 * h * k2)
            k4 = rhs(s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return s[1] - 1.0

    lo, hi = 0.30, 0.37
    flo = shoot(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = shoot(mid)
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    a0 = 0.5 * (lo + hi)
    fp = np.zeros(n + 1)
    s = np.array([0.0, 0.0, a0])
    for i in range(1, n + 1):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * h * k1)
        k3 = rhs(s + 0.5 * h * k2)
        k4 = rhs(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        fp[i] = s[1]
    return etas, fp, a0


def bfl_R(f, Delta, jf, jf2, ny):
    """BFL interpolated reflected values at the first fluid row jf (fluid
    above a wall at Delta below jf).  Returns (R2, R5, R6) arrays of length nx."""
    nx = f.shape[2]
    if Delta >= 0.5:
        c1 = 1.0 / (2 * Delta)
        c2 = (2 * Delta - 1.0) / (2 * Delta)
        R2 = c1 * f[4, jf, :] + c2 * f[2, jf2, :]
        # R5 uses f5 at (jf2, x+1) ; R6 uses f6 at (jf2, x-1)
        r5b = torch.zeros_like(R2)
        r5b[:-1] = f[5, jf2, 1:]
        r6b = torch.zeros_like(R2)
        r6b[1:] = f[6, jf2, :-1]
        R5 = c1 * f[8, jf, :] + c2 * r5b
        R6 = c1 * f[7, jf, :] + c2 * r6b
    else:
        lo = 1 - 2 * Delta
        R2 = 2 * Delta * f[4, jf, :] + lo * f[4, jf2, :]
        # node above along the link: (jf2, x+/-1)
        r5b = torch.roll(f[8, jf2, :], -1)  # f8 at (jf2, x+1)
        r6b = torch.roll(f[7, jf2, :], +1)  # f7 at (jf2, x-1)
        R5 = 2 * Delta * f[8, jf, :] + lo * r5b
        R6 = 2 * Delta * f[7, jf, :] + lo * r6b
    return R2, R5, R6


def wall_bc_bottom(f, Delta, mask):
    """Pre-stream BB on bottom solid row 0 under the plate (mask over x)."""
    f = f.clone()
    R2, R5, R6 = bfl_R(f, Delta, 1, 2, f.shape[1])
    f[2, 0, :] = torch.where(mask, R2, f[2, 0, :])
    f[5, 0, :-1] = torch.where(mask[:-1], R5[1:], f[5, 0, :-1])
    f[6, 0, 1:] = torch.where(mask[1:], R6[:-1], f[6, 0, 1:])
    return f


def wall_bc_mid(f, Delta, ys):
    """Pre-stream BB on mid thin plate solid row ys (both surfaces)."""
    f = f.clone()
    # upper surface: fluid above, wall at ys+Delta
    R2, R5, R6 = bfl_R(f, Delta, ys + 1, ys + 2, f.shape[1])
    f[2, ys, :] = R2
    f[5, ys, :-1] = R5[1:]
    f[6, ys, 1:] = R6[:-1]
    # lower surface: fluid below, wall at ys-Delta
    c1 = 1.0 / (2 * Delta) if Delta >= 0.5 else 2 * Delta
    if Delta >= 0.5:
        c2 = (2 * Delta - 1.0) / (2 * Delta)
        R4 = c1 * f[2, ys - 1, :] + c2 * f[4, ys - 2, :]
        r7b = torch.zeros_like(R4)
        r7b[:-1] = f[7, ys - 2, 1:]
        r8b = torch.zeros_like(R4)
        r8b[1:] = f[8, ys - 2, :-1]
        R7 = c1 * f[5, ys - 1, :] + c2 * r7b
        R8 = c1 * f[6, ys - 1, :] + c2 * r8b
    else:
        lo = 1 - 2 * Delta
        R4 = 2 * Delta * f[2, ys - 1, :] + lo * f[2, ys - 2, :]
        R7 = 2 * Delta * f[5, ys - 1, :] + lo * torch.roll(f[5, ys - 2, :], -1)
        R8 = 2 * Delta * f[6, ys - 1, :] + lo * torch.roll(f[6, ys - 2, :], +1)
    f[4, ys, :] = R4
    f[7, ys, 1:] = R7[:-1]
    f[8, ys, :-1] = R8[1:]
    return f


def run(args):
    nx, ny, le, pl, probe = args.nx, args.ny, args.le, args.plate_len, args.probe
    U, nu = args.U, args.nu
    tau = 3 * nu + 0.5
    dev = torch.device(args.device)
    torch.manual_seed(0)
    te = le + pl

    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    if args.wall == "bottom":
        y_solid = 0
    else:
        y_solid = ny // 2
    solid[y_solid, le:te] = True
    mask = solid[y_solid]

    f = equilibrium(
        torch.ones((ny, nx), device=dev),
        torch.full((ny, nx), U, device=dev),
        torch.zeros((ny, nx), device=dev),
    )
    if getattr(args, "seed", 0.0) > 0.0:
        # seed a Blasius-like boundary layer so the plate does not sit on the
        # spurious uniform-flow fixed point.  amp scales the BL thickness.
        etas_s, fp_s, _ = blasius()
        amp = args.seed
        ux_np = np.full((ny, nx), U)
        for x in range(le, te):
            sc = math.sqrt(nu * max(x - le, 1) / U) * amp
            yloc = np.arange(ny) - (args.q if args.wall == "bottom" else y_solid + 1 - args.q)
            if args.wall == "bottom":
                yloc = np.arange(ny) - args.q
            else:
                yloc = np.abs(np.arange(ny) - (y_solid + 0.5))
            fp_i = np.interp(yloc / sc, etas_s, fp_s)
            ux_np[:, x] = U * np.where(yloc <= 0, 0.0, fp_i)
        f = equilibrium(
            torch.ones((ny, nx), device=dev),
            torch.tensor(ux_np, device=dev, dtype=f.dtype),
            torch.zeros((ny, nx), device=dev),
        )
    m0 = float(f.sum().item())
    spec = SPEC.to(dev)
    feq_in = equilibrium(
        torch.ones((1, 1), device=dev),
        torch.full((1, 1), U, device=dev),
        torch.zeros((1, 1), device=dev),
    )[:, :, 0]  # (9,1) broadcastable

    def step(f_):
        f_ = collide_bgk(f_, tau)
        f_ = f_.clone()
        if args.wall == "bottom":
            # free-slip ghost in row 0, then no-slip BB under the plate
            f_[:, 0, :] = f_[:, 1, :][spec]
            f_ = wall_bc_bottom(f_, args.q, mask)
        else:
            f_ = wall_bc_mid(f_, args.q, y_solid)
            f_[:, 0, :] = f_[:, 1, :][spec]
        # top far-field
        if args.top == "specular":
            f_[:, -1, :] = f_[:, -2, :][spec]
        elif args.top == "neumann":
            f_[:, -1, :] = f_[:, -2, :]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        if args.outlet == "zouhe":
            f_ = zou_he_outlet_pressure(f_, 1.0)
        else:
            f_[:, :, -1] = f_[:, :, -2]
        if args.top == "freestream":
            f_[:, -1, :] = feq_in
        return f_

    t0 = time.time()
    umaxh = []
    for i in range(1, args.steps + 1):
        f = step(f)
        if i % 2000 == 0:
            _, ux, _ = macroscopic(f)
            umaxh.append(float(ux.max()))
            if args.verbose:
                print(
                    f"  step {i}: umax={umaxh[-1]:.5f} finite={bool(torch.isfinite(f).all())}",
                    flush=True,
                )
    elapsed = time.time() - t0
    ua = np.array(umaxh)
    drift = (
        (ua[-10:].max() - ua[-10:].min()) / abs(ua[-10:].mean()) if len(ua) >= 10 else float("nan")
    )

    # time-averaged profile at probe
    n_avg = 300
    prof = torch.zeros(ny, device=dev)
    for _ in range(n_avg):
        f = step(f)
        _, ux, _ = macroscopic(f)
        prof += ux[:, probe]
    prof /= n_avg
    _, ux_f, _ = macroscopic(f)
    mass_drift = (float(f.sum().item()) - m0) / m0 * 100.0

    up = prof.cpu().numpy().astype(np.float64)

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fp, a0 = blasius()

    y_w = args.q if args.wall == "bottom" else (y_solid + 1 - args.q)
    rows = np.arange(1, ny) if args.wall == "bottom" else np.arange(y_solid + 1, ny)
    y = rows - y_w
    eta = y / scale
    u = up[rows]
    ue = U  # Blasius far-field
    fpref = np.interp(eta, etas, fp)
    uref = U * fpref
    m = (eta > 0.05) & (eta < 5.0)
    rel = np.abs(u - uref) / np.maximum(uref, 1e-12)
    l2 = float(np.linalg.norm(rel[m]) / np.sqrt(m.sum()))

    # Cf: 1st-order at wall (u at y=q, distance q)
    Cf1 = 2 * nu * (u[0] / args.q) / U**2
    Cf_ref = 0.664 / math.sqrt(Rex)
    Cf_err1 = (Cf1 - Cf_ref) / Cf_ref * 100.0
    # 2nd/3rd via quadratic fit through first three nodes
    d3 = np.array([y[0], y[1], y[2]])
    A = np.vstack([np.ones(3), d3, d3**2]).T
    w = np.linalg.solve(A, np.array([0.0, 1.0, 0.0]))
    dudy2 = float(w @ u[:3])
    Cf2 = 2 * nu * dudy2 / U**2
    Cf_err2 = (Cf2 - Cf_ref) / Cf_ref * 100.0

    # integral quantities with u_e = U
    dstar = float(np.trapezoid(1 - u / ue, y))
    theta = float(np.trapezoid(u / ue * (1 - u / ue), y))
    H = dstar / theta if theta else float("nan")
    dstar_ref = 1.7208 * x_eff / math.sqrt(Rex)
    theta_ref = 0.664 * x_eff / math.sqrt(Rex)
    H_ref = 2.5911

    # overshoot metric
    os_max = float(np.max(u / ue))
    ue_local = float(np.mean(up[ny - 30 :])) if args.wall == "bottom" else None

    tab = []
    for et in (1.0, 2.0, 3.0, 4.0, 5.0):
        ut = float(np.interp(et * scale, y, u))
        fr = float(np.interp(et, etas, fp))
        tab.append(
            {
                "eta": et,
                "u_sim_over_U": round(ut / U, 5),
                "blasius": round(fr, 5),
                "rel_pct": round((ut / U - fr) / fr * 100.0, 2),
            }
        )

    res = dict(
        case="B25_dev",
        wall=args.wall,
        top=args.top,
        outlet=args.outlet,
        q=args.q,
        nx=nx,
        ny=ny,
        le=le,
        plate_len=pl,
        probe=probe,
        U=U,
        nu=nu,
        tau=tau,
        steps=args.steps,
        Rex=Rex,
        scale=scale,
        fpp0=a0,
        umax_drift=drift,
        mass_drift_pct=mass_drift,
        finite=bool(torch.isfinite(f).all()),
        u1_over_U=round(float(u[0] / U), 6),
        u1_blasius=round(float(np.interp(args.q / scale, etas, fp)), 6),
        overshoot_max=round(os_max, 5),
        Cf1=Cf1,
        Cf_ref=Cf_ref,
        Cf_err1_pct=Cf_err1,
        Cf2=Cf2,
        Cf_err2_pct=Cf_err2,
        delta_star=dstar,
        delta_star_ref=dstar_ref,
        theta=theta,
        theta_ref=theta_ref,
        H=H,
        H_ref=H_ref,
        l2_profile=l2,
        eta_tab=tab,
        profile=[[round(float(e), 4), round(float(uu) / U, 6)] for e, uu in zip(eta, u)][:120],
        elapsed_s=round(elapsed, 1),
    )
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(
        json.dumps(
            {
                k: res[k]
                for k in [
                    "wall",
                    "top",
                    "outlet",
                    "q",
                    "nx",
                    "ny",
                    "probe",
                    "steps",
                    "Rex",
                    "u1_over_U",
                    "u1_blasius",
                    "overshoot_max",
                    "Cf1",
                    "Cf_ref",
                    "Cf_err1_pct",
                    "Cf2",
                    "Cf_err2_pct",
                    "delta_star",
                    "delta_star_ref",
                    "theta",
                    "theta_ref",
                    "H",
                    "H_ref",
                    "l2_profile",
                    "mass_drift_pct",
                    "finite",
                    "elapsed_s",
                ]
            },
            indent=2,
        )
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=400)
    ap.add_argument("--ny", type=int, default=200)
    ap.add_argument("--le", type=int, default=20)
    ap.add_argument("--plate_len", type=int, default=180)
    ap.add_argument("--probe", type=int, default=200)
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--wall", default="bottom", choices=["bottom", "mid"])
    ap.add_argument("--top", default="freestream", choices=["specular", "neumann", "freestream"])
    ap.add_argument("--outlet", default="zouhe", choices=["zouhe", "convective"])
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--seed", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

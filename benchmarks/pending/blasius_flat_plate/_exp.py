#!/usr/bin/env python3
"""EXPERIMENT harness for the Blasius flat-plate benchmark (SDAA-capable).

Goal: find the configuration that drops the wall-friction error below 3%.
Tests BC variants (inlet free-stream Dirichlet, outlet Zou-He-pressure vs
convective/zero-gradient), wall placement (mid thin plate vs bottom wall),
and resolution.  Reports Cf via several wall-shear estimators.

Runnable now (library imports cleanly via PYTHONPATH=<repo>/src).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types as _types
from pathlib import Path

import numpy as np
import torch

from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

torch.set_num_threads(32)

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

GRIDS = {
    "mid200": dict(wall="mid", nx=240, ny=1400, le=20, plate_len=200, probe=200),
    "mid400": dict(wall="mid", nx=440, ny=1400, le=20, plate_len=400, probe=200),
    "bot400": dict(wall="bottom", nx=440, ny=700, le=20, plate_len=400, probe=200),
    "bot200": dict(wall="bottom", nx=240, ny=700, le=20, plate_len=200, probe=200),
}


def blasius_table(eta_max: float = 12.0, h: float = 0.002):
    def rhs(s):
        f, fp, fpp = s
        return np.array([fp, fpp, -0.5 * f * fpp])

    def rk4(s, dt):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    n = int(round(eta_max / h))

    def shoot(fpp0):
        s = np.array([0.0, 0.0, fpp0])
        for _ in range(n):
            s = rk4(s, h)
        return s[1]

    lo, hi = 0.30, 0.37
    flo = shoot(lo) - 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = shoot(mid) - 1.0
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    fpp0 = 0.5 * (lo + hi)
    etas = np.zeros(n + 1)
    fprimes = np.zeros(n + 1)
    s = np.array([0.0, 0.0, fpp0])
    for i in range(1, n + 1):
        s = rk4(s, h)
        etas[i] = i * h
        fprimes[i] = s[1]
    return etas, fprimes, fpp0


def plate_hwbb_mid(f, py, mask):
    f = f.clone()
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, :-1])
    return f


def plate_hwbb_bottom(f, mask):
    """Half-way BB on bottom wall (solid row 0), wall at y=0.5.

    fluid cell row 1 is at distance 0.5 from the wall.  Incoming populations
    (direction 2,5,6 head toward -y) are reflected to (4,7,8)."""
    f = f.clone()
    f[4, 1, :] = torch.where(mask, f[2, 1, :], f[4, 1, :])
    f[7, 1, 1:] = torch.where(mask[1:], f[5, 1, :-1], f[7, 1, 1:])
    f[8, 1, :-1] = torch.where(mask[:-1], f[6, 1, 1:], f[8, 1, :-1])
    return f


def run_case(args):
    g = GRIDS[args.grid]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    plate_end = le + plate_len
    wall = g["wall"]
    U, nu = args.U, args.nu
    tau = 3.0 * nu + 0.5
    device = torch.device(args.device)
    torch.manual_seed(0)

    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    if wall == "mid":
        py = ny // 2
        solid[py, le:plate_end] = True
        plate_row = solid[py]
    else:
        py = 0
        solid[0, le:plate_end] = True
        plate_row = solid[0]

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)
    initial_mass = float(f.sum().item())

    spec = SPEC.to(device)
    feq_in = equilibrium(
        torch.ones((ny, 1), device=device),
        torch.full((ny, 1), U, device=device),
        torch.zeros((ny, 1), device=device),
    )[:, :, 0].contiguous()

    def step_fn(f_):
        f_ = collide_bgk(f_, tau)
        if wall == "mid":
            f_ = plate_hwbb_mid(f_, py, plate_row)
            f_ = f_.clone()
            f_[:, 0, :] = f_[:, 1, :][spec]
            f_[:, -1, :] = f_[:, -2, :][spec]
        else:
            f_ = plate_hwbb_bottom(f_, plate_row)
            f_ = f_.clone()
            f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        # inlet: uniform free-stream equilibrium Dirichlet
        f_[:, :, 0] = feq_in
        # outlet
        if args.outlet == "zouhe":
            f_ = zou_he_outlet_pressure(f_, 1.0)
        else:  # convective / zero-gradient
            f_ = f_.clone()
            f_[:, :, -1] = f_[:, :, -2]
        return f_

    t0 = time.time()
    umax_hist = []
    for step in range(1, args.steps + 1):
        f = step_fn(f)
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            umax_hist.append(float(ux.max().item()))
    elapsed = time.time() - t0

    umax_arr = np.array(umax_hist)
    tail = umax_arr[-20:] if len(umax_arr) >= 20 else umax_arr
    umax_drift = (float(tail.max()) - float(tail.min())) / max(abs(float(tail.mean())), 1e-12)

    # time-averaged profile at probe
    prof = torch.zeros(ny, device=device)
    n_avg = 300
    for _ in range(n_avg):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof += ux[:, probe]
    prof /= n_avg

    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    finite = bool(torch.isfinite(f).all().item())

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()

    u_prof = prof.cpu().numpy()
    delta_star_ref = 1.7208 * x_eff / math.sqrt(Rex)
    theta_ref = 0.664 * x_eff / math.sqrt(Rex)
    Cf_ref = 0.664 / math.sqrt(Rex)

    def analyze(u_hi, y_hi, label):
        eta_hi = y_hi / scale
        fp_ref = np.interp(eta_hi, etas, fprimes)
        u_ref = U * fp_ref
        m = (eta_hi > 0.02) & (eta_hi < 5.0) & (fp_ref > 0.02) & (fp_ref < 0.995)
        rel = np.abs(u_hi - u_ref) / np.maximum(u_ref, 1e-12)
        l2 = float(np.linalg.norm(rel[m]) / np.linalg.norm(np.ones(m.sum()))) if m.any() else float("nan")
        maxp = float(rel[m].max() * 100.0) if m.any() else float("nan")
        dudy = float(np.linalg.solve(
            np.vstack([np.ones(3), [0.5, 1.5, 2.5], [0.25, 2.25, 6.25]]).T,
            np.array([0.0, 1.0, 0.0])) @ np.array([u_hi[0], u_hi[1], u_hi[2]]))
        Cf2 = 2.0 * nu * dudy / (U * U)
        dudy1 = u_hi[0] / 0.5
        Cf1 = 2.0 * nu * dudy1 / (U * U)
        # delta* integrand
        uedge = U
        ds = float(np.trapezoid(1.0 - u_hi / uedge, y_hi))
        return dict(label=label, l2=l2, max_rel_pct=maxp, Cf_2nd=Cf2, Cf_1st=Cf1,
                    Cf_err_2nd_pct=(Cf2 - Cf_ref) / Cf_ref * 100.0,
                    Cf_err_1st_pct=(Cf1 - Cf_ref) / Cf_ref * 100.0,
                    delta_star=ds, delta_star_ref=delta_star_ref,
                    u1_over_U=float(u_hi[0] / U))

    out = dict(grid=args.grid, wall=wall, outlet=args.outlet, nx=nx, ny=ny,
               U=U, nu=nu, tau=tau, Rex=Rex, scale=scale, steps=args.steps,
               elapsed_s=round(elapsed, 1), umax_drift=umax_drift,
               mass_drift_pct=mass_drift_pct, finite=finite, Cf_ref=Cf_ref,
               fpp0=fpp0, delta_star_ref=delta_star_ref, theta_ref=theta_ref)
    if wall == "mid":
        y_w = py + 0.5
        y_hi = np.arange(py + 1, ny, dtype=np.float64) - y_w
        out["upper"] = analyze(u_prof[py + 1:], y_hi, "upper")
        y_lo = y_w - np.arange(0, py, dtype=np.float64)  # 0.5.. 
        out["lower"] = analyze(u_prof[:py][::-1], y_lo, "lower")
    else:
        y_w = 0.5
        y_hi = np.arange(1, ny, dtype=np.float64) - y_w
        out["upper"] = analyze(u_prof[1:], y_hi, "main")
    # y+ contour sample of BL thickness (eta=5)
    out["u_edge_over_U"] = float(ux_f[ny - 2, probe].item() / U)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="bot400", choices=list(GRIDS))
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--outlet", default="convective", choices=["zouhe", "convective"])
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run_case(args)
    print(json.dumps(r, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
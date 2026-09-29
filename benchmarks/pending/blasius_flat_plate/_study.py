#!/usr/bin/env python3
"""Corrected Blasius flat-plate study (bottom or mid wall, full diagnostics).

Wall BC (pre-stream half-way bounce-back, pull-stream convention):
  solid row s, fluid row s+/-1, wall at +/-0.5.  Opp(2)=4, opp(5)=7, opp(6)=8.
  * fluid ABOVE solid row (bottom wall):   f[2,s]<-f[4,s+1]; f[5,s,:-1]<-f[7,s+1,1:];
                                            f[6,s,1:]<-f[8,s+1,:-1]
  * fluid BELOW solid row (upper surface): f[4,s]<-f[2,s-1]; f[7,s,1:]<-f[5,s-1,:-1];
                                            f[8,s,:-1]<-f[6,s-1,1:]
  (mid thin plate = both; bottom wall = first only)
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

torch.set_num_threads(32)
SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def blasius_tab(eta_max=14.0, h=0.002):
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def rhs(e, s):
        return [s[1], s[2], -0.5 * s[0] * s[2]]

    def shoot(a):
        return solve_ivp(rhs, [0, 14], [0, 0, a], rtol=1e-11, atol=1e-13, max_step=0.01).y[1, -1] - 1.0

    a = brentq(shoot, 0.3, 0.37, xtol=1e-13)
    n = int(round(eta_max / h))
    s = np.zeros((3, n + 1))
    s[:, 0] = [0, 0, a]
    etas = np.arange(n + 1) * h
    for i in range(n):
        e = etas[i]
        k1 = np.array(rhs(e, s[:, i]))
        k2 = np.array(rhs(e + h / 2, s[:, i] + h / 2 * k1))
        k3 = np.array(rhs(e + h / 2, s[:, i] + h / 2 * k2))
        k4 = np.array(rhs(e + h, s[:, i] + h * k3))
        s[:, i + 1] = s[:, i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return etas, s[1], s[0], a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=1100)
    ap.add_argument("--ny", type=int, default=400)
    ap.add_argument("--le", type=int, default=20)
    ap.add_argument("--plate_len", type=int, default=1000)
    ap.add_argument("--probes", type=str, default="200,470,920")
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--wall", default="bottom", choices=["bottom", "mid"])
    ap.add_argument("--outlet", default="zouhe", choices=["zouhe", "convective"])
    ap.add_argument("--device", default="sdaa:5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nx, ny, le, pl = args.nx, args.ny, args.le, args.plate_len
    probes = [int(v) for v in args.probes.split(",")]
    U, nu = args.U, args.nu
    tau = 3 * nu + 0.5
    dev = torch.device(args.device)
    plate_end = le + pl
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    if args.wall == "bottom":
        y_solid = 0
        solid[0, le:plate_end] = True
    else:
        y_solid = ny // 2
        solid[y_solid, le:plate_end] = True
    mask = solid[y_solid]

    f = equilibrium(torch.ones((ny, nx), device=dev), torch.full((ny, nx), U, device=dev),
                    torch.zeros((ny, nx), device=dev), device=dev)
    m0 = float(f.sum().item())
    spec = SPEC.to(dev)
    feq_in = equilibrium(torch.ones((ny, 1), device=dev), torch.full((ny, 1), U, device=dev),
                         torch.zeros((ny, 1), device=dev))[:, :, 0].contiguous()

    def step(f_):
        f_ = collide_bgk(f_, tau)
        f_ = f_.clone()
        s = torch.where
        if args.wall == "bottom":
            f_[2, 0, :] = s(mask, f_[4, 1, :], f_[2, 0, :])
            f_[5, 0, :-1] = s(mask[:-1], f_[7, 1, 1:], f_[5, 0, :-1])
            f_[6, 0, 1:] = s(mask[1:], f_[8, 1, :-1], f_[6, 0, 1:])
            f_[:, -1, :] = f_[:, -2, :][spec]
        else:
            y = y_solid
            f_[2, y, :] = s(mask, f_[4, y + 1, :], f_[2, y, :])
            f_[5, y, :-1] = s(mask[:-1], f_[7, y + 1, 1:], f_[5, y, :-1])
            f_[6, y, 1:] = s(mask[1:], f_[8, y + 1, :-1], f_[6, y, 1:])
            f_[4, y, :] = s(mask, f_[2, y - 1, :], f_[4, y, :])
            f_[7, y, 1:] = s(mask[1:], f_[5, y - 1, :-1], f_[7, y, 1:])
            f_[8, y, :-1] = s(mask[:-1], f_[6, y - 1, 1:], f_[8, y, :-1])
            f_[:, 0, :] = f_[:, 1, :][spec]
            f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        if args.outlet == "zouhe":
            f_ = zou_he_outlet_pressure(f_, 1.0)
        else:
            f_ = f_.clone()
            f_[:, :, -1] = f_[:, :, -2]
        return f_

    print(f"[study] wall={args.wall} outlet={args.outlet} nx={nx} ny={ny} "
          f"le={le} L={pl} probes={probes} U={U} nu={nu} steps={args.steps}", flush=True)
    for i in range(1, args.steps + 1):
        f = step(f)
        if i % 5000 == 0:
            _, ux, _ = macroscopic(f)
            print(f"  step {i}: umax={float(ux.max()):.5f} finite={bool(torch.isfinite(f).all())}",
                  flush=True)

    # average profiles over last 400 steps
    acc = torch.zeros((len(probes), ny), device=dev)
    navg = 400
    for _ in range(navg):
        f = step(f)
        _, ux, _ = macroscopic(f)
        for j, p in enumerate(probes):
            acc[j] += ux[:, p]
    acc /= navg

    _, ux_f, _ = macroscopic(f)
    mass_drift = (float(f.sum().item()) - m0) / m0 * 100.0
    etas, fprime, ffun, fpp0 = blasius_tab()

    def q_deriv(u3):
        return float(np.linalg.solve(
            np.vstack([np.ones(3), [0.5, 1.5, 2.5], [0.25, 2.25, 6.25]]).T,
            [0, 1, 0]) @ u3)

    res = dict(wall=args.wall, outlet=args.outlet, nx=nx, ny=ny, le=le, plate_len=pl,
               U=U, nu=nu, tau=tau, steps=args.steps, fpp0=fpp0,
               mass_drift_pct=mass_drift, finite=bool(torch.isfinite(f).all()), probes={})
    for j, p in enumerate(probes):
        x_eff = p - le
        Rex = U * x_eff / nu
        scale = math.sqrt(nu * x_eff / U)
        if args.wall == "bottom":
            y_w = 0.5
            rows = np.arange(1, ny)
        else:
            y_w = y_solid + 0.5
            rows = np.arange(y_solid + 1, ny)
        y = rows - y_w
        eta = y / scale
        prof = acc[j].cpu().numpy()[rows]
        ue = float(np.mean(acc[j].cpu().numpy()[-40:]))
        fp_ref = np.interp(eta, etas, fprime)
        u_ref = U * fp_ref
        # delta*, theta, H (u_e local)
        dstar = float(np.trapezoid(1 - prof / ue, y))
        theta = float(np.trapezoid(prof / ue * (1 - prof / ue), y))
        H = dstar / theta if theta else float("nan")
        m = (eta > 0.05) & (eta < 5.0)
        rel = np.abs(prof - u_ref) / np.maximum(u_ref, 1e-12)
        l2 = float(np.linalg.norm(rel[m]) / np.linalg.norm(np.ones(m.sum())))
        Cf_ref = 0.664 / math.sqrt(Rex)
        Cf1 = 2 * nu * (prof[0] / 0.5) / U ** 2
        Cf2 = 2 * nu * q_deriv(prof[0:3]) / U ** 2
        tab = []
        for et in (1.0, 2.0, 3.0, 4.0, 5.0):
            ut = float(np.interp(et * scale, y, prof))
            fr = float(np.interp(et, etas, fprime))
            tab.append({"eta": et, "u_sim": round(ut, 7), "u_blas": round(U * fr, 7),
                        "rel_pct": round((ut - U * fr) / (U * fr) * 100, 2)})
        res["probes"][str(p)] = dict(
            x_eff=x_eff, Rex=Rex, scale=scale, delta99_cells=4.91 * scale,
            u_edge_over_U=ue / U, l2=l2,
            Cf_ref=Cf_ref, Cf_1st=Cf1, Cf_err_1st_pct=(Cf1 - Cf_ref) / Cf_ref * 100,
            Cf_2nd=Cf2, Cf_err_2nd_pct=(Cf2 - Cf_ref) / Cf_ref * 100,
            dstar=dstar, dstar_ref=1.7208 * x_eff / math.sqrt(Rex),
            theta=theta, theta_ref=0.664 * x_eff / math.sqrt(Rex),
            H=H, eta_tab=tab,
            prof_in_eta=[[round(float(e), 4), round(float(uu) / U, 6), round(float(ur) / U, 6)]
                         for e, uu, ur in zip(eta, prof, u_ref)],
        )
    print(json.dumps(res, indent=2, default=float))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
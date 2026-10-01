#!/usr/bin/env python3
"""Fast diagnostic: watch the wall-normal profile + Cf evolve at the probe."""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def blasius(eta):
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def rhs(e, s):
        return [s[1], s[2], -0.5 * s[0] * s[2]]

    def shoot(a):
        sol = solve_ivp(rhs, [0, 12], [0.0, 0.0, a], rtol=1e-10, atol=1e-12, max_step=0.01)
        return sol.y[1, -1] - 1.0

    a = brentq(shoot, 0.3, 0.37, xtol=1e-12)
    sol = solve_ivp(
        rhs, [0, 12], [0.0, 0.0, a], rtol=1e-11, atol=1e-13, dense_output=True, max_step=0.005
    )
    return sol.sol(eta)[1], sol.sol(eta)[0], a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=440)
    ap.add_argument("--ny", type=int, default=700)
    ap.add_argument("--le", type=int, default=20)
    ap.add_argument("--plate_len", type=int, default=400)
    ap.add_argument("--probe", type=int, default=200)
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--tcol", type=int, default=200)
    ap.add_argument("--device", default="sdaa:4")
    args = ap.parse_args()

    nx, ny, le, pl, probe = args.nx, args.ny, args.le, args.plate_len, args.probe
    U, nu = args.U, args.nu
    tau = 3 * nu + 0.5
    dev = torch.device(args.device)
    plate_end = le + pl
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    solid[0, le:plate_end] = True
    mask = solid[0]

    f = equilibrium(
        torch.ones((ny, nx), device=dev),
        torch.full((ny, nx), U, device=dev),
        torch.zeros((ny, nx), device=dev),
        device=dev,
    )
    spec = SPEC.to(dev)
    feq_in = equilibrium(
        torch.ones((ny, 1), device=dev),
        torch.full((ny, 1), U, device=dev),
        torch.zeros((ny, 1), device=dev),
    )[:, :, 0].contiguous()

    def step(f_):
        f_ = collide_bgk(f_, tau)
        f_ = f_.clone()
        f_[4, 1, :] = torch.where(mask, f_[2, 1, :], f_[4, 1, :])
        f_[7, 1, 1:] = torch.where(mask[1:], f_[5, 1, :-1], f_[7, 1, 1:])
        f_[8, 1, :-1] = torch.where(mask[:-1], f_[6, 1, 1:], f_[8, 1, :-1])
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        f_[:, :, -1] = f_[:, :, -2]
        return f_

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    y_w = 0.5
    y = np.arange(1, ny) - y_w
    eta = y / scale
    fp_ref, f_ref, a = blasius(eta)
    u_ref = U * fp_ref
    Cf_ref = 0.664 / math.sqrt(Rex)

    for step_i in range(1, args.steps + 1):
        f = step(f)
        if step_i % args.tcol == 0:
            _, ux, _ = macroscopic(f)
            prof = ux[:, probe].cpu().numpy()
            u1 = prof[1]
            dudy1 = u1 / 0.5
            Cf1 = 2 * nu * dudy1 / U**2
            u123 = prof[1:4]
            dudy2 = float(
                np.linalg.solve(
                    np.vstack([np.ones(3), [0.5, 1.5, 2.5], [0.25, 2.25, 6.25]]).T, [0, 1, 0]
                )
                @ u123
            )
            Cf2 = 2 * nu * dudy2 / U**2
            n = 8
            print(
                f"step={step_i:6d} u1/U={u1 / U:.5f} (blas {fp_ref[0]:.5f}) "
                f"Cf1={Cf1:.5f} Cf2={Cf2:.5f} Cf_ref={Cf_ref:.5f} "
                f"err1={(Cf1 - Cf_ref) / Cf_ref * 100:+.2f}% err2={(Cf2 - Cf_ref) / Cf_ref * 100:+.2f}% "
                f"prof5={np.round(prof[1:6] / U, 4)}",
                flush=True,
            )
    print("fpp0_num=%.6f" % a)


if __name__ == "__main__":
    main()

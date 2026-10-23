#!/usr/bin/env python
"""方腔自然对流（de Vahl Davis）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/thermal_cavity/{run,driver}.py 使用同一组库入口
（driver 为纯胶水编排，本脚本按同序重排）：
  - tensorlbm.d2q9.equilibrium / macroscopic
  - tensorlbm.solver.stream
  - tensorlbm.thermal.{temperature_equilibrium, temperature_collision,
    temperature_stream, apply_temperature_boundaries, buoyancy_force,
    cavity_wall_mask, collide_bgk_force, pre_streaming_bounce_back,
    nusselt_number, thermal_params}

步序与 driver.advance（v6，mask_wall_u=True）逐项一致：温度碰撞前把壁环宏观 u
置零（固壁不输运温度），速度壁用 pre-streaming 半程反弹。参数换算与 driver
同构：H = nx−1、ν=(τ−1/2)/3、α=ν/Pr、τ_T=3α+1/2、gβ=Ra·ν·α/H³。

演示档 nx=96、Ra=1e4、固定步数（正式档六案 N=64/128/256 × Ra=1e3/1e4，含稳态
触发与验证段）。输出温度场/速度场、Nu 与中线速度时间序列。

用法：
    python docs/benchmarks/demos/thermal_cavity_demo.py \
        --nx 96 --ra 1e4 --steps 60000 --out demo_thermal_cavity.npz
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_SRC = os.environ.get("TENSORLBM_SRC") or str(
    Path(__file__).resolve().parents[3] / "src"
)
sys.path.insert(0, REPO_SRC)

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.thermal import (  # noqa: E402
    apply_temperature_boundaries,
    buoyancy_force,
    cavity_wall_mask,
    collide_bgk_force,
    nusselt_number,
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)
from tensorlbm.thermal import thermal_params as _lib_thermal_params  # noqa: E402


def lattice_params(nx, ra, pr, tau):
    p = _lib_thermal_params(nx, ra, pr, tau)
    h = float(nx - 1)
    return {
        "nu": p["nu"],
        "alpha": p["alpha"],
        "tau_T": p["tau_T"],
        "H": h,
        "g_beta": ra * p["nu"] * p["alpha"] / (h * h * h),
    }


def main():
    ap = argparse.ArgumentParser(description="thermal_cavity 场量演示")
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ra", type=float, default=1e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="demo_thermal_cavity.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    nx = args.nx
    ny = nx
    t_hot, t_cold = 1.0, 0.0
    p = lattice_params(nx, args.ra, args.pr, args.tau)
    H = p["H"]

    x = torch.arange(nx, device=device, dtype=dtype)
    T0 = (t_hot - (t_hot - t_cold) * x / H).unsqueeze(0).expand(ny, nx).contiguous()
    zero = torch.zeros_like(T0)
    f = equilibrium(torch.ones_like(T0), zero, zero)
    g = temperature_equilibrium(T0, zero, zero)
    g = apply_temperature_boundaries(g, t_hot, t_cold)
    wall = cavity_wall_mask(ny, nx, device)

    def advance(f, g):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_force(rho, T, p["g_beta"], t_ref=t_cold)
        fluid = 1.0 - wall.to(ux.dtype)
        ux = ux * fluid
        uy = uy * fluid
        g = temperature_collision(g, p["tau_T"], ux, uy)
        g = temperature_stream(g)
        g = apply_temperature_boundaries(g, t_hot, t_cold)
        f_pre = f
        f = collide_bgk_force(f, args.tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        return f, g

    hist = {"step": [], "nu": [], "u_max_nd": [], "v_max_nd": []}
    scale = H / p["alpha"]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        f, g = advance(f, g)
        if step % args.sample == 0 or step == args.steps:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu = nusselt_number(T, H, t_hot - t_cold, t_hot, t_cold, mode="grad1")
            xc = H / 2.0
            i0 = int(math.floor(xc))
            frac = xc - i0
            i1 = min(i0 + 1, nx - 1)
            u_line = (1.0 - frac) * ux[:, i0] + frac * ux[:, i1]
            v_line = (1.0 - frac) * uy[i0, :] + frac * uy[i1, :]
            hist["step"].append(step)
            hist["nu"].append(float(nu["nu"]))
            hist["u_max_nd"].append(float(u_line.max()) * scale)
            hist["v_max_nd"].append(float(v_line.max()) * scale)
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    nu_fin = nusselt_number(T, H, t_hot - t_cold, t_hot, t_cold, mode="grad1")
    xc = H / 2.0
    i0 = int(math.floor(xc))
    frac = xc - i0
    i1 = min(i0 + 1, nx - 1)
    u_line = (1.0 - frac) * ux[:, i0] + frac * ux[:, i1]
    v_line = (1.0 - frac) * uy[i0, :] + frac * uy[i1, :]

    np.savez(
        args.out,
        step=np.array(hist["step"]),
        nu=np.array(hist["nu"]),
        u_max_nd=np.array(hist["u_max_nd"]),
        v_max_nd=np.array(hist["v_max_nd"]),
        T=T.cpu().numpy().astype(np.float64),
        ux=ux.cpu().numpy().astype(np.float64),
        uy=uy.cpu().numpy().astype(np.float64),
        u_line=u_line.cpu().numpy().astype(np.float64),
        v_line=v_line.cpu().numpy().astype(np.float64),
        T_line=T[ny // 2, :].cpu().numpy().astype(np.float64),
        nu_final=nu_fin["nu"],
        nu_left=nu_fin["nu_left"],
        nu_right=nu_fin["nu_right"],
        u_max_nd_final=float(u_line.max()) * scale,
        v_max_nd_final=float(v_line.max()) * scale,
        nx=nx,
        ra=args.ra,
        pr=args.pr,
        tau=args.tau,
        steps=args.steps,
        params=p,
        scale=scale,
        elapsed_s=elapsed,
    )
    print(
        f"saved {args.out}: nx={nx} ra={args.ra:.0e} steps={args.steps} "
        f"nu={nu_fin['nu']:.5f} u_max={float(u_line.max()) * scale:.3f} "
        f"v_max={float(v_line.max()) * scale:.3f} t={elapsed:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

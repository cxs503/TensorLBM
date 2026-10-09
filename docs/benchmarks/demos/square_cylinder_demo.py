#!/usr/bin/env python
"""2D 自由流方柱 Re=100 涡脱落 — 瞬态场量可视化演示脚本。

与正式验证档 benchmarks/verified/square_cylinder/run.py 使用同一组库入口与
同一步进链（工程链复用 cylinder_re100）：
  - tensorlbm.solver.collide_mrt（tau_field 逐格松弛 = sponge）+ stream
  - tensorlbm.boundaries.far_field_bc_2d（入口/两侧自由流 + 出口零梯度 +
    obstacle bounce-back，一次调用）
  - tensorlbm.boundaries.make_sponge_strength（下游渐变吸收层）
  - tensorlbm.boundaries.compute_obstacle_forces（Ladd 动量交换 Cd/Cl）
  - tensorlbm.d2q9.equilibrium
  - 入口播种：2 周期 St_seed=0.14 正弦侧向扰动后回归自由流（同正式档）

演示档缩域缩径：D=16、域 16D×8D（正式档 D=32/48、40D×40D）、u_in=0.1
（正式档 0.05），仅用于给出涡街瞬态场量（|u|、涡量、p′）与 Cl 振荡形态；
Cd/St 定量判据一律取正式档 result.json（2D 自由流数值簇验证）。

用法：
    python docs/benchmarks/demos/square_cylinder_demo.py \
        --D 16 --steps 14000 --out demo.npz
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

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

ST_SEED = 0.14  # 播种 Strouhal（与正式档一致）
SEED_AMPL = 0.10  # 播种 uy 振幅（× u_in）
SEED_PERIODS = 2.0
SPONGE_ALPHA = 10.0  # τ_eff = τ·(1 + α·σ)


def square_mask(nx, ny, cx, cy, side, device):
    """正置方块掩码（与正式档 run.py 同法）：half-way BB → 有效边长 = side。"""
    x0 = int(round(cx - side / 2.0))
    x1 = x0 + int(round(side)) - 1
    y0 = int(round(cy - side / 2.0))
    y1 = y0 + int(round(side)) - 1
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def main():
    ap = argparse.ArgumentParser(description="方柱 Re=100 涡脱落瞬态演示")
    ap.add_argument("--D", type=int, default=16, help="边长格数（正式档 32/48）")
    ap.add_argument("--nx-D", type=float, default=16.0, help="域长（×D）")
    ap.add_argument("--ny-D", type=float, default=8.0, help="域高（×D）")
    ap.add_argument("--sq-x-D", type=float, default=4.0, help="方块中心距入口（×D）")
    ap.add_argument("--sponge-D", type=float, default=4.0, help="sponge 宽（×D）")
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u-in", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="square_cylinder_demo.npz")
    args = ap.parse_args()

    if args.device != "cpu":
        torch.set_num_threads(min(16, os.cpu_count() or 1))
    else:
        torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "0")) or min(32, os.cpu_count() or 1))
    dev = torch.device(args.device)

    D = args.D
    nx, ny = int(args.nx_D * D), int(args.ny_D * D)
    nu = args.u_in * D / args.re
    tau = 3.0 * nu + 0.5

    mask = square_mask(nx, ny, args.sq_x_D * D, ny / 2.0, D, dev)
    sigma = make_sponge_strength(
        ny, nx, int(nx - args.sponge_D * D), int(args.sponge_D * D),
        power=2.0, device=dev,
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    rho0 = torch.ones((ny, nx), device=dev)
    f = equilibrium(rho0, torch.full_like(rho0, args.u_in), torch.zeros_like(rho0))

    omega_seed = 2.0 * math.pi * ST_SEED * args.u_in / D
    seed_steps = int(round(SEED_PERIODS / (ST_SEED * args.u_in / D)))
    uy_amp = SEED_AMPL * args.u_in
    dyn_p = 0.5 * args.u_in**2 * D

    print(
        f"tensorlbm src ok; nx={nx} ny={ny} D={D} tau={tau:.4f} "
        f"seed_steps={seed_steps} steps={args.steps}",
        flush=True,
    )

    t0 = time.time()
    cd_hist, cl_hist = [], []
    for step in range(1, args.steps + 1):
        f = stream(collide_mrt(f, tau, tau_field=tau_field))
        fx, fy = compute_obstacle_forces(f, mask)  # post-stream, pre-bounce
        f = far_field_bc_2d(f, args.u_in, mask)
        if step <= seed_steps:
            # 入口播种（与正式档一致）：整列 feq(rho=1, u_in, uy_seed(t))
            uy_val = uy_amp * math.sin(omega_seed * step)
            f[:, :, 0] = equilibrium(
                torch.ones((ny, 1), device=f.device),
                torch.full((ny, 1), args.u_in, device=f.device),
                torch.full((ny, 1), uy_val, device=f.device),
            )[:, :, 0]
        cd_hist.append(float(fx.item()) / dyn_p)
        cl_hist.append(float(fy.item()) / dyn_p)
        if step % 1000 == 0:
            w = cd_hist[len(cd_hist) // 2:]
            print(
                f"  step {step}: Cd_mean(half)={sum(w)/len(w):.4f} "
                f"Cl_rms={float(np.sqrt(np.mean(np.square(cl_hist[-1000:])))):.4f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    np.savez(
        args.out,
        ux=ux.cpu().numpy(),
        uy=uy.cpu().numpy(),
        rho=rho.cpu().numpy(),
        mask=mask.cpu().numpy(),
        cd_hist=np.asarray(cd_hist),
        cl_hist=np.asarray(cl_hist),
        nx=nx,
        ny=ny,
        D=D,
        re=args.re,
        u_in=args.u_in,
        tau=tau,
        steps=args.steps,
        seed_steps=seed_steps,
        st_seed=ST_SEED,
        elapsed_s=elapsed,
    )
    w2 = cd_hist[len(cd_hist) // 2:]
    print(
        f"saved {args.out}: Cd_demo(mean 后半)={sum(w2)/len(w2):.4f} "
        f"t={elapsed:.1f}s（演示档，定量判据见入库 result.json）",
        flush=True,
    )


if __name__ == "__main__":
    main()

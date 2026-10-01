#!/usr/bin/env python
"""自由流圆柱 Re=200 Kármán 涡街 — 瞬态场量可视化演示脚本。

与正式验证档 benchmarks/verified/cylinder/re200/run.py 使用同一组库入口与
同一步进链（Re=100 工程链复用 + Re=200 播种频率）：
  - tensorlbm.solver.collide_mrt（tau_field 逐格松弛 = sponge）+ stream
  - 固体冻结（碰撞后以碰撞前分布回填固体格）
  - tensorlbm.boundaries.compute_obstacle_forces（表面格 Ladd 动量交换）
  - tensorlbm.boundaries.far_field_bc_2d（自由流远场 + 出口零梯度）
  - tensorlbm.boundaries.make_sponge_strength（下游渐变吸收层）
  - 每 2000 步质量重整化（与正式档一致）
  - 入口播种：前 4 列宏观量重构 feq + St_seed=0.195 正弦侧扰，2 周期

演示档缩域缩径：D=16、域 16D×8D（正式档 D=48/64、40D×40D）、u_in=0.1，
仅用于给出涡街瞬态场量（|u|、涡量、p′）与 Cl 振荡形态；Cd/St 定量判据
一律取正式档 result.json（Braza 1986 Re=200 验证）。

用法：
    python docs/benchmarks/demos/cylinder_re200_demo.py \
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
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import C, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

SEED_COLS = 4
SEED_AMPL = 0.10
SPONGE_ALPHA = 10.0


def main():
    ap = argparse.ArgumentParser(description="圆柱 Re=200 涡街瞬态演示")
    ap.add_argument("--D", type=int, default=16, help="直径格数（正式档 48/64）")
    ap.add_argument("--nx-D", type=float, default=16.0)
    ap.add_argument("--ny-D", type=float, default=8.0)
    ap.add_argument("--cyl-x-D", type=float, default=4.0)
    ap.add_argument("--sponge-D", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=200.0)
    ap.add_argument("--u-in", type=float, default=0.1)
    ap.add_argument("--st-seed", type=float, default=0.195)
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cylinder_re200_demo.npz")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "0")) or min(32, os.cpu_count() or 1))
    dev = torch.device(args.device)

    D = args.D
    nx, ny = int(args.nx_D * D), int(args.ny_D * D)
    nu = args.u_in * D / args.re
    tau = 3.0 * nu + 0.5

    solid = cylinder_mask(nx, ny, args.cyl_x_D * D, ny / 2.0, D / 2.0, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )
    sigma = make_sponge_strength(
        ny, nx, int(nx - args.sponge_D * D), int(args.sponge_D * D),
        power=2.0, device=dev,
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)

    T_seed = D / (args.st_seed * args.u_in)
    n_seed = int(2 * T_seed)
    dyn_p = 0.5 * args.u_in**2 * D

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, args.u_in)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))
    initial_mass = float(f.sum().item())

    # 播种用速度权重：直接取库常量 d2q9.C（正式档修复手写权重 bug 的同一配方）
    c2d = C.to(dev).float()
    cx_w = c2d[:, 0].view(9, 1, 1)
    cy_w = c2d[:, 1].view(9, 1, 1)

    print(
        f"tensorlbm src ok; nx={nx} ny={ny} D={D} tau={tau:.4f} "
        f"n_seed={n_seed} steps={args.steps}",
        flush=True,
    )

    t0 = time.time()
    cd_hist, cl_hist = [], []
    for step in range(1, args.steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau, tau_field=tau_field)
        f = torch.where(solid.unsqueeze(0), before, collided)  # 固体冻结
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, surface)
        f = far_field_bc_2d(f, args.u_in, obstacle_mask=solid)
        if step % 2000 == 0:
            f = f * (initial_mass / f.sum().item())  # 质量重整化（与正式档一致）
        if step <= n_seed:
            phase = 2 * math.pi * step / T_seed
            rho_col = f.sum(0)[:, :SEED_COLS]
            ux_col = (f * cx_w).sum(0)[:, :SEED_COLS] / rho_col.clamp(min=1e-12)
            uy_col = (f * cy_w).sum(0)[:, :SEED_COLS] / rho_col.clamp(min=1e-12)
            uy_col = uy_col + SEED_AMPL * args.u_in * math.sin(phase)
            f[:, :, :SEED_COLS] = equilibrium(rho_col, ux_col, uy_col)
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
        mask=solid.cpu().numpy(),
        cd_hist=np.asarray(cd_hist),
        cl_hist=np.asarray(cl_hist),
        nx=nx,
        ny=ny,
        D=D,
        re=args.re,
        u_in=args.u_in,
        tau=tau,
        steps=args.steps,
        seed_steps=n_seed,
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

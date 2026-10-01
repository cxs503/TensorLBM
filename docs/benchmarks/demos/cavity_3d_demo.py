#!/usr/bin/env python
"""3D 方腔流（展向周期近似 2D）Re=400 — 场量可视化演示脚本（D3Q19 MRT）。

与正式验证档 benchmarks/verified/cavity/3d/run.py 使用同一组库入口：
  - tensorlbm.solver3d.collide_mrt3d / stream3d（D3Q19 MRT 碰撞 + 流迁，
    z 展向天然周期）
  - tensorlbm.boundaries3d.zou_he_moving_lid_3d（D3Q19 顶盖 Zou/He 动壁，
    整层含角点）
  - 三静止壁（x=0 / x=nx-1 / y=0）pre-streaming 半程反弹（V3-3D 配方）

f 形状 (19, nz, ny, nx)：y 垂直（顶盖 y=ny-1 沿 +x 移动），z 展向周期。
本脚本粗网格/缩短步数用于快速出图（默认 48×48×12 × 40000 步，CPU 约
2 分钟；正式档为 96²×24 / 128²×32 × 100k 步），输出展向中间层与全场
数据供教程绘图；定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/cavity_3d_demo.py \
        --nx 48 --nz 12 --steps 40000 --out demo.npz
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 仓库内布局 docs/benchmarks/demos/ -> <repo>/src；可用环境变量覆盖
REPO_SRC = os.environ.get("TENSORLBM_SRC") or str(
    Path(__file__).resolve().parents[3] / "src"
)
sys.path.insert(0, REPO_SRC)

from tensorlbm.boundaries3d import zou_he_moving_lid_3d  # noqa: E402
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_mrt3d, stream3d  # noqa: E402


def stationary_pre_bounce3d(f_pre, f, wall):
    """pre-streaming 半程反弹（静止壁）。wall: (nz,ny,nx) bool。"""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def main():
    ap = argparse.ArgumentParser(description="3D 方腔（展向周期）Re=400 场量演示")
    ap.add_argument("--nx", type=int, default=48)
    ap.add_argument("--nz", type=int, default=12)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--u-lid", type=float, default=0.06)
    ap.add_argument("--re", type=float, default=400.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="demo_cavity_3d.npz")
    args = ap.parse_args()

    nx = args.nx
    ny = nx
    nz = args.nz
    tau = 3.0 * args.u_lid * nx / args.re + 0.5
    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    rho0 = torch.ones((nz, ny, nx), device=device)
    u0 = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho0, u0, u0, u0)
    mass0 = float(rho0.sum().item())

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True  # x=0
    wall[:, :, -1] = True  # x=nx-1
    wall[:, 0, :] = True  # y=0（底壁）；顶盖 y=ny-1 由 Zou-He 处理

    interior = ~wall
    interior[:, -1, :] = False  # 顶盖行不计残差

    t0 = time.time()
    ux_prev = uy_prev = None
    resid = None
    for step in range(1, args.steps + 1):
        f_pre = f
        f = collide_mrt3d(f, tau)
        f = stationary_pre_bounce3d(f_pre, f, wall)
        f = stream3d(f)
        f = zou_he_moving_lid_3d(f, args.u_lid)
        if step % 5000 == 0 or step == args.steps:
            _, ux, uy, _ = macroscopic3d(f)
            if ux_prev is not None:
                resid = torch.maximum(
                    (ux[interior] - ux_prev[interior]).abs().max(),
                    (uy[interior] - uy_prev[interior]).abs().max(),
                ).item()
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
    elapsed = time.time() - t0

    rho, ux, uy, uz = macroscopic3d(f)
    np.savez(
        args.out,
        ux=ux.cpu().numpy(),
        uy=uy.cpu().numpy(),
        uz=uz.cpu().numpy(),
        rho=rho.cpu().numpy(),
        nx=nx,
        ny=ny,
        nz=nz,
        steps=args.steps,
        u_lid=args.u_lid,
        re=args.re,
        tau=tau,
        elapsed_s=elapsed,
        last_resid=resid,
        mass_drift=float(rho.sum().item() - mass0),
    )
    print(
        f"saved {args.out}: nx={nx} nz={nz} steps={args.steps} tau={tau:.4f} "
        f"t={elapsed:.1f}s resid={resid:.2e} "
        f"mass_drift={float(rho.sum().item() - mass0):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

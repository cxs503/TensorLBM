#!/usr/bin/env python
"""顶盖驱动方腔流 Re=100 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/cavity/re100/run.py 使用同一组库入口：
  - tensorlbm.solver.collide_mrt / stream（D2Q9 MRT 碰撞 + 流迁）
  - tensorlbm.lid_driven_cavity.zou_he_moving_lid（顶盖 Zou/He 动壁）
  - 三静止壁 pre-streaming 半程反弹（V3 配方，见正式档 README）

本脚本缩短步数用于快速出图（默认 128² × 30000 步，CPU 约 1 分钟），
输出 ux / uy / rho 场与收敛信息供教程绘图；定量判据以正式档
result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/cavity_re100_demo.py \
        --nx 128 --steps 30000 --out demo.npz
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

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.lid_driven_cavity import zou_he_moving_lid  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


def stationary_pre_bounce(f_pre, f, wall):
    """三静止壁 pre-streaming 半程反弹（V3 配方，反射碰撞前流体侧分布）。"""
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)


def main():
    ap = argparse.ArgumentParser(description="cavity Re=100 场量演示")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--u-lid", type=float, default=0.06)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cavity_re100_demo.npz")
    args = ap.parse_args()

    nx = ny = args.nx
    tau = 3.0 * args.u_lid * nx / args.re + 0.5
    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True  # 底壁
    wall[:, 0] = True  # 左壁
    wall[:, -1] = True  # 右壁

    t0 = time.time()
    ux_prev = uy_prev = None
    resid = None
    for step in range(1, args.steps + 1):
        f_pre = f
        f = collide_mrt(f, tau=tau)
        f = stationary_pre_bounce(f_pre, f, wall)
        f = stream(f)
        f = zou_he_moving_lid(f, args.u_lid)
        if step % 5000 == 0 or step == args.steps:
            _, ux, uy = macroscopic(f)
            if ux_prev is not None:
                resid = torch.maximum(
                    (ux - ux_prev).abs().max(), (uy - uy_prev).abs().max()
                ).item()
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    np.savez(
        args.out,
        ux=ux.cpu().numpy(),
        uy=uy.cpu().numpy(),
        rho=rho.cpu().numpy(),
        nx=nx,
        steps=args.steps,
        u_lid=args.u_lid,
        re=args.re,
        tau=tau,
        elapsed_s=elapsed,
        last_resid=resid,
    )
    print(
        f"saved {args.out}: nx={nx} steps={args.steps} tau={tau:.4f} "
        f"t={elapsed:.1f}s resid={resid:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

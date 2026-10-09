#!/usr/bin/env python
"""Schäfer–Turek 2D-1 圆柱 Re=20 稳态绕流 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/cylinder/re20_st/run.py 使用同一组库入口：
  - tensorlbm.solver.collide_mrt / stream
  - tensorlbm.boundaries.zou_he_inlet_velocity（Poiseuille 剖面入口）
  - tensorlbm.boundaries.zou_he_outlet_pressure（rho=1 压力出口）
  - tensorlbm.boundaries.bounce_back_cells（半程反弹：通道壁 + 圆柱）
  - tensorlbm.boundaries.cylinder_mask / make_channel_wall_mask
  - tensorlbm.boundaries.compute_obstacle_forces（Ladd 动量交换 Cd/Cl）
  - tensorlbm.d2q9.equilibrium

演示档缩径：D=10 格（正式档 D=40/80），τ=0.8 不变；Cd 平台监测提前停机。
演示档 Cd/Cl 与入库判据数字有网格偏差，定量结论一律以正式档 result.json
（D=40/80，300k 步）为准。

用法：
    python docs/benchmarks/demos/cylinder_re20_st_demo.py \
        --D 10 --max-steps 20000 --out demo.npz
"""

import argparse
import math
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

from tensorlbm.boundaries import (  # noqa: E402
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Schäfer-Turek 2D-1 Re=20 场量演示")
    ap.add_argument("--D", type=int, default=10, help="直径格数（正式档 40/80）")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cylinder_re20_st_demo.npz")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "0")) or min(32, os.cpu_count() or 1))
    dev = torch.device(args.device)

    # 几何（与正式档同比例）：域 2.2×0.41，圆柱 (0.2,0.2) r=0.05（物理单位）
    lx, ly = 2.2, 0.41
    dx = 0.1 / args.D
    nx = int(round(lx / dx))
    ny = int(round(ly / dx))
    cx = round(0.2 / dx)
    cy = round(0.2 / dx)
    radius = 0.05 / dx

    tau = args.tau
    nu_lb = (tau - 0.5) / 3.0
    u_char = 20.0 * nu_lb / args.D  # U_mean（格子单位，Re=20）
    y_phys = torch.arange(ny, device=dev) * dx
    u_profile = 4.0 * 1.5 * y_phys * (0.41 - y_phys) / 0.41**2  # U_max=1.5·U_mean
    u_lb = u_char * u_profile

    solid = cylinder_mask(nx, ny, cx, cy, radius, dev)
    wall = make_channel_wall_mask(ny, nx, solid, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )
    dyn_p = 0.5 * u_char**2 * args.D

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = u_lb[:, None].expand(ny, nx).clone()
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))

    print(f"tensorlbm src ok; nx={nx} ny={ny} D={args.D} tau={tau} U_mean={u_char:.4f}",
          flush=True)

    t0 = time.time()
    cd_hist, cl_hist = [], []
    plateau_prev = None
    plateau_hits = 0
    steps_done = 0
    for step in range(1, args.max_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau)
        f = torch.where(solid.unsqueeze(0), before, collided)  # 固体冻结
        f = stream(f)
        f = zou_he_inlet_velocity(f, u_lb, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, wall)
        fx, fy = compute_obstacle_forces(f, surface)  # post-stream, pre-bounce
        f = bounce_back_cells(f, solid)
        cd_hist.append(float(fx.item()) / dyn_p)
        cl_hist.append(float(fy.item()) / dyn_p)
        steps_done = step
        if step % 1000 == 0:
            # 平台监测：最近 1000 步均值相对上一窗漂移
            cd_now = sum(cd_hist[-1000:]) / min(len(cd_hist), 1000)
            if plateau_prev is not None:
                drift = abs(cd_now - plateau_prev) / abs(cd_now)
                plateau_hits = plateau_hits + 1 if drift < 5e-4 else 0
                if plateau_hits >= 5 and step >= 8000:
                    print(f"  plateau at step {step}: Cd={cd_now:.4f}", flush=True)
                    break
            plateau_prev = cd_now
            print(f"  step {step}: Cd={cd_now:.4f} ({time.time()-t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    win = min(len(cd_hist), max(1000, len(cd_hist) // 4))
    cd_demo = sum(cd_hist[-win:]) / win
    cl_demo = sum(cl_hist[-win:]) / win
    np.savez(
        args.out,
        ux=ux.cpu().numpy(),
        uy=uy.cpu().numpy(),
        rho=rho.cpu().numpy(),
        solid=solid.cpu().numpy(),
        cd_hist=np.asarray(cd_hist),
        cl_hist=np.asarray(cl_hist),
        nx=nx,
        ny=ny,
        D=args.D,
        tau=tau,
        u_mean_lb=u_char,
        steps=steps_done,
        cd_demo=cd_demo,
        cl_demo=cl_demo,
        elapsed_s=elapsed,
    )
    print(
        f"saved {args.out}: nx={nx} ny={ny} steps={steps_done} "
        f"Cd_demo={cd_demo:.4f} Cl_demo={cl_demo:.4f} t={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

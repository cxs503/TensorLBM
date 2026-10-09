#!/usr/bin/env python
"""2D Couette 流（上壁运动）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/couette_2d/run.py 使用同一组库入口：
  - tensorlbm.solver.collide_bgk / stream（D2Q9 BGK 碰撞 + 周期流迁）
  - tensorlbm.d2q9.equilibrium / macroscopic
  - 边界 = pre-streaming 半程反弹 + 动壁动量注入
    （f_new[q] = f_pre[opp[q]] + 2*w_q*rho*(c_q·u_wall)/cs^2，
    与正式档 run.py 中 moving_wall_bounce_back 完全同一实现）

本脚本可缩短步数用于快速出图（默认 H=80 × 15000 步，CPU 数秒~数十秒），
输出 ux / rho 场、中线剖面与解析线性剖面对比；定量判据以正式档
result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/couette_2d_demo.py \
        --H 80 --steps 15000 --out demo.npz
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

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS2 = 1.0 / 3.0


def moving_wall_bounce_back(f_pre, f, wall, u_wall):
    """pre-streaming 半程反弹 + 动壁动量注入（与正式档 run.py 同一实现）。"""
    opp = OPPOSITE.to(f.device)
    c = C.to(f.device)
    w = W.to(f.device)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description="2D Couette 场量演示")
    ap.add_argument("--H", type=int, default=80)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="couette_2d_demo.npz")
    args = ap.parse_args()

    H = args.H
    ny = H + 2  # 壁面行 y=0 与 y=ny-1
    nx = H  # x 周期，nx 任意
    nu = (args.tau - 0.5) / 3.0
    H_eff = ny - 2.0  # 半程反弹有效缝隙（壁在 y=0.5 与 ny-1.5）
    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True  # 下壁：静止
    wall[-1, :] = True  # 上壁：以 U0 运动
    u_wall = torch.zeros((ny, nx), device=device)
    u_wall[-1, :] = args.u0

    # 初始条件：解析线性斜坡（缩短扩散暂态；稳态唯一，与初值无关）
    yy = torch.arange(ny, device=device, dtype=torch.float32)
    u0 = torch.clamp(
        args.u0 * (yy - 0.5) / H_eff, min=0.0, max=args.u0
    ).view(ny, 1).expand(ny, nx).clone()
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    f = equilibrium(torch.ones((ny, nx), device=device), u0, torch.zeros_like(u0))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        f_pre = f.clone()
        f = collide_bgk(f, tau=args.tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall)
        f = stream(f)
    elapsed = time.time() - t0

    # 末 200 步时间平均剖面（与正式档同款测量，降噪）
    prof_acc = torch.zeros(ny, device=device)
    for _ in range(200):
        f_pre = f.clone()
        f = collide_bgk(f, tau=args.tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall)
        f = stream(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, nx // 2]
    prof_acc /= 200.0

    rho, ux, uy = macroscopic(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = args.u0 * (y_phys / H_eff)

    np.savez(
        args.out,
        ux=ux.cpu().numpy(),
        uy=uy.cpu().numpy(),
        rho=rho.cpu().numpy(),
        u_profile=u_num,
        u_analytic=u_ana,
        y_phys=y_phys,
        H=H,
        ny=ny,
        nx=nx,
        tau=args.tau,
        nu=nu,
        U0=args.u0,
        steps=args.steps,
        elapsed_s=elapsed,
    )
    l2 = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    print(
        f"saved {args.out}: H={H} steps={args.steps} tau={args.tau} "
        f"t={elapsed:.1f}s l2_rel(demo)={l2:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

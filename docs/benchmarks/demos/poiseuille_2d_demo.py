#!/usr/bin/env python
"""2D Poiseuille 流（压力差驱动）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/poiseuille_2d/run.py 使用同一组库入口：
  - tensorlbm.solver.collide_bgk / stream（D2Q9 BGK 碰撞 + 周期流迁）
  - tensorlbm.boundaries.zou_he_outlet_pressure（Zou/He 压力出口，库函数）
  - 压力入口 = 标准 Zou/He 教材公式（正式档 run.py 同一 ~10 行实现，
    出口的镜像；Zou & He 1997）
  - 上下壁 pre-streaming 半程反弹（与正式档 run.py 同一实现）

本脚本可缩短步数用于快速出图（默认 H=60，ny=62 × nx=180 × 30000 步，
CPU 约一分钟），输出速度场/密度场/中线剖面；定量判据以正式档
result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/poiseuille_2d_demo.py \
        --H 60 --steps 30000 --out demo.npz
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

from tensorlbm.boundaries import zou_he_outlet_pressure  # noqa: E402
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS2 = 1.0 / 3.0


def zou_he_inlet_pressure(f: torch.Tensor, rho_in: float) -> torch.Tensor:
    """Zou-He 压力（密度）入口 x=0：与正式档 run.py 同一实现（Zou & He 1997）。"""
    f0, f2, f3, f4, f6, f7 = (
        f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0], f[6, :, 0], f[7, :, 0],
    )
    rho = torch.full_like(f0, rho_in)
    ux = 1.0 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho
    f_new = f.clone()
    f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux
    f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    return f_new


def main():
    ap = argparse.ArgumentParser(description="2D Poiseuille 场量演示")
    ap.add_argument("--H", type=int, default=60)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--umax", type=float, default=0.04)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="poiseuille_2d_demo.npz")
    args = ap.parse_args()

    H = args.H
    ny = H + 2
    nx = 3 * H  # L = 3H（与正式档同款长高比）
    nu = (args.tau - 0.5) / 3.0
    # u_max = dp*H^2/(8*nu*L), dp = d_rho*cs2, L = nx = 3H => d_rho = 24*nu*u_max/(cs2*H)
    delta_rho = 24.0 * nu * args.umax / (CS2 * H)
    rho_in = 1.0 + delta_rho / 2.0
    rho_out = 1.0 - delta_rho / 2.0
    u_max_ana = delta_rho * CS2 * H * H / (8.0 * nu * nx)
    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    # 初值：静止 + 线性密度斜坡 rho_in -> rho_out（正式档同款）
    xx = torch.arange(nx, device=device, dtype=torch.float32)
    rho0 = (rho_out + (rho_in - rho_out) * (1.0 - xx / (nx - 1))).view(1, nx).expand(ny, nx)
    f = equilibrium(rho0, torch.zeros((ny, nx), device=device), torch.zeros((ny, nx), device=device))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        f_pre = f.clone()
        f = collide_bgk(f, tau=args.tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(device)], f)
        f = stream(f)  # 周期 gather，边界列随后被 Zou-He 覆盖
        f = zou_he_inlet_pressure(f, rho_in)
        f = zou_he_outlet_pressure(f, rho_out)
    elapsed = time.time() - t0

    # 末 200 步时间平均剖面（与正式档同款测量，降噪）
    prof_acc = torch.zeros(ny, device=device)
    for _ in range(200):
        f_pre = f.clone()
        f = collide_bgk(f, tau=args.tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(device)], f)
        f = stream(f)
        f = zou_he_inlet_pressure(f, rho_in)
        f = zou_he_outlet_pressure(f, rho_out)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, nx // 2]
    prof_acc /= 200.0

    rho, ux, uy = macroscopic(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = 4.0 * u_max_ana * (y_phys / H) * (1.0 - y_phys / H)

    np.savez(
        args.out,
        ux=ux.cpu().numpy(),
        uy=uy.cpu().numpy(),
        rho=rho.cpu().numpy(),
        u_profile=u_num,
        u_analytic=u_ana,
        y_phys=y_phys,
        rho_x=rho[ny // 2, :].cpu().numpy(),
        H=H,
        ny=ny,
        nx=nx,
        tau=args.tau,
        nu=nu,
        rho_in=rho_in,
        rho_out=rho_out,
        delta_rho=delta_rho,
        u_max_ana=u_max_ana,
        steps=args.steps,
        elapsed_s=elapsed,
    )
    l2 = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    print(
        f"saved {args.out}: H={H} nx={nx} steps={args.steps} tau={args.tau} "
        f"t={elapsed:.1f}s l2_rel(demo)={l2:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

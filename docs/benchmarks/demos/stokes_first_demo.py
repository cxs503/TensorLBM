#!/usr/bin/env python
"""Stokes 第一问题（平板突然起动）— 时变边界层演示脚本。

与正式验证档 benchmarks/verified/stokes_first_problem/run.py 使用同一组
库入口（零手写物理内核）：
  - tensorlbm.d2q9.equilibrium / macroscopic（初值与测量）
  - tensorlbm.solver.collide_bgk / stream（D2Q9 BGK 碰撞 + 周期流迁）
  - 边界：下壁 pre-streaming 半程反弹 + 移动壁动量注入；
    上壁自由滑移镜面反射（SPECULAR 置换，应力自由远场）
    ——均为正式档同款内联范式（库内 OPPOSITE/C/W 常量构建）

本演示档与入库扫描同参数（H=100, tau=0.65, U=0.05）真实 CPU 步进，
输出三个时刻的 u(y) 剖面与近壁首格速度历史；定量判据以正式档
result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/stokes_first_demo.py \
        --H 100 --tau 0.65 --U 0.05 --steps 1000 4000 9000 --out demo.npz
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

from tensorlbm.d2q9 import OPPOSITE, C, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

try:
    from scipy.special import erfc
except ImportError:
    from math import erf as _erf

    def erfc(x):
        x = np.asarray(x, dtype=np.float64)
        return 1.0 - np.array([_erf(v) for v in x.ravel()]).reshape(x.shape)


CS2 = 1.0 / 3.0
# 自由滑移（镜面）反射置换：翻转 c_y、保留 c_x -> f_new[j] = f_pre[SPECULAR[j]]
SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def specular_replacement(f_pre):
    return f_pre[SPECULAR.to(f_pre.device)]


def moving_wall_replacement(f_pre, U):
    """移动壁 pre-streaming 半程反弹：f_new[opp] = f_pre[i] - 2 w_i rho_w (c_i·u_w)/cs^2。"""
    rho_w = f_pre.sum(dim=0)
    cx = C[:, 0].to(f_pre.device).float()
    w = W.to(f_pre.device).float()
    mom = 2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * (cx.view(9, 1, 1) * U) / CS2
    opp = OPPOSITE.to(f_pre.device)
    return f_pre[opp] - mom[opp]


def main():
    ap = argparse.ArgumentParser(description="Stokes 第一问题（erfc 边界层）演示")
    ap.add_argument("--H", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.65)
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--steps", type=int, nargs="+", default=[1000, 4000, 9000])
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="stokes_first_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("DEMO_THREADS", "32")), os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    H, nx, tau, U = args.H, args.nx, args.tau, args.U
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    max_step = max(args.steps)
    record_set = set(args.steps)

    wall_bottom = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_bottom[0, :] = True
    wall_top = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_top[-1, :] = True

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    profiles = {}
    first_row_hist = []  # (step, u(y=0.5))
    t0 = time.time()
    for step in range(1, max_step + 1):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = torch.where(wall_bottom.unsqueeze(0), moving_wall_replacement(f_pre, U), f)
        f = torch.where(wall_top.unsqueeze(0), specular_replacement(f_pre), f)
        f = stream(f)
        if step % 50 == 0:
            _, ux, _ = macroscopic(f)
            first_row_hist.append((step, float(ux[1, 0].item())))
        if step in record_set:
            _, ux, _ = macroscopic(f)
            profiles[step] = ux[1 : ny - 1, 0].cpu().numpy().astype(np.float64)
    elapsed = time.time() - t0

    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    ana = {s: U * erfc(y_phys / (2.0 * np.sqrt(nu * s))) for s in args.steps}

    np.savez(
        args.out,
        y=y_phys,
        steps=np.array(args.steps),
        u_num=np.stack([profiles[s] for s in args.steps]),
        u_ana=np.stack([ana[s] for s in args.steps]),
        first_row_steps=np.array([s for s, _ in first_row_hist]),
        first_row_u=np.array([u for _, u in first_row_hist]),
        H=H,
        ny=ny,
        nx=nx,
        tau=tau,
        nu=nu,
        U=U,
        elapsed_s=elapsed,
    )
    for i, s in enumerate(args.steps):
        m = ana[s] > 0.05 * U
        rel = np.abs(profiles[s][m] - ana[s][m]) / ana[s][m]
        print(
            f"t={s:5d}  max_rel={rel.max() * 100:.3f}%  delta={math.sqrt(nu * s):.2f}",
            flush=True,
        )
    print(f"saved {args.out}: elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()

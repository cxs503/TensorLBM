#!/usr/bin/env python
"""环形 Taylor-Couette 层流（2D r-θ 平面）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/taylor_couette/run.py 使用同一组库入口：
  - tensorlbm.rotating_cylinder.rotating_wall_velocity（内柱刚体壁速度）
  - tensorlbm.rotating_cylinder.moving_wall_bounce_back（Ladd 运动壁反弹）
  - tensorlbm.boundaries.bounce_back_cells / cylinder_mask（外柱静止反弹）
  - tensorlbm.solver.collide_bgk / stream + tensorlbm.d2q9.equilibrium/macroscopic

演示档把网格缩到 ri=12（正式档 ri=24/48/96），从静止 spin-up 到稳态，
输出速度场/切向速度场/径向分环剖面；定量判据以正式档 result.json 存档为准。

用法：
    python docs/benchmarks/demos/taylor_couette_demo.py \
        --ri 12 --steps 12000 --out demo.npz
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

from tensorlbm.boundaries import bounce_back_cells, cylinder_mask  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.rotating_cylinder import (  # noqa: E402
    moving_wall_bounce_back,
    rotating_wall_velocity,
)
from tensorlbm.solver import collide_bgk, stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Taylor-Couette 场量演示")
    ap.add_argument("--ri", type=int, default=12)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--re-i", type=float, default=20.0)
    ap.add_argument("--nu", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--avg-steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="taylor_couette_demo.npz")
    args = ap.parse_args()

    ri = args.ri
    ro = int(round(ri / args.eta))
    omega_i = args.re_i * args.nu / ri**2
    tau = 3.0 * args.nu + 0.5
    t_gap = (ro - ri) ** 2 / args.nu
    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    # 几何（与正式档 annulus_setup 同款）：方域 n=2*ro+3，圆心 ((n-1)/2,(n-1)/2)
    n = 2 * ro + 3
    cc = (n - 1) / 2.0
    inner = cylinder_mask(n, n, cc, cc, float(ri), device=device)  # d <= r_i
    outer = ~cylinder_mask(n, n, cc, cc, float(ro), device=device)  # d > r_o
    fluid = ~inner & ~outer
    yy, xx = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    d = torch.sqrt((xx - cc) ** 2 + (yy - cc) ** 2)
    ux_w, uy_w = rotating_wall_velocity(inner, cc, cc, omega_i)

    # 从静止 spin-up（与正式档一致，不用解析解初始化）
    rho0 = torch.ones((n, n), device=device, dtype=torch.float32)
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

    def step(f):
        g = stream(collide_bgk(f, tau))
        f2 = moving_wall_bounce_back(g, inner, ux_w, uy_w)
        return bounce_back_cells(f2, outer)

    t0 = time.time()
    for step_i in range(1, args.steps + 1):
        f = step(f)
        if step_i % 2000 == 0:
            g = stream(collide_bgk(f, tau))
            _, ux, uy = macroscopic(g)
            u_th = ((xx - cc) * uy - (yy - cc) * ux) / d
            print(
                f"step={step_i} umax={float(u_th[fluid].abs().max().item()):.6e}",
                flush=True,
            )
    elapsed = time.time() - t0

    # 末 avg_steps 步时间平均（正式档同款：post-stream、pre-bounce 的 g 上取样）
    acc_ux = torch.zeros((n, n), device=device)
    acc_uy = torch.zeros((n, n), device=device)
    for _ in range(args.avg_steps):
        g = stream(collide_bgk(f, tau))
        _, ux, uy = macroscopic(g)
        acc_ux += ux
        acc_uy += uy
        f = bounce_back_cells(moving_wall_bounce_back(g, inner, ux_w, uy_w), outer)
    acc_ux /= args.avg_steps
    acc_uy /= args.avg_steps
    u_th_mean = ((xx - cc) * acc_uy - (yy - cc) * acc_ux) / d

    # 径向分环（与正式档 analyse_profile 同款 bin：k <= d < k+1）
    d_np = d.cpu().numpy().astype(np.float64)
    fluid_np = fluid.cpu().numpy()
    u_np = u_th_mean.cpu().numpy().astype(np.float64)
    k = np.floor(d_np).astype(int)
    rs, us, ws = [], [], []
    for kk in range(int(np.floor(ri)) - 1, int(np.ceil(ro)) + 2):
        m = fluid_np & (k == kk)
        if m.sum() == 0:
            continue
        rs.append(float(d_np[m].mean()))
        us.append(float(u_np[m].mean()))
        ws.append(int(m.sum()))
    r_bins = np.asarray(rs)
    u_bins = np.asarray(us)
    w_bins = np.asarray(ws, dtype=np.float64)

    np.savez(
        args.out,
        ux=acc_ux.cpu().numpy(),
        uy=acc_uy.cpu().numpy(),
        u_th=u_np,
        d=d_np,
        fluid=fluid_np,
        r_bins=r_bins,
        u_bins=u_bins,
        w_bins=w_bins,
        ri=ri,
        ro=ro,
        n=n,
        omega_i=omega_i,
        tau=tau,
        nu=args.nu,
        re_i=args.re_i,
        steps=args.steps,
        elapsed_s=elapsed,
    )
    print(
        f"saved {args.out}: ri={ri} ro={ro} n={n} steps={args.steps} "
        f"tau={tau:.4f} omega_i={omega_i:.6e} t={elapsed:.1f}s "
        f"({args.steps / t_gap:.1f} t_gap)",
        flush=True,
    )


if __name__ == "__main__":
    main()

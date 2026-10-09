#!/usr/bin/env python
"""圆管 Poiseuille 流（3D D3Q19）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/poiseuille_3d_pipe/run.py 使用同一组库入口：
  - tensorlbm.solver3d.collide_bgk3d / stream3d
  - tensorlbm.d3q19.equilibrium3d / macroscopic3d
  - tensorlbm.boundaries3d.zou_he_inlet_velocity_3d / zou_he_outlet_pressure_3d
  - tensorlbm.boundaries3d.bounce_back_cells_3d（管壁半程反弹，post-streaming）

演示档把网格缩到 R=10（正式档 R=20/40），抛物线初始化 + 充分发展步数，
输出测量面速度剖面 / 轴向切片 / 沿程密度线 / 径向分环；定量判据以
正式档 result.json 存档为准。

用法：
    python docs/benchmarks/demos/poiseuille_3d_pipe_demo.py \
        --R 10 --steps 8000 --out demo.npz
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

from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402


def pipe_setup(R: int, L_over_R: int, device):
    """与正式档 run.py 的 pipe_setup 同款：方外框 + 圆截面 fluid d<=R。"""
    ny = nz = 2 * R + 3
    nx = L_over_R * R
    yc = zc = R + 1
    iz = torch.arange(nz, device=device, dtype=torch.float32).view(-1, 1)
    iy = torch.arange(ny, device=device, dtype=torch.float32).view(1, -1)
    d = torch.sqrt((iy - yc) ** 2 + (iz - zc) ** 2)
    fluid2d = d <= R
    wall_mask = (~fluid2d).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    return ny, nz, nx, yc, zc, d, fluid2d, wall_mask


def main():
    ap = argparse.ArgumentParser(description="3D 圆管 Poiseuille 场量演示")
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--L-over-R", type=int, default=6)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--avg-steps", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u-in", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="poiseuille_3d_pipe_demo.npz")
    args = ap.parse_args()

    R = args.R
    tau = args.tau
    u_in = args.u_in
    nu = (tau - 0.5) / 3.0
    R_eff = R + 0.5  # 半程反弹壁位（正式档同款）
    u_max_ana = 2.0 * u_in  # 质量守恒标称峰值
    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    ny, nz, nx, yc, zc, d, fluid2d, wall_mask = pipe_setup(R, args.L_over_R, device)

    # 抛物线初始化（正式档同款：rest density + 解析剖面，缩短入口发展段）
    d3 = d.unsqueeze(-1)
    ux0 = torch.where(
        fluid2d.unsqueeze(-1),
        u_max_ana * (1.0 - d3**2 / R_eff**2),
        torch.zeros_like(d3),
    )
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, ux0.expand(nz, ny, nx), torch.zeros_like(rho0),
                      torch.zeros_like(rho0), device=device)

    def step(f):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, 1.0)
        return bounce_back_cells_3d(f, wall_mask)

    t0 = time.time()
    x_meas = nx // 2
    for s in range(1, args.steps + 1):
        f = step(f)
        if s % 2000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            print(f"step={s} ucenter={float(ux[zc, yc, x_meas].item()):.6e}",
                  flush=True)
    elapsed = time.time() - t0

    # 末 avg_steps 步时间平均（正式档同款：测量面 x=nx//2）
    acc = torch.zeros((nz, ny), device=device)
    acc_rho = torch.zeros((nz, ny), device=device)
    for _ in range(args.avg_steps):
        f = step(f)
        rho, ux, _, _ = macroscopic3d(f)
        acc += ux[:, :, x_meas]
        acc_rho += rho[:, :, 0]
    acc /= args.avg_steps
    acc_rho /= args.avg_steps
    rho, ux, _, _ = macroscopic3d(f)

    # 径向分环（正式档 radial_profile 同款 bin：k <= d < k+1）
    d_np = d.cpu().numpy().astype(np.float64)
    fluid_np = fluid2d.cpu().numpy()
    u_np = acc.cpu().numpy().astype(np.float64)
    bin_idx = np.floor(d_np).astype(int)
    rs, us, cs_ = [], [], []
    for k in range(R + 1):
        m = fluid_np & (bin_idx == k)
        if m.sum() == 0:
            continue
        rs.append(float(d_np[m].mean()))
        us.append(float(u_np[m].mean()))
        cs_.append(int(m.sum()))
    r_bins = np.asarray(rs)
    u_bins = np.asarray(us)
    w_bins = np.asarray(cs_, dtype=np.float64)

    # 演示档 Q / 中心速度 / 水力半径 R_eff^Q（正式档同款公式）
    Q = float(u_np[fluid_np].sum())
    u_center = float(u_np[fluid_np & (d_np < 0.5)].mean())
    R_eff_Q_demo = math.sqrt(2.0 * Q / (math.pi * u_max_ana))
    u_center_pred = 2.0 * u_in * (float(R) / R_eff_Q_demo) ** 2

    np.savez(
        args.out,
        ux_mid=u_np,
        ux_axial=ux[zc].cpu().numpy(),  # x-y 平面（z=zc 切片）
        rho_axial=rho[zc].cpu().numpy(),
        rho_in_plane=acc_rho.cpu().numpy(),
        d=d_np,
        fluid=fluid_np,
        r_bins=r_bins,
        u_bins=u_bins,
        w_bins=w_bins,
        R=R,
        ny=ny,
        nz=nz,
        nx=nx,
        tau=tau,
        nu=nu,
        u_in=u_in,
        u_max_ana=u_max_ana,
        steps=args.steps,
        elapsed_s=elapsed,
        Q_demo=Q,
        u_center_demo=u_center,
        R_eff_Q_demo=R_eff_Q_demo,
        u_center_pred=u_center_pred,
    )
    print(
        f"saved {args.out}: R={R} grid={nz}x{ny}x{nx} steps={args.steps} "
        f"t={elapsed:.1f}s u_center={u_center:.6e} "
        f"R_eff_Q(demo)={R_eff_Q_demo:.4f} u_center_pred={u_center_pred:.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

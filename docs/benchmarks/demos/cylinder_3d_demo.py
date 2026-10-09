#!/usr/bin/env python
"""3D 展向周期圆柱 Re=40 稳态尾流 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/cylinder_3d/run.py 使用同一组库入口与
同一步进链：
  - tensorlbm.solver3d.collide_bgk3d（BGK）
  - NoDynamics（固体格恢复）+ tensorlbm.boundaries3d.bounce_back_cells_3d
    （half-way BB，stream 前）
  - tensorlbm.solver3d.stream3d_roll（z 周期输运）
  - tensorlbm.boundaries3d.far_field_bc_3d（y± 自由流 Dirichlet、
    x− 入流 / x+ 零梯度、z± 周期）
  - tensorlbm.obstacles.compute_obstacle_forces_3d（Ladd 动量交换，
    三口径：surface / all / interior —— 正式档验收口径 = surface）
  - tensorlbm.d3q19.equilibrium3d / macroscopic3d

演示档缩域缩径：D=20、横向 24D、nz=4（正式档 D=20/40、横向 40D、nz=1
z 周期），步数缩短；Re=40 低于涡脱阈值（Re_c≈47），尾流定常，量兴趣为
稳态 Cd。本演示输出中平面（z=nz/2）场量（|u|、展向涡量 ω_z、p′）与
三口径 Cd 收敛迹线；Cd 定量判据一律取正式档 result.json（2D 无限展向
数值簇验证）。

用法：
    python docs/benchmarks/demos/cylinder_3d_demo.py \
        --D 20 --steps 20000 --out demo.npz
"""

import argparse
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

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.obstacles import compute_obstacle_forces_3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402

RE = 40.0  # 低于涡脱阈值 Re_c≈47：定常尾流（与正式档一致）


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def surface_mask(solid):
    """壁面相邻固体格：固体格中至少一个 6-邻居为流体（同正式档定义）。"""
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def main():
    ap = argparse.ArgumentParser(description="3D 展向周期圆柱 Re=40 稳态演示")
    ap.add_argument("--D", type=int, default=20, help="直径格数（正式档 20/40）")
    ap.add_argument("--lateral", type=float, default=24.0, help="横向域（×D；正式档 40）")
    ap.add_argument("--nz", type=int, default=4, help="展向格数（z 周期）")
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="cylinder_3d_demo.npz")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "0")) or min(32, os.cpu_count() or 1))
    dev = torch.device(args.device)

    D = args.D
    R = D / 2.0
    nx = ny = int(round(args.lateral * D))
    nz = args.nz
    nu = args.u_in * D / RE
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * args.u_in**2 * (D * nz)

    solid = cylinder3d_mask(nx, ny, nz, nx // 2, ny // 2, R, dev)
    surf = surface_mask(solid)
    interior = solid & ~surf
    print(
        f"tensorlbm src ok; nx={nx} ny={ny} nz={nz} D={D} Re={RE} "
        f"tau={tau:.4f} solid={int(solid.sum())} surf={int(surf.sum())} "
        f"interior={int(interior.sum())} steps={args.steps}",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, args.u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0, ux0

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    def step(f):
        f_pre = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre  # NoDynamics：固体格恢复
        f = bounce_back_cells_3d(f, solid)  # half-way BB（stream 前）
        f = stream3d_roll(f)  # z 周期
        return far_field_bc_3d(f, args.u_in, bc_config=bc_config)

    cd_all_hist, cd_surf_hist, cd_int_hist = [], [], []
    t0 = time.time()
    for step_i in range(1, args.steps + 1):
        f = step(f)
        if step_i % args.sample == 0:
            # post-stream、pre-bounce 口径（此链 BB 在 stream 前施加，
            # 返回场的固体格即下一步将被反弹的 post-stream 分布）
            cd_all_hist.append(float(compute_obstacle_forces_3d(f, solid)[0].item()) / dpS)
            cd_surf_hist.append(float(compute_obstacle_forces_3d(f, surf)[0].item()) / dpS)
            cd_int_hist.append(float(compute_obstacle_forces_3d(f, interior)[0].item()) / dpS)
            n = min(20, len(cd_surf_hist))
            print(
                f"  step {step_i}: Cd_mem_surf={sum(cd_surf_hist[-n:])/n:.4f} "
                f"Cd_mem_all={sum(cd_all_hist[-n:])/n:.4f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
        if step_i % 1000 == 0 and not torch.isfinite(f).all():
            raise RuntimeError(f"diverged at step {step_i}")
    elapsed = time.time() - t0

    rho, ux, uy, uz = macroscopic3d(f)
    zk = nz // 2  # 中平面切片（z 向均匀，任取一层）
    n_win = max(1, min(50, len(cd_surf_hist)))
    print(
        f"saved {args.out}: Cd_surf_demo(末{n_win}样本)="
        f"{sum(cd_surf_hist[-n_win:])/n_win:.4f} t={elapsed:.1f}s"
        f"（演示档，定量判据见入库 result.json）",
        flush=True,
    )
    np.savez(
        args.out,
        ux=ux[zk].cpu().numpy(),
        uy=uy[zk].cpu().numpy(),
        uz_abs_max=float(uz.abs().max().item()),
        rho=rho[zk].cpu().numpy(),
        mask=solid[zk].cpu().numpy(),
        cd_hist=np.asarray(cd_surf_hist),
        cd_all_hist=np.asarray(cd_all_hist),
        cd_int_hist=np.asarray(cd_int_hist),
        nx=nx,
        ny=ny,
        nz=nz,
        D=D,
        re=RE,
        u_in=args.u_in,
        tau=tau,
        lateral_D=args.lateral,
        steps=args.steps,
        sample=args.sample,
        elapsed_s=elapsed,
    )


if __name__ == "__main__":
    main()

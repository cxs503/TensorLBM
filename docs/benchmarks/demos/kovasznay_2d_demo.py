#!/usr/bin/env python
"""Kovasznay 2D 稳态流 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/kovasznay_2d/run.py 使用同一组库入口与
同一步进链：
  - tensorlbm.solver.collide_bgk / stream（周期 gather，y 向周期）
  - tensorlbm.boundaries.zou_he_inlet_velocity（解析 Dirichlet 速度入口）
  - 出口零梯度 Neumann（f[:,:,-1] = f[:,:,-2]，与正式档同款内联，库无此函数）
  - tensorlbm.d2q9.equilibrium / macroscopic；初值 = 全场解析平衡态

解析解（Kovasznay 1948，与正式档同一公式实现）：
    u = U0·(1 − e^{λx'}·cos 2πy')，v = U0·(λ/2π)·e^{λx'}·sin 2πy')
    λ = Re/2 − sqrt(Re²/4 + 4π²)

演示档取 ny=64（= 正式验收扫描最粗档），其余条件（Re=40、U0=0.03、
x'∈[0,3]、20000 步、末 100 步时均）与正式档完全一致；输出 u/v 场与
演示档 L2 误差。定量判据一律取正式档 result.json。

用法：
    python docs/benchmarks/demos/kovasznay_2d_demo.py \
        --ny 64 --steps 20000 --out demo.npz
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

from tensorlbm.boundaries import zou_he_inlet_velocity  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

TWOPI = 2.0 * math.pi


def kov_lambda(re: float) -> float:
    """Kovasznay 衰减率 λ（负根，精确公式，与正式档同一实现）。"""
    return re / 2.0 - math.sqrt(re * re / 4.0 + 4.0 * math.pi * math.pi)


def analytic_field(nx, ny, u0, lam, xmax, dev, dt):
    """解析 u/v 场（晶格坐标：x'=i/nx·xmax，y'=j/ny，y 周期 1）。"""
    y = torch.arange(ny, device=dev, dtype=dt)
    x = torch.arange(nx, device=dev, dtype=dt)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xp = xx / nx * xmax
    yp = yy / ny
    e = torch.exp(lam * xp)
    u = u0 * (1.0 - e * torch.cos(TWOPI * yp))
    v = u0 * (lam / TWOPI) * e * torch.sin(TWOPI * yp)
    return u, v


def outlet_zero_gradient(f: torch.Tensor) -> torch.Tensor:
    """零梯度（Neumann）出口：出口列拷贝上游列全部 9 个分布函数。"""
    f = f.clone()
    f[:, :, -1] = f[:, :, -2]
    return f


def main():
    ap = argparse.ArgumentParser(description="Kovasznay 2D 稳态流演示")
    ap.add_argument("--ny", type=int, default=64, help="y 向格数（正式档 32/64/128）")
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--u0", type=float, default=0.03)
    ap.add_argument("--xmax", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="kovasznay_2d_demo.npz")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("DEMO_THREADS", "0")) or min(32, os.cpu_count() or 1))
    dev = torch.device(args.device)
    dt = torch.float32
    ny = args.ny
    nx = int(round(ny * args.xmax))

    lam = kov_lambda(args.re)
    nu = args.u0 * ny / args.re
    tau = 0.5 + 3.0 * nu

    u_ana, v_ana = analytic_field(nx, ny, args.u0, lam, args.xmax, dev, dt)
    f = equilibrium(torch.ones((ny, nx), device=dev, dtype=dt), u_ana, v_ana)
    u_in = u_ana[:, 0].contiguous()
    v_in = v_ana[:, 0].contiguous()

    print(
        f"tensorlbm src ok; nx={nx} ny={ny} tau={tau:.4f} lambda={lam:.6f} "
        f"steps={args.steps}",
        flush=True,
    )

    t0 = time.time()
    for step in range(1, args.steps + 1):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = outlet_zero_gradient(f)
        f = zou_he_inlet_velocity(f, u_in, v_in)
        if step % 5000 == 0:
            print(f"  step {step} ({time.time()-t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    # 末 100 步时间平均（稳态，去浮点噪声，与正式档同款测量）
    acc_u = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    acc_v = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    for _ in range(100):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = outlet_zero_gradient(f)
        f = zou_he_inlet_velocity(f, u_in, v_in)
        _, ux, uy = macroscopic(f)
        acc_u += ux.to(torch.float64)
        acc_v += uy.to(torch.float64)
    acc_u /= 100.0
    acc_v /= 100.0

    u_num = acc_u.cpu().numpy()
    v_num = acc_v.cpu().numpy()
    u_a = u_ana.cpu().numpy().astype(np.float64)
    v_a = v_ana.cpu().numpy().astype(np.float64)
    u_l2 = float(np.linalg.norm(u_num - u_a) / np.linalg.norm(u_a))
    v_l2 = float(np.linalg.norm(v_num - v_a) / np.linalg.norm(v_a))

    np.savez(
        args.out,
        u_num=u_num,
        v_num=v_num,
        u_ana=u_a,
        v_ana=v_a,
        nx=nx,
        ny=ny,
        re=args.re,
        u0=args.u0,
        tau=tau,
        lam=lam,
        steps=args.steps,
        u_l2_demo=u_l2,
        v_l2_demo=v_l2,
        elapsed_s=elapsed,
    )
    print(
        f"saved {args.out}: u_l2_demo={u_l2:.5f} v_l2_demo={v_l2:.5f} "
        f"t={elapsed:.1f}s（演示档；定量判据见入库 result.json）",
        flush=True,
    )


if __name__ == "__main__":
    main()

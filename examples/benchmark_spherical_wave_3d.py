#!/usr/bin/env python
"""3D球面波声学基准 (D3Q19 BGK, 密度缩放源 + 高斯脉冲).

三维脉动球辐射球面声波, 压力振幅按 1/r 衰减 (二维为 1/√r).
用于验证 D3Q19 格子上的三维声传播物理.

物理设置
--------
- 网格: nx=ny=nz=120 (D3Q19 BGK, tau=0.55)
- 源: 中心 (cx,cy,cz)=(60,60,60) 处半径 R0=5 的脉动球
  - delta_rho=0.01, omega=0.1
  - 高斯脉冲包络 (t0=80, sigma=40) 用于时间门控测量
- 吸收层 (海绵层): 目标平衡态, 宽度=25, 覆盖全部 6 个边界
- 监测点: 沿 x 轴距中心 r=15,20,25,30,35 处的压力

解析解 (三维球面波)
-------------------
  p'(r) = cs² · delta_rho · R0 / r · |exp(ikr)| / |exp(ikR0)|
  简化: p'(r) ∝ 1/r  (三维振幅按 1/r 衰减)
  空间衰减比: p(r2)/p(r1) = r1/r2

验证
----
1. 空间衰减: 测量各监测点峰值压力振幅, 与 1/r 衰减比较.
   比值 p(r2)/p(r1) 应等于 r1/r2. 目标: 误差 <10%.
2. 与二维对比: 三维衰减应比二维更陡 (1/r vs 1/√r).
   验证衰减指数 α ≈ -1 (二维为 -0.5). 目标: α ∈ [-1.2, -0.8].

运行
----
    PYTHONPATH=src python examples/benchmark_spherical_wave_3d.py --device cpu --steps 800
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d


def run_spherical_wave_3d(
    nx=120,
    ny=120,
    nz=120,
    tau=0.55,
    R0=5.0,
    delta_rho=0.01,
    omega=0.1,
    steps=800,
    device="cpu",
    log_every=100,
    sponge_width=25,
    pulse_t0=80.0,
    pulse_sigma=40.0,
):
    """运行三维球面波声学基准.

    返回包含峰值振幅、衰减比、衰减指数和通过/失败判定的字典.
    """
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    cs = math.sqrt(cs2)

    cx, cy, cz = nx // 2, ny // 2, nz // 2
    monitor_r = [15, 20, 25, 30, 35]
    # 监测点沿 x 轴: (cx+r, cy, cz) → 索引 [cz, cy, cx+r]
    monitor_pts = [(cz, cy, cx + r) for r in monitor_r]

    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nx, device=dev),
        indexing="ij",
    )
    dist = torch.sqrt(
        (xx.float() - cx) ** 2 + (yy.float() - cy) ** 2 + (zz.float() - cz) ** 2
    )

    # 源区域: 填充球 (紧凑单极子源)
    source = dist < R0  # [nz, ny, nx]

    # 海绵层: 在全部 3 个维度上向目标平衡态混合 (吸收声波)
    dist_x = torch.minimum(xx, nx - 1 - xx)
    dist_y = torch.minimum(yy, ny - 1 - yy)
    dist_z = torch.minimum(zz, nz - 1 - zz)
    dist_edge = torch.minimum(torch.minimum(dist_x, dist_y), dist_z).float()
    damping = torch.where(
        dist_edge < sponge_width,
        torch.sin(math.pi * dist_edge / (2 * sponge_width)) ** 2,
        torch.ones_like(dist_edge, device=dev),
    )
    feq_target = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # 压力时间序列
    p_ts = {r: [] for r in monitor_r}

    lam = 2 * math.pi * cs / omega
    k = omega / cs
    print("  三维球面波 (密度缩放源 + 高斯脉冲)")
    print(f"  网格: {nx}x{ny}x{nz} ({nx*ny*nz/1e6:.1f}M 网格)")
    print(f"  R0={R0}, delta_rho={delta_rho}, omega={omega}")
    print(f"  cs={cs:.4f}, lambda={lam:.1f}, k={k:.4f}")
    print(f"  脉冲: t0={pulse_t0}, sigma={pulse_sigma}")
    print(f"  海绵层: 宽度={sponge_width}")
    print(f"  {'step':>6s}  rho_max  " + "  ".join(f"p(r={r})" for r in monitor_r))

    for step in range(1, steps + 1):
        # --- LBM 步骤 ---
        f = collide_bgk3d(f, tau)
        f = stream3d(f)

        # --- 密度缩放源: 将 f 缩放到目标 rho, 保留速度 ---
        # 高斯包络正弦脉冲
        env = math.exp(-((step - pulse_t0) / pulse_sigma) ** 2)
        rho_src_val = 1.0 + delta_rho * math.sin(omega * step) * env

        rho_cur, ux_cur, uy_cur, uz_cur = macroscopic3d(f)
        scale = torch.where(
            source,
            rho_src_val / rho_cur.clamp(min=1e-10),
            torch.ones_like(rho_cur),
        )
        f = f * scale.unsqueeze(0)

        # --- 海绵层: 向目标平衡态混合 (吸收声波) ---
        f = feq_target + (f - feq_target) * damping

        # --- 记录监测点压力 ---
        rho_sp, ux_sp, uy_sp, uz_sp = macroscopic3d(f)
        for i, r in enumerate(monitor_r):
            mz, my, mx = monitor_pts[i]
            p_ts[r].append(float((rho_sp[mz, my, mx] - 1.0) * cs2))

        if step % log_every == 0 or step == steps:
            pvals = [p_ts[r][-1] for r in monitor_r]
            print(
                f"  {step:6d}  {float((rho_sp - rho0).abs().max()):.6f}  "
                + "  ".join(f"{v:+.6f}" for v in pvals),
                flush=True,
            )

    # --- 峰值振幅 (时间门控: 脉冲在反射到达前通过) ---
    peaks = {}
    for r in monitor_r:
        ts = p_ts[r]
        peaks[r] = max(abs(x) for x in ts) if ts else 0.0

    # --- 空间衰减比 (LBM vs 1/r 解析) ---
    print("\n  空间衰减比 (LBM vs 1/r 解析):")
    print(f"  {'pair':>12s}  {'LBM':>8s} {'1/r':>8s} {'err%':>6s}")
    decay_errs = []
    for i in range(len(monitor_r) - 1):
        r1, r2 = monitor_r[i], monitor_r[i + 1]
        if peaks[r1] > 0 and peaks[r2] > 0:
            lbm_ratio = peaks[r2] / peaks[r1]
            ana_ratio = r1 / r2  # 1/r 衰减: p(r2)/p(r1) = r1/r2
            d_err = abs(1 - lbm_ratio / ana_ratio) * 100
            decay_errs.append(d_err)
            print(f"  r={r1}->{r2:3d}  {lbm_ratio:8.3f} {ana_ratio:8.3f} {d_err:6.1f}")

    # --- 衰减指数拟合: p(r) = A * r^alpha ---
    # 对 log(p) vs log(r) 做线性回归
    r_arr = np.array(monitor_r, dtype=np.float64)
    p_arr = np.array([peaks[r] for r in monitor_r], dtype=np.float64)
    # 只用有效点
    valid = p_arr > 0
    if valid.sum() >= 2:
        log_r = np.log(r_arr[valid])
        log_p = np.log(p_arr[valid])
        # 线性回归: log_p = log_A + alpha * log_r
        alpha_fit, log_A = np.polyfit(log_r, log_p, 1)
    else:
        alpha_fit = 0.0

    print(f"\n  衰减指数拟合: p(r) = A * r^alpha")
    print(f"    alpha = {alpha_fit:.3f}  (三维解析: -1.0, 二维: -0.5)")

    # --- 验证 ---
    print("\n  验证:")
    checks = []

    # 1. 波生成
    gen_ok = max(peaks.values()) > 1e-6
    checks.append(("波生成", gen_ok, f"p_max={max(peaks.values()):.6f}"))
    print(f"    [{'PASS' if gen_ok else 'FAIL'}] 波生成: p_max={max(peaks.values()):.6f}")

    # 2. 波传播: 所有监测点有信号
    prop_ok = all(peaks[r] > 1e-7 for r in monitor_r)
    checks.append(("所有监测点波传播", prop_ok, ""))
    print(f"    [{'PASS' if prop_ok else 'FAIL'}] 波传播: 所有监测点有信号")

    # 3. 空间衰减: 误差 <10%
    decay_ok = len(decay_errs) > 0 and max(decay_errs) < 10.0
    decay_avg = sum(decay_errs) / len(decay_errs) if decay_errs else 0
    checks.append(("空间衰减 (1/r, 误差<10%)", decay_ok, f"avg_err={decay_avg:.1f}%"))
    print(f"    [{'PASS' if decay_ok else 'FAIL'}] 空间衰减: avg_err={decay_avg:.1f}% (1/r)")

    # 4. 衰减指数: alpha ∈ [-1.2, -0.8]
    alpha_ok = -1.2 <= alpha_fit <= -0.8
    checks.append(("衰减指数 (alpha∈[-1.2,-0.8])", alpha_ok, f"alpha={alpha_fit:.3f}"))
    print(f"    [{'PASS' if alpha_ok else 'FAIL'}] 衰减指数: alpha={alpha_fit:.3f} (目标: -1.0)")

    # 5. 三维 vs 二维: alpha 应接近 -1 而非 -0.5
    dim_ok = alpha_fit < -0.7
    checks.append(("三维衰减陡于二维", dim_ok, f"alpha={alpha_fit:.3f} < -0.7"))
    print(f"    [{'PASS' if dim_ok else 'FAIL'}] 三维衰减陡于二维: alpha={alpha_fit:.3f} < -0.7")

    all_pass = all(c[1] for c in checks)
    print(f"\n  {'PASS' if all_pass else 'FAIL'} — 三维球面波声传播")
    print(f"  空间衰减 {decay_avg:.1f}% 误差, 衰减指数 alpha={alpha_fit:.3f}")
    return {
        "peaks": peaks,
        "decay_errs": decay_errs,
        "alpha": alpha_fit,
        "checks": checks,
        "all_pass": all_pass,
    }


def main():
    p = argparse.ArgumentParser(
        description="三维球面波声学基准 (D3Q19 BGK, 密度缩放源 + 高斯脉冲)"
    )
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--ny", type=int, default=120)
    p.add_argument("--nz", type=int, default=120)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--R0", type=float, default=5.0)
    p.add_argument("--delta-rho", type=float, default=0.01)
    p.add_argument("--omega", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sponge-width", type=int, default=25)
    p.add_argument("--pulse-t0", type=float, default=80.0)
    p.add_argument("--pulse-sigma", type=float, default=40.0)
    args = p.parse_args()
    print("=" * 60)
    print("  三维球面波声学基准")
    print("  (D3Q19 BGK, 密度缩放源 + 高斯脉冲 + 时间门控测量)")
    print("=" * 60)
    run_spherical_wave_3d(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        R0=args.R0,
        delta_rho=args.delta_rho,
        omega=args.omega,
        steps=args.steps,
        device=args.device,
        log_every=args.log_every,
        sponge_width=args.sponge_width,
        pulse_t0=args.pulse_t0,
        pulse_sigma=args.pulse_sigma,
    )


if __name__ == "__main__":
    main()

#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B5-RE200: 2D cylinder Re=200 vortex shedding — MRT + seeding + downstream sponge layer.

直接复用已验证的 Re=100 工程链（/tmp/b5_sponge.py，benchmarks/verified/cylinder_re100）：
  1. 播种：入口 4 列正弦侧向扰动（St_seed=0.195, 10% 振幅, 仅 2 周期），
     对称破缺后频率自选；速度权重取库常量 d2q9.C 列（修复过的手写权重 bug）。
  2. sponge：下游最后 10D 列 tau_field 逐格渐变（alpha=10, 平方渐变），
     消除 far-field 零梯度出口的尾流反射（Re=100 v3 NaN 根因）。
  3. MRT 碰撞（collide_mrt 支持 tau_field）+ 固体冻结 + 表面格 MEM 测力 +
     每 2000 步质量修正。

物理：Re=200, u=0.08, 域 40D x 40D（阻塞 2.5%），圆柱中心 (12D, 20D)。
  nu = u*D/Re, tau = 3*nu + 0.5
    D=48: nu=0.0192,  tau=0.5576
    D=64: nu=0.0256,  tau=0.5768
参考：Braza et al. 1986 (JFM 165) Re=200：Cd≈1.33（文献 1.28-1.40）、St≈0.19-0.20。
St 提取：Cl 序列 FFT 峰值 + log 谱抛物线（parabolic）精化。
"""

import argparse
import json
import math
import sys
import time

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch

from tensorlbm.boundaries import (
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

REF_CD = 1.33  # Braza 1986 Re=200（文献窗 1.28-1.40）
REF_CD_LO, REF_CD_HI = 1.28, 1.40
REF_ST = 0.195  # Braza 1986 Re=200（文献窗 0.19-0.20）
REF_ST_LO, REF_ST_HI = 0.19, 0.20


def run_cylinder(
    D,
    u_in,
    re=200,
    n_steps=60000,
    warmup_frac=0.5,
    nx_D=40,
    ny_D=40,
    device="cuda:1",
    seed_amp=0.10,
    progress_every=5000,
    seed_cols=4,
    alpha=10.0,
    sponge_d=10.0,
    power=2.0,
    save_series=None,
    st_seed=0.195,
):
    dev = torch.device(device)
    radius = D / 2.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    nx, ny = nx_D * D, ny_D * D
    cx, cy = 12 * D, ny / 2.0
    print(
        f"[Re=200] D={D} nu={nu:.5f} tau={tau:.5f} grid={nx}x{ny} "
        f"device={device} steps={n_steps} St_seed={st_seed}",
        flush=True,
    )

    solid = cylinder_mask(nx, ny, cx, cy, radius, dev)
    fluid = ~solid
    surface = solid & (
        torch.roll(fluid, 1, 0)
        | torch.roll(fluid, -1, 0)
        | torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)
    )
    dyn_p = 0.5 * u_in**2 * D

    # ---- Sponge layer（下游最后 sponge_d*D 列，sigma 平方渐变） ----
    x0 = int(nx - sponge_d * D)  # sponge 起点
    W = nx - x0  # sponge 宽度（格）
    sigma = make_sponge_strength(ny, nx, x0, W, power=power, device=dev)
    tau_field = tau * (1.0 + alpha * sigma)  # (ny, nx) 逐格松弛时间
    n_sponge = W
    print(
        f"  [sponge] x0={x0} W={W} ({sponge_d}D) alpha={alpha} "
        f"tau={tau:.4f} -> tau_max={float(tau_field.max()):.3f} "
        f"(nu_max/nu={((float(tau_field.max()) - 0.5) / (tau - 0.5)):.1f}x)",
        flush=True,
    )

    St_seed = st_seed
    T_seed = D / (St_seed * u_in)  # 播种周期（格步）
    n_seed = int(2 * T_seed)  # 2 个周期

    rho0 = torch.ones((ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)
    initial_mass = float(f.sum().item())

    # 速度权重（D2Q9 c_x/c_y）——直接取自库常量 C（b4v3 手写权重 bug 的修复）
    from tensorlbm.d2q9 import C as C2D

    c2d = C2D.to(dev).float()
    cx_w = c2d[:, 0].view(9, 1, 1)
    cy_w = c2d[:, 1].view(9, 1, 1)

    t0 = time.time()
    cd_list, cl_list = [], []
    cl_series = []  # 每 10 步存 Cl 用于 FFT
    cd_series = []  # 每 10 步存 Cd 用于事后分段时均
    steps_done = 0
    nan_at = None

    for step in range(1, n_steps + 1):
        before = f.clone()
        collided = collide_mrt(f, tau, tau_field=tau_field)
        f = torch.where(solid.unsqueeze(0), before, collided)
        f = stream(f)
        fx, fy = compute_obstacle_forces(f, surface)
        f = far_field_bc_2d(f, u_in, obstacle_mask=solid)
        if step % 2000 == 0:
            f = f * (initial_mass / f.sum().item())

        # 播种窗口：入口列注入正弦侧向扰动（仅 2 周期）
        if step <= n_seed:
            phase = 2 * math.pi * step / T_seed
            rho_col = f.sum(0)[:, :seed_cols]
            ux_col = (f * cx_w).sum(0)[:, :seed_cols] / rho_col.clamp(min=1e-12)
            uy_col = (f * cy_w).sum(0)[:, :seed_cols] / rho_col.clamp(min=1e-12)
            uy_col += seed_amp * u_in * math.sin(phase)
            feq_col = equilibrium(rho_col, ux_col, uy_col)
            f[:, :, :seed_cols] = feq_col

        # 时间平均
        if step > int(n_steps * warmup_frac):
            cd_list.append(float(fx.item()) / dyn_p)
            cl_list.append(float(fy.item()) / dyn_p)
        if step % 10 == 0:
            cl_series.append(float(fy.item()) / dyn_p)
            cd_series.append(float(fx.item()) / dyn_p)
        if step % 1000 == 0 and torch.isnan(f).any():
            nan_at = step
            print(f"  *** NaN detected at step {step} ***", flush=True)
            break
        if step % progress_every == 0:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            cl_std = (
                sum((c - sum(cl_list) / max(len(cl_list), 1)) ** 2 for c in cl_list)
                / max(len(cl_list), 1)
            ) ** 0.5
            print(
                f"  step {step}: Cd_avg={cd_avg:.4f} Cl_std={cl_std:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        steps_done = step

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    cl_avg = sum(cl_list) / max(len(cl_list), 1)
    cl_std = (sum((c - cl_avg) ** 2 for c in cl_list) / max(len(cl_list), 1)) ** 0.5
    # 时均窗前后两半 Cd 漂移（统计收敛检查）
    half = max(len(cd_list) // 2, 1)
    cd_h1 = sum(cd_list[:half]) / half
    cd_h2 = sum(cd_list[half:]) / max(len(cd_list) - half, 1)

    # St: Cl 序列 FFT + parabolic（抛物线）精化
    st_raw, st = float("nan"), float("nan")
    if len(cl_series) > 64 and nan_at is None:
        import numpy as np

        sig = np.array(cl_series) - np.mean(cl_series)
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        freqs = np.fft.rfftfreq(len(sig), d=10.0)  # 每 10 步采样
        k = int(np.argmax(spec[1:])) + 1
        st_raw = freqs[k] * D / u_in
        st = st_raw
        if 0 < k < len(spec) - 1:
            lp = np.log(np.maximum(spec[k - 1 : k + 2], 1e-30))
            denom = lp[0] - 2.0 * lp[1] + lp[2]
            if abs(denom) > 1e-12:
                delta = 0.5 * (lp[0] - lp[2]) / denom  # 抛物线顶点偏移（bin 单位）
                delta = max(-0.5, min(0.5, delta))
                st = (freqs[k] + delta * (freqs[1] - freqs[0])) * D / u_in
    if save_series:
        import numpy as np

        np.save(save_series, np.array(cl_series))
        if save_series.endswith(".npy"):
            np.save(save_series.replace(".npy", "_cd.npy"), np.array(cd_series))

    return {
        "D": D,
        "re": re,
        "nu": nu,
        "nx_D": nx_D,
        "ny_D": ny_D,
        "tau": tau,
        "sponge": {
            "x0": x0,
            "width": n_sponge,
            "alpha": alpha,
            "power": power,
            "sigma": "quadratic",
            "tau_max": float(tau_field.max()),
        },
        "seed": {
            "st_seed": St_seed,
            "amp": seed_amp,
            "cols": seed_cols,
            "n_periods": 2,
            "T_seed_steps": T_seed,
            "n_seed_steps": n_seed,
        },
        "cd": cd_avg,
        "cd_half1": cd_h1,
        "cd_half2": cd_h2,
        "cl": cl_avg,
        "cl_std": cl_std,
        "st": float(st),
        "st_raw": float(st_raw),
        "steps_done": steps_done,
        "nan_at": nan_at,
        "warmup_frac": warmup_frac,
        "err_pct": (cd_avg - REF_CD) / REF_CD * 100,
        "st_err_pct": (st - REF_ST) / REF_ST * 100,
        "ref_cd": REF_CD,
        "ref_cd_range": [REF_CD_LO, REF_CD_HI],
        "ref_st": REF_ST,
        "ref_st_range": [REF_ST_LO, REF_ST_HI],
        "wall_s": time.time() - t0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=48)
    ap.add_argument("--u", type=float, default=0.08)
    ap.add_argument("--re", type=float, default=200)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--sponge-d", type=float, default=10.0)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--st-seed", type=float, default=0.195)
    ap.add_argument("--out", default="/tmp/b5_re200.json")
    ap.add_argument("--series", default=None, help="npz/npy path for Cl series")
    a = ap.parse_args()
    r = run_cylinder(
        a.D,
        a.u,
        a.re,
        a.steps,
        device=a.device,
        warmup_frac=a.warmup_frac,
        alpha=a.alpha,
        sponge_d=a.sponge_d,
        power=a.power,
        save_series=a.series,
        st_seed=a.st_seed,
    )
    print(json.dumps(r, indent=2))
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=2)

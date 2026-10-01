#!/usr/bin/env python
"""Womersley 振荡流（周期脉动管流/通道流）— 演示脚本。

与正式验证档 benchmarks/verified/womersley/run.py 使用同一组库入口
（零手写物理内核）：
  - tensorlbm.solver.collide_bgk / stream
  - tensorlbm.turbulent_channel._apply_body_force_2d（库自用通道体力路径，
    每步瞬时 a(t)=a0·cos(w·t)）
  - tensorlbm.d2q9.OPPOSITE + 预流半程 bounce-back 壁面范式
  - tensorlbm.d2q9.equilibrium / macroscopic（初值与测量）

演示档与入库扫描同参数（H=59, alpha=4, tau=0.8, u_peak=0.03）真实 CPU
步进，跳过 startup 瞬态后测 4 个整周期：输出 8 相位瞬时剖面与逐行
复振幅（phasor 解调）；解析解为 Womersley (1955) 复数双曲级数。
定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/womersley_demo.py \
        --H 59 --alpha 4 --tau 0.8 --upeak 0.03 --out demo.npz
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

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulent_channel import _apply_body_force_2d  # noqa: E402

MEASURE_PERIODS = 4
N_PHASES = 8


def womersley_phasor(y_phys, H, alpha, a0, omega):
    """Womersley 稳态周期解的复振幅 U(y)（numpy 复数算术，无 scipy）。"""
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H / 2.0) / (H / 2.0)
    return (a0 / (1j * omega)) * (1.0 - np.cosh(lam * yp) / np.cosh(lam))


def main():
    ap = argparse.ArgumentParser(description="Womersley 振荡通道流演示")
    ap.add_argument("--H", type=int, default=59)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--upeak", type=float, default=0.03)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="womersley_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("DEMO_THREADS", "32")), os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    H, alpha, tau, u_peak = args.H, args.alpha, args.tau, args.upeak
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)
    omega = 4.0 * alpha * alpha * nu / (H_eff * H_eff)
    T = 2.0 * math.pi / omega
    T_int = int(round(T))

    rows = np.arange(1, ny - 1, dtype=np.float64)
    y_phys = rows - 0.5
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H_eff / 2.0) / (H_eff / 2.0)
    shape_max = float(np.max(np.abs(1.0 - np.cosh(lam * yp) / np.cosh(lam))))
    a0 = u_peak * omega / shape_max  # 定标：解析峰值速度 = u_peak

    tau_diff = H_eff * H_eff / (math.pi**2 * nu)
    skip_periods = max(5, math.ceil(6.5 * tau_diff / T))
    skip_steps = skip_periods * T_int
    n_meas = MEASURE_PERIODS * T_int
    total_steps = skip_steps + n_meas

    wall = torch.zeros((ny, args.nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    rho0 = torch.ones((ny, args.nx), device=device)
    u0 = torch.zeros((ny, args.nx), device=device)
    f = equilibrium(rho0, u0, u0)

    Sc = torch.zeros(ny, device=device)
    Ss = torch.zeros(ny, device=device)
    snap_steps = [skip_steps + 1 + (m * T_int) // N_PHASES for m in range(N_PHASES)]
    snaps = {}

    t0 = time.time()
    for step in range(1, total_steps + 1):
        a_t = torch.tensor(a0 * math.cos(omega * step), device=device)
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(f.device)], f)
        f = _apply_body_force_2d(f, a_t)
        f = stream(f)
        if step > skip_steps:
            _, ux, _ = macroscopic(f)
            col_mean = ux.mean(dim=1)
            Sc += col_mean * math.cos(omega * step)
            Ss += col_mean * math.sin(omega * step)
            if step in snap_steps:
                snaps[step] = col_mean.detach().clone()
    elapsed = time.time() - t0

    ur = (2.0 / n_meas) * Sc.cpu().numpy().astype(np.float64)
    ui = (-(2.0 / n_meas)) * Ss.cpu().numpy().astype(np.float64)
    z_num = (ur + 1j * ui)[1 : ny - 1]
    z_ana = womersley_phasor(y_phys, H_eff, alpha, a0, omega)

    np.savez(
        args.out,
        y=y_phys,
        z_num_re=np.real(z_num),
        z_num_im=np.imag(z_num),
        z_ana_re=np.real(z_ana),
        z_ana_im=np.imag(z_ana),
        snap_steps=np.array(snap_steps),
        snap_u=np.stack([snaps[s].cpu().numpy().astype(np.float64) for s in snap_steps])[
            :, 1 : ny - 1
        ],
        H=H,
        ny=ny,
        nx=args.nx,
        H_eff=H_eff,
        alpha=alpha,
        omega=omega,
        T_int=T_int,
        a0=a0,
        u_peak_ana=u_peak,
        tau=tau,
        nu=nu,
        skip_steps=skip_steps,
        total_steps=total_steps,
        elapsed_s=elapsed,
    )
    row_mask = np.abs(z_ana) >= 0.10 * u_peak
    cx = np.abs(z_num / z_ana - 1.0)[row_mask] * 100.0
    print(
        f"complex_rel_max={cx.max():.3f}% steps={total_steps} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

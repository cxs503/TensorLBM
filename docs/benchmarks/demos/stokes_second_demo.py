#!/usr/bin/env python
"""Stokes 第二问题（振荡平板）— 受迫振荡边界层演示脚本。

与正式验证档 benchmarks/verified/stokes_second_problem/run.py 使用同一组
库入口（零手写物理内核）：
  - tensorlbm.solver.collide_bgk / stream
  - tensorlbm.lid_driven_cavity.zou_he_moving_lid（振荡平板=库移动盖，
    每步时变标量 u_lid(t)=U·cos(w·t)）
  - tensorlbm.d2q9.equilibrium / macroscopic（初值与测量）
  - 远场：自由滑移镜面反射（verified/stokes_first_problem 同款内联范式）

演示档与入库扫描同参数（H=100, kH=6, tau=0.8, U=0.05）真实 CPU 步进，
跳过 startup 瞬态后测 4 个整周期：输出 8 相位瞬时剖面、复振幅/相位
剖面（phasor 解调）、近/中/远三深度一个周期内的速度时序；
定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/stokes_second_demo.py \
        --H 100 --kH 6 --tau 0.8 --U 0.05 --out demo.npz
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

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.lid_driven_cavity import zou_he_moving_lid  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)
MEASURE_PERIODS = 4
N_PHASES = 8


def specular_replacement(f_pre):
    return f_pre[SPECULAR.to(f_pre.device)]


def main():
    ap = argparse.ArgumentParser(description="Stokes 第二问题（振荡平板）演示")
    ap.add_argument("--H", type=int, default=100)
    ap.add_argument("--kH", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="stokes_second_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("DEMO_THREADS", "32")), os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    H, kH, tau, U, nx = args.H, args.kH, args.tau, args.U, args.nx
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    k = kH / H
    omega = 2.0 * k * k * nu
    T = 2.0 * math.pi / omega
    T_int = int(round(T))
    tau_diff = H * H / (math.pi**2 * nu)
    skip_periods = max(10, math.ceil(8.0 * tau_diff / T))
    skip_steps = skip_periods * T_int
    n_meas = MEASURE_PERIODS * T_int
    total_steps = skip_steps + n_meas

    wall_bottom = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_bottom[0, :] = True

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    Sc = torch.zeros(ny, device=device)
    Ss = torch.zeros(ny, device=device)
    snap_steps = [skip_steps + 1 + (m * T_int) // N_PHASES for m in range(N_PHASES)]
    snaps = {}
    # 三个深度（板下 5 / 20 / 45 格）在最后一个测量周期内的逐相位时序
    trace_rows = [ny - 1 - d for d in (5, 20, 45)]
    trace_every = max(1, T_int // 64)
    traces = {r: [] for r in trace_rows}
    trace_steps = []

    t0 = time.time()
    for step in range(1, total_steps + 1):
        u_lid_t = torch.tensor(U * math.cos(omega * step), device=device)
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall_bottom.unsqueeze(0), specular_replacement(f_pre), f)
        f = stream(f)
        f = zou_he_moving_lid(f, u_lid_t)
        if step > skip_steps:
            _, ux, _ = macroscopic(f)
            col_mean = ux.mean(dim=1)
            Sc += col_mean * math.cos(omega * step)
            Ss += col_mean * math.sin(omega * step)
            if step in snap_steps:
                snaps[step] = col_mean.detach().clone()
            if step > total_steps - T_int and step % trace_every == 0:
                if not trace_steps or step != trace_steps[-1]:
                    trace_steps.append(step)
                    for r in trace_rows:
                        traces[r].append(float(col_mean[r].item()))
    elapsed = time.time() - t0

    ur = (2.0 / n_meas) * Sc.cpu().numpy().astype(np.float64)
    ui = (-(2.0 / n_meas)) * Ss.cpu().numpy().astype(np.float64)
    A_num = np.hypot(ur, ui)
    phi_num = -np.arctan2(ui, ur)

    rows = np.arange(ny)
    y = (ny - 1) - rows.astype(np.float64)  # 板下深度（Zou/He 板在节点上）
    A_ana = U * np.exp(-k * y)
    phi_ana = k * y

    np.savez(
        args.out,
        y=y[1 : ny - 1],
        A_num=A_num[1 : ny - 1],
        phi_num_deg=np.degrees(phi_num[1 : ny - 1]),
        A_ana=A_ana[1 : ny - 1],
        phi_ana_deg=np.degrees(phi_ana[1 : ny - 1]),
        snap_steps=np.array(snap_steps),
        snap_u=np.stack([snaps[s].cpu().numpy().astype(np.float64) for s in snap_steps])[
            :, 1 : ny - 1
        ],
        trace_steps=np.array(trace_steps),
        trace_u=np.stack([np.array(traces[r]) for r in trace_rows]),
        trace_depths=np.array([5.0, 20.0, 45.0]),
        H=H,
        ny=ny,
        nx=nx,
        kH=kH,
        k=k,
        omega=omega,
        T_int=T_int,
        skip_steps=skip_steps,
        total_steps=total_steps,
        tau=tau,
        nu=nu,
        U=U,
        elapsed_s=elapsed,
    )
    fluid = (rows >= 1) & (rows <= ny - 2)
    amp_mask = fluid & (A_ana >= 0.10 * U)
    phase_mask = amp_mask & (phi_ana >= 1.0)
    amp_rel = np.abs(A_num / A_ana - 1.0)[amp_mask] * 100.0
    dphi = np.angle(np.exp(1j * (phi_num - phi_ana)))
    phase_rel = (np.abs(dphi) / phi_ana)[phase_mask] * 100.0
    z = ur + 1j * ui
    za = A_ana * np.exp(-1j * phi_ana)
    cx_rel = np.abs(z / za - 1.0)[amp_mask] * 100.0
    print(
        f"amp={amp_rel.max():.3f}% phase={phase_rel.max():.3f}% "
        f"complex={cx_rel.max():.3f}% steps={total_steps} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

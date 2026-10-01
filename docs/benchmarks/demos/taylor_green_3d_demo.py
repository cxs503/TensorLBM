#!/usr/bin/env python
"""3D Taylor–Green 涡衰减 — 演示脚本。

与正式验证档 benchmarks/verified/taylor_green_3d/run.py 使用同一组库入口
（零手写物理内核）：
  - tensorlbm.d3q19.equilibrium3d / macroscopic3d（初值与测量）
  - tensorlbm.solver3d.collide_bgk3d / stream3d（周期模运算内建，
    步序 stream3d→collide_bgk3d）

初始场（OpenLB tgv3d 同款）：
    u = (U0·sin(kx)cos(ky)cos(kz), -U0·cos(kx)sin(ky)cos(kz), 0)
波矢 (±k,±k,±k) → |κ|²=3k²：速度衰减率 3νk²、能量衰减率 6νk²
（任务书 e^{-2νk²t} 是 2D 速度衰减率，3D 套用会报约 +200% 假误差）。
3D TG 非 NS 精确解：涡拉伸使 E_z 从 0 增长（演示档同步记录分分量能量）。

演示档 N=48³（入库档 N=64/96/128³）同 Re=24/U0=0.05，CPU 真跑；
定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/taylor_green_3d_demo.py \
        --n 48 --re 24 --u0 0.05 --steps 1200 --out demo.npz
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

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="3D Taylor-Green 涡衰减演示")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--re", type=float, default=24.0)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--record-every", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="taylor_green_3d_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    n, re, u0 = args.n, args.re, args.u0
    k = 2.0 * math.pi / n
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    gamma_e = 6.0 * nu * k * k
    gamma_vel = 3.0 * nu * k * k

    z, y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
    uy0 = -u0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
    uz0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium3d(rho, ux0, uy0, uz0)

    snap_steps = [int(round(f_ * args.steps)) for f_ in (0.1, 0.5, 1.0)]
    snap_set = set(snap_steps)
    times, energies, exs, eys, ezs, umaxs, snaps = [], [], [], [], [], [], {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        f = collide_bgk3d(stream3d(f), tau)
        if step % args.record_every == 0 or step in snap_set:
            _, uxm, uym, uzm = macroscopic3d(f)
            ex = float((0.5 * (uxm * uxm)).mean().item())
            ey = float((0.5 * (uym * uym)).mean().item())
            ez = float((0.5 * (uzm * uzm)).mean().item())
            if step % args.record_every == 0:
                times.append(step)
                energies.append(ex + ey + ez)
                exs.append(ex)
                eys.append(ey)
                ezs.append(ez)
                umaxs.append(
                    float(
                        (uxm * uxm + uym * uym + uzm * uzm).sqrt().max().item()
                    )
                )
            if step in snap_set:
                snaps[step] = (
                    uxm[n // 2].detach().clone(),
                    uzm[n // 2].detach().clone(),
                )
    elapsed = time.time() - t0

    np.savez(
        args.out,
        times=np.array(times),
        energies=np.array(energies),
        exs=np.array(exs),
        eys=np.array(eys),
        ezs=np.array(ezs),
        umaxs=np.array(umaxs),
        snap_steps=np.array(snap_steps),
        snap_ux_mid=np.stack(
            [snaps[s][0].cpu().numpy().astype(np.float64) for s in snap_steps]
        ),
        snap_uz_mid=np.stack(
            [snaps[s][1].cpu().numpy().astype(np.float64) for s in snap_steps]
        ),
        ux0_mid=ux0[n // 2].cpu().numpy().astype(np.float64),
        n=n,
        re=re,
        u0=u0,
        nu=nu,
        tau=tau,
        k=k,
        steps=args.steps,
        gamma_e_theory=gamma_e,
        gamma_vel_theory=gamma_vel,
        elapsed_s=elapsed,
    )
    t = np.asarray(times, dtype=np.float64)
    ge = -float(np.polyfit(t, np.log(np.asarray(energies)), 1)[0])
    gv = -float(
        np.polyfit(
            t, np.log(np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))), 1
        )[0]
    )
    print(
        f"gamma_E_sim={ge:.6e} theory={gamma_e:.6e} err_E={(ge / gamma_e - 1) * 100:+.4f}% "
        f"gamma_vel_sim={gv:.6e} err_vel={(gv / gamma_vel - 1) * 100:+.4f}% "
        f"ez_max={max(ezs):.3e} steps={args.steps} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

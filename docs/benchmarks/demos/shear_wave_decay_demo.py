#!/usr/bin/env python
"""2D 衰减剪切波 — 粘性耗散演示脚本。

与正式验证档 benchmarks/verified/shear_wave_decay/run.py 使用同一组库入口
（零手写物理内核）：
  - tensorlbm.d2q9.equilibrium / macroscopic（初值与测量）
  - tensorlbm.solver.collide_bgk / stream（周期 wrap 内建，步序 stream→collide）

解析解：u(x,y,t) = U0·sin(ky)·e^{-νk²t}，v≡0（不可压 NS 精确解，
纯剪切无非线性项）。演示档 H=64 与入库两档同物理参数（tau=0.8, U0=0.05），
输出多时刻 u(y) 剖面、能量/峰值速度历史与场快照；
定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/shear_wave_decay_demo.py \
        --n 64 --tau 0.8 --u0 0.05 --steps 3200 --out demo.npz
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
from tensorlbm.solver import collide_bgk, stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="2D 衰减剪切波演示")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="shear_wave_decay_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("DEMO_THREADS", "32")), os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    n, tau, u0 = args.n, args.tau, args.u0
    k = 2.0 * math.pi / n
    nu = (tau - 0.5) / 3.0
    gamma_vel = nu * k * k

    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * y)
    uy0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)

    snap_steps = [int(round(f_ * args.steps)) for f_ in (0.1, 0.3, 0.6, 1.0)]
    snap_set = set(snap_steps)
    times, energies, umaxs, profiles, snaps = [], [], [], {}, {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        f = collide_bgk(stream(f), tau)
        if step % args.record_every == 0 or step in snap_set:
            _, uxm, uym = macroscopic(f)
            if step % args.record_every == 0:
                times.append(step)
                energies.append(float((0.5 * (uxm * uxm + uym * uym)).mean().item()))
                umaxs.append(
                    float((uxm * uxm + uym * uym).sqrt().max().item())
                )
                profiles[step] = uxm.mean(dim=1).cpu().numpy().astype(np.float64)
            if step in snap_set:
                snaps[step] = uxm.detach().clone()
    elapsed = time.time() - t0

    yy = np.arange(n, dtype=np.float64)
    np.savez(
        args.out,
        y=yy,
        k=k,
        nu=nu,
        times=np.array(times),
        energies=np.array(energies),
        umaxs=np.array(umaxs),
        prof_steps=np.array(times[: len(profiles)]),
        prof_u=np.stack([profiles[s] for s in times]),
        snap_steps=np.array(snap_steps),
        snap_ux=np.stack(
            [snaps[s].cpu().numpy().astype(np.float64) for s in snap_steps]
        ),
        ux0=ux0.cpu().numpy().astype(np.float64),
        n=n,
        tau=tau,
        u0=u0,
        steps=args.steps,
        gamma_vel_theory=gamma_vel,
        elapsed_s=elapsed,
    )
    t = np.asarray(times, dtype=np.float64)
    gv = -float(np.polyfit(t, np.log(np.asarray(umaxs)), 1)[0])
    ge = -float(np.polyfit(t, np.log(np.asarray(energies)), 1)[0])
    print(
        f"gamma_vel_sim={gv:.6e} theory={gamma_vel:.6e} "
        f"err={(gv / gamma_vel - 1) * 100:+.4f}% gamma_E_sim={ge:.6e} "
        f"err_E={(ge / (2 * gamma_vel) - 1) * 100:+.4f}% elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

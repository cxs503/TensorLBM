#!/usr/bin/env python
"""2D Taylor–Green 涡衰减 — 演示脚本。

与正式验证档 benchmarks/verified/taylor_green_2d/run.py 使用同一组库入口
（零手写物理内核）：
  - tensorlbm.d2q9.equilibrium / macroscopic（初值与测量）
  - tensorlbm.solver.collide_bgk / stream（周期 wrap 内建，步序 stream→collide）

解析解：u=-U0·cos(kx)sin(ky)·e^{-2νk²t}，v=+U0·sin(kx)cos(ky)·e^{-2νk²t}
（不可压 NS 精确解）。演示档 N=64（入库档 N=128）同 Re=100/U0=0.05，
输出多时刻涡量/速度场快照、能量与峰值速度历史、y=N/4 线剖面；
定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/taylor_green_2d_demo.py \
        --n 64 --re 100 --u0 0.05 --out demo.npz
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
    ap = argparse.ArgumentParser(description="2D Taylor-Green 涡衰减演示")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=0, help="0 = auto (E/E0 -> e^-3)")
    ap.add_argument("--record-every", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="taylor_green_2d_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("DEMO_THREADS", "32")), os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    n, re, u0 = args.n, args.re, args.u0
    k = 2.0 * math.pi / n
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    gamma_e = 4.0 * nu * k * k
    gamma_vel = 2.0 * nu * k * k
    steps = args.steps or max(
        1000, int(math.ceil(3.0 / gamma_e / args.record_every) * args.record_every)
    )

    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux0 = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy0 = +u0 * torch.sin(k * x) * torch.cos(k * y)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)

    snap_steps = [int(round(f_ * steps)) for f_ in (0.1, 0.5, 1.0)]
    snap_set = set(snap_steps)
    times, energies, umaxs, snaps = [], [], [], {}
    t0 = time.time()
    for step in range(1, steps + 1):
        f = collide_bgk(stream(f), tau)
        if step % args.record_every == 0 or step in snap_set:
            _, uxm, uym = macroscopic(f)
            if step % args.record_every == 0:
                times.append(step)
                energies.append(float((0.5 * (uxm * uxm + uym * uym)).mean().item()))
                umaxs.append(
                    float((uxm * uxm + uym * uym).sqrt().max().item())
                )
            if step in snap_set:
                snaps[step] = (uxm.detach().clone(), uym.detach().clone())
    elapsed = time.time() - t0

    np.savez(
        args.out,
        x=np.arange(n, dtype=np.float64),
        times=np.array(times),
        energies=np.array(energies),
        umaxs=np.array(umaxs),
        snap_steps=np.array(snap_steps),
        snap_ux=np.stack(
            [snaps[s][0].cpu().numpy().astype(np.float64) for s in snap_steps]
        ),
        snap_uy=np.stack(
            [snaps[s][1].cpu().numpy().astype(np.float64) for s in snap_steps]
        ),
        ux0=ux0.cpu().numpy().astype(np.float64),
        uy0=uy0.cpu().numpy().astype(np.float64),
        n=n,
        re=re,
        u0=u0,
        nu=nu,
        tau=tau,
        k=k,
        steps=steps,
        gamma_e_theory=gamma_e,
        gamma_vel_theory=gamma_vel,
        elapsed_s=elapsed,
    )
    t = np.asarray(times, dtype=np.float64)
    ge = -float(np.polyfit(t, np.log(np.asarray(energies)), 1)[0])
    gv = -float(np.polyfit(t, np.log(np.asarray(umaxs)), 1)[0])
    print(
        f"gamma_E_sim={ge:.6e} theory={gamma_e:.6e} err_E={(ge / gamma_e - 1) * 100:+.4f}% "
        f"gamma_vel_sim={gv:.6e} err_vel={(gv / gamma_vel - 1) * 100:+.4f}% "
        f"steps={steps} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

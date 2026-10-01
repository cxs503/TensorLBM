#!/usr/bin/env python
"""启动 Poiseuille 流（突加恒定体力的通道瞬态）— 演示脚本。

与正式验证档 benchmarks/verified/startup_poiseuille/run.py 使用同一组
库入口（零手写物理内核）：
  - tensorlbm.solver.collide_bgk / stream
  - tensorlbm.turbulent_channel._apply_body_force_2d（常量体力 a）
  - tensorlbm.d2q9.OPPOSITE + 预流半程 bounce-back 壁面范式
  - tensorlbm.d2q9.equilibrium / macroscopic（初值与测量）

演示档与入库扫描同参数（H=59, tau=0.8, u_max=0.03）真实 CPU 步进，
瞬态本身即被测对象（无跳过窗口）：输出 7 个 t* 时刻的 u(y) 剖面与
中心线速度历史；解析解为经典 Fourier 级数瞬态解。
定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/startup_poiseuille_demo.py \
        --H 59 --tau 0.8 --umax 0.03 --out demo.npz
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

TSTAR_MARKS = [0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5]
TSTAR_CENTER_MIN = 0.02
CENTER_EVERY = 20
N_ODD = np.arange(1, 1000, 2)


def startup_profile(y_phys, H, nu, a, t):
    """突加恒定体力的 Fourier 级数瞬态解（float64）。"""
    u_max = a * H * H / (8.0 * nu)
    yh = y_phys / H
    n = N_ODD.astype(np.float64)
    lam = np.minimum((n * np.pi / H) ** 2 * nu * t, 700.0)
    fac = (1.0 / n**3)[:, None] * np.sin(np.outer(n, yh) * np.pi) * np.exp(-lam)[:, None]
    s = fac.sum(axis=0)
    steady = 4.0 * yh * (1.0 - yh)
    return u_max * (steady - (32.0 / np.pi**3) * s)


def main():
    ap = argparse.ArgumentParser(description="启动 Poiseuille 流演示")
    ap.add_argument("--H", type=int, default=59)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--umax", type=float, default=0.03)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="startup_poiseuille_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("DEMO_THREADS", "32")), os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    H, tau, u_max = args.H, args.tau, args.umax
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    H_eff = float(ny - 2)
    a = 8.0 * nu * u_max / (H_eff * H_eff)
    t_end = int(math.ceil(TSTAR_MARKS[-1] * H_eff * H_eff / nu)) + CENTER_EVERY

    wall = torch.zeros((ny, args.nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    rho0 = torch.ones((ny, args.nx), device=device)
    u0 = torch.zeros((ny, args.nx), device=device)
    f = equilibrium(rho0, u0, u0)

    rows = np.arange(1, ny - 1, dtype=np.float64)
    y_phys = rows - 0.5
    center_row = ny // 2
    mark_steps = {int(round(ts * H_eff * H_eff / nu)): ts for ts in TSTAR_MARKS}
    tstar_min_step = int(round(TSTAR_CENTER_MIN * H_eff * H_eff / nu))

    center_hist = []
    snaps = {}
    t0 = time.time()
    a_t = torch.tensor(a, device=device)
    for step in range(1, t_end + 1):
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(f.device)], f)
        f = _apply_body_force_2d(f, a_t)
        f = stream(f)
        if step % CENTER_EVERY == 0 and step >= tstar_min_step:
            _, ux, _ = macroscopic(f)
            center_hist.append(
                (step, float(ux[center_row, args.nx // 2].item()))
            )
        if step in mark_steps:
            _, ux, _ = macroscopic(f)
            snaps[step] = ux.mean(dim=1).cpu().numpy().astype(np.float64)[1 : ny - 1]
    elapsed = time.time() - t0

    np.savez(
        args.out,
        y=y_phys,
        mark_steps=np.array(sorted(mark_steps)),
        mark_tstar=np.array(
            [mark_steps[s] for s in sorted(mark_steps)]
        ),
        u_num=np.stack([snaps[s] for s in sorted(mark_steps)]),
        u_ana=np.stack(
            [
                startup_profile(y_phys, H_eff, nu, a, float(s))
                for s in sorted(mark_steps)
            ]
        ),
        center_steps=np.array([s for s, _ in center_hist]),
        center_u=np.array([u for _, u in center_hist]),
        center_u_ana=np.array(
            [
                float(
                    startup_profile(
                        np.array([H_eff / 2.0]), H_eff, nu, a, float(s)
                    )[0]
                )
                for s, _ in center_hist
            ]
        ),
        H=H,
        ny=ny,
        nx=args.nx,
        H_eff=H_eff,
        tau=tau,
        nu=nu,
        a=a,
        u_max=u_max,
        t_end=t_end,
        elapsed_s=elapsed,
    )
    c_err = np.abs(
        np.array([u for _, u in center_hist])
        - np.array(
            [
                float(
                    startup_profile(
                        np.array([H_eff / 2.0]), H_eff, nu, a, float(s)
                    )[0]
                )
                for s, _ in center_hist
            ]
        )
    ) / u_max
    print(
        f"centerline_err_max={c_err.max() * 100:.3f}% steps={t_end} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

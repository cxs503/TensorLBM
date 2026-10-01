#!/usr/bin/env python
"""平面声波色散与衰减 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/acoustics/run.py 使用同一组库入口：
  - tensorlbm.solver.collide_bgk / stream（D2Q9 BGK + 周期流迁）
  - tensorlbm.d2q9.equilibrium / macroscopic

演示与正式档同构（N∈{16,32,64}、τ=1、ε=1e-3、fp64、CPU 单线程、域=恰一
波长 k=2π/N、ny=8），输出密度脉动快照序列与基波复振幅 Z(n) 时间序列，
供教程绘制传播/衰减/相位图。定量判据以正式档 result.json 为准。

用法：
    python docs/benchmarks/demos/acoustics_demo.py \
        --N 16 32 64 --out demo_acoustics.npz
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

from tensorlbm.d2q9 import C, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

DEVICE = torch.device("cpu")
DTYPE = torch.float64
CS = 1.0 / math.sqrt(3.0)
SQ3 = math.sqrt(3.0)

_W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
_CX9 = C[:, 0].numpy().astype(float)


def eig_acoustic(k, tau):
    """右行声学本征值 s = log(λ)（线性 collide-then-stream 算子，9×9 每波数）。"""
    r_ = 1.0 / tau
    I9 = np.eye(9)
    A = np.zeros((9, 9), dtype=complex)
    for q in range(9):
        A[q, :] = np.exp(1j * k * _CX9[q]) * (
            (1.0 - r_) * I9[q] + r_ * _W9[q] * (np.ones(9) + 3.0 * _CX9[q] * _CX9)
        )
    lam = np.linalg.eigvals(A)
    j = int(np.argmin(np.abs(np.log(lam) - 1j * k * CS)))
    return complex(np.log(lam[j]))


def plane_wave_run(N, tau, eps, ny=8, n_periods=16.0, snap_every_T=0.25):
    nx = N
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, device=DEVICE, dtype=DTYPE)
    phase = k * x
    rho0 = 1.0 + eps * torch.sin(phase)
    ux0 = CS * eps * torch.sin(phase)
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(
        rho0.unsqueeze(0).expand(ny, nx).contiguous(),
        ux0.unsqueeze(0).expand(ny, nx).contiguous(),
        uy0.unsqueeze(0).expand(ny, nx).contiguous(),
    )
    period = SQ3 * nx
    nsteps = int(math.ceil(n_periods * period))
    e_minus = np.exp(-1j * k * np.arange(nx))

    zp = np.empty(nsteps + 1, dtype=complex)
    snaps = []
    snap_at = []
    snap_step = 0
    snap_stride = max(1, int(round(snap_every_T * period)))
    t0 = time.time()
    for n in range(nsteps + 1):
        rho = macroscopic(f)[0]
        rp = (rho[0].numpy()).astype(np.float64) - 1.0
        zp[n] = rp @ e_minus / nx
        if n >= snap_step and len(snaps) < 64:
            snaps.append(rp.copy())
            snap_at.append(n)
            snap_step = n + snap_stride
        if n == nsteps:
            break
        f = collide_bgk(f, tau)
        f = stream(f)
    wall = time.time() - t0
    s_star = eig_acoustic(k, tau)
    nu = CS * CS * (tau - 0.5)
    return {
        "N": N,
        "k": k,
        "tau": tau,
        "eps": eps,
        "nsteps": nsteps,
        "period": period,
        "zp": zp,
        "snaps": np.array(snaps),
        "snap_at": np.array(snap_at),
        "c_cont": CS,
        "delta_cont": nu * k * k,
        "c_discrete": s_star.imag / k,
        "delta_discrete": -s_star.real,
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser(description="acoustics 场量演示")
    ap.add_argument("--N", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--periods", type=float, default=16.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="demo_acoustics.npz")
    args = ap.parse_args()

    torch.set_num_threads(1)
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    out = {}
    t_all = 0.0
    for N in args.N:
        r = plane_wave_run(N, args.tau, args.eps, n_periods=args.periods)
        tag = f"N{N}"
        for key in ("N", "k", "tau", "eps", "nsteps", "period", "c_cont",
                    "delta_cont", "c_discrete", "delta_discrete", "wall_s"):
            out[f"{key}_{tag}"] = r[key]
        out[f"zp_{tag}"] = r["zp"]
        out[f"snaps_{tag}"] = r["snaps"]
        out[f"snap_at_{tag}"] = r["snap_at"]
        t_all += r["wall_s"]
        print(
            f"N={N}: steps={r['nsteps']} period={r['period']:.1f} "
            f"c_disc={r['c_discrete']:.9f} delta_disc={r['delta_discrete']:.6e} "
            f"({r['wall_s']:.1f}s)",
            flush=True,
        )
    out["tau"] = args.tau
    out["eps"] = args.eps
    out["elapsed_s"] = t_all
    np.savez(args.out, **out)
    print(f"saved {args.out} (total {t_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

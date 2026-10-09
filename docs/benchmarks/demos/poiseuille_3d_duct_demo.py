#!/usr/bin/env python
"""方形与 2:1 矩形管 Poiseuille 流（3D D3Q19）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/poiseuille_3d_duct/run.py（主档 aspect=1.0
方形 + 子档 duct_ar05 aspect=0.5）使用同一组库入口与解析级数实现：
  - tensorlbm.solver3d.collide_bgk3d / stream3d
  - tensorlbm.d3q19.equilibrium3d / macroscopic3d
  - tensorlbm.boundaries3d.zou_he_inlet_velocity_3d / zou_he_outlet_pressure_3d
    / bounce_back_cells_3d
  - 解析双重奇正弦级数 duct_S / duct_I0（run.py 同款逐字实现）

一个脚本跑两个演示档：方形 W=16×H=16 与 2:1 矩形 W=20×H=10（正式档分别为
W=32/64/128 与 W=64/128），级数初始化 + 充分发展步数；定量判据以正式档
result.json（两份）存档为准。

用法：
    python docs/benchmarks/demos/poiseuille_3d_duct_demo.py \
        --steps 5000 --out demo.npz
"""

import argparse
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

from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

_M = 121  # 级数截断（run.py 同款默认）


# ---------- 解析级数（run.py 逐字实现） ----------
def duct_S(yy: np.ndarray, zz: np.ndarray, a: float, b: float, M: int = _M) -> np.ndarray:
    """-lap S = 1（u = G/nu·S）在 (-a,a)×(-b,b) 上的级数解 S。"""
    m = np.arange(1, M + 1, 2, dtype=np.float64)
    n = np.arange(1, M + 1, 2, dtype=np.float64)
    sy = np.sin(np.pi * n[None, :] * (yy[..., None] + b) / (2.0 * b))
    sz = np.sin(np.pi * m[None, :] * (zz[..., None] + a) / (2.0 * a))
    lam = (np.pi * m / (2.0 * a))[:, None] ** 2 + (np.pi * n / (2.0 * b))[None, :] ** 2
    coef = 16.0 / (np.pi**2 * np.outer(m, n) * lam)
    return np.einsum("...m,mn,...n->...", sz, coef, sy)


def duct_I0(a: float, b: float, M: int = 1201) -> float:
    """I0 = duct_S 在矩形上的积分（逐项解析）。"""
    m = np.arange(1, M + 1, 2, dtype=np.float64)
    n = np.arange(1, M + 1, 2, dtype=np.float64)
    lam = (np.pi * m / (2.0 * a))[:, None] ** 2 + (np.pi * n / (2.0 * b))[None, :] ** 2
    coef = 16.0 / (np.pi**2 * np.outer(m, n) * lam)
    proj = np.outer(4.0 * a / (m * np.pi), 4.0 * b / (n * np.pi))
    return float((coef * proj).sum())


def duct_u(yy, zz, a, b, G, nu):
    return G / nu * duct_S(yy, zz, a, b)


def duct_setup(W: int, H: int, L_over_W: int, device):
    """run.py 的 duct_setup 同款：fluid y∈[1,H]、z∈[1,W]，截面数组按 [z,y]。"""
    ny, nz = H + 2, W + 2
    nx = L_over_W * W
    yy = np.arange(1, ny - 1, dtype=np.float64) - (H + 1) / 2.0
    zz = np.arange(1, nz - 1, dtype=np.float64) - (W + 1) / 2.0
    wall2d = np.zeros((nz, ny), dtype=bool)
    wall2d[:, 0] = wall2d[:, -1] = True
    wall2d[0, :] = wall2d[-1, :] = True
    wall_mask = (
        torch.from_numpy(wall2d).to(device).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    )
    return ny, nz, nx, yy, zz, wall_mask


def run_demo(W: int, H: int, tau: float, u_in: float, steps: int, avg_steps: int,
             device) -> dict:
    nu = (tau - 0.5) / 3.0
    a, b = W / 2.0, H / 2.0
    ny, nz, nx, yy, zz, wall_mask = duct_setup(W, H, 4, device)
    G_nom = nu * u_in * (4.0 * a * b) / duct_I0(a, b)
    zz2, yy2 = np.meshgrid(zz, yy, indexing="ij")  # (W, H)
    u0_cell = duct_u(yy2, zz2, a, b, G_nom, nu)

    ux3 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    u0_t = torch.from_numpy(u0_cell.astype(np.float32)).to(device).unsqueeze(-1)
    ux3[1:-1, 1:-1, :] = u0_t.expand(nz - 2, ny - 2, nx)
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, ux3, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device)

    def _step(f):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, 1.0)
        return bounce_back_cells_3d(f, wall_mask)

    x_meas = nx // 2
    t0 = time.time()
    for s in range(1, steps + 1):
        f = _step(f)
        if s % 1000 == 0:
            _, ux, _, _ = macroscopic3d(f)
            print(f"  W={W} H={H} step={s} "
                  f"umax={float(ux[1:-1, 1:-1, x_meas].max().item()):.6e}",
                  flush=True)
    elapsed = time.time() - t0

    acc = torch.zeros((nz, ny), device=device)
    for _ in range(avg_steps):
        f = _step(f)
        _, ux, _, _ = macroscopic3d(f)
        acc += ux[:, :, x_meas]
    acc /= avg_steps
    rho, ux, _, _ = macroscopic3d(f)

    u_plane = acc[1:-1, 1:-1].cpu().numpy().astype(np.float64)  # (W, H) [z,y]
    rho_x = rho[nz // 2, ny // 2, :].cpu().numpy().astype(np.float64)
    Q = float(u_plane.sum())
    u_mean = Q / (W * H)
    u_max = float(u_plane.max())
    return {
        "u_plane": u_plane,
        "rho_x": rho_x,
        "yy": yy,
        "zz": zz,
        "a": a,
        "b": b,
        "G_nom": G_nom,
        "W": W,
        "H": H,
        "nx": nx,
        "Q": Q,
        "u_mean": u_mean,
        "u_max": u_max,
        "umax_over_umean": u_max / u_mean,
        "steps": steps,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="方形+2:1 矩形管 Poiseuille 场量演示")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--avg-steps", type=int, default=400)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u-in", type=float, default=0.005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="poiseuille_3d_duct_demo.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))

    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    out = {}
    print("demo 1/2: 方形 W=16 H=16", flush=True)
    sq = run_demo(16, 16, args.tau, args.u_in, args.steps, args.avg_steps, device)
    print("demo 2/2: 2:1 矩形 W=20 H=10", flush=True)
    ar = run_demo(20, 10, args.tau, args.u_in, args.steps, args.avg_steps, device)

    for tag, d in (("sq_", sq), ("ar_", ar)):
        for k, v in d.items():
            out[tag + k] = v
    out["tau"] = args.tau
    out["nu"] = (args.tau - 0.5) / 3.0
    out["u_in"] = args.u_in
    np.savez(args.out, **out)
    print(
        f"saved {args.out}: sq W=16xH=16 & ar W=20xH=10 steps={args.steps} "
        f"t={sq['elapsed_s'] + ar['elapsed_s']:.1f}s "
        f"umax/umean sq={sq['umax_over_umean']:.5f} ar={ar['umax_over_umean']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

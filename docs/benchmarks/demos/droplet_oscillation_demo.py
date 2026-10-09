#!/usr/bin/env python
"""液滴毛细振荡（m=2 模态）— 场量可视化演示脚本。

与正式验证档 benchmarks/verified/droplet_oscillation/run.py 使用同一组库入口：
  - tensorlbm.multiphase.collide_sc_single_component（D2Q9 SCMP BGK 碰撞）
  - tensorlbm.multiphase.psi_exp（伪势 psi = 1 - exp(-rho)）
  - tensorlbm.solver.stream（周期拉氏流迁）
  - tensorlbm.d2q9.equilibrium / macroscopic

物理常数与正式档一致（预注册）：tau=1.0、物理 G_eff=-5.0（库参数 G=+5.0，
后向 gather 符号约定）、初场椭圆 tanh 剖面 W=4、eps=0.05、m=2、域 L=4R 周期。
本脚本用缩小半径 R=64（正式档为 128/160/224）并缩短演化时长（约 2.3 个
理论周期），CPU 数十秒出图；输出界面.extent 时间序列 RxRy(t)=R_x-R_y 与
相场快照。定量判据以正式档 result.json 存档扫描为准。

用法：
    python docs/benchmarks/demos/droplet_oscillation_demo.py \
        --R 64 --periods 2.3 --out demo_droplet_oscillation.npz
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 仓库内布局 docs/benchmarks/demos/ -> <repo>/src；可用环境变量覆盖
REPO_SRC = os.environ.get("TENSORLBM_SRC") or str(
    Path(__file__).resolve().parents[3] / "src"
)
sys.path.insert(0, REPO_SRC)

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

# ---- 正式档预注册常数（run.py 同源） ----
TAU = 1.0
G_LIB = 5.0  # 库参数；物理 G_eff = -5.0（后向 gather 符号约定）
RHO_L_INIT, RHO_V_INIT = 1.957, 0.1596
W_INT = 4.0
EPS = 0.05
SKIP_STEPS = 500
SIGMA_A, RHO_L_A, RHO_V_A = 0.056112, 1.957, 0.1596  # 演示档理论线（basis A）


def init_ellipse_rho(L, R0, eps, rho_l, rho_v, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx, dy = xx - L / 2.0, yy - L / 2.0
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx)
    r_surf = R0 * (1.0 + eps * torch.cos(2.0 * theta))
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((r_surf - r) / W_INT))
    return rho.clamp(min=1e-3)


def _interface_extent(rho, mid, L, axis):
    line = rho[L // 2, :] if axis == 0 else rho[:, L // 2]
    cross = (line[:-1] - mid) * (line[1:] - mid) < 0
    idx = torch.nonzero(cross).flatten().tolist()
    if len(idx) < 2:
        return float("nan")
    xint = []
    for i in idx:
        r0, r1 = float(line[i]), float(line[i + 1])
        xint.append(float(i) if abs(r1 - r0) < 1e-12 else i + (mid - r0) / (r1 - r0))
    return float(max(xint) - min(xint)) / 2.0


def measure_osc(f, R_guess, xx, yy, rr, L):
    rho, ux, uy = macroscopic(f)
    umag = torch.sqrt(ux * ux + uy * uy)
    r_eq = R_guess
    mid = 0.5 * (RHO_L_INIT + RHO_V_INIT)
    rho_in = rho_out = float("nan")
    for _ in range(3):
        inside, outside = rr <= r_eq * 0.5, rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
        rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        r_eq_new = math.sqrt(n_liq / math.pi)
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    mask = rho > mid
    rho_liq = rho[mask]
    m_liq = float(rho_liq.sum().item())
    q = float((rho_liq * (xx[mask] ** 2 - yy[mask] ** 2)).sum().item()) / m_liq
    return {
        "Q": q,
        "R_eq": r_eq,
        "R_x": _interface_extent(rho, mid, L, 0),
        "R_y": _interface_extent(rho, mid, L, 1),
        "rho_in": rho_in,
        "rho_out": rho_out,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


def main():
    ap = argparse.ArgumentParser(description="droplet_oscillation 场量演示")
    ap.add_argument("--R", type=float, default=64.0)
    ap.add_argument("--periods", type=float, default=2.3)
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="demo_droplet_oscillation.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    R0 = args.R
    L = int(4 * R0)
    # 演示档理论周期（basis A 常数；入库判据用 basis C 的 R 匹配静态值）
    rho_lv = RHO_L_A + RHO_V_A
    omega_A = math.sqrt(6.0 * SIGMA_A / (rho_lv * R0**3))
    T_A = 2.0 * math.pi / omega_A
    steps = int(math.ceil((SKIP_STEPS + args.periods * T_A) / args.sample) * args.sample)
    snap_steps = [
        int(round(SKIP_STEPS + k * 0.25 * T_A / args.sample) * args.sample)
        for k in range(4 * int(math.ceil(args.periods)) + 1)
        if int(round(SKIP_STEPS + k * 0.25 * T_A)) <= steps
    ]
    snap_set = set(snap_steps)

    rho0 = init_ellipse_rho(L, R0, EPS, RHO_L_INIT, RHO_V_INIT, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx, yy = xx - L / 2.0, yy - L / 2.0
    rr = torch.sqrt(xx * xx + yy * yy)

    hist = []
    snaps = []
    snap_at = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f = stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        if step in snap_set:
            rho_now = f.sum(dim=0)
            snaps.append(rho_now.cpu().numpy().astype(np.float32))
            snap_at.append(step)
            snap_set.discard(step)
        if step % args.sample == 0:
            rho_cur = f.sum(dim=0)
            if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                raise RuntimeError(f"NaN/negative rho at step {step}")
            m = measure_osc(f, R0, xx, yy, rr, L)
            m["step"] = step
            hist.append(m)
    elapsed = time.time() - t0

    np.savez(
        args.out,
        step=np.array([h["step"] for h in hist]),
        Q=np.array([h["Q"] for h in hist]),
        R_eq=np.array([h["R_eq"] for h in hist]),
        R_x=np.array([h["R_x"] for h in hist]),
        R_y=np.array([h["R_y"] for h in hist]),
        rho_in=np.array([h["rho_in"] for h in hist]),
        rho_out=np.array([h["rho_out"] for h in hist]),
        max_u=np.array([h["max_u"] for h in hist]),
        mass=np.array([h["mass"] for h in hist]),
        snapshots=np.stack(snaps),
        snap_steps=np.array(snap_at),
        R=R0,
        L=L,
        eps=EPS,
        steps=steps,
        sample=args.sample,
        skip=SKIP_STEPS,
        omega_theory_A=omega_A,
        T_theory_A=T_A,
        mass0=mass0,
        elapsed_s=elapsed,
    )
    rxry = np.array([h["R_x"] for h in hist]) - np.array([h["R_y"] for h in hist])
    print(
        f"saved {args.out}: R={R0:.0f} L={L} steps={steps} T_A={T_A:.0f} "
        f"t={elapsed:.1f}s RxRy[0]={rxry[0]:.2f} RxRy[-1]={rxry[-1]:.2f} "
        f"mass_drift={abs(hist[-1]['mass'] - mass0) / mass0:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

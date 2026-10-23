#!/usr/bin/env python
"""静态液滴 Young–Laplace 定律 — 场量可视化演示脚本（2D）。

与正式验证档 benchmarks/verified/laplace_droplet/run.py 使用同一组库入口：
  - tensorlbm.multiphase.collide_sc_single_component（D2Q9 SCMP 碰撞）
  - tensorlbm.solver.stream（周期流迁）
  - tensorlbm.d2q9.equilibrium / macroscopic
（正式档另含 3D 轨：multiphase3d.collide_sc_single_component_3d + solver3d.stream3d，
 本 CPU 演示只跑 2D。）

物理常数与正式档一致：tau=1.0、物理 G_eff=-5.0（库参数 G=+5.0，后向 gather
符号约定）、tanh 液滴剖面 W=4、离散共存初密度 1.957/0.1596、L=4R 周期。
稳态判据与测量口径（带均值→中密度阈→R_eq→0.5R 内/1.5R 外压力环带）
与正式档同构。输出三半径 Δp(t) 收敛史与终态密度/压力场。

用法：
    python docs/benchmarks/demos/laplace_droplet_demo.py \
        --radii 15,25,40 --out demo_laplace_droplet.npz
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
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

TAU = 1.0
G_LIB = 5.0  # 库参数；物理 G_eff = -5.0
G_EFF = -5.0
RHO_L, RHO_V = 1.957, 0.1596
W_INT = 4.0


def eos_pressure(rho, g=G_EFF):
    psi = 1.0 - torch.exp(-rho)
    return rho / 3.0 + g * psi * psi / 6.0


def init_rho(L, R, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = RHO_V + 0.5 * (RHO_L - RHO_V) * (1.0 + torch.tanh((R - r) / W_INT))
    return rho.clamp(min=1e-3)


def radius_field(L, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)


def measure(f, R_guess, rr):
    rho, ux, uy = macroscopic(f)
    umag = torch.sqrt(ux**2 + uy**2)
    p = eos_pressure(rho)
    r_eq = R_guess
    for _ in range(3):
        inside = rr <= r_eq * 0.5
        outside = rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean().item())
        rho_out = float(rho[outside].mean().item())
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        r_eq_new = math.sqrt(n_liq / math.pi)
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    inside = rr <= r_eq * 0.5
    outside = rr >= r_eq * 1.5
    p_in = float(p[inside].mean().item())
    p_out = float(p[outside].mean().item())
    return {
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "rho_in": float(rho[inside].mean().item()),
        "rho_out": float(rho[outside].mean().item()),
        "R_eq": r_eq,
        "max_u": float(umag.max().item()),
        "mass": float(rho.sum().item()),
    }


def run_droplet(R, L, device, max_steps, min_steps, sample_interval):
    t0 = time.perf_counter()
    rho0 = init_rho(L, R, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    rr = radius_field(L, device)
    hist = []
    converged = False
    step = 0
    for step in range(1, max_steps + 1):
        f = stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        if step % sample_interval == 0:
            rho_cur = f.sum(dim=0)
            if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                raise RuntimeError(f"R={R}: NaN/negative rho at step {step}")
            m = measure(f, R, rr)
            m["step"] = step
            hist.append(m)
            if len(hist) >= 2:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                if d_dp < 2e-4 and d_r < 2e-3 and step >= min_steps:
                    converged = True
                    break
    tail = hist[-3:]
    rho, ux, uy = macroscopic(f)
    final = {
        "step": int(step),
        "converged": converged,
        "p_in": float(np.mean([s["p_in"] for s in tail])),
        "p_out": float(np.mean([s["p_out"] for s in tail])),
        "dp": float(np.mean([s["dp"] for s in tail])),
        "rho_in": float(np.mean([s["rho_in"] for s in tail])),
        "rho_out": float(np.mean([s["rho_out"] for s in tail])),
        "R_eq": float(np.mean([s["R_eq"] for s in tail])),
        "max_u": float(np.mean([s["max_u"] for s in tail])),
        "mass_drift": abs(float(tail[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
        "rho_field": rho.cpu().numpy().astype(np.float32),
        "p_field": eos_pressure(rho).cpu().numpy().astype(np.float32),
        "ux": ux.cpu().numpy().astype(np.float32),
        "uy": uy.cpu().numpy().astype(np.float32),
    }
    return final, hist


def main():
    ap = argparse.ArgumentParser(description="laplace_droplet 场量演示（2D）")
    ap.add_argument("--radii", default="15,25,40")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--min-steps", type=int, default=6000)
    ap.add_argument("--sample-interval", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="demo_laplace_droplet.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    out = {"radii": np.array([float(x) for x in args.radii.split(",")])}
    for R in out["radii"]:
        R = float(R)
        L = int(4 * R)
        final, hist = run_droplet(
            R, L, device, args.max_steps, args.min_steps, args.sample_interval
        )
        tag = f"R{int(R)}"
        out[f"step_{tag}"] = np.array([h["step"] for h in hist])
        out[f"dp_{tag}"] = np.array([h["dp"] for h in hist])
        out[f"R_eq_{tag}"] = np.array([h["R_eq"] for h in hist])
        for k in ("step", "converged", "p_in", "p_out", "dp", "rho_in", "rho_out",
                  "R_eq", "max_u", "mass_drift", "dt_s"):
            out[f"{k}_{tag}"] = final[k]
        out[f"rho_{tag}"] = final["rho_field"]
        out[f"p_{tag}"] = final["p_field"]
        out[f"L_{tag}"] = L
        print(
            f"R={R:.0f}: steps={final['step']} conv={final['converged']} "
            f"R_eq={final['R_eq']:.3f} dp={final['dp']:.6f} "
            f"sigma_i={final['dp'] * final['R_eq']:.5f} ({final['dt_s']:.0f}s)",
            flush=True,
        )
    np.savez(args.out, **out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

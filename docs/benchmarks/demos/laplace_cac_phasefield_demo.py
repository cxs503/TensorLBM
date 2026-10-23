#!/usr/bin/env python
"""CAC 相场两相静态液滴 — 场量可视化演示脚本（3D，D3Q19 双分布 fp64）。

与正式验证档 benchmarks/verified/laplace_cac_phasefield/run.py 使用同一组
库入口与同一臂构造（tensorlbm.cac_lbm.CACSim + init_phi_droplet +
_macro_fp64，fp64、harmonic mu(phi)、W=4、M=0.05、ν=0.1、p0=0.01、周期
立方域、Route-B σ_model = 0.01/0.9238958795895513）：
  - B1 臂（R 迁移）：128³，R = 20/32/48，密度比 100
  - B3 臂（密度比迁移）：96³，R = 20，密度比 10/100
采样（每 500 步 + 首步）、估计窗（末 2000 步）、压差带（φ>0.9 内 /
φ<0.1 外）与 σ_rec = Δp·R/2 口径均与正式档一致。

正式档共 5 臂全跑；本演示默认 2 臂（b1_R20 + b3_rho100，覆盖 R 阶梯与
密度比迁移两个方向），并额外输出终态 φ/压力/密度/|u| 中面切片与中线
剖面（npz），供教程云图画图。

用法：
    python docs/benchmarks/demos/laplace_cac_phasefield_demo.py \
        --arms b1_R20,b3_rho100 --out demo_laplace_cac_phasefield.npz
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

import tensorlbm  # noqa: E402
from tensorlbm import cac_lbm as m  # noqa: E402

SIGMA_TARGET = 0.01
BAR_EFF_W4 = 0.9238958795895513  # Route-B 标定常数（README/档案锁值）
SIGMA_ROUTE_B = SIGMA_TARGET / BAR_EFF_W4
N_STEPS = 12_000
MOB = 0.05
NU = 0.1
W_INT = 4.0
P0 = 0.01
DEFAULT_ARMS = ("b1_R20", "b3_rho100")


def run_arm(kind, arg, sigma, device, dtype, n_steps, sample=500, dump_fields=True):
    """与 run.py run_arm 同构造/同采样；末尾加中面场切片与中线剖面。"""
    if kind == "b1":
        shape = (128, 128, 128)
        radius = float(arg)
        rho_l, rho_g = 1.0, 0.01
    else:
        shape = (96, 96, 96)
        radius = 20.0
        rho_l, rho_g = 1.0, 1.0 / float(arg)
    mu_l, mu_g = NU * rho_l, NU * rho_g

    sim = m.CACSim(
        shape,
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=mu_l,
        mu_g=mu_g,
        sigma=sigma,
        width=W_INT,
        mobility=MOB,
        p0=P0,
        axes="ppp",
        mu_law="harmonic",
        dtype=dtype,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        gravity_mode="rho",
    )
    center = tuple(s / 2.0 for s in shape)
    sim.initialize(
        m.init_phi_droplet(shape, radius=radius, width=W_INT, center=center, dtype=dtype)
    )

    t0 = time.time()
    series = []
    phi0_int = None
    for it in range(1, n_steps + 1):
        sim.step()
        if it == 1 or it % sample == 0:
            phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            if phi0_int is None:
                phi0_int = phi.sum().item()
            inside = phi > 0.9
            outside = phi < 0.1
            p_in = p[inside].mean().item() if inside.any() else float("nan")
            p_out = p[outside].mean().item() if outside.any() else float("nan")
            umax = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt().max().item()
            series.append(
                {
                    "step": it,
                    "phi_integral": phi.sum().item(),
                    "phi_drift_rel": abs(phi.sum().item() - phi0_int) / phi0_int,
                    "p_in": p_in,
                    "p_out": p_out,
                    "dP": p_in - p_out,
                    "sigma_rec": (p_in - p_out) * radius / 2.0,
                    "u_max": umax,
                    "rho_min": rho.min().item(),
                    "rho_max": rho.max().item(),
                    "nan_f": int(torch.isnan(sim.f).sum().item()),
                    "nan_g": int(torch.isnan(sim.g).sum().item()),
                }
            )
            if it % 2000 == 0:
                s = series[-1]
                print(
                    f"[{kind}/{arg:g}] step {it}  sig_rec {s['sigma_rec']:.5f}  "
                    f"u_max {s['u_max']:.2e}  drift {s['phi_drift_rel']:.2e}",
                    flush=True,
                )

    out = {
        "kind": kind,
        "arg": arg,
        "shape": shape,
        "radius": radius,
        "rho_l": rho_l,
        "rho_g": rho_g,
        "mu_l": mu_l,
        "mu_g": mu_g,
        "sigma_model": sigma,
        "n_steps": n_steps,
        "nan_free": all(s["nan_f"] == 0 and s["nan_g"] == 0 for s in series),
        "dt_s": time.time() - t0,
    }
    win = [s for s in series if s["step"] > n_steps - 2000]
    out["sigma_rec_window_mean"] = sum(s["sigma_rec"] for s in win) / len(win)
    out["dP_window_mean"] = sum(s["dP"] for s in win) / len(win)
    out["u_max_window"] = max(s["u_max"] for s in win)
    out["tail_sigma_rec"] = [s["sigma_rec"] for s in win]
    out["drift_end"] = series[-1]["phi_drift_rel"]
    out["series"] = series

    if dump_fields:
        phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
        umag = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2).sqrt()
        zc = shape[0] // 2
        out["phi_mid"] = phi[zc].cpu().numpy().astype(np.float32)
        out["p_mid"] = p[zc].cpu().numpy().astype(np.float32)
        out["rho_mid"] = rho[zc].cpu().numpy().astype(np.float32)
        out["umag_mid"] = umag[zc].cpu().numpy().astype(np.float32)
        out["phi_line"] = phi[zc, shape[1] // 2, :].cpu().numpy().astype(np.float64)
        out["p_line"] = p[zc, shape[1] // 2, :].cpu().numpy().astype(np.float64)
        # 展示用等效半径（φ>0.5 体积反推；σ_rec 口径仍用名义 R，同正式档）
        vol = float((phi > 0.5).sum().item())
        out["R_eff_phi05"] = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="CAC 相场液滴 Laplace 场量演示（3D）")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--sigma", default="auto",
                    help="模型 σ；默认 Route-B = 0.01/bar_eff(W=4)")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="demo_laplace_cac_phasefield.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    sigma = SIGMA_ROUTE_B if args.sigma == "auto" else float(args.sigma)
    dtype = torch.float64
    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    out = {"sigma_target": np.array([SIGMA_TARGET]),
           "sigma_model": np.array([sigma]),
           "bar_eff_w4": np.array([BAR_EFF_W4])}
    for arm in [a.strip() for a in args.arms.split(",")]:
        kind = "b1" if arm.startswith("b1") else "b3"
        arg = float(arm.split("_R")[1] if kind == "b1" else arm.split("_rho")[1])
        print(f"===== {arm} on {device} (sigma={sigma:.15g}) =====", flush=True)
        r = run_arm(kind, arg, sigma, device, dtype, args.steps)
        tag = arm.replace(".", "p")
        for k in ("radius", "rho_l", "rho_g", "sigma_rec_window_mean",
                  "dP_window_mean", "u_max_window", "drift_end", "nan_free",
                  "R_eff_phi05", "dt_s", "n_steps"):
            out[f"{k}_{tag}"] = r[k]
        out[f"step_{tag}"] = np.array([s["step"] for s in r["series"]])
        for k in ("sigma_rec", "dP", "u_max", "phi_drift_rel"):
            out[f"{k}_{tag}"] = np.array([s[k] for s in r["series"]])
        if "phi_mid" in r:
            for k in ("phi_mid", "p_mid", "rho_mid", "umag_mid", "phi_line",
                      "p_line"):
                out[f"{k}_{tag}"] = r[k]
        print(
            f"[{arm}] sigma_rec={r['sigma_rec_window_mean']:.8f} "
            f"err={abs(r['sigma_rec_window_mean'] / SIGMA_TARGET - 1) * 100:.3f}%  "
            f"u_max={r['u_max_window']:.2e}  drift={r['drift_end']:.2e}  "
            f"R_eff(φ>0.5)={r.get('R_eff_phi05', float('nan')):.2f}  "
            f"({r['dt_s']:.0f}s)",
            flush=True,
        )
    np.savez(args.out, **out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

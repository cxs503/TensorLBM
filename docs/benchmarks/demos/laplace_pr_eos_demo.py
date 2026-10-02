#!/usr/bin/env python
"""PR-EOS 高密度比伪势静态液滴 — 场量可视化演示脚本（2D）。

与正式验证档 benchmarks/verified/laplace_pr_eos/run.py 使用同一组库入口
与同一协议（主档 L=128 层级的 R = 12/16/20/24 四半径完整复跑，含同一
稳态判据与尾窗测量口径）：
  - tensorlbm.multiphase.make_psi_eos（修正符号 PR EOS 伪势，YS2006 参数）
  - tensorlbm.multiphase.collide_sc_single_component（forcing="edm"，
    精确差分 EDM 力）
  - tensorlbm.solver.stream（周期流迁，D2Q9）
Maxwell 共存锁值（等面积构造，机器拷贝自 run.py 的 MAXWELL_LOCK）：
rho_l = 7.997835999814006 / rho_v = 0.0649052793822428
（rho_cross = 9.708178500398901），T_r = 0.55。

本演示额外输出终态密度/机械压力/速度场与全程收敛史（npz），供教程
云图画图；正式档覆盖主/副两个读法共 24 个 run，本演示只覆盖判决主档
的 L=128 层级（4 run）。

用法：
    python docs/benchmarks/demos/laplace_pr_eos_demo.py \
        --radii 12,16,20,24 --L 128 --out demo_laplace_pr_eos.npz
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
from tensorlbm.d2q9 import C, equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, make_psi_eos  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

CS2 = 1.0 / 3.0
T_C_PR = 0.07291903717301021  # PR 临界温度（YS2006 参数）
T_R = 0.55
T = T_R * T_C_PR
# Maxwell 等面积锁值（机器拷贝自 run.py MAXWELL_LOCK["pr"]，勿手改）
RHO_L_MAXWELL = 7.997835999814006
RHO_V_MAXWELL = 0.0649052793822428
RHO_CROSS = 9.708178500398901
G = 1.0  # 库后向 gather 约定下 G>0 为吸引（修正符号伪势）
TAU = 1.0
W_INT = 4.0


def init_droplet(L, R, rho_l, rho_v, device, width=W_INT):
    """与 run.py init_droplet 同式：tanh 液滴剖面 + 平衡初场。"""
    ys = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((R - r) / width))
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


def run_droplet(L, R, device, psi_fn, max_steps=40000, min_steps=4000,
                sample=1000, conv_tol=1e-5, dump_fields=False):
    """与 run.py run_droplet 同协议的步进/采样/稳态判据/尾窗测量。"""
    t0 = time.perf_counter()
    f = init_droplet(L, R, RHO_L_MAXWELL, RHO_V_MAXWELL, device)
    mass0 = float(f.sum(dim=(0, 1, 2)).item())
    ys = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    core0 = r <= 0.25 * L / 2
    far0 = r >= 0.85 * L / 2
    far = r >= 0.80 * L / 2

    hist, stable, err_msg, step_i = [], True, None, 0
    for step_i in range(1, max_steps + 1):
        f = stream(
            collide_sc_single_component(f, G=G, tau=TAU, psi_fn=psi_fn,
                                        forcing="edm")
        )
        if step_i % sample:
            continue
        rho = f.sum(dim=0)
        if not torch.isfinite(rho).all() or float(rho.min().item()) < 0.0:
            stable, err_msg = False, f"nonfinite/negative rho at step {step_i}"
            break
        rho_l = float(rho[core0].mean().item())
        rho_v = float(rho[far0].mean().item())
        for _ in range(2):  # 等面积 R_eff + 带定义两轮精化
            r_eff = float(
                torch.sqrt(
                    torch.clamp((rho - rho_v).sum(), min=0.0)
                    / (math.pi * max(rho_l - rho_v, 1e-12))
                ).item()
            )
            if r_eff <= 0:
                break
            core = r <= 0.5 * r_eff
            rho_l = float(rho[core].mean().item())
            rho_v = float(rho[far].mean().item())
        if r_eff <= 0:
            stable, err_msg = False, f"degenerate droplet at step {step_i}"
            break
        psi = psi_fn(rho)
        p_mech = rho * CS2 - 0.5 * G * CS2 * psi * psi
        dp = float(p_mech[core].mean().item()) - float(p_mech[far].mean().item())
        mom_x = (f * C.to(device)[:, 0].view(9, 1, 1)).sum(dim=0)
        mom_y = (f * C.to(device)[:, 1].view(9, 1, 1)).sum(dim=0)
        rho_c = torch.clamp(rho, min=1e-12)
        umax = float(
            torch.sqrt((mom_x / rho_c) ** 2 + (mom_y / rho_c) ** 2).max().item()
        )
        mass = float(rho.sum().item())
        hist.append(
            dict(
                step=step_i,
                rho_l=rho_l,
                rho_v=rho_v,
                dp=dp,
                r_eff=r_eff,
                u_max=umax,
                mass=mass,
                rho_min=float(rho.min().item()),
                rho_max=float(rho.max().item()),
                mass_drift=abs(mass - mass0) / mass0,
            )
        )
        if len(hist) >= 4 and step_i >= min_steps:
            rec = hist[-3:]
            if all(
                abs(h["rho_l"] - hp["rho_l"]) / max(abs(h["rho_l"]), 1e-12) <= conv_tol
                and abs(h["rho_v"] - hp["rho_v"]) / max(abs(h["rho_v"]), 1e-12) <= conv_tol
                for h, hp in zip(rec[1:], rec[:-1])
            ):
                break

    out = dict(
        L=L,
        R=R,
        max_steps=max_steps,
        steps_run=int(step_i),
        stable=stable,
        error=err_msg,
        converged=bool(len(hist) >= 4 and step_i < max_steps and stable),
        hist=hist,
    )
    tail = hist[-5:] if hist else []
    if tail:
        for k in ("rho_l", "rho_v", "dp", "r_eff"):
            m = sum(t[k] for t in tail) / len(tail)
            out[k] = m
            out[k + "_std"] = (sum((t[k] - m) ** 2 for t in tail) / len(tail)) ** 0.5
        out["ratio"] = out["rho_l"] / out["rho_v"]
        out["u_max"] = max(t["u_max"] for t in tail)
        out["rho_max_field"] = max(t["rho_max"] for t in tail)
        out["rho_min_field"] = min(t["rho_min"] for t in tail)
        out["mass_drift"] = max(t["mass_drift"] for t in tail)
        out["mass_rate_per_step"] = out["mass_drift"] / out["steps_run"]
    out["dt_s"] = time.perf_counter() - t0
    if dump_fields and stable:
        rho = f.sum(dim=0)
        psi = psi_fn(rho)
        p_mech = rho * CS2 - 0.5 * G * CS2 * psi * psi
        mom_x = (f * C.to(device)[:, 0].view(9, 1, 1)).sum(dim=0)
        mom_y = (f * C.to(device)[:, 1].view(9, 1, 1)).sum(dim=0)
        rho_c = torch.clamp(rho, min=1e-12)
        ux = mom_x / rho_c
        uy = mom_y / rho_c
        out["rho_field"] = rho.cpu().numpy().astype(np.float32)
        out["p_field"] = p_mech.cpu().numpy().astype(np.float32)
        out["ux_field"] = ux.cpu().numpy().astype(np.float32)
        out["uy_field"] = uy.cpu().numpy().astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description="PR-EOS 液滴 Laplace 场量演示（2D）")
    ap.add_argument("--radii", default="12,16,20,24")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=40000)
    ap.add_argument("--min-steps", type=int, default=4000)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="demo_laplace_pr_eos.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)
    psi_fn = make_psi_eos("pr", T)

    radii = [float(x) for x in args.radii.split(",")]
    out = {
        "radii": np.array(radii),
        "L": np.array([args.L]),
        "T": np.array([T]),
        "T_r": np.array([T_R]),
        "rho_l_maxwell": np.array([RHO_L_MAXWELL]),
        "rho_v_maxwell": np.array([RHO_V_MAXWELL]),
        "rho_cross": np.array([RHO_CROSS]),
    }
    for i, R in enumerate(radii):
        r = run_droplet(
            args.L,
            R,
            device,
            psi_fn,
            max_steps=args.max_steps,
            min_steps=args.min_steps,
            sample=args.sample,
            dump_fields=(i == len(radii) - 1),
        )
        tag = f"R{int(R)}"
        for k in (
            "steps_run",
            "stable",
            "converged",
            "rho_l",
            "rho_v",
            "dp",
            "r_eff",
            "ratio",
            "u_max",
            "rho_max_field",
            "rho_min_field",
            "mass_drift",
            "mass_rate_per_step",
            "dt_s",
        ):
            out[f"{k}_{tag}"] = r[k]
        out[f"step_{tag}"] = np.array([h["step"] for h in r["hist"]])
        for k in ("dp", "r_eff", "rho_l", "rho_v", "u_max", "mass_drift"):
            out[f"hist_{k}_{tag}"] = np.array([h[k] for h in r["hist"]])
        if "rho_field" in r:
            out["rho_field_Rmax"] = r["rho_field"]
            out["p_field_Rmax"] = r["p_field"]
            out["ux_field_Rmax"] = r["ux_field"]
            out["uy_field_Rmax"] = r["uy_field"]
            out["R_field"] = np.array([R])
        print(
            f"R={R:g}: steps={r['steps_run']} conv={r['converged']} "
            f"r_eff={r['r_eff']:.3f} dp={r['dp']:.6f} "
            f"ratio={r['ratio']:.1f} u_max={r['u_max']:.2e} "
            f"({r['dt_s']:.0f}s)",
            flush=True,
        )

    # 与正式档同式的 LSQ 自由截距拟合（L=128 层级）
    xs = np.array([1.0 / out[f"r_eff_R{int(R)}"] for R in radii])
    ys = np.array([out[f"dp_R{int(R)}"] for R in radii])
    n = len(xs)
    mx, my = xs.mean(), ys.mean()
    sxx = ((xs - mx) ** 2).sum()
    slope = float(((xs - mx) * (ys - my)).sum() / sxx)
    intercept = float(my - slope * mx)
    resid = ys - (slope * xs + intercept)
    rms = float(np.sqrt((resid**2).sum() / n))
    out["sigma_fit"] = np.array([slope])
    out["intercept"] = np.array([intercept])
    out["rms_residual"] = np.array([rms])
    out["dp_max"] = np.array([ys.max()])
    print(
        f"sigma(L={args.L}) = {slope:.6f}  intercept={intercept:.2e}  "
        f"rms/max(dP) = {rms / ys.max() * 100:.4f}%",
        flush=True,
    )
    np.savez(args.out, **out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

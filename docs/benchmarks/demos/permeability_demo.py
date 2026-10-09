#!/usr/bin/env python
"""周期圆柱方阵渗透率 — 场量可视化演示脚本。

与正式验证档 benchmarks/verified/permeability/run.py 使用同一组库入口：
  - tensorlbm.solver.collide_bgk / stream（D2Q9 BGK + 周期流迁）
  - tensorlbm.boundaries.bounce_back_cells（固体胞元反弹）
  - tensorlbm.turbulent_channel._apply_body_force_2d（体力注入）
  - tensorlbm.d2q9.equilibrium / macroscopic

步序与正式档（库 turbulent_channel 同序）：collide → stream → 流体域掩码
体力 → bounce-back。几何（周期折叠距离的布尔圆柱掩码，风格同
porous_media.make_random_cylinder_medium）非物理核，本脚本按正式档同式构造。
稳态探测/测量窗与正式档同构（末 20 采样均值、漂移 1e-5 连续 3 窗触发）。

演示档跑 φ=0.3（d=26/52）与 φ=0.5（d=20/40）四案（正式档为 26/52/104 与
20/40/80/160/320 全阶梯，GPU）；输出掩码几何、终态速度场与 ⟨u_x⟩_fluid
收敛史。定量判据以正式档 result.json 为准。

用法：
    python docs/benchmarks/demos/permeability_demo.py \
        --cases 0.3:26,0.3:52,0.5:20,0.5:40 --out demo_permeability.npz
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

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.turbulent_channel import _apply_body_force_2d  # noqa: E402

TAU = 1.0
NU = (TAU - 0.5) / 3.0
RE_TARGET = 0.05
SANGANI_TABLE = {
    0.05: 15.56, 0.10: 24.83, 0.20: 51.53, 0.30: 102.90, 0.40: 217.89,
    0.50: 532.55, 0.60: 1763.0, 0.70: 13520.0, 0.75: 126300.0,
}
SAMPLE_EVERY = 200
DRIFT_SPAN = 10
DRIFT_TOL = 1e-5
STEADY_REPEATS = 3
POST_STEADY = 8000
MEAS_SAMPLES = 20


def cylinder_array_mask(d, phi, device):
    idx = np.arange(d, dtype=np.float64)
    dx = np.abs(idx - d / 2.0)
    dx = np.minimum(dx, d - dx)
    dist2 = dx[None, :] ** 2 + dy_fold(d)[:, None] ** 2
    r_real = d * math.sqrt(phi / math.pi)
    mask = dist2 <= r_real * r_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def dy_fold(d):
    idx = np.arange(d, dtype=np.float64)
    dy = np.abs(idx - d / 2.0)
    return np.minimum(dy, d - dy)


def run_case(phi, d, force_scale, device, max_steps):
    f_table = SANGANI_TABLE[phi]
    r_real = d * math.sqrt(phi / math.pi)
    k_lu = d * d / f_table
    a_body = RE_TARGET * NU * NU / (k_lu * r_real) * force_scale

    solid = cylinder_array_mask(d, phi, device)
    fluid = ~solid
    n_solid = int(solid.sum().item())
    phi_actual = n_solid / (d * d)
    solid_b = solid.unsqueeze(0)

    rho0 = torch.ones(d, d, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, u0, u0, device=device)

    def step(f_t):
        f_t = collide_bgk(f_t, TAU)
        f_t = stream(f_t)
        f_t = torch.where(solid_b, f_t, _apply_body_force_2d(f_t, a_body))
        return bounce_back_cells(f_t, solid)

    hist_step, hist_uxf, hist_uyf = [], [], []
    recent_drifts = []
    steady_step = None
    t0 = time.perf_counter()
    for step_i in range(1, max_steps + 1):
        f = step(f)
        if step_i % SAMPLE_EVERY == 0:
            _, ux, uy = macroscopic(f)
            uxf = float(ux[fluid].mean().item())
            uyf = float(uy[fluid].mean().item())
            hist_step.append(step_i)
            hist_uxf.append(uxf)
            hist_uyf.append(uyf)
            if steady_step is None and len(hist_uxf) >= DRIFT_SPAN + 1:
                drift = abs(hist_uxf[-1] - hist_uxf[-1 - DRIFT_SPAN]) / abs(hist_uxf[-1])
                recent_drifts.append(drift)
                if len(recent_drifts) >= STEADY_REPEATS and all(
                    v < DRIFT_TOL for v in recent_drifts[-STEADY_REPEATS:]
                ):
                    steady_step = step_i
        if steady_step is not None and step_i >= steady_step + POST_STEADY:
            break
    wall = time.perf_counter() - t0

    uxf_meas = float(np.mean(hist_uxf[-MEAS_SAMPLES:]))
    _, ux, uy = macroscopic(f)
    k_sim = NU * uxf_meas / a_body
    k_ref = d * d / ((1.0 - phi) * f_table)
    return {
        "phi": phi,
        "d": d,
        "a_body": a_body,
        "phi_actual": phi_actual,
        "N_solid": n_solid,
        "k_sim": k_sim,
        "k_ref": k_ref,
        "err_pct": abs(k_sim / k_ref - 1.0) * 100.0,
        "steady_step": steady_step,
        "steps_run": hist_step[-1],
        "wall_s": wall,
        "solid": solid.cpu().numpy(),
        "ux": ux.cpu().numpy().astype(np.float32),
        "uy": uy.cpu().numpy().astype(np.float32),
        "hist_step": np.array(hist_step),
        "hist_uxf": np.array(hist_uxf),
        "hist_uyf": np.array(hist_uyf),
    }


def main():
    ap = argparse.ArgumentParser(description="permeability 场量演示")
    ap.add_argument("--cases", default="0.3:26,0.3:52,0.5:20,0.5:40")
    ap.add_argument("--force-scale", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=400000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="demo_permeability.npz")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    import tensorlbm

    print(f"tensorlbm: {tensorlbm.__file__}", flush=True)

    out = {}
    t_all = 0.0
    for spec in args.cases.split(","):
        phi_s, d_s = spec.split(":")
        phi, d = float(phi_s), int(d_s)
        r = run_case(phi, d, args.force_scale, device, args.max_steps)
        tag = f"phi{phi}_d{d}"
        for k in ("phi", "d", "a_body", "phi_actual", "N_solid", "k_sim", "k_ref",
                  "err_pct", "steady_step", "steps_run", "wall_s"):
            out[f"{k}_{tag}"] = r[k]
        out[f"solid_{tag}"] = r["solid"]
        out[f"ux_{tag}"] = r["ux"]
        out[f"uy_{tag}"] = r["uy"]
        out[f"hist_step_{tag}"] = r["hist_step"]
        out[f"hist_uxf_{tag}"] = r["hist_uxf"]
        t_all += r["wall_s"]
        print(
            f"phi={phi} d={d}: steps={r['steps_run']} steady={r['steady_step']} "
            f"k_sim={r['k_sim']:.4f} k_ref={r['k_ref']:.4f} err={r['err_pct']:.3f}% "
            f"({r['wall_s']:.0f}s)",
            flush=True,
        )
    out["force_scale"] = args.force_scale
    out["elapsed_s"] = t_all
    np.savez(args.out, **out)
    print(f"saved {args.out} (total {t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

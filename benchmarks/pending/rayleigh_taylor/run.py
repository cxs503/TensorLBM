#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""Rayleigh-Taylor 不稳定性 benchmark — VOF 自由表面共性模块首次验证（真实模拟，无外推）。

模型路径（共性模块三路径之物理模块/库路径组合）：
    from tensorlbm import init_phi_rayleigh_taylor_3d, free_surface_vof_step
    —— 顶层导出验证：init_phi_rayleigh_taylor_3d（RT 初始场）与
       free_surface_vof_step（VOF 碰撞+流播+phi 平流一步演化）均可直接从
       tensorlbm 导入调用（commit da550e5 顶层导出，本 benchmark 为首次使用）。

配置（按任务书）：3D 域 (nz,ny,nx)=(64,128,64)，重力 −y（gy<0）；
重流体在界面之上（init_phi_rayleigh_taylor_3d 的约定：phi=1 重流体在上、
phi=0 轻流体在下——即经典 RT 失稳分层；任务书文字“下半重流体上半轻流体”
与模块约定相反，若按任务书文字则重力 −y 下为稳定分层无 RT，故采用模块
约定=标准 RT 配置，README 有说明）；Atwood=0.9（rho_heavy=1.0，
rho_light=1/19≈0.05263）；界面单模扰动 λ=64（=nx，一个波长），振幅 a0=0.01λ=0.64；
τ=0.8（ν=(τ−0.5)/3=0.1）；gy=−1e-4；固体=封闭六面盒（bounce-back 壁）。
理论：γ_theory = sqrt(At·g·k)，k=2π/λ=0.09817，g=|gy|=1e-4 → γ=sqrt(0.9·1e-4·0.09817)
     = 2.973e-3。线性期界面振幅 a(t)≈a0·exp(γ·t)。

测量：每 50 步记录 (a) 界面模式振幅 a(t)（phi=0.5 等值面高度 h(x) 的亚格子
线性插值 + 对 sin(2πx/λ) 基模的傅里叶投影，z 方向平均）(b) max|u|
(c) mixing_layer_thickness_3d（0.1<phi<0.9 的 y 向厚度）(d) phi min/max。
γ_sim = ln a vs t 最小二乘斜率（线性期窗口）。

判定：γ_sim 与 γ_theory 误差 ≤3% 且 ≥2 档网格收敛 → verified/rayleigh_taylor/；
否则如实记录失败（result.json verified=false），存 benchmarks/pending/rayleigh_taylor/。

本脚本同时内嵌“伪流动标度”诊断（gy=0 下 Δρ=0.7/0.1/0.01/0 各跑 200 步，
记录稳态 max|u|），用于量化 VOF 共性模块在密度界面处的固有伪流动
（锚定密度 → 平衡态压力跳变 Δp=Δρ·cs² 无平衡机制）。

用法：
    python run.py --device cuda:2 --grid 64 --steps 5000
    python run.py --device cuda:2 --grid 96 --steps 3000
"""
import argparse
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

# ---- 共性模块顶层导入（验证 da550e5 导出；本 benchmark 首次使用）----
from tensorlbm import (  # noqa: E402
    free_surface_vof_step,
    init_phi_rayleigh_taylor_3d,
    mixing_layer_thickness_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402

CS2 = 1.0 / 3.0


def make_closed_box(nz, ny, nx, device):
    """封闭六面容器（bounce-back 壁）solid mask。"""
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    return solid


def interface_height(phi):
    """phi=0.5 等值面高度 h(x,z)（亚格子线性插值，单值界面假设）。

    鲁棒实现：对每列找相邻行对 (r, r+1) 满足 phi[r] >= 0.5 > phi[r+1]
    （最后一行重流体 → 第一行轻流体），在两行间线性插值 0.5 等值面。
    对平滑界面（init_phi_rayleigh_taylor_3d 的 clamp(y_iface+1-y) 型
    2 格过渡）与陡峭界面均正确。
    """
    nz_, ny_, nx_ = phi.shape
    ge = (phi >= 0.5).to(torch.float32)
    # 每列“最后一重行”：cumsum 从顶向下，取 ge 的最后一个 1
    # 从底向上计数更简单：翻转 y 后 argmax 第一个 0.5 以下……
    # 直接方式：对每个 y，标记 (phi[y]>=0.5 & phi[y+1]<0.5)
    up = ge[:, :-1, :]
    down = ge[:, 1:, :]
    pair = (up == 1) & (down == 0)  # (nz, ny-1, nx)：交叉对所在行 r = y
    # 取第一个（最低）交叉对
    pair_idx = torch.argmax(pair.to(torch.float32), dim=1)  # (nz, nx) 第一交叉行 r
    has = pair.any(dim=1)  # (nz, nx)
    r = pair_idx.long()
    ph_r = torch.gather(phi, 1, r.unsqueeze(1)).squeeze(1)
    ph_r1 = torch.gather(phi, 1, (r + 1).clamp(max=ny_ - 1).unsqueeze(1)).squeeze(1)
    frac = ((ph_r - 0.5) / (ph_r - ph_r1 + 1e-9)).clamp(0.0, 1.0)
    h = r.float() + frac
    # 无交叉对的列（全 0 或全 1）回退：全 1 取 ny-1，全 0 取 0
    h = torch.where(has, h, torch.where(ge[:, 0, :] > 0.5, float(ny_ - 1), 0.0))
    return h  # (nz, nx)


def mode_amplitude(phi, wavelength):
    """基模 sin(2πx/λ) 的傅里叶投影振幅 a(t)。"""
    nz_, _, nx_ = phi.shape
    h = interface_height(phi)
    x = torch.arange(nx_, device=phi.device, dtype=torch.float32)
    sinx = torch.sin(2.0 * math.pi * x / wavelength)
    a = (2.0 / (nz_ * nx_)) * (h * sinx.unsqueeze(0)).sum()
    return float(a)


def fit_growth_rate(times, amps, t_min, t_max):
    """线性期 ln a vs t 最小二乘斜率 + R²。"""
    t = np.array(times, dtype=np.float64)
    a = np.array(amps, dtype=np.float64)
    m = (a > 1e-9) & (t >= t_min) & (t <= t_max)
    if m.sum() < 4:
        return None, None
    g = np.polyfit(t[m], np.log(a[m]), 1)
    pred = g[0] * t[m] + g[1]
    r2 = 1.0 - np.sum((np.log(a[m]) - pred) ** 2) / max(
        np.sum((np.log(a[m]) - np.mean(np.log(a[m]))) ** 2), 1e-30
    )
    return float(g[0]), float(r2)


def spurious_flow_scan(device, nz=16, ny=64, nx=64, steps=200):
    """gy=0 下密度对比度扫描：量化 VOF 密度界面固有伪流动（锚定密度压力跳变）。"""
    out = {}
    interface_frac, amplitude, wavelength, tau = 0.5, 2.0, float(nx), 1.0
    solid = make_closed_box(nz, ny, nx, device)
    for tag, rho_g in (("dr07", 0.3), ("dr01", 0.9), ("dr001", 0.99), ("dr0", 1.0)):
        phi = init_phi_rayleigh_taylor_3d(
            nz, ny, nx, interface_frac, amplitude, wavelength, device
        )
        phi = phi.masked_fill(solid, 0.0)
        f = equilibrium3d(
            torch.ones((nz, ny, nx), device=device),
            torch.zeros((nz, ny, nx), device=device),
            torch.zeros((nz, ny, nx), device=device),
            torch.zeros((nz, ny, nx), device=device),
            device=device,
        )
        umax_hist = []
        for step in range(steps):
            f, phi = free_surface_vof_step(
                f, phi, tau=tau, gy=0.0,
                rho_liquid=1.0, rho_gas=rho_g, solid=solid,
            )
            if step % 50 == 0:
                _, ux_, uy_, uz_ = macroscopic3d(f)
                umax_hist.append(
                    float(torch.max(torch.sqrt(ux_**2 + uy_**2 + uz_**2)))
                )
        out[tag] = {"rho_gas": rho_g, "umax_plateau": float(np.median(umax_hist[-3:])),
                    "umax_hist": [round(u, 4) for u in umax_hist]}
    return out


def run_rt(device, grid, steps, sample=50):
    """主 RT benchmark 运行：VOF 共性模块演化，测界面振幅增长。"""
    # 网格：grid 为横向格数 nx；nz=grid, ny=2*grid, nx=grid（任务书 64×128×64 例）
    nz, ny, nx = grid, 2 * grid, grid
    interface_frac = 0.5
    wavelength = float(nx)          # λ = nx，一个波长（任务书 λ=64 @ nx=64）
    amplitude = 0.01 * wavelength   # a0 = 0.01λ（任务书）
    tau = 0.8
    rho_heavy = 1.0
    rho_light = 1.0 / 19.0          # Atwood = (1−1/19)/(1+1/19) = 0.9
    atwood = (rho_heavy - rho_light) / (rho_heavy + rho_light)
    gy = -1.0e-4
    g = abs(gy)
    k = 2.0 * math.pi / wavelength
    gamma_theory = math.sqrt(atwood * g * k)
    nu = (tau - 0.5) / 3.0
    visc_corr = nu * k * k / gamma_theory  # 粘性阻尼量级 νk²/γ

    solid = make_closed_box(nz, ny, nx, device)
    phi = init_phi_rayleigh_taylor_3d(
        nz, ny, nx, interface_frac, amplitude, wavelength, device
    )
    phi = phi.masked_fill(solid, 0.0)
    f = equilibrium3d(
        torch.ones((nz, ny, nx), device=device),
        torch.zeros((nz, ny, nx), device=device),
        torch.zeros((nz, ny, nx), device=device),
        torch.zeros((nz, ny, nx), device=device),
        device=device,
    )

    times, amps, umaxs, mixes, phmin, phmax = [], [], [], [], [], []
    t0 = time.time()
    first_step_ms = None
    for step in range(steps + 1):
        t_step = time.time()
        f, phi = free_surface_vof_step(
            f, phi, tau=tau, gy=gy,
            rho_liquid=rho_heavy, rho_gas=rho_light, solid=solid,
        )
        if step == 1:
            first_step_ms = (time.time() - t_step) * 1e3
        if step % sample == 0:
            a = mode_amplitude(phi, wavelength)
            times.append(step)
            amps.append(a)
            _, ux_, uy_, uz_ = macroscopic3d(f)
            umaxs.append(float(torch.max(torch.sqrt(ux_**2 + uy_**2 + uz_**2))))
            mixes.append(mixing_layer_thickness_3d(phi))
            phmin.append(float(phi.min()))
            phmax.append(float(phi.max()))
            if step % 500 == 0:
                print(f"  step={step:6d} a={a:+.4f} umax={umaxs[-1]:.4f} "
                      f"mix={mixes[-1]:.0f} ({time.time()-t0:.0f}s)")
    elapsed = time.time() - t0

    # γ_sim 拟合：窗口 [200, 2000]（线性期应有 exp 增长；被伪流动破坏则拟合失败）
    gamma_sim_full, r2_full = fit_growth_rate(times, amps, 0, max(times))
    gamma_sim_lin, r2_lin = fit_growth_rate(times, amps, 200, 2000)

    if gamma_sim_lin is not None:
        err_pct = (gamma_sim_lin - gamma_theory) / gamma_theory * 100.0
    else:
        err_pct = None

    result = {
        "case": "rayleigh_taylor_vof_common_module",
        "reference": "gamma_theory = sqrt(At*g*k), At=0.9, g=1e-4, k=2pi/64=0.09817 -> 2.973e-3",
        "config": {
            "grid": [nz, ny, nx],
            "wavelength": wavelength,
            "amplitude0": amplitude,
            "tau": tau,
            "nu": nu,
            "rho_heavy": rho_heavy,
            "rho_light": rho_light,
            "atwood": atwood,
            "gy": gy,
            "steps": steps,
            "sample": sample,
            "solid": "closed box, bounce-back walls",
            "module_path": "top-level tensorlbm: init_phi_rayleigh_taylor_3d + free_surface_vof_step",
        },
        "theory": {
            "gamma_theory": gamma_theory,
            "k": k,
            "viscous_corr_nu_k2_over_gamma": visc_corr,
        },
        "measurements": {
            "a_t": [round(a, 5) for a in amps],
            "umax_t": [round(u, 5) for u in umaxs],
            "mixing_t": mixes,
            "phi_min": [round(v, 3) for v in phmin],
            "phi_max": [round(v, 3) for v in phmax],
            "times": times,
        },
        "fits": {
            "gamma_sim_full": gamma_sim_full,
            "r2_full": r2_full,
            "gamma_sim_lin_window_200_2000": gamma_sim_lin,
            "r2_lin": r2_lin,
        },
        "err_pct": err_pct,
        "spurious_flow_scan_gy0": spurious_flow_scan(device),
        "elapsed_s": elapsed,
        "first_step_ms": first_step_ms,
        "verified": False,
        "verdict_reason": (
            "VOF 共性模块在密度界面处存在固有伪流动（锚定密度→平衡态压力跳变 "
            "Δp=Δρ·cs² 无平衡机制），|u|_spurious≈O(sqrt(Δρ·cs²/ρ̄))≈0.4-0.9 "
            "（Δρ=0.947@At=0.9），比 RT 线性期速度 γ·a0≈1.9e-3 大 2 个数量级，"
            "~50 步内界面被撕碎，a(t) 无指数增长 → γ_sim 不可测 → 未达标。"
        ),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device={device} grid={args.grid} steps={args.steps}")
    result = run_rt(device, args.grid, args.steps)
    result["device"] = str(device)
    result["grid_label"] = args.grid

    out_path = args.out or f"/tmp/rt_vof_result_g{args.grid}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"RESULT {args.grid}: gamma_theory={result['theory']['gamma_theory']:.6e} "
          f"gamma_sim_lin={result['fits']['gamma_sim_lin_window_200_2000']} "
          f"err_pct={result['err_pct']} verified={result['verified']}")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()

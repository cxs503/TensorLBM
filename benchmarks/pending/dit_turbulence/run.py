#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""DIT benchmark — 高 Re Taylor–Green 能量级联（D3Q19 周期域，Re=1600）。

物理问题
--------
3D Taylor–Green 涡在 Re=1600（转捩到湍流）下的能量级联：初始大尺度涡
拉伸产生小尺度结构，耗散率 ε(t) 在 t_c 处达到峰值（Brachet et al. 1983,
JFM 130:411——谱 DNS Re=1600, 256³：t_c ≈ 9 个大尺度翻转时间）。

初始场（OpenLB tgv3d 同款）：
    u = ( U0·sin(kx)·cos(ky)·cos(kz), -U0·cos(kx)·sin(ky)·cos(kz), 0 )
    k = 2π/N（周期域 [0,N)³），ν = U0·N/Re，τ = 0.5 + 3ν
    U0 = 0.1（Ma≈0.17），Re = U0·N/ν = 1600

无量纲时间：t* = t·k·U0 = t·(2π/N)·U0 = t / (L/(2πU0))（大尺度翻转时间
τ_conv = L/(2πU0) = N/(2πU0) 的倍数）。Brachet 1983: ε(t) 峰值在
t_c* ≈ 9.0（Re=1600，256³ 谱 DNS）。

测量
----
· E(t) = 0.5·mean(u²)（macroscopic3d 实测，每 record_every 步）
· ε(t) = -dE/dt（中心差分，5-9 点 Savitzky-Golay 平滑）→ 峰值时间 t_c
· 交叉验证：ε_ω = ν·mean(ω²)，ω = ∇×u（FFT 谱导数，周期域精确）
· 诊断：E(t_c)/E0、ε_max·L/U0³、质量漂移、w_max 历史（涡拉伸证据）

判定（任务书）：|t_c* - 9.0| ≤ 3%（t_c* ∈ [8.73, 9.27]）且 ≥2 档网格
（N=128/256，同 Re）t_c 单调收敛 → verified。

用法：
    python run.py                        # 完整 benchmark（N=128/256 两档）→ result.json
    python run.py --n 128 --collision mrt --device cuda:2   # 单案例
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

# ── 兼容 shim：并行 agent 正在重构 thermal.py（工作区未提交中间态缺旧 API）──
# 只在本进程内把 git HEAD 版 thermal 挂到 tensorlbm.thermal，不修改任何仓库文件。
def _install_thermal_legacy() -> None:
    import importlib.util
    import subprocess

    src = subprocess.run(
        ["git", "show", "HEAD:src/tensorlbm/thermal.py"],
        cwd="/home/wxsc/cxs/TensorLBM", capture_output=True, text=True,
    ).stdout
    with open("/tmp/thermal_legacy.py", "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("tensorlbm.thermal", "/tmp/thermal_legacy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["tensorlbm.thermal"] = mod


try:
    import tensorlbm  # noqa: F401
except ImportError:
    _install_thermal_legacy()

from tensorlbm.solver3d import (  # noqa: E402
    collide_bgk3d,
    collide_mrt3d,
    collide_rlbm3d,
    stream3d_roll,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.turbulence import collide_smagorinsky_bgk3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = dict(re=1600.0, u0=0.1, t_c_ref=9.0)


def auto_steps(n: int, u0: float, t_star_max: float = 20.0, min_steps: int = 500) -> int:
    """跑到 t* = t_star_max（无量纲），任务范围 500–20000 步。"""
    k = 2.0 * math.pi / n
    steps = int(math.ceil(t_star_max / (k * u0)))
    return max(steps, min_steps)


def sg_smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay 平滑（0 阶=移动平均，用中心点窗口，防峰值偏移）。"""
    w = window if window % 2 == 1 else window + 1
    half = w // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = y[lo:hi].mean()
    return out


def peak_time_parabolic(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """抛物线插值找峰值时间：在离散最大值邻域 3 点拟合。"""
    imax = int(np.argmax(y))
    if imax == 0 or imax == len(y) - 1:
        return float(t[imax]), float(y[imax])
    a, b, c = y[imax - 1], y[imax], y[imax + 1]
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-30:
        return float(t[imax]), float(b)
    dt = t[1] - t[0]
    offset = 0.5 * (a - c) / denom * dt
    t_pk = t[imax] + offset
    y_pk = b - 0.25 * (a - c) * (a - c) / denom
    return float(t_pk), float(y_pk)


def spectral_vorticity_dissipation(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, nu: float,
) -> float:
    """ε_ω = ν·mean(ω²)，ω = ∇×u 用 FFT 谱导数（周期域精确）。"""
    nz, ny, nx = ux.shape
    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=1.0).to(ux.device)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=1.0).to(ux.device)
    kz = 2.0 * torch.pi * torch.fft.fftfreq(nz, d=1.0).to(ux.device)
    KX = kx.view(1, 1, nx)
    KY = ky.view(1, ny, 1)
    KZ = kz.view(nz, 1, 1)
    uxh = torch.fft.fftn(ux)
    uyh = torch.fft.fftn(uy)
    uzh = torch.fft.fftn(uz)
    # ω = ∇×u
    wx = 1j * (KY * uzh - KZ * uyh)
    wy = 1j * (KZ * uxh - KX * uzh)
    wz = 1j * (KX * uyh - KY * uxh)
    om2 = (wx.abs() ** 2 + wy.abs() ** 2 + wz.abs() ** 2).mean().item()
    return float(nu * om2)


def run_case(
    n: int,
    re: float,
    u0: float,
    steps: int,
    record_every: int = 0,
    collision: str = "mrt",
    device: str = "cuda:2",
    dtype: str = "float32",
    threads: int = 32,
    t_star_max: float = 20.0,
) -> dict:
    torch.set_num_threads(threads)
    dt = torch.float64 if dtype == "float64" else torch.float32

    k = 2.0 * math.pi / float(n)
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    t_conv = n / (2.0 * math.pi * u0)      # 大尺度翻转时间 L/(2πU0)（步）
    if record_every == 0:
        record_every = max(1, int(round(0.05 * t_conv)))  # 无量纲采样 Δt*=0.05

    collide = {
        "bgk": collide_bgk3d,
        "mrt": collide_mrt3d,
        "rlbm": collide_rlbm3d,
        "smag": lambda f, t: collide_smagorinsky_bgk3d(f, t, C_s=0.1),
    }[collision]

    # ── 初始场：3D TG 涡（周期域，f 形状 (19,nz,ny,nx)）──────────────
    z, y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * x) * torch.cos(k * y) * torch.cos(k * z)
    uy0 = -u0 * torch.cos(k * x) * torch.sin(k * y) * torch.cos(k * z)
    uz0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium3d(rho, ux0, uy0, uz0)

    mass0 = float(f.sum().item())
    e0 = float((0.5 * (ux0 * ux0 + uy0 * uy0)).mean().item())

    # ── 主循环：stream（roll 周期流播）→ collide ──────────────────────
    times: list[int] = []
    energies: list[float] = []
    eps_kin: list[float] = []     # ε = -dE/dt（后处理）
    eps_w: list[float] = []       # ε_ω = ν·mean(ω²)（FFT，交叉验证）
    wmaxs: list[float] = []
    wall0 = time.time()
    for step in range(1, steps + 1):
        f = stream3d_roll(f)
        f = collide(f, tau)
        if step % record_every == 0:
            _, uxm, uym, uzm = macroscopic3d(f)
            e = float((0.5 * (uxm * uxm + uym * uym + uzm * uzm)).mean().item())
            if not math.isfinite(e):
                raise RuntimeError(f"NaN/Inf at step {step}")
            times.append(step)
            energies.append(e)
            eps_w.append(spectral_vorticity_dissipation(uxm, uym, uzm, nu))
            wmaxs.append(float(uzm.abs().max().item()))
    wall = time.time() - wall0

    t_arr = np.asarray(times, dtype=np.float64) / t_conv   # 无量纲 t*
    E = np.asarray(energies, dtype=np.float64)

    # ε_kin = -dE/dt（中心差分）
    eps_kin_arr = np.empty_like(E)
    eps_kin_arr[0] = -(E[1] - E[0]) / (t_arr[1] - t_arr[0])
    eps_kin_arr[-1] = -(E[-1] - E[-2]) / (t_arr[-1] - t_arr[-2])
    eps_kin_arr[1:-1] = -(E[2:] - E[:-2]) / (t_arr[2:] - t_arr[:-2])
    # 平滑（窗口≈0.3 无量纲时间），再找峰值
    sw = max(3, int(round(0.3 / (t_arr[1] - t_arr[0]))))
    if sw % 2 == 0:
        sw += 1
    eps_sm = sg_smooth(eps_kin_arr, sw)

    t_c_sim, eps_max = peak_time_parabolic(t_arr, eps_sm)
    # ε_ω 峰值时间（交叉验证）
    eps_w_arr = np.asarray(eps_w, dtype=np.float64)
    eps_w_sm = sg_smooth(eps_w_arr, sw)
    t_c_w, eps_w_max = peak_time_parabolic(t_arr, eps_w_sm)

    err_tc = (t_c_sim - CANONICAL["t_c_ref"]) / CANONICAL["t_c_ref"] * 100.0

    # 诊断：t_c 时刻能量占比（插值）
    e_at_tc = float(np.interp(t_c_sim, t_arr, E) / E[0])
    eps_max_norm = eps_max / (u0**3 / n)   # ε* = ε·L/U0³
    mass_end = float(f.sum().item())

    return {
        "n": n, "re": re, "u0": u0, "nu": nu, "tau": tau, "k": k,
        "t_conv_steps": t_conv, "steps": steps, "record_every": record_every,
        "collision": collision, "dtype": dtype, "device": device,
        "t_c_ref": CANONICAL["t_c_ref"],
        "t_c_star": t_c_sim, "err_tc_pct": err_tc,
        "t_c_wale_star": t_c_w,
        "eps_max": eps_max, "eps_max_star": eps_max_norm,
        "eps_w_max": eps_w_max,
        "e_over_e0_at_tc": e_at_tc,
        "e0": e0, "w_max": max(wmaxs),
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times),
        "wall_sec": wall,
        "times_star": t_arr.tolist(),
        "energy": E.tolist(),
        "eps_kin_smoothed": eps_sm.tolist(),
        "eps_omega": eps_w_arr.tolist(),
        "wmaxs": wmaxs,
    }


def judge(cases: list[dict], tol_pct: float = 3.0) -> dict:
    """网格收敛判定：|err_tc| ≤3% 且 |err_tc| 随 N 单调下降（≥2 档）。"""
    by_n = {c["n"]: c for c in cases}
    ns = sorted(by_n)
    errs = [by_n[n]["err_tc_pct"] for n in ns]
    tcs = [by_n[n]["t_c_star"] for n in ns]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs)
    # 档间差判据（t_c 值本身两档接近）
    spread = (max(tcs) - min(tcs)) / CANONICAL["t_c_ref"] * 100.0
    verified = bool(within and converged and len(ns) >= 2)
    return {
        "grids": ns,
        "t_c_star_per_grid": dict(zip(ns, tcs)),
        "err_tc_pct_per_grid": dict(zip(ns, errs)),
        "grid_spread_pct": spread,
        "converged_monotone": converged,
        "within_tol": within,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": (
            "PASS: t_c* 偏差≤3% 且随 N 单调收敛 → 保存 verified/"
            if verified else
            "FAIL: 未达 ≤3% 或未收敛 → 不保存 verified/"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="High-Re Taylor-Green energy cascade benchmark (D3Q19)")
    ap.add_argument("--n", type=int, default=0, help="网格 N（0=完整 benchmark 的 128/256 两档）")
    ap.add_argument("--re", type=float, default=CANONICAL["re"])
    ap.add_argument("--u0", type=float, default=CANONICAL["u0"])
    ap.add_argument("--steps", type=int, default=0, help="0 = auto (t*到 20)")
    ap.add_argument("--record-every", type=int, default=0)
    ap.add_argument("--collision", choices=["bgk", "mrt", "rlbm", "smag"], default="mrt")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grids = [128, 256] if args.n == 0 else [args.n]
    results, case_files = [], []
    for n in grids:
        steps = args.steps or auto_steps(n, args.u0)
        print(f"=== N={n}³  Re={args.re}  U0={args.u0}  steps={steps}  "
              f"collision={args.collision} ===", flush=True)
        res = run_case(n, args.re, args.u0, steps,
                       record_every=args.record_every, collision=args.collision,
                       dtype=args.dtype, device=args.device, threads=args.threads)
        keep = {k: v for k, v in res.items()
                if k not in ("times_star", "energy", "eps_kin_smoothed",
                             "eps_omega", "wmaxs")}
        name = f"case_N{n}.json"
        with open(os.path.join(args.out, name), "w") as fh:
            json.dump(keep, fh, indent=2, ensure_ascii=False)
        np.savetxt(
            os.path.join(args.out, f"energy_history_N{n}.csv"),
            np.column_stack([res["times_star"], res["energy"],
                             res["eps_kin_smoothed"], res["eps_omega"], res["wmaxs"]]),
            header="t_star,energy,eps_kin_smoothed,eps_omega,wmax",
            delimiter=",", comments="",
        )
        case_files.append(name)
        results.append(res)
        print(f"  t_c*={res['t_c_star']:.4f}  err_tc={res['err_tc_pct']:+.3f}%  "
              f"(t_c_wale={res['t_c_wale_star']:.4f})  E(tc)/E0={res['e_over_e0_at_tc']:.4f}  "
              f"eps_max*={res['eps_max_star']:.4f}  wall={res['wall_sec']:.0f}s", flush=True)

    if len(grids) >= 2:
        verdict = judge(results)
        out = {
            "benchmark": "dit_turbulence",
            "description": (
                "High-Re Taylor-Green energy cascade (D3Q19 periodic cube): "
                "u=(U0 sin(kx)cos(ky)cos(kz), -U0 cos(kx)sin(ky)cos(kz), 0), k=2π/N, "
                "Re=1600, U0=0.1. Dissipation-rate peak time t_c vs Brachet 1983 "
                "(spectral DNS: t_c*≈9 large-eddy turnover times L/(2πU0))."
            ),
            "lattice": "D3Q19", "collision": args.collision,
            "boundary": "periodic (stream3d_roll)",
            "extrap": "none",
            "common_modules": ["solver3d.collide_mrt3d", "solver3d.stream3d_roll",
                               "d3q19.equilibrium3d", "d3q19.macroscopic3d"],
            "formula": (
                "t* = t·k·U0 (k=2π/N), t_conv = N/(2πU0); Brachet 1983 Re=1600: "
                "t_c* ≈ 9.0; ε(t) = -dE/dt, E=0.5·mean(u²); cross-check ε_ω=ν·mean(ω²)"
            ),
            "reference": ("Brachet, Meiron, Orszag, Nickel, Morf & Frisch (1983), "
                          "'Small-scale structure of the Taylor-Green vortex', "
                          "JFM 130, 411-452: dissipation peak t_c*≈9 at Re=1600 (256³ spectral DNS)"),
            "cases": [{k: v for k, v in r.items()
                       if k not in ("times_star", "energy", "eps_kin_smoothed",
                                    "eps_omega", "wmaxs")} for r in results],
            "case_files": case_files,
            "convergence": verdict,
        }
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("\n=== 判定 ===")
        print("  t_c*: " + ", ".join(f"N{n}={tc:.4f}" for n, tc in
                                     zip(verdict["grids"], verdict["t_c_star_per_grid"].values())))
        print("  err_tc: " + ", ".join(f"N{n}={e:+.3f}%" for n, e in
                                       zip(verdict["grids"], verdict["err_tc_pct_per_grid"].values())))
        print(f"  converged_monotone={verdict['converged_monotone']}  "
              f"within_tol={verdict['within_tol']}  grid_spread={verdict['grid_spread_pct']:.2f}%")
        print(f"  {verdict['judgment']}")
        print(f"  -> {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()

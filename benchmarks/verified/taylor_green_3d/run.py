#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B17-3D: 3D Taylor–Green 涡衰减 benchmark（D3Q19 周期域，OpenLB tgv3d 同款初始场）。

真实模拟（无外推、无人工修正）：全部能量/速度由 macroscopic3d 每步实测。

物理问题
--------
周期域 [0,N)³（晶格单位 Δx=Δt=1），k=2π/N，ν=(τ−1/2)/3，Re=U0·N/ν，
初始场（OpenLB tgv3d 同款）：

    u = ( U0·sin(kx)·cos(ky)·cos(kz),  -U0·cos(kx)·sin(ky)·cos(kz),  0 )

⚠️ 衰减率推导（关键，与 2D 不同）：
    · 该场由波矢 (±k,±k,±k) 的 8 支 Fourier 模叠加，每支模 |κ|² = 3k²
      （2D TG 是 (±k,±k) → |κ|²=2k²）。
    · 线性粘性衰减 ⇒ 速度衰减率 γ_vel = ν|κ|² = **3νk²**，
      动能衰减率 γ_E = 2·γ_vel = **6νk²**；E(t) = (U0²/8)·e^{−6νk²t}。
    · 任务书/问题清单的 "e^{−2νk²t}" 是 2D TG 的速度衰减率，3D 场套用会报
      +200% 假误差（先数波矢范数，见 skill trap 1）——本脚本同时记录该
      对比（err_vs_task_formula_pct）仅作文档，不作判据。

与 2D TG 的关键区别（物理真实性）：
    · 2D TG 是 NS 精确解（u·∇u 恰为压力梯度），衰减纯指数；
    · 3D TG 不是精确解——u·∇u 的旋度部分立即驱动涡拉伸，w 从 0 增长，
      小尺度模（|κ|²≥4k²）使 Z/E 略升 ⇒ γ_sim 略高于 6νk²（实测 w_max
      约 0.0026·U0、ez_max≈4e-7@Re=24——准稳态拉伸效应，随 Re 增大而增大，
      随能量衰减回归 6νk²）。如实测量并报告。

测量与判定：
    · γ_E_sim = -d(ln E)/dt（线性拟合）对比 γ_E_theory = 6νk²（主指标）；
    · 判定：三档网格（N=64/96/128，同一 Re）|err_E| ≤ 3% 且 |err_E| 随 N
      单调下降 → 达标；R² ≥ 0.999 指数性检查；
    · 真实性证据：E_x/E_y/E_z 分分量历史（w 能量从 0 增长再衰减）、w_max
      历史、质量漂移、E0_meas vs U0²/8。

规范配置（2026-08-19 扫描定案，扫描见 README）：
    U0=0.05, Re=24（τ=0.9/1.1/1.3 @ N=64/96/128）——低 Re 纯层流衰减，
    实测 err_E = +0.309% → +0.279% → +0.250%（单调收敛，R²≥0.99999）。

用法：
    python run.py                        # 完整 benchmark（N=64/96/128 三档）→ result.json
    python run.py --n 64 --re 24 --steps 2000    # 单案例
    python run.py --scan                 # Re×N 扫描 → scan_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = dict(re=24.0, u0=0.05, dtype="float32")


def auto_steps(
    n: int,
    re: float,
    u0: float,
    record_every: int,
    target_efolds: float = 10.0,
    cap: int = 10000,
    min_steps: int = 2000,
) -> int:
    """按能量目标衰减 E/E0 → e^{-target_efolds} 自动确定步数（任务范围 2000–10000）。"""
    nu = u0 * n / re
    k = 2.0 * math.pi / n
    gamma_e = 6.0 * nu * k * k  # γ_E = 6νk²（3D TG，|κ|²=3k²）
    steps = int(math.ceil(target_efolds / gamma_e / record_every) * record_every)
    return min(max(steps, min_steps), cap)


def run_case(
    n: int,
    re: float,
    u0: float,
    steps: int,
    record_every: int = 50,
    device: str = "cpu",
    dtype: str = "float32",
    threads: int = 32,
    compile_mode: str | None = "default",
) -> dict:
    torch.set_num_threads(threads)
    dt = torch.float64 if dtype == "float64" else torch.float32

    k = 2.0 * math.pi / float(n)
    nu = u0 * n / re  # 晶格单位运动粘度（Re = U0·N/ν）
    tau = 0.5 + 3.0 * nu  # BGK 弛豫时间
    gamma_e_theory = 6.0 * nu * k * k  # 能量衰减率 6νk²（|κ|²=3k²）
    gamma_vel_theory = 3.0 * nu * k * k  # 速度衰减率 3νk²
    gamma_task_formula = 2.0 * nu * k * k  # 任务书公式（2D TG 速度衰减率）——仅记录
    re_eff = u0 / (nu * k)  # 非线性强度 U0/(νk) = Re/(2π)

    # ── 初始场：3D TG 涡（周期域，z,y,x 顺序，f 形状 (19,nz,ny,nx)）────────
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
    f = equilibrium3d(rho, ux0, uy0, uz0)  # 初值 f = f_eq（标准做法）

    mass0 = float(f.sum().item())
    e0_theory = u0 * u0 / 8.0  # mean(sin²cos²cos²)=1/8，两个分量各半

    # ── 主循环：stream（周期模运算内建）→ collide（BGK）───────────────────
    # 整步步进函数经共性 compile 路径；步序号与采样监测留在编译域外。
    def _step(f):
        return collide_bgk3d(stream3d(f), tau)

    step_fn = route_step(_step, compile_mode, name=f"taylor_green_3d[N{n}]")

    times: list[int] = [0]  # t=0 占位，循环后填初值
    energies: list[float] = [0.0]
    exs: list[float] = [0.0]
    eys: list[float] = [0.0]
    ezs: list[float] = [0.0]
    umaxs: list[float] = [0.0]
    wmaxs: list[float] = [0.0]
    wall0 = time.time()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % record_every == 0:
            _, uxm, uym, uzm = macroscopic3d(f)
            ex = float((0.5 * (uxm * uxm)).mean().item())
            ey = float((0.5 * (uym * uym)).mean().item())
            ez = float((0.5 * (uzm * uzm)).mean().item())
            um = float((uxm * uxm + uym * uym + uzm * uzm).sqrt().max().item())
            wm = float(uzm.abs().max().item())
            times.append(step)
            energies.append(ex + ey + ez)
            exs.append(ex)
            eys.append(ey)
            ezs.append(ez)
            umaxs.append(um)
            wmaxs.append(wm)
    wall = time.time() - wall0

    # 记录 t=0 初值（由初始速度场直接给出，E0 实测）
    ex0 = float((0.5 * (ux0 * ux0)).mean().item())
    ey0 = float((0.5 * (uy0 * uy0)).mean().item())
    ez0 = 0.0
    energies[0] = ex0 + ey0 + ez0
    exs[0] = ex0
    eys[0] = ey0
    ezs[0] = ez0
    umaxs[0] = float((ux0 * ux0 + uy0 * uy0).sqrt().max().item())
    wmaxs[0] = 0.0

    # ── 拟合（t≥record_every 起的实测点；polyfit 返回 [斜率, 截距]）────────
    t = np.asarray(times[1:], dtype=np.float64)
    lnE = np.log(np.asarray(energies[1:], dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))

    urms = np.sqrt(2.0 * np.asarray(energies, dtype=np.float64))
    lnU = np.log(urms[1:])
    a_u, _ = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u

    # 分段一致性检查（纯指数验证；3D 涡拉伸会使其略有差异，如实报告）
    half = len(t) // 2
    g_h1 = -float(np.polyfit(t[:half], lnE[:half], 1)[0])
    g_h2 = -float(np.polyfit(t[half:], lnE[half:], 1)[0])

    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0
    err_task_pct = (gamma_e_sim - gamma_task_formula) / gamma_task_formula * 100.0
    mass_end = float(f.sum().item())

    return {
        "n": n,
        "re": re,
        "u0": u0,
        "nu": nu,
        "tau": tau,
        "k": k,
        "re_eff": re_eff,
        "steps": steps,
        "record_every": record_every,
        "dtype": dtype,
        "device": device,
        "compile_mode": compile_mode,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": err_e_pct,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": err_vel_pct,
        "gamma_task_formula_2d": gamma_task_formula,
        "err_vs_task_formula_pct": err_task_pct,
        "r2": r2,
        "gamma_e_half1": g_h1,
        "gamma_e_half2": g_h2,
        "e0_theory": e0_theory,
        "e0_meas": energies[0],
        "ez0_meas": ezs[0],
        "ez_max": max(ezs),
        "w_max": max(wmaxs),
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times) - 1,
        "wall_sec": wall,
        "times": times,
        "energies": energies,
        "exs": exs,
        "eys": eys,
        "ezs": ezs,
        "umaxs": umaxs,
        "wmaxs": wmaxs,
    }


def save_case(res: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    keep = {
        k: v
        for k, v in res.items()
        if k not in ("times", "energies", "exs", "eys", "ezs", "umaxs", "wmaxs")
    }
    name = f"case_N{res['n']}.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(keep, fh, indent=2, ensure_ascii=False)
    hist = np.column_stack(
        [
            res["times"],
            res["energies"],
            res["exs"],
            res["eys"],
            res["ezs"],
            res["umaxs"],
            res["wmaxs"],
        ]
    )
    np.savetxt(
        os.path.join(out_dir, f"energy_history_N{res['n']}.csv"),
        hist,
        header="step,energy,ex,ey,ez,umax,wmax",
        delimiter=",",
        comments="",
    )
    return name


def judge(cases: list[dict], tol_pct: float = 3.0) -> dict:
    """网格收敛判定：|err_E| ≤3% 且 |err_E| 随 N 单调下降（≥2 档）。"""
    by_n = {c["n"]: c for c in cases}
    ns = sorted(by_n)
    errs = [by_n[n]["err_e_pct"] for n in ns]
    errs_v = [by_n[n]["err_vel_pct"] for n in ns]
    r2s = [by_n[n]["r2"] for n in ns]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs)
    within_v = all(abs(e) <= tol_pct for e in errs_v)
    expo = all(r2 >= 0.999 for r2 in r2s)
    verified = bool(within and within_v and converged and expo and len(ns) >= 2)
    return {
        "grids": ns,
        "err_e_pct_per_grid": dict(zip(ns, errs)),
        "err_vel_pct_per_grid": dict(zip(ns, errs_v)),
        "r2_per_grid": dict(zip(ns, r2s)),
        "converged_monotone": converged,
        "within_tol": within,
        "within_tol_vel": within_v,
        "exponential_r2_ok": expo,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": (
            "PASS: 网格 γ_E 偏差≤3% 且随 N 收敛 → 保存 verified/"
            if verified
            else "FAIL: 未达 ≤3% 或未收敛（见各案例 err_e_pct 与收敛趋势）→ 不保存 verified/"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="B17-3D Taylor-Green vortex decay benchmark (D3Q19)")
    ap.add_argument("--n", type=int, default=0, help="网格 N（0=完整 benchmark 的 64/96/128 三档）")
    ap.add_argument("--re", type=float, default=CANONICAL["re"])
    ap.add_argument("--u0", type=float, default=CANONICAL["u0"])
    ap.add_argument("--steps", type=int, default=0, help="0 = auto (E/E0→e^-10, 范围 2000-10000)")
    ap.add_argument("--record-every", type=int, default=50)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--scan", action="store_true")
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    compile_mode = compile_mode_from_args(args)

    if args.scan:
        cases = []
        for re in (16, 24, 32, 48):
            for n in (64, 96):
                steps = args.steps or auto_steps(n, re, args.u0, args.record_every)
                res = run_case(
                    n,
                    re,
                    args.u0,
                    steps,
                    record_every=args.record_every,
                    dtype=args.dtype,
                    device=args.device,
                    threads=args.threads,
                )
                cases.append(
                    {
                        k: v
                        for k, v in res.items()
                        if k not in ("times", "energies", "exs", "eys", "ezs", "umaxs", "wmaxs")
                    }
                )
                print(
                    f"Re={re:3d} N={n:3d}: err_E={res['err_e_pct']:+.4f}%  "
                    f"R2={res['r2']:.6f}  half=({res['gamma_e_half1']:.4e}/{res['gamma_e_half2']:.4e})",
                    flush=True,
                )
        with open(os.path.join(args.out, "scan_summary.json"), "w") as fh:
            json.dump(cases, fh, indent=2, ensure_ascii=False)
        print(f"\nscan done -> {os.path.join(args.out, 'scan_summary.json')}")
        return

    # ── 完整 benchmark：N=64/96/128 三档网格（同一 Re/U0）────────────────
    if args.n == 0:
        grids = [64, 96, 128]
        results, case_files = [], []
        for n in grids:
            steps = args.steps or auto_steps(n, args.re, args.u0, args.record_every)
            print(f"=== N={n}³  Re={args.re}  U0={args.u0}  steps={steps} ===", flush=True)
            res = run_case(
                n,
                args.re,
                args.u0,
                steps,
                record_every=args.record_every,
                dtype=args.dtype,
                device=args.device,
                threads=args.threads,
                compile_mode=compile_mode,
            )
            case_files.append(save_case(res, args.out))
            results.append(res)
            print(
                f"  gamma_E_sim={res['gamma_e_sim']:.6e} theory={res['gamma_e_theory']:.6e} "
                f"err_E={res['err_e_pct']:+.4f}%  R2={res['r2']:.6f}  "
                f"wall={res['wall_sec']:.1f}s",
                flush=True,
            )
        verdict = judge(results)
        out = {
            "benchmark": "taylor_green_3d",
            "description": (
                "3D Taylor-Green vortex decay: u=(U0·sin(kx)cos(ky)cos(kz), "
                "-U0·cos(kx)sin(ky)cos(kz), 0), k=2π/N, periodic N³ domain, "
                "laminar viscous decay, OpenLB tgv3d-style initial field"
            ),
            "lattice": "D3Q19",
            "collision": "bgk",
            "boundary": "periodic (stream3d 模运算, 库内建)",
            "extrap": "none",
            "common_modules": [
                "solver3d.collide_bgk3d",
                "solver3d.stream3d",
                "d3q19.equilibrium3d",
                "d3q19.macroscopic3d",
            ],
            "formula": (
                "gamma_vel_theory = 3·nu·k², gamma_E_theory = 6·nu·k² "
                "(modes (±k,±k,±k), |κ|²=3k²; nu=(tau-1/2)/3, k=2π/N); "
                "task-formula 2νk² is the 2D velocity rate — documented, not used"
            ),
            "task_formula_note": (
                "任务书/问题清单 'e^{-2νk²t}' 是 2D TG（|κ|²=2k²）的速度衰减率；"
                "3D 场波矢 (±k,±k,±k) → |κ|²=3k²，正确对比为 γ_vel=3νk²、γ_E=6νk²。"
                "err_vs_task_formula_pct ≈ +200%（仅记录不作判据）。"
            ),
            "physics_note": (
                "3D TG 非 NS 精确解（2D 是）：u·∇u 的旋部分立即驱动涡拉伸，"
                "w 从 0 增长（w_max≈0.0026U0、ez_max≈4e-7@Re=24），小尺度模使 "
                "Z/E 略升 ⇒ γ_sim 略高于 6νk²（+0.25~0.31%），随 Re 增大而增大"
                "（Re=48: +1.3~1.5%）——如实测量，不修正。"
            ),
            "cases": [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("times", "energies", "exs", "eys", "ezs", "umaxs", "wmaxs")
                }
                for r in results
            ],
            "case_files": case_files,
            "convergence": verdict,
        }
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("\n=== 判定 ===")
        print(
            "  err_E: "
            + ", ".join(
                f"N{n}={e:+.4f}%"
                for n, e in zip(verdict["grids"], verdict["err_e_pct_per_grid"].values())
            )
        )
        print(
            f"  converged_monotone={verdict['converged_monotone']}  within_tol={verdict['within_tol']}  "
            f"r2_ok={verdict['exponential_r2_ok']}"
        )
        print(f"  {verdict['judgment']}")
        print(f"  -> {os.path.join(args.out, 'result.json')}")
        return

    # ── 单案例模式 ──────────────────────────────────────────────────────
    steps = args.steps or auto_steps(args.n, args.re, args.u0, args.record_every)
    res = run_case(
        args.n,
        args.re,
        args.u0,
        steps,
        record_every=args.record_every,
        dtype=args.dtype,
        device=args.device,
        threads=args.threads,
        compile_mode=compile_mode,
    )
    save_case(res, args.out)
    print(
        f"N={args.n}³ Re={args.re} U0={args.u0} steps={steps} dtype={args.dtype} device={args.device}"
    )
    print(
        f"  nu={res['nu']:.6f}  tau={res['tau']:.6f}  k={res['k']:.6f}  Re_eff={res['re_eff']:.2f}"
    )
    print(
        f"  gamma_E_theory={res['gamma_e_theory']:.6e}  gamma_E_sim={res['gamma_e_sim']:.6e}  err_E={res['err_e_pct']:+.4f}%"
    )
    print(
        f"  gamma_vel_theory={res['gamma_vel_theory']:.6e}  gamma_vel_sim={res['gamma_vel_sim']:.6e}  err_vel={res['err_vel_pct']:+.4f}%"
    )
    print(
        f"  R2={res['r2']:.6f}  half=({res['gamma_e_half1']:.4e}/{res['gamma_e_half2']:.4e})  "
        f"mass_drift_rel={res['mass_drift_rel']:.2e}  wall={res['wall_sec']:.1f}s"
    )
    print(f"  -> {os.path.join(args.out, f'case_N{args.n}.json')}")


if __name__ == "__main__":
    main()

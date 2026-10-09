#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""2D 衰减剪切波 benchmark（粘性耗散验证，周期 D2Q9 BGK，真实模拟，禁外推）。

解析解（不可压 Navier–Stokes 的精确解；纯剪切 u·∇u ≡ 0，无涡拉伸）：
    u(x,y,t) = U0·sin(ky)·e^{-νk²t},   v ≡ 0
    ν = (τ − 1/2)/3,   k = 2π/H（晶格单位 H=N, Δx=Δt=1）

与 Taylor–Green（B17）的关键区别：
    · TG 是涡场（波矢 (±k,±k)，|κ|² = 2k²，速度衰减率 2νk²，能量 4νk²）；
    · 本案例是单 Fourier 模 (0, k) 纯剪切波，|κ|² = k²：
        速度衰减率 γ_vel = νk²，动能衰减率 γ_E = 2νk²。
    · 无涡拉伸、无非线性项（u·∇u≡0），衰减应更接近解析（BGK 离散修正 O(k⁴)）。

测量与判定：
    · γ_vel_sim = -d(ln u_max)/dt（线性拟合）对比 γ_vel_theory = νk²（主指标）；
    · γ_E_sim = -d(ln E)/dt 对比 γ_E_theory = 2νk²（交叉验证）；
    · 判定：两档网格（H=64/128）|γ_sim/γ_theory − 1| ≤ 3% 且误差随 H 收敛 → 达标；
    · 偏差符号如实记录（BGK 离散耗散可为正或负）。

用法：
    python run.py                      # 完整 benchmark（H=64/128 两档网格收敛）→ result.json
    python run.py --n 64 --steps 1000  # 单案例
    python run.py --scan               # H×U0 扫描 → scan_summary.json
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

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = dict(n=128, tau=0.8, u0=0.05, dtype="float32")


def auto_steps(
    n: int, tau: float, u0: float, record_every: int, cap: int = 10000, min_steps: int = 2000
) -> int:
    """按速度振幅目标衰减 u_max/U0 → e^-3 自动确定步数（任务范围 2000–10000）。"""
    nu = (tau - 0.5) / 3.0
    k = 2.0 * math.pi / n
    gamma = nu * k * k
    steps = max(min_steps, int(math.ceil(3.0 / gamma / record_every) * record_every))
    return min(steps, cap)


def run_case(
    n: int,
    tau: float,
    u0: float,
    steps: int,
    record_every: int = 100,
    device: str = "cpu",
    dtype: str = "float32",
    threads: int = 32,
    compile_mode: str | None = "default",
) -> dict:
    torch.set_num_threads(threads)
    dt = torch.float64 if dtype == "float64" else torch.float32

    L = float(n)
    k = 2.0 * math.pi / L
    nu = (tau - 0.5) / 3.0
    gamma_vel_theory = nu * k * k  # 速度衰减率 νk²
    gamma_e_theory = 2.0 * nu * k * k  # 动能衰减率 2νk²（E ∝ u²）

    # ── 初始场：纯剪切波 u = U0·sin(ky)，v=0，周期域 x,y ∈ [0,N) ────────
    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    ux0 = u0 * torch.sin(k * y)
    uy0 = torch.zeros_like(ux0)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)  # feq 初值即可（瞬态 ~10τ 内消失）

    mass0 = float(f.sum().item())
    e0_theory = u0 * u0 / 4.0  # 0.5·mean(sin²) = 1/4
    umax0_meas = float(ux0.abs().max().item())  # y=H/4 处 sin 峰值恰在格点上

    # ── 主循环：stream（周期 wrap 内建）→ collide（BGK）──────────────────
    # 整步步进函数经共性 compile 路径；步序号与采样监测留在编译域外。
    def _step(f):
        return collide_bgk(stream(f), tau)

    step_fn = route_step(_step, compile_mode, name=f"shear_wave_decay[H{n}]")

    times: list[int] = []
    energies: list[float] = []
    umaxs: list[float] = []
    wall0 = time.time()
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % record_every == 0:
            _, uxm, uym = macroscopic(f)
            e = float((0.5 * (uxm * uxm + uym * uym)).mean().item())
            um = float((uxm * uxm + uym * uym).sqrt().max().item())
            times.append(step)
            energies.append(e)
            umaxs.append(um)
    wall = time.time() - wall0

    # ── 拟合：polyfit 返回 [斜率, 截距]，γ = -斜率 = -a ───────────────────
    t = np.asarray(times, dtype=np.float64)
    lnE = np.log(np.asarray(energies, dtype=np.float64))
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid_e = lnE - (b_e + a_e * t)
    r2_e = 1.0 - float(np.sum(resid_e**2) / np.sum((lnE - lnE.mean()) ** 2))

    lnU = np.log(np.asarray(umaxs, dtype=np.float64))
    a_u, b_u = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u
    resid_u = lnU - (b_u + a_u * t)
    r2_u = 1.0 - float(np.sum(resid_u**2) / np.sum((lnU - lnU.mean()) ** 2))

    # 半窗口一致性检查（验证纯指数衰减，防拟合假象）
    half = len(t) // 2
    gv_h1 = -float(np.polyfit(t[:half], lnU[:half], 1)[0])
    gv_h2 = -float(np.polyfit(t[half:], lnU[half:], 1)[0])

    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0
    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    mass_end = float(f.sum().item())

    return {
        "H": n,
        "tau": tau,
        "u0": u0,
        "nu": nu,
        "k": k,
        "steps": steps,
        "record_every": record_every,
        "dtype": dtype,
        "device": device,
        "compile_mode": compile_mode,
        "gamma_vel_theory": gamma_vel_theory,
        "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": err_vel_pct,
        "r2_vel": r2_u,
        "gamma_vel_half1": gv_h1,
        "gamma_vel_half2": gv_h2,
        "gamma_e_theory": gamma_e_theory,
        "gamma_e_sim": gamma_e_sim,
        "err_e_pct": err_e_pct,
        "r2_e": r2_e,
        "e0_theory": e0_theory,
        "e0_meas": energies[0],
        "umax0_theory": u0,
        "umax0_meas": umax0_meas,
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times),
        "wall_sec": wall,
        "times": times,
        "energies": energies,
        "umaxs": umaxs,
    }


def save_case(res: dict, out_dir: str, tag: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    keep = {k: v for k, v in res.items() if k not in ("times", "energies", "umaxs")}
    name = f"case_H{res['H']}{tag}.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(keep, fh, indent=2, ensure_ascii=False)
    hist = np.column_stack([res["times"], res["energies"], res["umaxs"]])
    np.savetxt(
        os.path.join(out_dir, f"energy_history_H{res['H']}{tag}.csv"),
        hist,
        header="step,energy,umax",
        delimiter=",",
        comments="",
    )
    return name


def judge(cases: list[dict], tol_pct: float = 3.0) -> dict:
    """两档网格收敛判定：误差 ≤3% 且 |err| 随 H 单调下降。"""
    by_h = {c["H"]: c for c in cases}
    hs = sorted(by_h)
    errs = [by_h[h]["err_vel_pct"] for h in hs]
    errs_e = [by_h[h]["err_e_pct"] for h in hs]
    r2s = [by_h[h]["r2_vel"] for h in hs]
    converged = all(abs(errs[i + 1]) < abs(errs[i]) for i in range(len(errs) - 1))
    within = all(abs(e) <= tol_pct for e in errs) and all(abs(e) <= tol_pct for e in errs_e)
    expo = all(r2 >= 0.999 for r2 in r2s)
    verified = bool(within and converged and expo and len(hs) >= 2)
    return {
        "grids": hs,
        "err_vel_pct_per_grid": dict(zip(hs, errs)),
        "err_e_pct_per_grid": dict(zip(hs, errs_e)),
        "r2_vel_per_grid": dict(zip(hs, r2s)),
        "converged_monotone": converged,
        "within_tol": within,
        "exponential_r2_ok": expo,
        "tol_pct": tol_pct,
        "verified": verified,
        "judgment": (
            "PASS: 两档网格 γ 偏差≤3% 且随 H 收敛 → 保存 verified/"
            if verified
            else "FAIL: 未达 ≤3% 或未收敛（见各案例 err_vel_pct 与收敛趋势）→ 不保存 verified/"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2D decaying shear wave benchmark (viscous dissipation)"
    )
    ap.add_argument("--n", type=int, default=0, help="网格 H（0=完整 benchmark 的 64/128 两档）")
    ap.add_argument("--tau", type=float, default=CANONICAL["tau"])
    ap.add_argument("--u0", type=float, default=CANONICAL["u0"])
    ap.add_argument("--steps", type=int, default=0, help="0 = auto (u_max/U0→e^-3, cap 10000)")
    ap.add_argument("--record-every", type=int, default=100)
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
        for n in (64, 128):
            for u0 in (0.05, 0.10):
                steps = args.steps or auto_steps(n, args.tau, u0, args.record_every)
                res = run_case(
                    n,
                    args.tau,
                    u0,
                    steps,
                    record_every=args.record_every,
                    dtype=args.dtype,
                    device=args.device,
                    threads=args.threads,
                    compile_mode=compile_mode,
                )
                cases.append(
                    {k: v for k, v in res.items() if k not in ("times", "energies", "umaxs")}
                )
                print(
                    f"H={n} U0={u0}: err_vel={res['err_vel_pct']:+.4f}%  "
                    f"err_E={res['err_e_pct']:+.4f}%  R2={res['r2_vel']:.6f}",
                    flush=True,
                )
        with open(os.path.join(args.out, "scan_summary.json"), "w") as fh:
            json.dump(cases, fh, indent=2, ensure_ascii=False)
        print(f"\nscan done -> {os.path.join(args.out, 'scan_summary.json')}")
        return

    # ── 完整 benchmark：H=64/128 两档网格（同一 τ/U0）────────────────────
    if args.n == 0:
        grids, steps_list = [64, 128], []
        results, case_files = [], []
        for n in grids:
            steps = args.steps or auto_steps(n, args.tau, args.u0, args.record_every)
            steps_list.append(steps)
            print(f"=== H={n}  tau={args.tau}  U0={args.u0}  steps={steps} ===", flush=True)
            res = run_case(
                n,
                args.tau,
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
                f"  gamma_vel_sim={res['gamma_vel_sim']:.6e} theory={res['gamma_vel_theory']:.6e} "
                f"err_vel={res['err_vel_pct']:+.4f}%  err_E={res['err_e_pct']:+.4f}%  "
                f"R2={res['r2_vel']:.6f}  wall={res['wall_sec']:.1f}s",
                flush=True,
            )
        verdict = judge(results)
        out = {
            "benchmark": "shear_wave_decay",
            "description": "2D decaying shear wave: u=U0·sin(2πy/H)·e^{-νk²t}, viscous dissipation check",
            "lattice": "D2Q9",
            "collision": "bgk",
            "boundary": "periodic (stream mod wrap, 库内建)",
            "extrap": "none",
            "common_modules": [
                "solver.collide_bgk",
                "solver.stream",
                "d2q9.equilibrium",
                "d2q9.macroscopic",
            ],
            "formula": "gamma_vel_theory = nu·k², nu=(tau-1/2)/3, k=2π/H; gamma_e_theory = 2·gamma_vel_theory",
            "cases": [
                {k: v for k, v in r.items() if k not in ("times", "energies", "umaxs")}
                for r in results
            ],
            "case_files": case_files,
            "convergence": verdict,
        }
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print("\n=== 判定 ===")
        print(
            "  err_vel: "
            + ", ".join(
                f"H{h}={e:+.4f}%"
                for h, e in zip(verdict["grids"], verdict["err_vel_pct_per_grid"].values())
            )
        )
        print(
            "  err_E:   "
            + ", ".join(
                f"H{h}={e:+.4f}%"
                for h, e in zip(verdict["grids"], verdict["err_e_pct_per_grid"].values())
            )
        )
        print(
            f"  converged_monotone={verdict['converged_monotone']}  within_tol={verdict['within_tol']}  r2_ok={verdict['exponential_r2_ok']}"
        )
        print(f"  {verdict['judgment']}")
        print(f"  -> {os.path.join(args.out, 'result.json')}")
        return

    # ── 单案例模式 ──────────────────────────────────────────────────────
    steps = args.steps or auto_steps(args.n, args.tau, args.u0, args.record_every)
    res = run_case(
        args.n,
        args.tau,
        args.u0,
        steps,
        record_every=args.record_every,
        dtype=args.dtype,
        device=args.device,
        threads=args.threads,
        compile_mode=compile_mode,
    )
    save_case(res, args.out)
    print(f"H={args.n} tau={args.tau} U0={args.u0} steps={steps} dtype={args.dtype}")
    print(f"  nu={res['nu']:.6f}  k={res['k']:.6f}")
    print(
        f"  gamma_vel_theory={res['gamma_vel_theory']:.6e}  gamma_vel_sim={res['gamma_vel_sim']:.6e}  err_vel={res['err_vel_pct']:+.4f}%"
    )
    print(
        f"  gamma_e_theory={res['gamma_e_theory']:.6e}  gamma_e_sim={res['gamma_e_sim']:.6e}  err_E={res['err_e_pct']:+.4f}%"
    )
    print(
        f"  R2_vel={res['r2_vel']:.6f}  mass_drift_rel={res['mass_drift_rel']:.2e}  wall={res['wall_sec']:.1f}s"
    )
    print(f"  -> {os.path.join(args.out, f'case_H{args.n}.json')}")


if __name__ == "__main__":
    main()

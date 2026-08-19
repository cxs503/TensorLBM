#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B17: 2D Taylor–Green 涡衰减 benchmark（周期 D2Q9 BGK，真实模拟，禁外推）。

解析解（不可压 Navier–Stokes 的精确解，TG 涡场满足 u·∇u ≡ 0）：
    u(x,y,t) = -U0·cos(kx)·sin(ky)·e^{-2νk²t}
    v(x,y,t) = +U0·sin(kx)·cos(ky)·e^{-2νk²t}
    ν = (τ − 1/2)/3,  k = 2π/L（晶格单位 L=N, Δx=Δt=1）

衰减率推导（关键：速度场与动能衰减率差 2 倍）：
    · 速度场每支 Fourier 模波矢为 (±k,±k)，|κ|² = 2k²，
      ∂u/∂t = ν∇²u ⇒ 速度衰减率 γ_vel = 2νk²（即任务/文献中 e^{-2νk²t}）。
    · 动能 E = 0.5·⟨u²+v²⟩ ∝ u² ⇒ 能量衰减率 γ_E = 2·γ_vel = 4νk²。
    · E(t) = (U0²/4)·e^{-4νk²t}，故拟合 ln(E) vs t 应与 4νk² 对比。

测量与判定：
    γ_E_sim = -d(ln E)/dt（线性拟合）对比 γ_E_theory = 4νk²；
    同时拟合 ln(|u|_max) 得 γ_vel_sim 对比 γ_vel_theory = 2νk²（交叉验证）。
    |γ_sim/γ_theory − 1| ≤ 1% → 达标（LBM 数值耗散可忽略）。
    偏差符号如实记录（BGK 离散耗散可为正或负）。

用法：
    python run.py                      # 标准案例（N=128, Re=100, U0=0.05）→ result.json
    python run.py --scan               # N×Re×U0 扫描 → scan_summary.json + scan_histories/
    python run.py --n 64 --re 100 --u0 0.05 [--steps N] [--dtype float32] [--out DIR]
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

from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = dict(n=128, re=100, u0=0.05, dtype="float32")


def auto_steps(n: int, re: int, u0: float, record_every: int, cap: int = 200000) -> int:
    """按能量目标衰减 E/E0 = e^-3 自动确定步数（保证拟合窗口覆盖明显衰减）。"""
    gamma_e = 16.0 * math.pi * math.pi * u0 / (re * n)  # 4νk² = 16π²U0/(Re·N)
    steps = max(1000, int(math.ceil(3.0 / gamma_e / record_every) * record_every))
    return min(steps, cap)


def run_case(
    n: int,
    re: int,
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
    nu = u0 * L / re          # 晶格单位运动粘度（Re = U0·L/ν）
    tau = 0.5 + 3.0 * nu      # BGK 弛豫时间
    gamma_e_theory = 4.0 * nu * k * k     # 能量衰减率 4νk²
    gamma_vel_theory = 2.0 * nu * k * k   # 速度衰减率 2νk²

    # ── 初始场：TG 涡（周期域 x,y ∈ [0,N)，Δx=1）────────────────────────
    y, x = torch.meshgrid(
        torch.arange(n, device=device, dtype=dt),
        torch.arange(n, device=device, dtype=dt),
        indexing="ij",
    )
    ux0 = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy0 = +u0 * torch.sin(k * x) * torch.cos(k * y)
    rho = torch.ones_like(ux0)
    f = equilibrium(rho, ux0, uy0)        # 初值 f = f_eq（标准做法）

    mass0 = float(f.sum().item())
    e0_theory = u0 * u0 / 4.0
    umax_theory = u0

    # ── 主循环：stream（周期 wrap 内建）→ collide（BGK）──────────────────
    # 整步步进函数经共性 compile 路径；步序号与采样监测留在编译域外。
    def _step(f):
        return collide_bgk(stream(f), tau)

    step_fn = route_step(_step, compile_mode, name=f"taylor_green_2d[N{n}]")

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

    # ── 拟合 ln(E) vs t：斜率 = −γ_E_sim；拟合 ln(|u|_max) vs t：斜率 = −γ_vel_sim ──
    t = np.asarray(times, dtype=np.float64)
    lnE = np.log(np.asarray(energies, dtype=np.float64))
    # polyfit 返回 [最高次系数, …] = [斜率, 截距]
    a_e, b_e = np.polyfit(t, lnE, 1)
    gamma_e_sim = -a_e
    resid = lnE - (b_e + a_e * t)
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((lnE - lnE.mean()) ** 2))

    lnU = np.log(np.asarray(umaxs, dtype=np.float64))
    a_u, _ = np.polyfit(t, lnU, 1)
    gamma_vel_sim = -a_u

    # 分段一致性检查（验证纯指数衰减，而非拟合假象）
    half = len(t) // 2
    g_h1 = -float(np.polyfit(t[:half], lnE[:half], 1)[0])
    g_h2 = -float(np.polyfit(t[half:], lnE[half:], 1)[0])

    err_e_pct = (gamma_e_sim - gamma_e_theory) / gamma_e_theory * 100.0
    err_vel_pct = (gamma_vel_sim - gamma_vel_theory) / gamma_vel_theory * 100.0
    mass_end = float(f.sum().item())

    return {
        "n": n, "re": re, "u0": u0, "nu": nu, "tau": tau, "k": k,
        "steps": steps, "record_every": record_every, "dtype": dtype, "device": device,
        "compile_mode": compile_mode,
        "gamma_e_theory": gamma_e_theory, "gamma_e_sim": gamma_e_sim,
        "err_e_pct": err_e_pct,
        "gamma_vel_theory": gamma_vel_theory, "gamma_vel_sim": gamma_vel_sim,
        "err_vel_pct": err_vel_pct,
        "r2": r2, "gamma_e_half1": g_h1, "gamma_e_half2": g_h2,
        "e0_theory": e0_theory, "e0_meas": energies[0],
        "umax_theory": umax_theory, "umax_meas": umaxs[0],
        "mass_drift_rel": (mass_end - mass0) / mass0,
        "n_samples": len(times),
        "wall_sec": wall,
        "times": times, "energies": energies, "umaxs": umaxs,
    }


def save_case(res: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    keep = {k: v for k, v in res.items() if k not in ("times", "energies", "umaxs")}
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(keep, fh, indent=2, ensure_ascii=False)
    hist = np.column_stack([res["times"], res["energies"], res["umaxs"]])
    np.savetxt(
        os.path.join(out_dir, "energy_history.csv"), hist,
        header="step,energy,umax", delimiter=",", comments="",
    )


def scan(record_every: int = 100, max_steps: int = 200000, threads: int = 32) -> list[dict]:
    cases = []
    for n in (64, 96, 128):
        for re in (100, 500, 1000):
            cases.append((n, re, 0.05, "float32"))
    cases += [
        (64, 100, 0.10, "float32"),      # Ma 敏感性（U0=0.1, Ma≈0.17）
        (128, 100, 0.10, "float32"),
        (128, 100, 0.01, "float32"),     # 线性区检查（U0→0）
        (128, 100, 0.05, "float64"),     # fp32 舍入误差检查
    ]
    results = []
    hist_dir = os.path.join(HERE, "scan_histories")
    os.makedirs(hist_dir, exist_ok=True)
    for (n, re, u0, dtype) in cases:
        steps = auto_steps(n, re, u0, record_every, cap=max_steps)
        tag = f"N{n}_Re{re}_U0{u0}_{dtype}"
        print(f"=== {tag}: steps={steps}, tau={0.5+3*u0*n/re:.6f} ===", flush=True)
        try:
            res = run_case(n, re, u0, steps, record_every=record_every, dtype=dtype, threads=threads)
            np.savetxt(
                os.path.join(hist_dir, f"history_{tag}.csv"),
                np.column_stack([res["times"], res["energies"], res["umaxs"]]),
                header="step,energy,umax", delimiter=",", comments="",
            )
            summary = {k: v for k, v in res.items() if k not in ("times", "energies", "umaxs")}
            results.append(summary)
            print(
                f"  gamma_E_sim={res['gamma_e_sim']:.6e} theory={res['gamma_e_theory']:.6e} "
                f"err_E={res['err_e_pct']:+.4f}%  err_vel={res['err_vel_pct']:+.4f}%  "
                f"R2={res['r2']:.6f}  wall={res['wall_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", flush=True)
            results.append({"case": tag, "error": str(exc)})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="B17 Taylor-Green vortex decay benchmark")
    ap.add_argument("--n", type=int, default=CANONICAL["n"])
    ap.add_argument("--re", type=float, default=CANONICAL["re"])
    ap.add_argument("--u0", type=float, default=CANONICAL["u0"])
    ap.add_argument("--steps", type=int, default=0, help="0 = auto (E/E0→e^-3)")
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--scan", action="store_true")
    add_compile_mode_arg(ap)
    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)

    if args.scan:
        results = scan(record_every=args.record_every, max_steps=args.max_steps, threads=args.threads)
        with open(os.path.join(args.out, "scan_summary.json"), "w") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\nscan done -> {os.path.join(args.out, 'scan_summary.json')}")
        return

    steps = args.steps if args.steps > 0 else auto_steps(args.n, args.re, args.u0, args.record_every, cap=args.max_steps)
    res = run_case(args.n, args.re, args.u0, steps,
                   record_every=args.record_every, device=args.device,
                   dtype=args.dtype, threads=args.threads, compile_mode=compile_mode)
    save_case(res, args.out)
    print(f"N={args.n} Re={args.re} U0={args.u0} steps={steps} dtype={args.dtype}")
    print(f"  nu={res['nu']:.6f}  tau={res['tau']:.6f}  k={res['k']:.6f}")
    print(f"  gamma_E_theory={res['gamma_e_theory']:.6e}  gamma_E_sim={res['gamma_e_sim']:.6e}  err_E={res['err_e_pct']:+.4f}%")
    print(f"  gamma_vel_theory={res['gamma_vel_theory']:.6e}  gamma_vel_sim={res['gamma_vel_sim']:.6e}  err_vel={res['err_vel_pct']:+.4f}%")
    print(f"  R2={res['r2']:.6f}  mass_drift_rel={res['mass_drift_rel']:.2e}  wall={res['wall_sec']:.1f}s")
    print(f"  -> {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()

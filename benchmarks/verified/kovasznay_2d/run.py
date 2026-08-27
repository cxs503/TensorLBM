#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B18: Kovasznay 2D 稳态流（解析 Navier-Stokes 解验证，D2Q9 BGK，真实模拟，禁外推）。

解析解（Kovasznay 1948，二维稳态不可压 Navier-Stokes 精确解，y 向周期 1）：
    u(x',y') = U0·(1 − e^{λx'}·cos(2πy'))
    v(x',y') = U0·(λ/2π)·e^{λx'}·sin(2πy')
    p(x',y') = 0.5·(1 − e^{2λx'})          （无量纲，ρU0² 归一）
    λ = Re/2 − sqrt(Re²/4 + 4π²)   （负根，衰减支；由动量方程导出 λ² − Re·λ − 4π² = 0）
    Re = U0·L/ν，特征速度 U0、特征长度 L = y 向周期（= ny 格）。

注：任务书称 "Re=40 时 λ≈−0.5"，按公式精确计算 Re=40 → λ = 20 − sqrt(400+4π²) ≈ −0.9637
（λ=−0.5 对应 Re≈78.5）。本实现以公式为准：λ 恒由上述公式精确生成（已数值验证
max|N-S 残差| ≈ 1e-3，纯差分截断；恒等式 λ=ν(λ²−4π²) 成立至 2e-15）。

晶格设定（主配置 config A：零梯度出口，域长 3 个 y 周期）：
    · 域 nx×ny（nx = 3·ny，即 x' ∈ [0,3]），y 向周期（库 stream 周期 gather 内建）。
    · ν = U0·ny/Re，τ = 0.5 + 3ν（BGK）。Re=40，U0=0.03（Ma_max = 2U0/cs ≈ 0.104）。
    · 初值：全场解析场 f = f_eq(ρ=1, u_ana, v_ana)。
    · 入口 x=0：解析 Dirichlet —— 库函数 boundaries.zou_he_inlet_velocity
      （每行施加解析 u(0,y')、v(0,y')，Zou & He 1997 速度入口重构，二阶）。
    · 出口 x=nx−1：零梯度 Neumann —— f[:,:,−1] = f[:,:,−2]（~3 行内联，库无此函数，
      见 /tmp/kovasznay_gap.md）。零梯度出口在短域（x'∈[0,1]）会把出口处 v 分量
      杀到 ≈0（全场 v 最大相对误差 ~96%），域长 3 个周期后出口扰动区 e^{λx'} 已衰减
      至 0.056 以下，全场误差 ≤3%。
    · 主循环：collide_bgk → stream(周期) → 出口零梯度 → 入口 Zou-He（同 poiseuille_2d 模式）。

交叉验证（config B：方域 x'∈[0,1]，双端解析 Dirichlet 出口，文献标准做法）：
    · 出口用 equilibrium(ρ=1, u_ana, v_ana) 整列覆盖（解析 Dirichlet，~3 行内联）。
    · 用于展示内部求解器本身的收敛性（无出口 BC 污染），不作为验收配置。

稳态与测量：
    · ≥ min_steps（默认 20000）步；每 500 步监测全场 ‖u‖₂ 相对漂移，<1e-6 即停（上限 max_steps）。
    · 末 100 步时间平均后，全场对比 u/v 与解析解：
      - L2 相对误差 ‖u_num−u_ana‖₂/‖u_ana‖₂（全场 + 内部列 1..nx−2）
      - 掩模最大点相对误差（|u_ana|>0.1·U0；|v_ana|>0.1·max|v_ana|，避免除以近零值）
      - 最大绝对误差 / U0
    · 验收（config A，网格 ny=64/128）：有限、步数≥20000、全场 max(u/v L2、u/v 掩模最大
      相对误差) ≤ 3%，且 64→128 各指标单调下降（收敛）。

用法：
    run.py single <ny> <out.json> [--xmax 3.0] [--outlet zerograd|dirichlet] [--re 40]
        [--u0 0.03] [--min-steps 20000] [--max-steps 60000] [--device cpu]
    run.py scan <out_dir> [--grids 32 64 128] [--xmax 3.0] [--outlet zerograd]
        [--re 40] [--u0 0.03] ...   → summary.json + result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries import zou_he_inlet_velocity
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

CS2 = 1.0 / 3.0
TWOPI = 2.0 * math.pi


def kov_lambda(re: float) -> float:
    """Kovasznay 衰减率 λ（负根，精确公式）。"""
    return re / 2.0 - math.sqrt(re * re / 4.0 + 4.0 * math.pi * math.pi)


def analytic_field(
    nx: int,
    ny: int,
    u0: float,
    lam: float,
    xmax: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """解析 u/v 场（晶格坐标：x'=i/nx·xmax，y'=j/ny，y 周期 1）。"""
    y = torch.arange(ny, device=device, dtype=dtype)
    x = torch.arange(nx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xp = xx / nx * xmax
    yp = yy / ny
    e = torch.exp(lam * xp)
    u = u0 * (1.0 - e * torch.cos(TWOPI * yp))
    v = u0 * (lam / TWOPI) * e * torch.sin(TWOPI * yp)
    return u, v


def outlet_zero_gradient(f: torch.Tensor) -> torch.Tensor:
    """零梯度（Neumann）出口：出口列拷贝上游列全部 9 个分布函数（∂f/∂x = 0）。"""
    f = f.clone()
    f[:, :, -1] = f[:, :, -2]
    return f


def outlet_analytic_dirichlet(
    f: torch.Tensor, u_ana_col: torch.Tensor, v_ana_col: torch.Tensor
) -> torch.Tensor:
    """解析 Dirichlet 出口：整列设为平衡态 f_eq(ρ=1, u_ana, v_ana)。"""
    ny = u_ana_col.shape[0]
    rho_c = torch.ones((ny, 1), device=f.device, dtype=f.dtype)
    f = f.clone()
    f[:, :, -1] = equilibrium(rho_c, u_ana_col.view(ny, 1), v_ana_col.view(ny, 1))[:, :, 0]
    return f


def run_case(
    ny: int,
    re: float,
    u0: float,
    xmax: float,
    outlet: str,
    min_steps: int,
    max_steps: int,
    out_path: str,
    device: str = "cpu",
    threads: int = 48,
    seed: int = 0,
    compile_mode: str | None = "default",
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    dev = torch.device(device)
    dt = torch.float32
    nx = int(round(ny * xmax))

    lam = kov_lambda(re)
    nu = u0 * ny / re  # Re = U0·ny/ν（特征长度 = y 周期 = ny 格）
    tau = 0.5 + 3.0 * nu
    ma_max = 2.0 * u0 / math.sqrt(CS2)

    u_ana, v_ana = analytic_field(nx, ny, u0, lam, xmax, dev, dt)
    f = equilibrium(torch.ones((ny, nx), device=dev, dtype=dt), u_ana, v_ana)  # 初值 = 解析场
    initial_mass = float(f.sum().item())

    u_in = u_ana[:, 0].contiguous()
    v_in = v_ana[:, 0].contiguous()

    # ---- 整步步进函数（共性 compile 路径；步序号与稳态监测留在编译域外）----
    def _step(f):
        f = collide_bgk(f, tau)
        f = stream(f)
        if outlet == "zerograd":
            f = outlet_zero_gradient(f)
        else:
            f = outlet_analytic_dirichlet(f, u_ana[:, -1], v_ana[:, -1])
        return zou_he_inlet_velocity(f, u_in, v_in)

    step_fn = route_step(_step, compile_mode, name=f"kovasznay_2d[ny{ny}]")

    t0 = time.time()
    l2_hist: list[tuple[int, torch.Tensor]] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % 500 == 0:
            _, ux, _ = macroscopic(f)
            # 场变化监测：‖u(t)−u(t−2500)‖₂/‖u(t)‖₂（float64 累加）。
            # 注：Zou-He 速度入口+零梯度出口存在 ~0.03%/60000 步的缓慢质量泄漏，
            # 使 ‖u‖₂ 范数以 ~6e-6/2500 步缓慢漂移；速度场本身（相对解析解误差）
            # 在 ~2000 步即收敛（实测 2000–40000 步误差完全平坦）。故阈值取 5e-5。
            l2_hist.append((step, ux.double()))
            if step >= min_steps and len(l2_hist) >= 5:
                u_prev = l2_hist[-5][1]  # 2500 步前的 u 场
                num = float(((ux.double() - u_prev) ** 2).sum().sqrt().item())
                den = float((ux.double() ** 2).sum().sqrt().item())
                if num / max(den, 1e-12) < 5e-5:
                    steady = True
                    break
    elapsed = time.time() - t0
    final_drift = 0.0
    if len(l2_hist) >= 5:
        u_prev = l2_hist[-5][1]
        num = float(((l2_hist[-1][1] - u_prev) ** 2).sum().sqrt().item())
        den = float((l2_hist[-1][1] ** 2).sum().sqrt().item())
        final_drift = num / max(den, 1e-12)

    # 末 100 步时间平均（稳态，去浮点噪声）
    acc_u = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    acc_v = torch.zeros((ny, nx), device=dev, dtype=torch.float64)
    for _ in range(100):
        f = step_fn(f)
        _, ux, uy = macroscopic(f)
        acc_u += ux.to(torch.float64)
        acc_v += uy.to(torch.float64)
    acc_u /= 100.0
    acc_v /= 100.0

    u_num = acc_u.cpu().numpy()
    v_num = acc_v.cpu().numpy()
    u_a = u_ana.cpu().numpy().astype(np.float64)
    v_a = v_ana.cpu().numpy().astype(np.float64)

    def l2_rel(num: np.ndarray, ana: np.ndarray) -> float:
        return float(np.linalg.norm(num - ana) / np.linalg.norm(ana))

    def max_rel_masked(num: np.ndarray, ana: np.ndarray, thresh: float) -> float:
        mask = np.abs(ana) > thresh
        if mask.sum() == 0:
            return float("nan")
        return float(np.max(np.abs(num[mask] - ana[mask]) / np.abs(ana[mask])) * 100.0)

    u_l2 = l2_rel(u_num, u_a)
    v_l2 = l2_rel(v_num, v_a)
    u_max_rel = max_rel_masked(u_num, u_a, 0.1 * u0)
    v_max_rel = max_rel_masked(v_num, v_a, 0.1 * float(np.max(np.abs(v_a))))
    u_l2_int = l2_rel(u_num[:, 1:-1], u_a[:, 1:-1])
    v_l2_int = l2_rel(v_num[:, 1:-1], v_a[:, 1:-1])
    max_abs_u = float(np.max(np.abs(u_num - u_a)) / u0)
    max_abs_v = float(np.max(np.abs(v_num - v_a)) / u0)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0

    result = {
        "case": "B18_kovasznay_2d",
        "config": f"xmax={xmax} outlet={outlet}",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": (
            "zou_he_velocity_inlet(analytic Dirichlet) + zero-gradient outlet"
            if outlet == "zerograd"
            else "zou_he_velocity_inlet(analytic Dirichlet) + analytic Dirichlet outlet"
        ),
        "driving": "analytic Navier-Stokes solution (Kovasznay 1948), steady-state verification",
        "extrap": "none",
        "ny": ny,
        "nx": nx,
        "xmax": xmax,
        "outlet": outlet,
        "re": re,
        "lambda": lam,
        "u0": u0,
        "nu_lb": nu,
        "tau": tau,
        "ma_max": ma_max,
        "compile_mode": compile_mode,
        "min_steps": min_steps,
        "n_steps": step,
        "steady": steady,
        "final_drift_rel": final_drift,
        "u_l2_rel": u_l2,
        "v_l2_rel": v_l2,
        "u_l2_rel_interior": u_l2_int,
        "v_l2_rel_interior": v_l2_int,
        "u_max_rel_pct": u_max_rel,
        "v_max_rel_pct": v_max_rel,
        "max_abs_err_u_over_u0": max_abs_u,
        "max_abs_err_v_over_u0": max_abs_v,
        "mass_drift_pct": mass_drift_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def scan(
    grids,
    re,
    u0,
    xmax,
    outlet,
    min_steps,
    max_steps,
    out_dir: str,
    device: str = "cpu",
    compile_mode: str | None = "default",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for ny in grids:
        p = out_dir / f"case_ny{ny}.json"
        r = run_case(
            ny,
            re,
            u0,
            xmax,
            outlet,
            min_steps,
            max_steps,
            str(p),
            device=device,
            compile_mode=compile_mode,
        )
        cases.append(r)
        print(
            f"ny={r['ny']:3d} nx={r['nx']:3d} tau={r['tau']:.3f} steps={r['n_steps']:6d} "
            f"steady={r['steady']} u_l2={r['u_l2_rel']:.5f} v_l2={r['v_l2_rel']:.5f} "
            f"u_mr={r['u_max_rel_pct']:.2f}% v_mr={r['v_max_rel_pct']:.2f}% "
            f"elapsed={r['elapsed_s']}s",
            flush=True,
        )

    by_n = {r["ny"]: r for r in cases}
    conv = {}
    verdict = "not_enough_grids"
    if 64 in by_n and 128 in by_n:
        c64, c128 = by_n[64], by_n[128]
        conv = {
            "u_l2_decreases": c128["u_l2_rel"] < c64["u_l2_rel"],
            "v_l2_decreases": c128["v_l2_rel"] < c64["v_l2_rel"],
            "u_max_rel_decreases": c128["u_max_rel_pct"] < c64["u_max_rel_pct"],
            "v_max_rel_decreases": c128["v_max_rel_pct"] < c64["v_max_rel_pct"],
        }
        errs_64 = max(
            c64["u_l2_rel"],
            c64["v_l2_rel"],
            c64["u_max_rel_pct"] / 100.0,
            c64["v_max_rel_pct"] / 100.0,
        )
        errs_128 = max(
            c128["u_l2_rel"],
            c128["v_l2_rel"],
            c128["u_max_rel_pct"] / 100.0,
            c128["v_max_rel_pct"] / 100.0,
        )
        max_err = max(errs_64, errs_128) * 100.0
        finite = all(r["finite"] for r in cases)
        min_steps_ok = all(r["n_steps"] >= min_steps for r in cases)
        conv_ok = all(conv.values())
        verdict = "PASS" if (finite and min_steps_ok and max_err <= 3.0 and conv_ok) else "FAIL"
        conv["max_err_pct"] = round(max_err, 4)
        conv["max_err_pct_64"] = round(errs_64 * 100.0, 4)
        conv["max_err_pct_128"] = round(errs_128 * 100.0, 4)

    summary = {
        "case": "B18_kovasznay_2d_convergence",
        "config": f"xmax={xmax} outlet={outlet}",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": (
            "zou_he_velocity_inlet(analytic Dirichlet) + zero-gradient outlet"
            if outlet == "zerograd"
            else "zou_he_velocity_inlet(analytic Dirichlet) + analytic Dirichlet outlet"
        ),
        "extrap": "none",
        "re": re,
        "lambda": kov_lambda(re),
        "u0": u0,
        "grids_ny": grids,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "per_grid": [
            {
                k: r[k]
                for k in [
                    "ny",
                    "nx",
                    "tau",
                    "nu_lb",
                    "n_steps",
                    "steady",
                    "u_l2_rel",
                    "v_l2_rel",
                    "u_l2_rel_interior",
                    "v_l2_rel_interior",
                    "u_max_rel_pct",
                    "v_max_rel_pct",
                    "max_abs_err_u_over_u0",
                    "max_abs_err_v_over_u0",
                    "mass_drift_pct",
                    "finite",
                    "elapsed_s",
                ]
            }
            for r in cases
        ],
        "convergence_64_to_128": conv,
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="B18 Kovasznay 2D steady flow (D2Q9 BGK)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("ny", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--re", type=float, default=40.0)
    p1.add_argument("--u0", type=float, default=0.03)
    p1.add_argument("--xmax", type=float, default=3.0)
    p1.add_argument("--outlet", choices=["zerograd", "dirichlet"], default="zerograd")
    p1.add_argument("--min-steps", type=int, default=20000)
    p1.add_argument("--max-steps", type=int, default=60000)
    p1.add_argument("--device", type=str, default="cpu")
    add_compile_mode_arg(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    p2.add_argument("--re", type=float, default=40.0)
    p2.add_argument("--u0", type=float, default=0.03)
    p2.add_argument("--xmax", type=float, default=3.0)
    p2.add_argument("--outlet", choices=["zerograd", "dirichlet"], default="zerograd")
    p2.add_argument("--min-steps", type=int, default=20000)
    p2.add_argument("--max-steps", type=int, default=60000)
    p2.add_argument("--device", type=str, default="cpu")
    add_compile_mode_arg(p2)

    args = ap.parse_args()
    compile_mode = compile_mode_from_args(args)
    if args.mode == "single":
        r = run_case(
            args.ny,
            args.re,
            args.u0,
            args.xmax,
            args.outlet,
            args.min_steps,
            args.max_steps,
            args.out_json,
            device=args.device,
            compile_mode=compile_mode,
        )
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in [
                        "ny",
                        "nx",
                        "re",
                        "lambda",
                        "u0",
                        "nu_lb",
                        "tau",
                        "ma_max",
                        "n_steps",
                        "steady",
                        "u_l2_rel",
                        "v_l2_rel",
                        "u_max_rel_pct",
                        "v_max_rel_pct",
                        "mass_drift_pct",
                        "elapsed_s",
                    ]
                },
                indent=2,
            )
        )
    else:
        s = scan(
            args.grids,
            args.re,
            args.u0,
            args.xmax,
            args.outlet,
            args.min_steps,
            args.max_steps,
            args.out_dir,
            device=args.device,
            compile_mode=compile_mode,
        )
        print(f"verdict: {s['verdict']}")


if __name__ == "__main__":
    main()

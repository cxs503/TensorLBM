#!/usr/bin/env python3
"""CASE-LBM-KOVASZNAY-EXACT — 独立复核与误差预算（不复用 tensorlbm 求解器代码）。

目的（锚点 `CASE-LBM-KOVASZNAY-EXACT`，容差 <3% @ Re=40, N>=128）：

1. **独立复现**：本文件自带一份 D2Q9 BGK 实现（y 向周期 roll、x 向 Zou-He 解析速度入口、
   出口零梯度 Neumann），**不 import tensorlbm 的任何求解器函数**，按算例 README 记载的
   算法（U0=0.03、nx=3·ny、x'∈[0,3]）重跑 ny=32/64/128，与
   `benchmarks/verified/kovasznay_2d/result.json` 的既有数字对账（`reproduce` 节）。
2. **误差预算**：既有 PASS 的残余误差（u 最大绝对误差 ~1.1%·U0，随加密**不降**）被
   归因于 BGK 的 O(Ma²) 可压缩性底噪（U0=0.03 ⇒ Ma_max≈0.104 ⇒ Ma²≈1.08%）。
   本文件用 **同一 ny=64、只变 U0（0.03 / 0.015 / 0.0075）** 的臂做可证伪检验：
   若误差 ∝ Ma²，则 U0 减半应使 max|Δu|/U0 降到 ~1/4；并据此把误差外推到 Ma→0，
   给出**去掉可压缩性底噪后的离散误差**（这才是"网格收敛阶"该看的量）。
3. **参考解自检**：Kovasznay 解析解是否真的满足不可压 N-S（离散残差 = 差分量级），
   以及 λ² − Re·λ − 4π² = 0 恒等式；同时量化"任务书写 λ(Re=40)≈−0.5"的口径错误。

输出：`artifacts/independent_audit.json`（全部数字程序化写入，零手抄）。

用法：
    PYTHONPATH=src python3 benchmarks/verified/kovasznay_2d/independent_audit.py \
        --out artifacts/independent_audit.json [--grids 32 64 128] [--quick]

    --quick   仅 ny=32、4000 步、跳过 Ma 臂（pytest 用，~15 s）
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

DT = torch.float32  # 与算例 run.py 一致；度量前统一升到 float64

# D2Q9（标准序：0 静止，1..4 轴向，5..8 对角）
EX = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.long)
EY = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.long)
W = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=DT)
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------- 解析解
def kovasznay_lambda(re: float) -> float:
    """λ 的负根：λ² − Re·λ − 4π² = 0（由 x 向动量方程导出）。"""
    return 0.5 * re - math.sqrt(0.25 * re * re + 4.0 * math.pi ** 2)


def analytic_profiles(ny: int, nx: int, re: float, u0: float) -> dict:
    """Kovasznay 1948 二维定常不可压 N-S 精确解（L = ny = y 向周期，x' = x/L）。"""
    lam = kovasznay_lambda(re)
    # 坐标约定与算例 run.py 一致（节点对齐：x'=i/nx·xmax，y'=j/ny），保证可逐项对账
    y = torch.arange(ny, dtype=DT) / ny
    x = torch.arange(nx, dtype=DT) / nx * (nx / ny)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    e = torch.exp(lam * xx)
    u = u0 * (1.0 - e * torch.cos(2 * math.pi * yy))
    v = u0 * (lam / (2 * math.pi)) * e * torch.sin(2 * math.pi * yy)
    p = u0 ** 2 * 0.5 * (1.0 - torch.exp(2.0 * lam * xx))
    return {"u": u, "v": v, "p": p, "lam": lam, "xp": xx, "yp": yy}


def ns_residual(u: torch.Tensor, v: torch.Tensor, p: torch.Tensor, nu: float) -> dict:
    """不可压 N-S 残差（周期差分 y、内部列中心差分 x），按 U0²/L 无量纲化在调用侧做。"""

    def dx(a):
        return (torch.roll(a, -1, dims=1) - torch.roll(a, 1, dims=1)) * 0.5

    def dy(a):
        return (torch.roll(a, -1, dims=0) - torch.roll(a, 1, dims=0)) * 0.5

    def dxx(a):
        return torch.roll(a, -1, dims=1) - 2 * a + torch.roll(a, 1, dims=1)

    def dyy(a):
        return torch.roll(a, -1, dims=0) - 2 * a + torch.roll(a, 1, dims=0)

    rx = u * dx(u) + v * dy(u) + dx(p) - nu * (dxx(u) + dyy(u))
    ry = u * dx(v) + v * dy(v) + dy(p) - nu * (dxx(v) + dyy(v))
    div = dx(u) + dy(v)
    return {"rx": rx[:, 1:-1], "ry": ry[:, 1:-1], "div": div[:, 1:-1]}


# ---------------------------------------------------------------- 独立 LBM
def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    eu = (EX.view(9, 1, 1) * ux + EY.view(9, 1, 1) * uy).to(DT)
    usq = ux * ux + uy * uy
    return W.view(9, 1, 1) * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)


def stream(f: torch.Tensor) -> torch.Tensor:
    """f_post 沿 e_i 迁移；两个方向都用 roll（y 周期正确；x 的环绕列随后被边界覆盖）。"""
    src = torch.empty_like(f)
    for i in range(9):
        src[i] = torch.roll(torch.roll(f[i], int(EY[i]), dims=0), int(EX[i]), dims=1)
    return src


def zou_he_inlet(f: torch.Tensor, ux_row: torch.Tensor, uy_row: torch.Tensor) -> None:
    """Zou & He (1997) 速度入口（x=0 列，原地写入未知的 1/5/8）。"""
    f0, f1, f2, f3, f4 = f[0, :, 0], f[1, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0]
    f5, f6, f7, f8 = f[5, :, 0], f[6, :, 0], f[7, :, 0], f[8, :, 0]
    rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_row)
    f[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux_row
    # 对角未知量按「非平衡反弹」配对：f5(c=+1,+1) 与 f7(-1,-1)、f8(+1,-1) 与 f6(-1,+1)
    f[5, :, 0] = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux_row + 0.5 * rho * uy_row
    f[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux_row - 0.5 * rho * uy_row


def run(ny: int, re: float = 40.0, u0: float = 0.03, *, xmax: float = 3.0,
        min_steps: int = 20000, max_steps: int = 60000, check_every: int = 500,
        drift_tol: float = 1e-6, outlet: str = "zerograd") -> dict:
    nx = int(round(xmax * ny))
    lam = kovasznay_lambda(re)
    nu = u0 * ny / re
    tau = 0.5 + 3.0 * nu
    ana = analytic_profiles(ny, nx, re, u0)
    ua, va = ana["u"], ana["v"]

    # 初值 = 解析场平衡态
    rho0 = torch.ones((ny, nx), dtype=DT)
    f = equilibrium(rho0, ua, va)
    ux_in = ua[:, 0].clone()
    uy_in = va[:, 0].clone()
    ux_out = ua[:, -1].clone()
    uy_out = va[:, -1].clone()

    drift_hist: list[float] = []
    u_prev = None
    n = 0
    steady = False
    t0 = time.time()
    while n < max_steps:
        f = f - (f - equilibrium(*(macroscopic(f)))) / tau
        f = stream(f)
        if outlet == "zerograd":
            f[:, :, -1] = f[:, :, -2]      # 出口零梯度 Neumann
        else:                              # 解析 Dirichlet 出口（配对臂，隔离出口 BC 的贡献）
            f[:, :, -1] = equilibrium(torch.ones((ny, 1), dtype=DT),
                                      ux_out.view(ny, 1), uy_out.view(ny, 1))[:, :, -1]
        zou_he_inlet(f, ux_in, uy_in)      # 入口解析 Zou-He
        n += 1
        if n % check_every == 0:
            ux, uy = macroscopic(f)[1:]
            norm = torch.sqrt((ux * ux + uy * uy).sum())
            if u_prev is not None:
                drift = float((norm - u_prev) / u_prev)
                drift_hist.append(drift)
                if n >= min_steps and abs(drift) < drift_tol:
                    steady = True
                    break
            u_prev = norm

    rho, ux, uy = macroscopic(f)
    dt = time.time() - t0
    met = metrics(ux, uy, rho, ana, nu)
    met.update({"ny": ny, "nx": nx, "tau": tau, "nu_lb": nu, "lam": lam, "outlet": outlet,
                "n_steps": n, "steady": steady, "elapsed_s": round(dt, 1),
                "drift_last": drift_hist[-1] if drift_hist else None,
                "finite": bool(torch.isfinite(f).all())})
    return met


def macroscopic(f: torch.Tensor):
    rho = f.sum(0)
    ux = (EX.view(9, 1, 1) * f).sum(0) / rho
    uy = (EY.view(9, 1, 1) * f).sum(0) / rho
    return rho, ux, uy


def metrics(ux, uy, rho, ana, nu) -> dict:
    ux, uy, rho = ux.double(), uy.double(), rho.double()
    ua, va = ana["u"].double(), ana["v"].double()
    u0 = float(ua.max() - ua.min()) / 2.0  # 幅值 = U0
    du, dv = ux - ua, uy - va

    def l2(a):
        return float(torch.sqrt((a * a).sum()))

    mask_u = ua.abs() > 0.1 * u0
    vmax = float(va.abs().max())
    mask_v = va.abs() > 0.1 * vmax
    interior = slice(1, ux.shape[1] - 1)

    out = {
        "u_l2_rel": l2(du) / l2(ua),
        "v_l2_rel": l2(dv) / l2(va),
        "u_l2_rel_interior": l2(du[:, interior]) / l2(ua[:, interior]),
        "v_l2_rel_interior": l2(dv[:, interior]) / l2(va[:, interior]),
        "u_max_rel_pct": 100.0 * float((du[mask_u].abs() / ua[mask_u].abs()).max()),
        "v_max_rel_pct": 100.0 * float((dv[mask_v].abs() / va[mask_v].abs()).max()),
        # 绝对误差（按 U0 归一）——随加密应当收敛的量
        "max_abs_err_u_over_u0": float(du.abs().max()) / u0,
        "max_abs_err_v_over_u0": float(dv.abs().max()) / u0,
        "l2_abs_err_u_over_u0": l2(du) / math.sqrt(du.numel()) / u0,
        "ma_max": 2.0 * u0 / math.sqrt(CS2),
        "ma2": (2.0 * u0 / math.sqrt(CS2)) ** 2,
        "mass_drift_pct": 100.0 * (float(rho.mean()) - 1.0),
    }
    # 压力诊断：解析 δp = U0²·0.5(1−e^{2λx'})，LBM p = (ρ−ρ_ref)/3，ρ_ref 取入口列均值
    p_ana = ana["p"].double()
    dp_num = (rho - rho[:, 0].mean()) / 3.0
    dp_ana = p_ana - p_ana[:, 0].mean()
    out["p_l2_rel_vs_analytic"] = l2(dp_num - dp_ana) / l2(dp_ana)
    out["p_amp_over_u0sq"] = float(dp_ana.abs().max()) / (u0 * u0)
    # 无参考量检查：数值场与解析场在**同一离散算子**下的 N-S 残差之比
    r_num = ns_residual(ux, uy, dp_num, nu)
    r_ana = ns_residual(ua, va, dp_ana, nu)
    scale = u0 * u0 / (ux.shape[0] / 1.0)  # U0²/L，L=ny 格
    out["ns_res_max_u"] = float(r_num["rx"].abs().max()) / scale
    out["ns_res_max_u_analytic_floor"] = float(r_ana["rx"].abs().max()) / scale
    out["ns_res_max_v"] = float(r_num["ry"].abs().max()) / scale
    out["ns_res_max_v_analytic_floor"] = float(r_ana["ry"].abs().max()) / scale
    out["div_max"] = float(r_num["div"].abs().max()) * (ux.shape[0] / 1.0) / u0
    # 逐列误差剖面（定位误差到底在哪儿：出口区 vs 内部）
    prof_u = (du.abs().max(dim=0).values / u0)
    prof_v = (dv.abs().max(dim=0).values / u0)
    nx = ux.shape[1]
    tail = slice(max(1, int(round(0.9 * nx))), nx)
    body = slice(1, max(2, int(round(0.9 * nx))))
    out["err_profile_u_over_u0"] = [round(float(x), 6) for x in prof_u]
    out["err_profile_v_over_u0"] = [round(float(x), 6) for x in prof_v]
    out["u_err_interior90_over_u0"] = float(prof_u[body].max())
    out["u_err_last10pct_over_u0"] = float(prof_u[tail].max())
    out["v_err_interior90_over_u0"] = float(prof_v[body].max())
    out["v_err_last10pct_over_u0"] = float(prof_v[tail].max())
    return out


# ---------------------------------------------------------------- 复现对账 / 误差预算
def cross_check(**kw) -> dict:
    """解析解自检：λ 恒等式 + N-S 残差（应只剩差分量级）+ 任务书 λ 口径。"""
    re = kw.get("re", 40.0)
    lam = kovasznay_lambda(re)
    identity = lam * lam - re * lam - 4 * math.pi ** 2
    ny, u0 = 128, 0.03
    ana = analytic_profiles(ny, 3 * ny, re, u0)
    nu = u0 * ny / re
    r = ns_residual(ana["u"], ana["v"], ana["p"], nu)
    scale = u0 * u0 / ny
    lam_task = -0.5
    return {
        "re": re, "lambda": lam, "lambda_identity_residual": identity,
        "lambda_for_task_value": 0.5 * (re + math.sqrt(re * re + 16 * math.pi ** 2)),
        "task_lambda": lam_task,
        "task_lambda_implicit_re": -(4 * math.pi ** 2 + lam_task * lam_task) / lam_task,
        "ns_res_max_analytic_u": float(r["rx"].abs().max()) / scale,
        "ns_res_max_analytic_v": float(r["ry"].abs().max()) / scale,
        "div_max_analytic": float(r["div"].abs().max()) * ny / u0,
    }


def reproduce(reported: dict, got: dict) -> dict:
    """与 result.json 既有 per-grid 数字逐项对账（相对偏差）。"""
    rows = []
    for got_g in got:
        match = [r for r in reported["primary_config"]["per_grid"] if r["ny"] == got_g["ny"]]
        if not match:
            continue
        ref = match[0]
        row = {"ny": got_g["ny"]}
        for key in ("u_l2_rel", "v_l2_rel", "u_max_rel_pct", "v_max_rel_pct"):
            a, b = got_g[key], ref[key]
            row[key] = {"independent": a, "reported": b,
                        "rel_diff_pct": 100.0 * (a - b) / b if b else None}
        rows.append(row)
    return {"rows": rows,
            "max_abs_rel_diff_pct": max(abs(r[k]["rel_diff_pct"])
                                        for r in rows
                                        for k in ("u_l2_rel", "v_l2_rel")
                                        if r[k]["rel_diff_pct"] is not None)}


def ma_budget(ny: int = 64, u0s=(0.03, 0.015, 0.0075), steps=None, **kw) -> dict:
    """固定 ny，只变 U0：检验 max|Δu|/U0 ∝ Ma² 并外推 Ma→0。"""
    steps = steps or [int(round(6000 * (0.03 / u))) for u in u0s]
    arms = []
    for u0, ns in zip(u0s, steps):
        m = run(ny, kw.get("re", 40.0), u0, min_steps=min(6000, ns), max_steps=ns,
                xmax=kw.get("xmax", 3.0))
        arms.append({"u0": u0, "ma2": m["ma2"], "n_steps": m["n_steps"],
                     "max_abs_err_u_over_u0": m["max_abs_err_u_over_u0"],
                     "max_abs_err_v_over_u0": m["max_abs_err_v_over_u0"],
                     "u_l2_rel": m["u_l2_rel"], "v_l2_rel": m["v_l2_rel"],
                     "elapsed_s": m["elapsed_s"]})
    arms = sorted(arms, key=lambda a: a["u0"])
    # 线性回归 err = a + b·Ma²；a = Ma→0 的离散误差（含两侧 BC 影响）
    xs = [a["ma2"] for a in arms]
    ys = [a["max_abs_err_u_over_u0"] for a in arms]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    a0 = my - b * mx
    return {"ny": ny, "arms": arms, "fit": {"intercept": a0, "slope": b,
                                            "predicted_err_at_u0_0.03": a0 + b * xs[0]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/independent_audit.json")
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--u0", type=float, default=0.03)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--steps", type=int, nargs="+", default=None,
                    help="逐网格步数上限（与 --grids 等长；accepted 目标：不污染稳态）")
    ap.add_argument("--append", action="store_true", help="把结果并入既有 --out 文件")
    ap.add_argument("--outlet", default="zerograd", choices=["zerograd", "dirichlet"])
    ap.add_argument("--outlet-b", default=None, choices=["zerograd", "dirichlet"],
                    help="配对臂出口（隔离出口 BC 对误差的贡献）")
    ap.add_argument("--ma-ny", type=int, default=48)
    ap.add_argument("--ma-steps", type=int, nargs="+", default=[6000, 12000, 24000])
    ap.add_argument("--ma-u0s", type=float, nargs="+", default=[0.03, 0.015, 0.0075])
    ap.add_argument("--skip-grids", action="store_true")
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--quick", action="store_true", help="ny=32、4000 步、跳过 Ma 臂")
    ap.add_argument("--skip-ma", action="store_true")
    here = Path(__file__).resolve().parent
    args = ap.parse_args(argv)

    if args.quick:
        args.grids, args.min_steps, args.max_steps = [32], 4000, 8000
        args.skip_ma = True

    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))

    t0 = time.time()
    steps = args.steps or [args.min_steps] * len(args.grids)
    assert len(steps) == len(args.grids), "--steps 与 --grids 必须等长"
    grids = []
    if not args.skip_grids:
        for ny, ns in zip(args.grids, steps):
            grids.append(run(ny, args.re, args.u0, min_steps=min(args.min_steps, ns),
                             max_steps=ns, outlet=args.outlet))
    out_path = Path(args.out)
    out = {}
    if args.append and out_path.exists():
        out = json.loads(out_path.read_text())
    out.setdefault("case", "CASE-LBM-KOVASZNAY-EXACT")
    out.setdefault("kind", "independent_audit")
    out.update({"re": args.re, "u0": args.u0,
                "implementation": "self-contained D2Q9 BGK (no tensorlbm imports)"})
    out["self_check"] = cross_check(re=args.re)
    if grids:
        key = "grids" if args.outlet == "zerograd" else "grids_dirichlet"
        prev = {g["ny"]: g for g in out.get(key, [])}
        for g in grids:
            prev[g["ny"]] = g
        out[key] = [prev[k] for k in sorted(prev)]

    res_path = here / "result.json"
    if res_path.exists() and not args.quick and out.get("grids"):
        out["reproduce"] = reproduce(json.loads(res_path.read_text()), out["grids"])

    if not args.skip_ma:
        prev_arms = {a["u0"]: a for a in out.get("ma_budget", {}).get("arms", [])}
        mb = ma_budget(ny=args.ma_ny, u0s=tuple(args.ma_u0s),
                       steps=args.ma_steps, re=args.re)
        for a in mb["arms"]:
            prev_arms[a["u0"]] = a
        mb["arms"] = [prev_arms[k] for k in sorted(prev_arms)]
        out["ma_budget"] = mb

    out["wall_seconds"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""方柱 Re=100 自由流涡脱落 — Okajima 1982 验证（播种+sponge 工程链复用）。

基于 benchmarks/verified/cylinder/re100/run.py（Braza 验证）的完整工程链
直接改造：圆形掩码 → 方块掩码（half-way BB 精确边长 D），其余环节
（播种 + sponge + MRT + far_field + Ladd MEM 测力 + FFT 测 St）全部复用。

工程链（库 solver + 库 BC，零手写 collide/stream/equilibrium）：
- tensorlbm.solver.collide_mrt（tau_field 逐格松弛 = sponge）+ stream
- tensorlbm.boundaries.far_field_bc_2d（入口/两侧自由流 Dirichlet + 出口
  零梯度 + obstacle bounce-back，一次调用）
- tensorlbm.boundaries.make_sponge_strength（下游 10D 渐变吸收层）
- tensorlbm.boundaries.compute_obstacle_forces（Ladd 动量交换 Cd/Cl，
  post-stream、pre-bounce-back 采样）
- tensorlbm.d2q9.equilibrium

物理设置：
- 自由流方柱（正置，尖锐前缘分离），Re = U·D/ν = 100
- 域 40D×40D（blockage 2.5%），方块中心距入口 10D
- 下游 10D sponge：τ_eff = τ·(1 + α·σ)，α=10，σ 二次渐变
- 入口播种：2 个 St_seed=0.14 周期、10% 振幅正弦横向扰动后回归自由流
- ν = U·D/Re，τ = 3ν + 0.5

参考解（任务口径，Okajima 1982）：
- St ≈ 0.14（方柱 Re=100 文献范围 0.13-0.15）
- Cd ≈ 1.6（自由流方柱 Re=100 文献范围 ~1.5-1.7；Sohankar 1997 的 2.05
  是通道配置参考，不适用自由流）

compile 路径（tensorlbm.compile_utils，lesson 2 双变体）：
- 播种期（前 2 个 St_seed 周期）与自由流期是两种步进模式 → 编译两个
  变体（_step_seeded / _step_plain），eager 驱动循环按步序号选择
- Cd/Cl 为编译函数的 per-step 张量输出，留在 GPU 上累积，结束后一次性
  .cpu()——逐步 .item() 同步留在编译域外

用法：
    run.py --D 32 48 --device cuda:0 [--compile-mode default|eager] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks (compile_route)

import numpy as np
import torch
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries import (
    compute_obstacle_forces,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium
from tensorlbm.solver import collide_mrt, stream

REF_CD = 1.6  # Okajima 1982（任务口径，自由流方柱 Re=100 文献 1.5-1.7）
REF_ST = 0.14  # Okajima 1982（方柱 Re=100 St 文献 0.13-0.15）

# 域/播种参数（与 cylinder_re100 完全同口径，仅 St_seed 换方柱值）
DOMAIN_D = 40.0  # 域边长 = 40D
SQUARE_X_D = 10.0  # 方块中心距入口 10D
SPONGE_D = 10.0  # sponge 宽 10D（下游）
SPONGE_ALPHA = 10.0  # τ_eff = τ·(1 + α·σ)，collide_mrt docstring 公式
ST_SEED = 0.14  # 播种 Strouhal（方柱文献 St 0.13-0.15 中值）
SEED_AMPL = 0.10  # 播种 uy 振幅（× u_in）
SEED_PERIODS = 2.0  # 播种持续 2 个 St_seed 周期


def square_mask(
    nx: int, ny: int, cx: float, cy: float, side: float, device: torch.device
) -> torch.Tensor:
    """正置方块掩码，half-way BB → 有效边长精确等于 side。

    与 cylinder_mask 同接口风格（nx, ny, cx, cy, size, device）。
    x0..x1 覆盖 side 个固体格，half-way bounce-back 把壁面放在
    x0-0.5 / x1+0.5，流体-固体交界面的有效尺寸恰为 side 格。
    """
    x0 = int(round(cx - side / 2.0))
    x1 = x0 + int(round(side)) - 1
    y0 = int(round(cy - side / 2.0))
    y1 = y0 + int(round(side)) - 1
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    return (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)


def run_case(
    D: int,
    re: float,
    u_in: float,
    steps: int,
    device: torch.device,
    compile_mode: str | None = "default",
    warmup_frac: float = 0.5,
    out_path: str | None = None,
    series_path: str | None = None,
) -> dict:
    nx = int(DOMAIN_D * D)
    ny = nx
    nu = u_in * D / re
    tau = 0.5 + 3.0 * nu

    mask = square_mask(nx, ny, SQUARE_X_D * D, ny / 2.0, D, device)
    sigma = make_sponge_strength(
        ny, nx, int(nx - SPONGE_D * D), int(SPONGE_D * D), power=2.0, device=device
    )
    tau_field = tau * (1.0 + SPONGE_ALPHA * sigma)  # τ_eff = τ·(1 + α·σ)

    # 初值：自由流平衡态（固体格内分布由首步 BB 覆盖，同 cylinder 口径）
    rho0 = torch.ones((ny, nx), device=device)
    f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))

    # ---- 整步步进函数（共性 compile 路径，双变体：播种期 / 自由流期）----
    def _forces_and_farfield(f):
        fx, fy = compute_obstacle_forces(f, mask)  # post-stream, pre-bounce-back
        return far_field_bc_2d(f, u_in, mask), fx, fy

    def _step_plain(f):
        return _forces_and_farfield(stream(collide_mrt(f, tau, tau_field=tau_field)))

    def _step_seeded(f, uy_col):
        f, fx, fy = _forces_and_farfield(stream(collide_mrt(f, tau, tau_field=tau_field)))
        # 入口播种：整列覆盖为 feq(rho=1, u_in, uy_seed(t))；两周期后由
        # _step_plain 回归纯自由流
        f[:, :, 0] = equilibrium(
            torch.ones((ny, 1), device=f.device, dtype=f.dtype),
            torch.full((ny, 1), u_in, device=f.device, dtype=f.dtype),
            uy_col.view(ny, 1),
        )[:, :, 0]
        return f, fx, fy

    step_plain = route_step(_step_plain, compile_mode, name=f"square_re100[D{D}]plain")
    step_seeded = route_step(_step_seeded, compile_mode, name=f"square_re100[D{D}]seed", quiet=True)

    omega_seed = 2.0 * math.pi * ST_SEED * u_in / D
    seed_steps = int(round(SEED_PERIODS / (ST_SEED * u_in / D)))
    uy_amp = SEED_AMPL * u_in

    fx_hist: list[torch.Tensor] = []
    fy_hist: list[torch.Tensor] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        if step <= seed_steps:
            uy_val = uy_amp * math.sin(omega_seed * step)
            uy_col = torch.full((ny,), uy_val, device=device, dtype=f.dtype)
            f, fx, fy = step_seeded(f, uy_col)
        else:
            f, fx, fy = step_plain(f)
        fx_hist.append(fx)
        fy_hist.append(fy)
    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"D={D}: non-finite populations after {steps} steps")

    fx_np = torch.stack(fx_hist).detach().cpu().numpy().astype(np.float64)
    fy_np = torch.stack(fy_hist).detach().cpu().numpy().astype(np.float64)

    q_dyn = 0.5 * 1.0 * u_in * u_in * D  # 动压尺度（rho=1，D 为边长格数）
    w0 = int(warmup_frac * steps)  # 稳态分析窗起点
    cd_series = fx_np / q_dyn
    cd_mean = float(cd_series[w0:].mean())
    cl_series = fy_np / q_dyn

    # St：升力信号谱峰主频（FFT；对逐格噪声鲁棒），滞回过零作交叉验证。
    cl_w = cl_series[w0:] - cl_series[w0:].mean()
    st_val = float("nan")
    st_cross = float("nan")
    if cl_w.size >= 256:
        spec = np.abs(np.fft.rfft(cl_w * np.hanning(cl_w.size)))
        k = int(np.argmax(spec[1:])) + 1  # 跳过 DC
        delta = 0.0
        if 0 < k < spec.size - 1:
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-30:
                delta = float(0.5 * (a - c) / den)
        st_val = float((k + delta) / cl_w.size * D / u_in)
        thr = 0.25 * float(np.abs(cl_w).max())
        sig = np.where(cl_w >= thr, 1, np.where(cl_w <= -thr, -1, 0))
        idx = np.where(sig != 0, np.arange(sig.size), -1)
        last = np.maximum.accumulate(idx)
        state = np.where(last >= 0, sig[np.maximum(last, 0)], 0)
        crossings = np.where((state[:-1] < 0) & (state[1:] > 0))[0] + 1
        if len(crossings) >= 3:
            t_shed = float(np.median(np.diff(crossings)))
            st_cross = D / (u_in * t_shed)

    err_cd = (cd_mean - REF_CD) / REF_CD * 100.0
    err_st = (st_val - REF_ST) / REF_ST * 100.0

    if series_path:
        np.savez(series_path, cd=cd_series, cl=cl_series, warmup_from=w0)

    result = {
        "case": "square_cylinder_re100_free_stream_vortex_shedding",
        "D": D,
        "nx": nx,
        "ny": ny,
        "re": re,
        "u_in": u_in,
        "nu_lb": nu,
        "tau": round(tau, 6),
        "sponge": {
            "x0": int(nx - SPONGE_D * D),
            "width": int(SPONGE_D * D),
            "alpha": SPONGE_ALPHA,
            "power": 2.0,
        },
        "seed": {
            "st_seed": ST_SEED,
            "ampl_frac": SEED_AMPL,
            "periods": SEED_PERIODS,
            "steps": seed_steps,
        },
        "steps": steps,
        "warmup_frac": warmup_frac,
        "analyze_from": w0,
        "compile_mode": compile_mode,
        "cd": round(cd_mean, 4),
        "cd_ref": REF_CD,
        "err_pct": round(err_cd, 2),
        "st": round(st_val, 4) if math.isfinite(st_val) else None,
        "st_ref": REF_ST,
        "st_err_pct": round(err_st, 2) if math.isfinite(err_st) else None,
        "st_crossing": round(st_cross, 4) if math.isfinite(st_cross) else None,
        "cl_amp": float(np.abs(cl_w).max()),
        "finite": True,
        "elapsed_s": round(elapsed, 1),
    }
    print(
        f"[square_re100 D={D}] steps={steps} t={elapsed:.0f}s Cd={cd_mean:.4f} "
        f"({err_cd:+.2f}%) St={st_val:.4f} ({err_st:+.2f}%) "
        f"cl_amp={result['cl_amp']:.4f} seed_steps={seed_steps}",
        flush=True,
    )
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="square cylinder Re=100 free-stream vortex shedding (Okajima)"
    )
    ap.add_argument("--D", type=int, nargs="+", default=[32, 48])
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="per-D step counts; default 60000 (D=32) / 90000 (D=48)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup-frac", type=float, default=0.5)
    ap.add_argument("--out", default="")
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    device = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    default_steps = {32: 60000, 48: 90000}
    steps_list = args.steps or [default_steps.get(D, 60000) for D in args.D]

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    grids = {}
    for D, steps in zip(args.D, steps_list):
        grids[str(D)] = run_case(
            D,
            args.re,
            args.u_in,
            steps,
            device,
            compile_mode=compile_mode,
            warmup_frac=args.warmup_frac,
            out_path=str(out / f"case_D{D}.json") if out else None,
        )

    cds = [g["cd"] for g in grids.values()]
    summary = {
        "case": "square_cylinder_re100_free_stream_vortex_shedding",
        "reference": "Okajima 1982: St~0.14 (0.13-0.15), Cd~1.6 (free-stream)",
        "grids": grids,
        "convergence": {
            "cd": cds,
            "err_decreased": abs(grids[str(args.D[-1])]["err_pct"])
            <= abs(grids[str(args.D[0])]["err_pct"])
            if len(cds) > 1
            else None,
            "cd_within_3pct": all(abs(g["err_pct"]) <= 3.0 for g in grids.values()),
            "st_within_3pct": all(abs(g["st_err_pct"]) <= 3.0 for g in grids.values()),
        },
    }
    if out:
        (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["convergence"]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

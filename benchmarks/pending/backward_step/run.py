#!/usr/bin/env python3
"""Gartling (1990) 2-D backward-facing step benchmark — Re = 800, expansion ratio 2:1.

观测量: 一级(下壁)再附着长度 X1, 以**下游通道高度 H** 归一化。
参考值 (多源核对见 REFERENCE_AUDIT.md):
    X1 / H     = 6.10     (Gartling 1990; Gresho et al.)
    X1 / h_step= 12.20    (同一物理量的台阶高归一化; Keskar&Lyn 12.19, Grigoriev&Dargush 12.18)
    X2 / H     = 4.85     (上壁分离点, 二次核对)
    X3 / H     = 10.48    (上壁再附着, 二次核对)

Gartling 口径 (关键 — 见 REFERENCE_AUDIT.md):
    通道高 H, 台阶高 h = H/2 (ER = 2), 入口 = 上半通道, 充分发展抛物线
    u(y) = 24 y (H/2 - y)  →  ū = 1, u_max = 1.5
    Re = ū H / ν = 800      (**基于通道高 H**, 不是台阶高)
    入口剖面在台阶平面处施加 (台阶上游无通道).

格点映射:
    ny = 2m + 1 (奇数)  →  下游壁到壁高 H = ny - 1 = 2m, 入口壁到壁高 = m = H/2.
    台阶固体占 y = 0..m (共 m+1 行), 台阶下游立面在 x = x_step - 0.5.
    下壁在 y=0.5, 上壁在 y=ny-0.5  (half-way bounce-back).
    x_step = 8 个上游格 (入口在 x=0, 施加的抛物线在短上游通道内是精确定常解,
    故等价于 Gartling 在台阶平面施加剖面).

用法:
    run.py --m 64 96 --device sdaa:0 [--u 0.06] [--steps N N] [--compile-mode eager] [--out DIR]
    run.py --smoke --device sdaa:0          # 小网格短跑冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks (compile_route)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    compile_status_of,
    route_step,
)

from tensorlbm.boundaries import (  # noqa: E402
    bounce_back_cells,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, collide_rlbm, stream  # noqa: E402

# ---------------------------------------------------------------------------
# 参考值 (Gartling 1990, Re=800, ER=2) — 多源核对见 REFERENCE_AUDIT.md
# ---------------------------------------------------------------------------
REF_X1_H = 6.10  # X1 / H_channel  (先用 H 归一化, 恒等于 X1/h_step = 12.20)
REF_X2_H = 4.85  # 上壁分离点 / H
REF_X3_H = 10.48  # 上壁再附着 / H
ERR_TOL_PCT = 3.0

X_STEP = 8  # 上游格数 (入口 x=0 .. 台阶立面 x=X_STEP-0.5)


def _default_device() -> str:
    try:
        import torch_sdaa  # noqa: F401

        if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
            return "sdaa:0"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# ---------------------------------------------------------------------------
# 几何 / 入口剖面
# ---------------------------------------------------------------------------
def build_solid(ny: int, nx: int, m: int, x_step: int, device: torch.device) -> torch.Tensor:
    """固体掩码: 上壁 (y=ny-1), 下壁 (y=0, x>=x_step), 台阶块 (y=0..m, x<x_step)."""
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    solid[-1, :] = True
    solid[0, x_step:] = True
    solid[0 : m + 1, :x_step] = True
    return solid


def inlet_profile(ny: int, m: int, u_mean: float, device: torch.device, dtype) -> torch.Tensor:
    """充分发展抛物线入口剖面 (整列 ny 行; 台阶固体行取 0).

    入口通道壁到壁高 a = m, 底壁在 y = m + 0.5, 顶壁在 y = ny - 0.5.
    u(s) ∝ 6 s (a - s) / a²,  s = y - (m + 0.5);  离散归一化使入口行均值 == u_mean.
    """
    a = float(m)
    y = np.arange(ny, dtype=np.float64)
    s = y - (m + 0.5)
    prof = np.zeros(ny, dtype=np.float64)
    rows = np.arange(m + 1, ny - 1)  # 入口流体行: m+1 .. ny-2
    ss = s[rows]
    raw = 6.0 * ss * (a - ss) / (a * a)
    raw_mean = raw.mean()
    prof[rows] = raw / raw_mean * u_mean
    return torch.tensor(prof, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# 再附着/分离点测量 (壁面剪应力 ∝ 第一层流体行的 ux; 线性亚格插值)
# ---------------------------------------------------------------------------
def _crossings(row: np.ndarray) -> list[tuple[float, int]]:
    """返回 row 中所有过零点的 (亚格列坐标(相对 row[0]), 符号方向).

    方向 +1: 由负到正 (- -> +); -1: 由正到负 (+ -> -).
    row 为 ux[y_wall_row, x_step:].
    """
    out: list[tuple[float, int]] = []
    for i in range(len(row) - 1):
        a, b = float(row[i]), float(row[i + 1])
        if a == b:
            continue
        if a <= 0.0 < b:
            frac = (0.0 - a) / (b - a)
            out.append((i + frac, +1))
        elif a >= 0.0 > b:
            frac = (0.0 - a) / (b - a)
            out.append((i + frac, -1))
    return out


def measure(ux: torch.Tensor, x_step: int, ny: int, m: int) -> dict[str, float]:
    """从 ux 场测量 X1 (下壁再附), X2/X3 (上壁分离/再附). 归一化: H = 2m, h = m."""
    H = float(ny - 1)  # 下游通道壁到壁高 = 2m
    # 下壁: 第一层流体行 y=1; 距离立案面 x = x_step - 0.5
    lower = ux[1, x_step:].detach().cpu().numpy().astype(np.float64)
    cross = _crossings(lower)
    x1 = None
    for xc, d in cross:
        if d == +1:
            x1 = (x_step + xc) - (x_step - 0.5)
            break
    # 上壁: 第一层流体行 y=ny-2
    upper = ux[ny - 2, x_step:].detach().cpu().numpy().astype(np.float64)
    uc = _crossings(upper)
    x2 = x3 = None
    for k, (xc, d) in enumerate(uc):
        if d == -1 and x2 is None:  # + -> - 分离
            x2 = (x_step + xc) - (x_step - 0.5)
        elif d == +1 and x2 is not None and x3 is None:  # - -> + 再附
            x3 = (x_step + xc) - (x_step - 0.5)
            break
    return {
        "X1_H": (x1 / H) if x1 is not None else float("nan"),
        "X1_h": (x1 / m) if x1 is not None else float("nan"),
        "X2_H": (x2 / H) if x2 is not None else float("nan"),
        "X3_H": (x3 / H) if x3 is not None else float("nan"),
        "X1_cells": x1 if x1 is not None else float("nan"),
    }


# ---------------------------------------------------------------------------
# 单档网格运行
# ---------------------------------------------------------------------------
def run_grid(
    m: int,
    re: float,
    u_in: float,
    steps: int,
    out_interval: int,
    device: torch.device,
    compile_mode: str | None,
    L_over_H: float,
    collision: str,
    x_step: int = X_STEP,
) -> dict:
    ny = 2 * m + 1
    H = ny - 1  # = 2m
    L = int(round(L_over_H * H))
    nx = x_step + L
    nu = u_in * H / re
    tau = 0.5 + 3.0 * nu

    solid = build_solid(ny, nx, m, x_step, device)
    u_prof = inlet_profile(ny, m, u_in, device, torch.float32)

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.zeros((ny, nx), device=device)
    ux0[:, :] = u_in  # 均匀初值（流体区）
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(ux0))

    collide = {"mrt": collide_mrt, "rlbm": collide_rlbm}[collision]

    def _step(f):
        f = collide(f, tau=tau)
        f = stream(f)
        f = zou_he_inlet_velocity(f, u_prof, 0.0)
        f = zou_he_outlet_pressure(f, 1.0)
        f = bounce_back_cells(f, solid)
        return f

    step_fn = route_step(_step, compile_mode, name=f"bfs_gartling[m{m}]")

    series: list[dict] = []
    t0 = time.time()
    mass0 = None
    for step in range(1, steps + 1):
        f = step_fn(f)
        if step % out_interval == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            ux = ux.masked_fill(solid, 0.0)
            mm = measure(ux, x_step, ny, m)
            mass = float(rho.sum().item())
            if mass0 is None:
                mass0 = mass
            mm["step"] = step
            mm["max_speed"] = float(torch.sqrt(ux * ux + uy * uy).max().item())
            mm["mass_drift"] = mass - mass0
            series.append(mm)
            print(
                f"[m={m} H={H}] step={step:>7d} X1/H={mm['X1_H']:.4f} "
                f"X1/h={mm['X1_h']:.4f} X2/H={mm['X2_H']:.3f} X3/H={mm['X3_H']:.3f} "
                f"max|u|={mm['max_speed']:.4f} drift={mm['mass_drift']:+.3e} "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )
    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"m={m}: non-finite populations")

    x1h = series[-1]["X1_H"]
    x1h_series = [s["X1_H"] for s in series]
    tail = x1h_series[-3:] if len(x1h_series) >= 3 else x1h_series
    span_tail = max(tail) - min(tail)
    err = (x1h - REF_X1_H) / REF_X1_H * 100.0

    cs = compile_status_of(step_fn)
    return {
        "m": m,
        "ny": ny,
        "nx": nx,
        "H": H,
        "ER": float(H) / float(m),
        "L_over_H": float(L) / float(H),
        "u_in": u_in,
        "re": re,
        "nu_lb": nu,
        "tau": round(tau, 6),
        "collision": collision,
        "steps": steps,
        "X1_H": round(x1h, 4),
        "X1_h": round(series[-1]["X1_h"], 4),
        "X2_H": round(series[-1]["X2_H"], 4),
        "X3_H": round(series[-1]["X3_H"], 4),
        "X1_H_series": [round(v, 4) for v in x1h_series],
        "err_pct": round(err, 3),
        "tail_span_H": round(span_tail, 4),
        "max_speed": round(series[-1]["max_speed"], 5),
        "mass_drift_rel": series[-1]["mass_drift"] / float(ny * nx),
        "elapsed_s": round(elapsed, 1),
        "compile_status": cs.get("compile_status"),
        "compile_status_reason": cs.get("compile_status_reason"),
        "finite": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gartling 1990 BFS Re=800 ER=2 benchmark")
    ap.add_argument("--m", type=int, nargs="+", default=[64, 96], help="台阶半高 m (格); H=2m")
    ap.add_argument("--u", type=float, default=0.06, help="入口平均速度 (格)")
    ap.add_argument("--re", type=float, default=800.0)
    ap.add_argument("--steps", type=int, nargs="+", default=None)
    ap.add_argument("--out-interval", type=int, default=None)
    ap.add_argument("--L-over-H", type=float, default=15.0)
    ap.add_argument("--collision", choices=["mrt", "rlbm"], default="mrt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    if args.smoke:
        args.m = [8]
        args.steps = [400]
        args.out_interval = 200
        args.L_over_H = 15.0

    device = torch.device(args.device if args.device else _default_device())
    compile_mode = compile_mode_from_args(args)

    default_steps = {m: max(40000, int(2.5 * (args.L_over_H * (2 * m)) / args.u)) for m in args.m}
    steps_list = args.steps or [default_steps[m] for m in args.m]
    out_interval = args.out_interval or max(2000, steps_list[0] // 20)

    print(
        f"=== Gartling BFS Re={args.re} ER=2 ref X1/H={REF_X1_H} (X1/h={REF_X1_H*2:.2f}) "
        f"u={args.u} collision={args.collision} L/H={args.L_over_H} device={device} "
        f"compile={compile_mode!r} ===",
        flush=True,
    )
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    grids: dict[str, dict] = {}
    for m, steps in zip(args.m, steps_list):
        g = run_grid(
            m,
            args.re,
            args.u,
            steps,
            out_interval,
            device,
            compile_mode,
            args.L_over_H,
            args.collision,
        )
        grids[str(m)] = g
        if out:
            (out / f"case_m{m}.json").write_text(json.dumps(g, indent=2))

    x1 = [g["X1_H"] for g in grids.values()]
    errs = [g["err_pct"] for g in grids.values()]
    span = (max(x1) - min(x1)) / (sum(x1) / len(x1)) * 100.0 if len(x1) > 1 else 0.0
    within = all(abs(e) <= ERR_TOL_PCT for e in errs)
    conv = span <= ERR_TOL_PCT if len(x1) > 1 else None

    summary = {
        "case": "backward_step_gartling_re800_er2",
        "reference": {
            "source": "Gartling 1990 IJNMF 11(7):953-967; corroborated by ECN/TNO ECN-E-11-042 "
            "§4 (X1=6.10, X2=4.85, X3=10.48) and arXiv:2507.16509 table "
            "(Gartling/Gresho 12.20, Keskar&Lyn 12.19, Grigoriev&Dargush 12.18 in step-height units)",
            "re_definition": "Re = u_mean * H_channel / nu  (= 800)",
            "X1_over_H": REF_X1_H,
            "X1_over_h_step": REF_X1_H * 2.0,
            "X2_over_H": REF_X2_H,
            "X3_over_H": REF_X3_H,
            "err_tol_pct": ERR_TOL_PCT,
        },
        "grids": grids,
        "convergence": {
            "X1_H": x1,
            "err_pct": errs,
            "span_pct": round(span, 3),
            "all_within_3pct": within,
            "grid_span_within_3pct": conv,
        },
        "verified": bool(within and (conv if conv is not None else True)),
    }
    if out:
        (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["convergence"], indent=2), flush=True)
    print(f"VERIFIED={summary['verified']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
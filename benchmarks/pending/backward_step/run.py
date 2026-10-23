#!/usr/bin/env python3
"""Gartling (1990) 2-D backward-facing step benchmark — Re = 800, expansion ratio 2:1.

观测量: 一级(下壁)再附着长度 X1, 以**下游通道高度 H** 归一化。
参考值 (多源核对见 REFERENCE_AUDIT.md):
    X1 / H     = 6.10     (Gartling 1990; Gresho et al.; ECN/TNO ECN-E-11-042)
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

★ mode K 稳定化 (本文件; 见 STATUS.md / README §根因):
    封闭域内 硬速度入口 (Zou/He velocity Dirichlet) + Zou/He 压力出口
    (pressure Dirichlet) 会激发**封闭声学驻波** (周期 = L/c_s, 与实测振荡
    周期精确吻合), 使质量漂移 ~ +4e-2/步、近壁剪应力信号被声学噪声淹没.
    修复 (mode K):
      (1) **pre-stream 半程 bounce-back** (库 Poiseuille 验证口径): 碰撞后、
          流之前把固体格设为其前碰撞分布的反向 (fp[opp]), 再 stream —
          壁面反射时相与库 Poiseuille 基准一致; 取代原来的 post-stream BB.
      (2) **非平衡态海绵层** (make_sponge_strength): 下游末端 W 格外
          f <- f - σ(x)·(f - f_eq), 只吸收非平衡(声学)分量, 逐格守恒
          (Σ(f-f_eq)=0), 把出口附近的声学波吸收掉, 不扰动 X1 所在区域.
    效果: 质量漂移 +4e-2/步 → +2e-7/步; rho 稳定 [0.992,1.008]; max|u|≈0.086.

★ 鲁棒化测量 (本文件):
    (a) **壁面剪应力多行探测**: 近壁 K 行 (K = min(6, m//2)) 的最小二乘线性
        拟合斜率 du/dy 作为壁面剪应力 (∝ ν·du/dy), 而非单行 ux — 抑制
        单行离散噪声; 同时报告 row1 口径作对照.
    (b) **时间窗口平均**: 每隔 avg_stride 步采集一次近壁 K 行的瞬时剖面,
        对末尾 avg_steps 步 (滑窗, 覆盖 ≥1 个声学周期) 求平均后再取零穿越,
        消除残余声学振荡.
    X1 = 下壁 du/dy 的 − → + 零穿越 (亚格线性插值), 距台阶立面 x=x_step−0.5,
    以 H 归一化.

用法:
    run.py --m 64 96 --device sdaa:0 [--u 0.06] [--steps N N] [--compile-mode eager]
    run.py --smoke --device sdaa:0          # 小网格短跑冒烟
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
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
    make_sponge_strength,
    zou_he_inlet_velocity,
    zou_he_outlet_pressure,
)
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, collide_rlbm, stream  # noqa: E402

# ---------------------------------------------------------------------------
# 参考值 (Gartling 1990, Re=800, ER=2) — 多源核对见 REFERENCE_AUDIT.md
# ---------------------------------------------------------------------------
REF_X1_H = 6.10  # X1 / H_channel  (恒等于 X1/h_step = 12.20)
REF_X2_H = 4.85  # 上壁分离点 / H
REF_X3_H = 10.48  # 上壁再附着 / H
ERR_TOL_PCT = 3.0

X_STEP = 8  # 上游格数 (入口 x=0 .. 台阶立面 x=X_STEP-0.5)

# mode K 默认参数 (由 /tmp/test_bc3.py 验证: drift +2e-7/步, rho∈[0.992,1.008])
DEFAULT_SPONGE_PEAK = 0.3
DEFAULT_SPONGE_WIDTH_H = 3.0


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
# 鲁棒再附着/分离测量: 多行壁面剪应力 + 亚格零穿越
# ---------------------------------------------------------------------------
def _shear_weights(K: int, wall_first: bool) -> np.ndarray:
    """K 行最小二乘斜率 du/dy 的线性权重 (对行序号按 y 递增).

    profile_shear(x) = Σ_i w_i · ux[row_i, x] ∝ du/dy(x) (∝ 壁面剪应力/ν).
    权重和为 0 ⇒ 对均匀偏移不敏感; 只在近壁线性区使用 (K 小).
    """
    # 相对 y 权重与绝对偏移无关 (斜率), 用 0..K-1 作坐标
    y = np.arange(K, dtype=np.float64)
    ym = y.mean()
    w = y - ym
    return w / float((w * w).sum())


def _crossings(profile: np.ndarray, start_col: int) -> list[tuple[float, int]]:
    """profile 中 col >= start_col 的所有过零点 (亚格列坐标, 符号方向).

    方向 +1: − → +;  方向 −1: + → −.
    """
    out: list[tuple[float, int]] = []
    p = np.asarray(profile, dtype=np.float64)
    for c in range(start_col, len(p) - 1):
        a, b = p[c], p[c + 1]
        if a == b:
            continue
        if a <= 0.0 < b:
            out.append((c + (0.0 - a) / (b - a), +1))
        elif a >= 0.0 > b:
            out.append((c + (0.0 - a) / (b - a), -1))
    return out


def _x1_from_shear(shear: np.ndarray, x_step: int, H: float) -> tuple[float, float]:
    """下壁 − → + 零穿越 → (X1/H, X1_cells). 找不到 → nan."""
    for xc, d in _crossings(shear, x_step):
        if d == +1:
            x1_cells = (xc) - (x_step - 0.5)
            return x1_cells / H, x1_cells
    return float("nan"), float("nan")


def _x2x3_from_shear(shear: np.ndarray, x_step: int, H: float) -> tuple[float, float]:
    """上壁 (+ → − 分离, 随后 − → + 再附) → (X2/H, X3/H)."""
    x2 = x3 = float("nan")
    for xc, d in _crossings(shear, x_step):
        if d == -1 and x2 != x2:
            x2 = (xc - (x_step - 0.5)) / H
        elif d == +1 and x2 == x2 and x3 != x3:
            x3 = (xc - (x_step - 0.5)) / H
            break
    return x2, x3


def _rows_near_walls(ny: int, m: int, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (下壁近壁行, 下壁权重(按交付顺序), 上壁近壁行, 上壁权重).

    交付顺序统一为"距壁由近到远", 使权重 w[0] 对应最近壁行 (斜率方向正确).
    下壁 (y=0.5): 行 1,2,...,K  → y 递增 → 直接 _shear_weights.
    上壁 (y=ny-0.5): 行 ny-2, ny-3, ..., ny-1-K → y 递减;
        交付顺序"由近到远"= ny-2, ny-3, ... 即 y 递减, 需把坐标反向:
        对 y 递减序列, 斜率 du/dy 权重取 _shear_weights(K)[::-1].
    """
    lo_rows = np.arange(1, K + 1)
    up_rows = np.arange(ny - 2, ny - 2 - K, -1)  # 由近(ny-2)到远
    w = _shear_weights(K, True)
    w_lo = w                      # lo_rows: y 递增 (1..K), 由近到远
    w_up = w[::-1]                # up_rows: y 递减, 行顺序由近到远 ⇒ 坐标反向
    return lo_rows, w_lo, up_rows, w_up


def measure_profile(
    ux: torch.Tensor,
    solid: torch.Tensor,
    x_step: int,
    ny: int,
    m: int,
    K: int,
    lo_rows: np.ndarray,
    w_lo: np.ndarray,
    up_rows: np.ndarray,
    w_up: np.ndarray,
) -> dict[str, float | None]:
    """从 (可含时间平均的) ux 场测量 X1/X2/X3.

    主口径 = 多行最小二乘剪应力; 对照口径 = row1 (单行 ux).
    """
    H = float(ny - 1)  # = 2m
    uxm = ux.masked_fill(solid, 0.0)

    lo = uxm[lo_rows].detach().cpu().numpy().astype(np.float64)  # (K, nx)
    up = uxm[up_rows].detach().cpu().numpy().astype(np.float64)

    shear_lo = (w_lo[:, None] * lo).sum(axis=0)  # ∝ du/dy 下壁
    shear_up = (w_up[:, None] * up).sum(axis=0)  # ∝ du/dy 上壁

    x1_h, x1_cells = _x1_from_shear(shear_lo, x_step, H)
    x2_h, x3_h = _x2x3_from_shear(shear_up, x_step, H)

    # 对照: row1 (最近壁单行) / row2
    row1 = uxm[1].detach().cpu().numpy().astype(np.float64)
    r1_h, _ = _x1_from_shear(row1, x_step, H)
    r1u = uxm[ny - 2].detach().cpu().numpy().astype(np.float64)
    r1x2, r1x3 = _x2x3_from_shear(r1u, x_step, H)

    return {
        "X1_H": x1_h,
        "X1_h": (x1_cells / m) if x1_cells == x1_cells else float("nan"),
        "X1_cells": x1_cells,
        "X2_H": x2_h,
        "X3_H": x3_h,
        "X1_H_row1": r1_h,
        "X2_H_row1": r1x2,
        "X3_H_row1": r1x3,
    }


# ---------------------------------------------------------------------------
# 单档网格运行
# ---------------------------------------------------------------------------
def run_grid(
    m: int,
    re: float,
    u_in: float,
    steps: int,
    min_steps: int,
    out_interval: int,
    avg_steps: int,
    avg_stride: int,
    device: torch.device,
    compile_mode: str | None,
    L_over_H: float,
    collision: str,
    sponge_peak: float,
    sponge_width_H: float,
    prestream_bb: bool,
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

    # --- mode K 组件 (2): 非平衡态海绵层 ---
    sponge_w = int(round(sponge_width_H * H))
    sponge_x0 = nx - sponge_w
    if sponge_peak > 0.0 and sponge_w > 0:
        sigma = sponge_peak * make_sponge_strength(
            ny, nx, sponge_x0, sponge_w, power=2.0, device=device
        )
    else:
        sigma = torch.zeros((ny, nx), device=device)

    # 初值: 上游抛物线, 下游 ~ u_in/2 (下游均值), 减小启动瞬态
    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.zeros((ny, nx), device=device)
    ux0[:, :x_step] = u_prof.view(ny, 1)
    ux0[:, x_step:] = u_in / 2.0
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(ux0))

    collide = {"mrt": collide_mrt, "rlbm": collide_rlbm}[collision]
    opp = OPPOSITE.to(device)
    solid3 = solid.unsqueeze(0)

    def _noneq_sponge(f: torch.Tensor) -> torch.Tensor:
        if sponge_peak <= 0.0:
            return f
        rho, ux, uy = macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        return f - sigma.unsqueeze(0) * (f - feq)

    if prestream_bb:

        def _step(f):
            fp = f.clone()
            fc = collide(f, tau=tau)
            # pre-stream 半程 BB (库 Poiseuille 验证口径)
            fc = torch.where(solid3, fp[opp], fc)
            f = stream(fc)
            f = zou_he_inlet_velocity(f, u_prof, 0.0)
            f = zou_he_outlet_pressure(f, 1.0)
            return _noneq_sponge(f)

    else:

        def _step(f):
            f = collide(f, tau=tau)
            f = stream(f)
            f = zou_he_inlet_velocity(f, u_prof, 0.0)
            f = zou_he_outlet_pressure(f, 1.0)
            f = torch.where(solid3, f[opp], f)  # post-stream BB (旧口径 A)
            return _noneq_sponge(f)

    step_fn = route_step(_step, compile_mode, name=f"bfs_gartling[m{m}]")

    K = max(2, min(6, m // 2))
    lo_rows, w_lo, up_rows, w_up = _rows_near_walls(ny, m, K)
    K = len(lo_rows)
    n_win = max(1, int(round(avg_steps / max(1, avg_stride))))
    win: deque = deque(maxlen=n_win)
    acc_lo = np.zeros(nx, dtype=np.float64)  # 全程累积 (时间平均)
    acc_up = np.zeros(nx, dtype=np.float64)
    n_acc = 0

    series: list[dict] = []
    t0 = time.time()
    mass0 = None
    plateau = False
    prev_win_x1: float | None = None
    stable_hits = 0

    for step in range(1, steps + 1):
        f = step_fn(f)
        # 采集近壁 K 行 (多行剪应力剖面)
        if step % avg_stride == 0:
            ux = macroscopic(f)[1]
            uxm = ux.masked_fill(solid, 0.0)
            lo = uxm[lo_rows].detach().cpu().numpy().astype(np.float64)
            up = uxm[up_rows].detach().cpu().numpy().astype(np.float64)
            s_lo = (w_lo[:, None] * lo).sum(axis=0)
            s_up = (w_up[:, None] * up).sum(axis=0)
            win.append((s_lo, s_up))
            acc_lo += s_lo
            acc_up += s_up
            n_acc += 1

        if step % out_interval == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            mm = measure_profile(ux, solid, x_step, ny, m, K, lo_rows, w_lo, up_rows, w_up)
            mass = float(rho.sum().item())
            if mass0 is None:
                mass0 = mass
            # 滑窗时间平均
            wlo = np.mean([s[0] for s in win], axis=0) if win else None
            wup = np.mean([s[1] for s in win], axis=0) if win else None
            if wlo is not None:
                x1w, x1wc = _x1_from_shear(wlo, x_step, H)
                x2w, x3w = _x2x3_from_shear(wup, x_step, H)
            else:
                x1w = x1wc = x2w = x3w = float("nan")
            mm.update(
                {
                    "step": step,
                    "X1_H_win": x1w,
                    "X2_H_win": x2w,
                    "X3_H_win": x3w,
                    "max_speed": float(torch.sqrt(ux * ux + uy * uy).max().item()),
                    "mass_drift": mass - mass0,
                    "rho_min": float(rho.min().item()),
                    "rho_max": float(rho.max().item()),
                    "n_win": len(win),
                }
            )
            series.append(mm)
            print(
                f"[m={m} H={H} nx={nx} K={K} win={n_win}] step={step:>7d} "
                f"X1/H(win)={x1w:.4f} X1/H(row1)={mm['X1_H_row1']:.4f} "
                f"X1/H(inst)={mm['X1_H']:.4f} "
                f"X1/h={mm['X1_h']:.4f} X2/H={x2w:.3f} X3/H={x3w:.3f} "
                f"max|u|={mm['max_speed']:.4f} drift={mm['mass_drift']:+.3e} "
                f"rho[{mm['rho_min']:.4f},{mm['rho_max']:.4f}] "
                f"t={time.time() - t0:.0f}s",
                flush=True,
            )
            # 平台判据: 滑窗 X1/H 相对变化 < 0.2% 连续 3 次 (且 >= min_steps)
            if step >= min_steps and x1w == x1w and prev_win_x1 is not None and prev_win_x1 == prev_win_x1:
                rel = abs(x1w - prev_win_x1) / max(abs(prev_win_x1), 1e-12)
                stable_hits = stable_hits + 1 if rel < 0.002 else 0
                if stable_hits >= 3:
                    plateau = True
            prev_win_x1 = x1w
            if plateau and step >= min_steps:
                print(f"[m={m}] plateau reached at step={step}", flush=True)
                break

    elapsed = time.time() - t0
    if not bool(torch.isfinite(f).all().item()):
        raise RuntimeError(f"m={m}: non-finite populations")

    # 全程时间平均 (最终稳健值)
    mean_lo = acc_lo / max(1, n_acc)
    mean_up = acc_up / max(1, n_acc)
    x1_full, x1_full_cells = _x1_from_shear(mean_lo, x_step, H)
    x2_full, x3_full = _x2x3_from_shear(mean_up, x_step, H)
    # 滑窗 (末尾 avg_steps) 时间平均
    wlo = np.mean([s[0] for s in win], axis=0)
    wup = np.mean([s[1] for s in win], axis=0)
    x1_win, x1_win_cells = _x1_from_shear(wlo, x_step, H)
    x2_win, x3_win = _x2x3_from_shear(wup, x_step, H)

    x1h = series[-1]["X1_H_win"] if series[-1]["X1_H_win"] == series[-1]["X1_H_win"] else x1_win
    x1h_row1 = series[-1]["X1_H_row1"]
    x1h_series = [s["X1_H_win"] for s in series]
    tail = [v for v in x1h_series[-5:] if v == v]
    tail_span = (max(tail) - min(tail)) if len(tail) >= 2 else float("nan")
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
        "mode": "K" if (prestream_bb and sponge_peak > 0) else "A",
        "prestream_bb": prestream_bb,
        "sponge_peak": sponge_peak,
        "sponge_width_H": sponge_width_H,
        "steps_run": series[-1]["step"],
        "min_steps": min_steps,
        "plateau": plateau,
        "avg_steps": avg_steps,
        "avg_stride": avg_stride,
        "n_win_samples": n_win,
        "K_rows": K,
        "X1_H": round(x1h, 4),
        "X1_h": round(x1h * 2.0, 4),
        "X1_H_row1": round(x1h_row1, 4) if x1h_row1 == x1h_row1 else None,
        "X1_H_fullavg": round(x1_full, 4) if x1_full == x1_full else None,
        "X2_H": round(x2_win, 4) if x2_win == x2_win else None,
        "X3_H": round(x3_win, 4) if x3_win == x3_win else None,
        "X1_H_series": [round(v, 4) if v == v else None for v in x1h_series],
        "err_pct": round(err, 3),
        "tail_span_H": round(tail_span, 4) if tail_span == tail_span else None,
        "max_speed": round(series[-1]["max_speed"], 5),
        "rho_min": round(series[-1]["rho_min"], 5),
        "rho_max": round(series[-1]["rho_max"], 5),
        "mass_drift_rel": series[-1]["mass_drift"] / float(ny * nx),
        "elapsed_s": round(elapsed, 1),
        "compile_status": cs.get("compile_status"),
        "compile_mode_effective": cs.get("compile_mode_effective"),
        "compile_status_reason": cs.get("compile_status_reason"),
        "finite": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gartling 1990 BFS Re=800 ER=2 benchmark (mode K)")
    ap.add_argument("--m", type=int, nargs="+", default=[64, 96], help="台阶半高 m (格); H=2m")
    ap.add_argument("--u", type=float, default=0.06, help="入口平均速度 (格)")
    ap.add_argument("--re", type=float, default=800.0)
    ap.add_argument("--steps", type=int, nargs="+", default=None)
    ap.add_argument("--min-steps", type=int, default=30000)
    ap.add_argument("--out-interval", type=int, default=None)
    ap.add_argument("--avg-steps", type=int, default=20000, help="时间平均滑窗步数 (>=1 声学周期)")
    ap.add_argument("--avg-stride", type=int, default=20, help="时间平均采样间隔")
    ap.add_argument("--L-over-H", type=float, default=20.0)
    ap.add_argument("--collision", choices=["mrt", "rlbm"], default="mrt")
    ap.add_argument("--sponge-peak", type=float, default=DEFAULT_SPONGE_PEAK)
    ap.add_argument("--sponge-width-H", type=float, default=DEFAULT_SPONGE_WIDTH_H)
    ap.add_argument("--no-prestream-bb", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    add_compile_mode_arg(ap)
    args = ap.parse_args()

    if args.smoke:
        args.m = [16]
        args.steps = [6000]
        args.min_steps = 6000
        args.out_interval = 1000
        args.L_over_H = 20.0
        args.avg_steps = 4000

    device = torch.device(args.device if args.device else _default_device())
    compile_mode = compile_mode_from_args(args)

    # 默认步数: 至少覆盖若干气泡对流时 + 声学建立; 上限放宽到 200k
    default_steps = {
        m: max(40000, int(3.0 * (args.L_over_H * (2 * m)) / (args.u / 2.0))) for m in args.m
    }
    steps_list = args.steps or [default_steps[m] for m in args.m]
    out_interval = args.out_interval or max(2000, min(steps_list[0] // 20, 5000))

    print(
        f"=== Gartling BFS Re={args.re} ER=2 ref X1/H={REF_X1_H} (X1/h={REF_X1_H * 2:.2f}) "
        f"u={args.u} collision={args.collision} L/H={args.L_over_H} device={device} "
        f"compile={compile_mode!r} mode={'K' if not args.no_prestream_bb else 'A'} "
        f"sponge(peak={args.sponge_peak},W={args.sponge_width_H}H) "
        f"avg={args.avg_steps}min_steps={args.min_steps} ===",
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
            args.min_steps,
            out_interval,
            args.avg_steps,
            args.avg_stride,
            device,
            compile_mode,
            args.L_over_H,
            args.collision,
            args.sponge_peak,
            args.sponge_width_H,
            not args.no_prestream_bb,
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
        "mode": "K (pre-stream halfway bounce-back + non-equilibrium sponge)",
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
        "measurement": {
            "primary": "multi-row (K rows) least-squares wall shear du/dy ~ tau_w, time-window averaged over avg_steps",
            "secondary": "row1 ux sign crossing (single-row reference)",
            "avg_steps": args.avg_steps,
            "avg_stride": args.avg_stride,
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
#!/usr/bin/env python
"""基准测试: 声波反射 — D3Q19 BGK LBM。

验证平面声波从左向右传播并在刚性壁面（反弹回边界条件）上反射的物理过程。
入射波与反射波的叠加形成驻波。对于刚性壁面，反射系数 |R|=1，壁面处压力加倍。

设置
----
  * 网格: nx=400, ny=4, nz=1 (一维 x 方向传播, D3Q19 BGK)
  * τ = 0.8 (低粘度，尖锐共振)
  * 左边界 (x=0): 平面波源
      ρ = 1 + δρ·sin(ω·t),  ux = c_s·δρ·sin(ω·t)
  * 右边界 (x=nx-1): 刚性壁面 (反弹回 bounce-back)
  * 上下边界: 周期 (ny=4, 最小化)
  * 无海绵层 (我们需要反射！)
  * 监测点: x=0, nx/4, nx/2, 3·nx/4, nx-1 (壁面)
  * δρ = 0.001, ω = 0.1
  * 步数: 2000 (足以达到稳态驻波)

解析解（含粘性衰减）
--------------------
入射波从源 (x=0) 出发，振幅 A = δρ，以速率 Γ 衰减：
  p_i(x,t) = A·e^{-Γ·x/c_s}·sin(ω·t - k·x)

反射波从壁面 (x=L) 出发，振幅 A·e^{-Γ·L/c_s}（入射波到达壁面时的振幅），
再衰减回传：
  p_r(x,t) = A·e^{-Γ·(2L-x)/c_s}·sin(ω·t + k·(x-L))

驻波振幅（峰值）：
  |p(x)| = √[a_i² + a_r² + 2·a_i·a_r·cos(2k(x-L))]

其中：
  a_i(x) = A·e^{-Γ·x/c_s}       (入射波振幅)
  a_r(x) = A·e^{-Γ·(2L-x)/c_s}  (反射波振幅)
  Γ = ν·k²  (衰减率, ν = (τ-½)/3, k = ω/c_s)
  L = nx - 0.5  (半程反弹回壁面位置)

壁面 (x=L)：a_i = a_r = A·e^{-Γ·L/c_s}
  |p(L)| = 2·A·e^{-Γ·L/c_s}  (压力加倍，因子 2 来自反射)

验证
----
  1. 驻波形态: 测量压力振幅 vs x, 与含衰减解析解比较。目标: <10% 误差。
  2. 压力加倍: |p_wall| / |p_incident_wall| ≈ 2，
     其中 |p_incident_wall| = A·e^{-Γ·L/c_s}（壁面处入射波振幅）。
     压力加倍是壁面局部效应：总压 = 入射 + 反射 = 2 × 入射，与衰减无关。
     目标: <10% 误差。
  3. 反射系数: |R| = (S_max - S_min)/(S_max + S_min) = 1 (驻波比法)。
     目标: <5% 误差。

运行
----
    PYTHONPATH=src python examples/benchmark_acoustic_reflection.py \\
        --device cpu --steps 2000

关于源边界条件的说明
--------------------
左边界采用硬源（Dirichlet 型）：将 x=0 处全部分布函数设为源平衡分布。
由于源值 ρ 和 ux 满足 ux = c_s·(ρ-1)（纯右行波条件），硬源等效于
无反射边界：入射波通过入射方向分布函数注入，反射波通过出射方向
分布函数被覆盖吸收。因此不会形成共振腔，驻波振幅由单次反射决定。

关于反弹回边界条件的说明
------------------------
右壁面采用半程反弹回（half-way bounce-back）：周期流（torch.roll）后，
将在壁面处从域外绕回的未知分布函数（cx < 0 方向）替换为其反向
（cx > 0 方向，从内部流来）。壁面物理位置在边界节点与幽灵域之间，
即 x = nx - 0.5。
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# 使 tensorlbm 可导入（从仓库根目录运行时）。
# --------------------------------------------------------------------------- #
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

# 在多核机器上 PyTorch 默认使用全部核心，对小张量的线程管理开销过大。
# 限制为 8 线程可显著加速。
_DEFAULT_THREADS = min(8, os.cpu_count() or 1)
torch.set_num_threads(_DEFAULT_THREADS)

# =========================================================================== #
# 常量
# =========================================================================== #

CS2 = 1.0 / 3.0          # 格子声速平方
CS = math.sqrt(CS2)      # c_s = 1/√3 ≈ 0.5774

# =========================================================================== #
# D3Q19 反弹回方向表
# =========================================================================== #
# D3Q19 格子速度 (cx, cy, cz):
#   0:(0,0,0)   1:(1,0,0)   2:(-1,0,0)   3:(0,1,0)   4:(0,-1,0)
#   5:(0,0,1)   6:(0,0,-1)  7:(1,1,0)    8:(-1,-1,0)  9:(1,-1,0)
#  10:(-1,1,0) 11:(1,0,1)  12:(-1,0,-1) 13:(1,0,-1) 14:(-1,0,1)
#  15:(0,1,1)  16:(0,-1,-1)17:(0,1,-1)  18:(0,-1,1)
#
# OPPOSITE = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]

# 右壁面 (x=nx-1): 未知方向为 cx < 0
_RIGHT_UNKNOWN = [2, 8, 10, 12, 14]
_RIGHT_KNOWN = [1, 7, 9, 11, 13]   # OPPOSITE[_RIGHT_UNKNOWN]


def apply_right_wall_bounceback(f: torch.Tensor) -> torch.Tensor:
    """在右壁面 (x=nx-1) 施加半程反弹回。

    周期流（torch.roll）后，壁面处从域外绕回的未知分布函数
    （cx < 0 方向）被替换为其反向（cx > 0 方向，从内部流来）。
    壁面物理位置在边界节点与幽灵域之间，即 x = nx - 0.5。
    """
    f_new = f.clone()
    f_new[_RIGHT_UNKNOWN, :, :, -1] = f[_RIGHT_KNOWN, :, :, -1]
    return f_new


def apply_source(
    f: torch.Tensor,
    rho_val: float,
    ux_val: float,
    ny: int,
    nz: int,
    device: torch.device,
) -> torch.Tensor:
    """在左边界 x=0 施加硬源（Dirichlet 型）。

    将 x=0 处全部 19 个分布函数设为源平衡分布
    feq(ρ_src, ux_src, 0, 0)。由于源值满足 ux = c_s·(ρ-1)
    （纯右行波条件），此边界等效于无反射边界：注入入射波、
    吸收反射波。
    """
    rho_src = torch.full((nz, ny, 1), rho_val, device=device, dtype=f.dtype)
    ux_src = torch.full((nz, ny, 1), ux_val, device=device, dtype=f.dtype)
    uy_src = torch.zeros((nz, ny, 1), device=device, dtype=f.dtype)
    uz_src = torch.zeros((nz, ny, 1), device=device, dtype=f.dtype)
    feq_src = equilibrium3d(rho_src, ux_src, uy_src, uz_src, device=device)
    f[:, :, :, 0:1] = feq_src
    return f


# =========================================================================== #
# ASCII 可视化
# =========================================================================== #

def ascii_plot_1d(
    y_num: np.ndarray,
    y_ana: np.ndarray | None = None,
    width: int = 72,
    height: int = 14,
    title: str = "",
    x_max: float | None = None,
) -> None:
    """将一维剖面打印为 ASCII 图，可选叠加解析解。

    ``█`` = 数值, ``·`` = 解析, ``╬`` = 两者重合。
    """
    n = len(y_num)
    if y_ana is not None:
        ymin = min(float(y_num.min()), float(y_ana.min()))
        ymax = max(float(y_num.max()), float(y_ana.max()))
    else:
        ymin = float(y_num.min())
        ymax = float(y_num.max())
    span = max(ymax - ymin, 1e-12)

    def _resample(y: np.ndarray) -> list[float]:
        idx = np.minimum((np.arange(width) * n / width).astype(int), n - 1)
        return [float(y[i]) for i in idx]

    num_s = _resample(y_num)
    ana_s = _resample(y_ana) if y_ana is not None else None

    if title:
        print(f"  {title}", flush=True)

    half = span / (2 * height)
    for row in range(height, 0, -1):
        y_val = ymin + (row - 0.5) * span / height
        line: list[str] = []
        for col in range(width):
            ch = " "
            if abs(num_s[col] - y_val) <= half:
                ch = "█"
            if ana_s is not None and abs(ana_s[col] - y_val) <= half:
                ch = "·" if ch == " " else "╬"
            line.append(ch)
        y_lbl = ymin + row * span / height
        print(f"  {y_lbl:12.6e} |" + "".join(line) + "|", flush=True)

    print(f"  {'':12} +" + "-" * width + "+", flush=True)
    if x_max is not None:
        lbl = [" "] * width
        step = max(width // 8, 1)
        for i in range(0, width, step):
            xv = int(i * x_max / width)
            for j, ch in enumerate(str(xv)):
                if i + j < width:
                    lbl[i + j] = ch
        print(f"  {'':12}  " + "".join(lbl), flush=True)


# =========================================================================== #
# 含衰减的解析驻波振幅
# =========================================================================== #

def damped_standing_wave_amplitude(
    x_arr: np.ndarray,
    A: float,
    gamma: float,
    k: float,
    cs: float,
    L: float,
) -> np.ndarray:
    """计算含粘性衰减的驻波振幅剖面。

    入射波振幅: a_i(x) = A·exp(-Γ·x/c_s)
    反射波振幅: a_r(x) = A·exp(-Γ·(2L-x)/c_s)
    驻波振幅:   |p(x)| = √[a_i² + a_r² + 2·a_i·a_r·cos(2k(x-L))]

    在壁面 (x=L): a_i = a_r, cos(0)=1, |p| = 2·a_i = 2·A·exp(-Γ·L/c_s)
    """
    a_i = A * np.exp(-gamma * x_arr / cs)
    a_r = A * np.exp(-gamma * (2.0 * L - x_arr) / cs)
    phase = 2.0 * k * (x_arr - L)
    return np.sqrt(a_i ** 2 + a_r ** 2 + 2.0 * a_i * a_r * np.cos(phase))


# =========================================================================== #
# 主模拟
# =========================================================================== #

def run_reflection_benchmark(
    nx: int = 400,
    ny: int = 4,
    nz: int = 1,
    tau: float = 0.8,
    delta_rho: float = 0.001,
    omega: float = 0.1,
    n_steps: int = 2000,
    device: str = "cpu",
    log_every: int = 200,
    measure_window: int = 500,
) -> dict:
    """运行声波反射基准测试。

    Parameters
    ----------
    nx, ny, nz : int
        网格尺寸。nz=1 将 D3Q19 降为二维求解器；ny 保持较小值。
    tau : float
        BGK 弛豫时间 (τ > 0.5)。运动粘度 ν = (τ−½)/3。
    delta_rho : float
        源密度扰动振幅（入射波振幅 A = δρ）。
    omega : float
        源角频率 ω (rad/步)。
    n_steps : int
        LBM 时间步数。
    device : str
        ``"cpu"`` 或 ``"cuda"``。
    log_every : int
        每隔多少步打印一行诊断信息。
    measure_window : int
        末尾多少步用于振幅测量（稳态窗口）。
    """
    dev = torch.device(device)
    cs = CS
    cs2 = CS2
    nu = (tau - 0.5) / 3.0               # LBM 运动粘度
    L = nx - 0.5                         # 半程反弹回壁面位置
    k = omega / cs                       # 波数
    lam = 2.0 * math.pi / k              # 波长
    period = 2.0 * math.pi / omega       # 振荡周期
    n_oscillations = n_steps / period

    # 粘性衰减率 Γ = ν·k² (参见 benchmark_acoustic_wave_1d.py)
    gamma_damp = nu * k * k
    # 壁面处入射波振幅 (经衰减)
    p_incident_wall = delta_rho * math.exp(-gamma_damp * L / cs)
    # 壁面处预期驻波振幅 (压力加倍)
    p_wall_expected = 2.0 * p_incident_wall
    # 衰减因子
    damp_factor = math.exp(-gamma_damp * L / cs)

    measure_start = max(n_steps - measure_window, 1)

    # ---- 监测点 ----
    monitor_x = [0, nx // 4, nx // 2, 3 * nx // 4, nx - 1]
    monitor_labels = [
        "x=0(源)",
        "x=nx/4",
        "x=nx/2",
        "x=3nx/4",
        "x=壁面",
    ]

    # ---- 初始条件: 静止流体 ----
    rho_init = torch.ones((nz, ny, nx), device=dev)
    ux_init = torch.zeros((nz, ny, nx), device=dev)
    uy_init = torch.zeros((nz, ny, nx), device=dev)
    uz_init = torch.zeros((nz, ny, nx), device=dev)
    f = equilibrium3d(rho_init, ux_init, uy_init, uz_init, device=dev)

    # ---- 数据存储 ----
    # 振幅测量: 运行最大/最小值
    rho_max = torch.full((nx,), -1e30, device=dev, dtype=torch.float64)
    rho_min = torch.full((nx,), 1e30, device=dev, dtype=torch.float64)
    # 监测点时间序列
    monitor_series = torch.zeros(
        n_steps + 1, len(monitor_x), device=dev, dtype=torch.float64
    )

    # ---- 初始监测 ----
    for i, mx in enumerate(monitor_x):
        monitor_series[0, i] = f[:, 0, 0, mx].sum().double()

    # ---- 头部信息 ----
    print("=" * 72, flush=True)
    print("  声波反射基准测试 — D3Q19 BGK LBM", flush=True)
    print("=" * 72, flush=True)
    print(f"  网格            : {nx} × {ny} × {nz}", flush=True)
    print(f"  τ               : {tau}", flush=True)
    print(f"  ν = (τ−½)/3     : {nu:.6f}", flush=True)
    print(f"  c_s = 1/√3      : {cs:.6f}", flush=True)
    print(f"  扰动幅度 δρ     : {delta_rho}", flush=True)
    print(f"  角频率 ω        : {omega}", flush=True)
    print(f"  波长 λ          : {lam:.2f}", flush=True)
    print(f"  周期 T          : {period:.2f} 步", flush=True)
    print(f"  波数 k = ω/c_s  : {k:.6f} rad/格子", flush=True)
    print(f"  壁面位置 L      : {L} (半程反弹回)", flush=True)
    print(f"  k·L             : {k * L:.4f} (= {k * L / math.pi:.4f}·π)",
          flush=True)
    print(f"  衰减率 Γ = νk²  : {gamma_damp:.6e} /步", flush=True)
    print(f"  衰减长度 c_s/Γ  : {cs / gamma_damp:.1f} 格子", flush=True)
    print(f"  衰减因子 e^(-ΓL/c_s): {damp_factor:.4f}", flush=True)
    print(f"  边界条件        : 左=硬源(无反射), 右=反弹回, 上下=周期",
          flush=True)
    print(f"  监测点          : {monitor_x}", flush=True)
    print(f"  步数            : {n_steps}", flush=True)
    print(f"  测量窗口        : {measure_start}–{n_steps} ({measure_window} 步)",
          flush=True)
    print(f"  振荡次数        : {n_oscillations:.1f}", flush=True)
    print(f"  设备            : {dev}", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)
    print("  解析解 (含粘性衰减):", flush=True)
    print("    入射波: p_i = A·e^{-Γx/c_s}·sin(ωt - kx)", flush=True)
    print("    反射波: p_r = A·e^{Γ(2L-x)/c_s}·sin(ωt + k(x-L))",
          flush=True)
    print("    驻波振幅: |p(x)| = √[a_i² + a_r² + 2·a_i·a_r·cos(2k(x-L))]",
          flush=True)
    print(f"    壁面入射振幅: A·e^(-ΓL/c_s) = {p_incident_wall:.6e}",
          flush=True)
    print(f"    壁面驻波振幅: 2×入射 = {p_wall_expected:.6e} (压力加倍)",
          flush=True)
    print(flush=True)

    # ---- 日志表头 ----
    hdr = f"  {'步数':>6}"
    for lbl in monitor_labels:
        hdr += f"  {lbl:>14}"
    print(hdr, flush=True)
    print("  " + "-" * (6 + 16 * len(monitor_x)), flush=True)

    # ---- 时间循环 ----
    has_nan = False
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            # === 碰撞 ===
            f = collide_bgk3d(f, tau)

            # === 流动 (周期, torch.roll) ===
            f = stream3d(f)

            # === 右壁面反弹回 ===
            f = apply_right_wall_bounceback(f)

            # === 左边界源 ===
            t = float(step)
            rho_src_val = 1.0 + delta_rho * math.sin(omega * t)
            ux_src_val = cs * delta_rho * math.sin(omega * t)
            f = apply_source(f, rho_src_val, ux_src_val, ny, nz, dev)

            # === 振幅测量 (运行 max/min) ===
            if step >= measure_start:
                rho_1d = f[:, 0, 0, :].sum(dim=0).double()
                rho_max = torch.maximum(rho_max, rho_1d)
                rho_min = torch.minimum(rho_min, rho_1d)

            # === 监测点 ===
            for i, mx in enumerate(monitor_x):
                monitor_series[step, i] = f[:, 0, 0, mx].sum().double()

            # === 定期日志 + NaN 检查 ===
            if step % log_every == 0 or step == n_steps:
                vals = [monitor_series[step, i].item()
                        for i in range(len(monitor_x))]
                line = f"  {step:>6}"
                for v in vals:
                    line += f"  {v - 1.0:14.8f}"
                print(line, flush=True)

                if torch.isnan(f).any().item() or torch.isinf(f).any().item():
                    has_nan = True
                    print(f"\n  ✗ 检测到 NaN/Inf (步 {step})!", flush=True)
                    break

    # ---- 转换为 numpy ----
    monitor_np = monitor_series.cpu().numpy()

    # ---- 提前退出 ----
    if has_nan:
        print("=" * 72, flush=True)
        print("  ✗ FAIL — 模拟发散 (NaN/Inf)", flush=True)
        print("=" * 72, flush=True)
        return {"pass": False, "error": "NaN/Inf"}

    # ======================================================================= #
    # 分析
    # ======================================================================= #
    amp_profile = ((rho_max - rho_min) / 2.0).cpu().numpy()

    # ---- 含衰减的解析振幅 ----
    x_arr = np.arange(nx, dtype=np.float64)
    amp_ana = damped_standing_wave_amplitude(
        x_arr, delta_rho, gamma_damp, k, cs, L
    )

    # ---- 1. 驻波形态误差 ----
    # 排除源附近区域 (x < 10)，源边界会扭曲振幅剖面
    exclude_src = 10
    mask = x_arr >= exclude_src
    amp_diff = amp_profile[mask] - amp_ana[mask]
    l2_num = np.sqrt(np.sum(amp_diff ** 2))
    l2_den = np.sqrt(np.sum(amp_ana[mask] ** 2))
    pattern_error = (l2_num / max(l2_den, 1e-15)) * 100.0

    # ---- 2. 压力加倍 ----
    # 壁面处: |p_wall| / |p_incident_wall| ≈ 2
    # |p_incident_wall| = A·exp(-Γ·L/c_s) (入射波到达壁面时的振幅)
    # 压力加倍是壁面局部效应: 总压 = 入射 + 反射 = 2 × 入射
    p_wall = amp_profile[nx - 1]
    doubling_ratio = p_wall / p_incident_wall
    doubling_error = abs(doubling_ratio - 2.0) / 2.0 * 100.0

    # ---- 3. 反射系数 (驻波比法) ----
    # |R| = (S_max - S_min) / (S_max + S_min)
    # 排除源附近区域
    amp_excl = amp_profile[exclude_src:]
    s_max = float(np.max(amp_excl))
    s_min = float(np.min(amp_excl))
    if s_max + s_min > 1e-15:
        R_coeff = (s_max - s_min) / (s_max + s_min)
    else:
        R_coeff = 0.0
    R_error = abs(R_coeff - 1.0) * 100.0

    # ======================================================================= #
    # 结果
    # ======================================================================= #
    print(flush=True)
    print("=" * 72, flush=True)
    print("  结果", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)

    # ---- 1. 驻波形态 ----
    print("  ┌─── 1. 驻波形态 ──────────────────────────────────────────┐",
          flush=True)
    print(f"  │  解析: |p(x)| = √[a_i²+a_r²+2·a_i·a_r·cos(2k(x-L))]  │",
          flush=True)
    print(f"  │  (含粘性衰减 Γ = νk² = {gamma_damp:.2e})              │",
          flush=True)
    print(f"  │  L2 相对误差: {pattern_error:.2f}%                              │",
          flush=True)
    print(f"  │  阈值:        10%                                        │",
          flush=True)
    status1 = "✓ PASS" if pattern_error < 10.0 else "✗ FAIL"
    print(f"  │  状态:        {status1}                                        │",
          flush=True)
    print("  └──────────────────────────────────────────────────────────┘",
          flush=True)
    print(flush=True)

    # ---- 2. 压力加倍 ----
    print("  ┌─── 2. 压力加倍 ──────────────────────────────────────────┐",
          flush=True)
    print(f"  │  壁面入射振幅 |p_i(L)| = {p_incident_wall:.6e}          │",
          flush=True)
    print(f"  │  壁面总振幅   |p_wall|  = {p_wall:.6e}          │",
          flush=True)
    print(f"  │  比值 |p_wall|/|p_i(L)| = {doubling_ratio:.4f}              │",
          flush=True)
    print(f"  │  误差: {doubling_error:.2f}%                              │",
          flush=True)
    print(f"  │  阈值: 10%                                        │",
          flush=True)
    status2 = "✓ PASS" if doubling_error < 10.0 else "✗ FAIL"
    print(f"  │  状态: {status2}                                        │",
          flush=True)
    print("  └──────────────────────────────────────────────────────────┘",
          flush=True)
    print(flush=True)

    # ---- 3. 反射系数 ----
    print("  ┌─── 3. 反射系数 ──────────────────────────────────────────┐",
          flush=True)
    print(f"  │  S_max (波腹) = {s_max:.6e}                        │",
          flush=True)
    print(f"  │  S_min (波节) = {s_min:.6e}                        │",
          flush=True)
    print(f"  │  |R| = (S_max - S_min)/(S_max + S_min) = {R_coeff:.4f}  │",
          flush=True)
    print(f"  │  误差: {R_error:.2f}%                                        │",
          flush=True)
    print(f"  │  阈值: 5%                                          │",
          flush=True)
    status3 = "✓ PASS" if R_error < 5.0 else "✗ FAIL"
    print(f"  │  状态: {status3}                                        │",
          flush=True)
    print("  └──────────────────────────────────────────────────────────┘",
          flush=True)
    print(flush=True)

    # ---- 总结 ----
    all_pass = (
        pattern_error < 10.0
        and doubling_error < 10.0
        and R_error < 5.0
    )
    print("=" * 72, flush=True)
    if all_pass:
        print("  ✓✓ 全部通过 — 声波反射基准测试 PASS", flush=True)
        print(f"     驻波形态误差 = {pattern_error:.2f}% < 10%", flush=True)
        print(f"     压力加倍误差 = {doubling_error:.2f}% < 10% "
              f"(比值 {doubling_ratio:.4f})", flush=True)
        print(f"     反射系数误差 = {R_error:.2f}% < 5% "
              f"(|R| = {R_coeff:.4f})", flush=True)
    else:
        print("  ✗ 未通过 — 声波反射基准测试 FAIL", flush=True)
        print(f"     驻波形态误差 = {pattern_error:.2f}% "
              f"({'✓' if pattern_error < 10.0 else '✗'} 阈值 10%)", flush=True)
        print(f"     压力加倍误差 = {doubling_error:.2f}% "
              f"({'✓' if doubling_error < 10.0 else '✗'} 阈值 10%)",
              flush=True)
        print(f"     反射系数误差 = {R_error:.2f}% "
              f"({'✓' if R_error < 5.0 else '✗'} 阈值 5%)", flush=True)
    print("=" * 72, flush=True)

    # ======================================================================= #
    # ASCII 图
    # ======================================================================= #

    # ---- 振幅剖面 ----
    print(flush=True)
    print("  压力振幅剖面 (█ = 数值, · = 解析含衰减):", flush=True)
    ascii_plot_1d(
        amp_profile, amp_ana,
        width=72, height=14,
        title="|p(x)| 振幅 vs x",
        x_max=nx,
    )

    # ---- 监测点时间序列 ----
    print(flush=True)
    print("  监测点压力时间序列 (█ = ρ−1):", flush=True)
    for i, (mx, lbl) in enumerate(zip(monitor_x, monitor_labels)):
        print(flush=True)
        ascii_plot_1d(
            monitor_np[:, i] - 1.0,
            width=72, height=8,
            title=f"ρ'(t) at {lbl} (x={mx})",
            x_max=n_steps,
        )

    return {
        "pattern_error": pattern_error,
        "doubling_ratio": doubling_ratio,
        "doubling_error": doubling_error,
        "R": R_coeff,
        "R_error": R_error,
        "s_max": s_max,
        "s_min": s_min,
        "p_wall": p_wall,
        "p_incident_wall": p_incident_wall,
        "gamma_damp": gamma_damp,
        "damp_factor": damp_factor,
        "pass": all_pass,
    }


# =========================================================================== #
# CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="声波反射基准测试 (D3Q19 BGK LBM)"
    )
    parser.add_argument(
        "--nx", type=int, default=400,
        help="x 方向网格数 (默认 400)",
    )
    parser.add_argument(
        "--ny", type=int, default=4,
        help="y 方向网格数 (默认 4)",
    )
    parser.add_argument(
        "--nz", type=int, default=1,
        help="z 方向网格数 (默认 1, 二维)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.8,
        help="弛豫时间 τ (默认 0.8)",
    )
    parser.add_argument(
        "--delta", type=float, default=0.001,
        help="密度扰动振幅 δρ (默认 0.001)",
    )
    parser.add_argument(
        "--omega", type=float, default=0.1,
        help="角频率 ω (默认 0.1)",
    )
    parser.add_argument(
        "--steps", type=int, default=2000,
        help="时间步数 (默认 2000)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="设备: 'cpu' 或 'cuda' (默认 cpu)",
    )
    parser.add_argument(
        "--log-every", type=int, default=200,
        help="打印间隔 (默认 200)",
    )
    parser.add_argument(
        "--measure-window", type=int, default=500,
        help="末尾测量窗口步数 (默认 500)",
    )
    args = parser.parse_args()

    run_reflection_benchmark(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        delta_rho=args.delta,
        omega=args.omega,
        n_steps=args.steps,
        device=args.device,
        log_every=args.log_every,
        measure_window=args.measure_window,
    )


if __name__ == "__main__":
    main()

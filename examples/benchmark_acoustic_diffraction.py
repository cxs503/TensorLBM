#!/usr/bin/env python
"""声学衍射基准测试 — D3Q19 BGK LBM

平面声波通过刚性壁面上的狭缝衍射，衍射波图案与解析解进行比较。

物理设置
--------
  - 网格: nx=400, ny=300, nz=1 (2D D3Q19 BGK)
  - 刚性壁面: x=nx/3处的垂直壁面，中心开有宽度为a的狭缝
  - 壁面 = 反弹边界 (bounce-back)
  - 平面波源: 左边界 x=0,
      rho = 1 + δρ·sin(ωt),  ux = cs·δρ·sin(ωt)  (右行波)
  - 海绵层: 右、上、下边界 (宽度50), 目标平衡态 (rho=1, u=0)

解析解
------
远场 (Fraunhofer) 衍射:
  |p(θ)| ∝ |sinc(ka·sin(θ)/2)|,  sinc(x) = sin(x)/x

精确 2D 衍射 (Rayleigh-Sommerfeld 积分):
  p(r,θ) ∝ |∫_{-a/2}^{a/2} H₀⁽¹⁾(k·R(y')) dy'|
  其中 R(y') = √(r² - 2r·y'·sin(θ) + y'²)

  Fraunhofer 近似在菲涅尔数 F=(a/2)²/(λr)≪1 时有效。
  当 F 较大时 (近场)，使用精确积分更准确。

注: 为产生旁瓣需 ka > 2π，即 a > λ/π ≈ 36.3 (ω=0.1时)。
    默认 a=60 使 ka≈10.4，产生清晰的主瓣和第一旁瓣。
    任务指定 a=20 时 ka≈3.5 < 2π，无旁瓣，仅可验证主瓣形状。

验证
----
  1. 在固定距离r处测量不同角度θ的压力幅值 (Fourier投影提取ω分量)
  2. 与精确2D衍射积分 (Rayleigh-Sommerfeld) 比较
  3. 同时展示Fraunhofer sinc图案作为参考
  4. 目标: 主瓣和第一旁瓣误差 < 15%

运行
----
    PYTHONPATH=src python examples/benchmark_acoustic_diffraction.py \\
        --device cpu --steps 1200
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
from scipy.special import hankel1

# --------------------------------------------------------------------------- #
# Make tensorlbm importable when running from the repo root.
# --------------------------------------------------------------------------- #
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

# =========================================================================== #
# Constants
# =========================================================================== #

CS2 = 1.0 / 3.0          # lattice speed of sound squared
CS = math.sqrt(CS2)      # c_s = 1/√3 ≈ 0.5774


# =========================================================================== #
# Helper functions
# =========================================================================== #

def _sinc(x: float) -> float:
    """sinc(x) = sin(x)/x, with sinc(0)=1."""
    if abs(x) < 1e-10:
        return 1.0
    return math.sin(x) / x


def _exact_diffraction_2d(
    k: float, r: float, theta: float, a: float, n_points: int = 500
) -> float:
    """Exact 2D diffraction amplitude (Rayleigh-Sommerfeld integral).

    Uses the 2D Green's function (Hankel function H₀⁽¹⁾) for the Helmholtz
    equation.  The diffracted field from a uniformly illuminated slit of
    width *a* (Kirchhoff approximation, aperture field = 2·U_incident) is::

        p(r, θ) ∝ ∫_{-a/2}^{a/2} H₀⁽¹⁾(k·R(y')) dy'

    where ``R(y') = √(r² − 2r·y'·sin(θ) + y'²)`` is the distance from
    source point *y'* on the slit to the observation point at *(r, θ)*.

    In the far field (Fresnel number F ≪ 1) this reduces to the
    Fraunhofer sinc pattern.  In the near field the exact integral is
    significantly different.

    Returns the absolute amplitude |p| (arbitrary normalisation).
    """
    y_prime = np.linspace(-a / 2.0, a / 2.0, n_points)
    sin_t = math.sin(theta)
    # Distance from each source point on the slit to the observer
    R = np.sqrt(r ** 2 - 2.0 * r * y_prime * sin_t + y_prime ** 2)
    R = np.maximum(R, 0.5)  # avoid singularity at R = 0
    integrand = hankel1(0, k * R)
    integral = np.trapezoid(integrand, y_prime)
    return abs(float(integral))


def _ascii_plot_angular(
    angles_deg: list[int],
    lbm_norm: np.ndarray,
    exact_norm: np.ndarray,
    sinc_norm: np.ndarray,
    width: int = 72,
    height: int = 14,
) -> None:
    """Print angular diffraction pattern as ASCII art.

    ``█`` = LBM numerical, ``·`` = exact (Rayleigh-Sommerfeld),
    ``-`` = sinc (Fraunhofer), ``╬`` = overlap.
    """
    n = len(angles_deg)
    ymin = 0.0
    ymax = max(float(lbm_norm.max()), float(exact_norm.max()),
               float(sinc_norm.max()), 1.0)
    span = max(ymax - ymin, 1e-12)

    def _resample(y: np.ndarray) -> list[float]:
        idx = np.minimum((np.arange(width) * n / width).astype(int), n - 1)
        return [float(y[i]) for i in idx]

    lbm_s = _resample(lbm_norm)
    exact_s = _resample(exact_norm)
    sinc_s = _resample(sinc_norm)

    print("  衍射图案 (█=LBM, ·=精确解, -=sinc):", flush=True)
    half = span / (2 * height)
    for row in range(height, 0, -1):
        y_val = ymin + (row - 0.5) * span / height
        line: list[str] = []
        for col in range(width):
            ch = " "
            if abs(sinc_s[col] - y_val) <= half:
                ch = "-"
            if abs(exact_s[col] - y_val) <= half:
                ch = "·" if ch == " " else ch
            if abs(lbm_s[col] - y_val) <= half:
                ch = "█" if ch == " " else "╬"
            line.append(ch)
        y_lbl = ymin + row * span / height
        print(f"  {y_lbl:7.4f} |" + "".join(line) + "|", flush=True)

    print(f"  {'':7} +" + "-" * width + "+", flush=True)

    # Angle labels
    lbl = [" "] * width
    step_lbl = max(width // 7, 1)
    for i in range(0, width, step_lbl):
        xv = angles_deg[min(i * n // width, n - 1)]
        for j, ch in enumerate(str(xv)):
            if i + j < width:
                lbl[i + j] = ch
    print(f"  {'':7}  " + "".join(lbl) + " (度)", flush=True)


# =========================================================================== #
# Main simulation
# =========================================================================== #

def run_diffraction_benchmark(
    nx: int = 400,
    ny: int = 300,
    nz: int = 1,
    tau: float = 0.55,
    slit_width: int = 60,
    delta_rho: float = 0.001,
    omega: float = 0.1,
    steps: int = 1200,
    device: str = "cpu",
    log_every: int = 200,
    sponge_width: int = 50,
    monitor_r: int = 110,
    source_width: int = 3,
) -> dict:
    """Run the acoustic diffraction benchmark.

    Returns dict with amplitudes, errors, and pass/fail status.
    """
    dev = torch.device(device)
    cs = CS
    cs2 = CS2

    wall_x = nx // 3          # vertical wall position
    slit_cy = ny // 2         # slit centre in y
    slit_half = slit_width // 2
    slit_lo = slit_cy - slit_half
    slit_hi = slit_cy + slit_half

    k = omega / cs
    lam = 2.0 * math.pi * cs / omega
    ka = k * slit_width

    # First zero angle of the sinc pattern
    if ka > 2.0 * math.pi:
        theta_zero_deg = math.degrees(math.asin(min(2.0 * math.pi / ka, 1.0)))
    else:
        theta_zero_deg = 90.0
    has_side_lobe = ka > 2.0 * math.pi

    # Fresnel number (half-width convention): F = (a/2)² / (λr)
    fresnel_num = (slit_width / 2.0) ** 2 / (lam * monitor_r)

    # --- Monitor angles (degrees) ---
    max_angle = 60
    angles_deg = list(range(-max_angle, max_angle + 1, 10))
    angles_rad = [math.radians(a) for a in angles_deg]
    n_mon = len(angles_deg)

    # Monitor positions at fixed distance r from slit centre
    monitor_pts: list[tuple[int, int]] = []
    for theta in angles_rad:
        mx = int(round(wall_x + monitor_r * math.cos(theta)))
        my = int(round(slit_cy + monitor_r * math.sin(theta)))
        # Clamp to stay inside grid and outside sponge / wall
        mx = max(source_width + 1, min(nx - sponge_width - 2, mx))
        my = max(sponge_width + 2, min(ny - sponge_width - 3, my))
        monitor_pts.append((mx, my))

    # --- Initialization ---
    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # Grid coordinates
    _zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing="ij")

    # --- Wall mask: vertical wall at x=wall_x with slit opening ---
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall_mask[:, :, wall_x] = True
    wall_mask[:, slit_lo:slit_hi, wall_x] = False

    # --- Sponge layer (right, top, bottom only — left is source) ---
    dist_right = (nx - 1 - xx).float()
    dist_top = (ny - 1 - yy).float()
    dist_bottom = yy.float()
    dist_sponge = torch.minimum(dist_right, torch.minimum(dist_top, dist_bottom))
    damping = torch.where(
        dist_sponge < sponge_width,
        torch.sin(math.pi * dist_sponge / (2.0 * sponge_width)) ** 2,
        torch.ones_like(dist_sponge),
    )

    feq_target = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)
    opp = OPPOSITE.to(dev)

    # --- Print header ---
    print("=" * 72, flush=True)
    print("  声学衍射基准测试 — D3Q19 BGK LBM", flush=True)
    print("  (平面波通过狭缝衍射, 精确2D衍射积分对比)", flush=True)
    print("=" * 72, flush=True)
    print(f"  网格: {nx} × {ny} × {nz}", flush=True)
    print(f"  壁面位置: x={wall_x}", flush=True)
    print(f"  狭缝宽度 a: {slit_width}", flush=True)
    print(f"  狭缝范围: y=[{slit_lo}, {slit_hi})", flush=True)
    print(f"  声速 cs: {cs:.6f}", flush=True)
    print(f"  角频率 ω: {omega}", flush=True)
    print(f"  波长 λ: {lam:.2f}", flush=True)
    print(f"  波数 k: {k:.6f}", flush=True)
    print(f"  ka: {ka:.4f}", flush=True)
    print(f"  第一零点角度: θ={theta_zero_deg:.1f}°", flush=True)
    print(f"  菲涅尔数 F: {fresnel_num:.4f} (<<1为远场)", flush=True)
    if fresnel_num < 0.1:
        print(f"  → 远场条件满足, Fraunhofer近似有效", flush=True)
    else:
        print(f"  → 近场效应显著, 使用精确Rayleigh-Sommerfeld积分", flush=True)
    print(f"  密度扰动 δρ: {delta_rho}", flush=True)
    print(f"  τ: {tau}", flush=True)
    print(f"  海绵层宽度: {sponge_width}", flush=True)
    print(f"  监测距离 r: {monitor_r}", flush=True)
    print(f"  监测角度: {angles_deg}", flush=True)
    print(f"  步数: {steps}", flush=True)
    print(f"  设备: {dev}", flush=True)
    print("=" * 72, flush=True)

    # --- Measurement timing ---
    # Wave must travel source→wall + slit→farthest monitor, plus transient
    travel_steps = (wall_x + monitor_r) / cs
    measure_start = max(500, int(1.5 * travel_steps))
    measure_start = min(measure_start, steps - 100)
    print(f"  测量起始步: {measure_start}  (波传播≈{travel_steps:.0f}步)", flush=True)
    print(flush=True)

    # Print monitor positions
    print("  监测点位置:", flush=True)
    for i, (a_deg, (mx, my)) in enumerate(zip(angles_deg, monitor_pts)):
        print(f"    θ={a_deg:+3d}°: (x={mx}, y={my})", flush=True)
    print(flush=True)

    # --- Time series storage ---
    p_ts: list[list[float]] = [[] for _ in range(n_mon)]
    t_meas: list[int] = []

    # --- Logging header ---
    hdr = f"  {'步数':>6s}  {'ρ_max':>10s}"
    for a_deg in angles_deg:
        hdr += f"  p(θ={a_deg:+d}°)"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    # --- Time loop ---
    has_nan = False
    for step in range(1, steps + 1):
        # === Collision ===
        f = collide_bgk3d(f, tau)

        # === Streaming (periodic via torch.roll) ===
        f = stream3d(f)

        # === Bounce-back at wall (full-way) ===
        # At wall cells, swap f[q] ↔ f[opp[q]]  (reflects all populations)
        f = torch.where(wall_mask.unsqueeze(0), f[opp], f)

        # === Source: set equilibrium at left boundary ===
        # Right-travelling plane wave:  ρ' = δρ sin(ωt),  ux = cs·δρ sin(ωt)
        rho_src = 1.0 + delta_rho * math.sin(omega * step)
        ux_src = cs * delta_rho * math.sin(omega * step)
        rho_s = torch.full((nz, ny, source_width), rho_src, device=dev)
        ux_s = torch.full((nz, ny, source_width), ux_src, device=dev)
        uy_s = torch.zeros((nz, ny, source_width), device=dev)
        uz_s = torch.zeros((nz, ny, source_width), device=dev)
        feq_s = equilibrium3d(rho_s, ux_s, uy_s, uz_s, device=dev)
        f[:, :, :, :source_width] = feq_s

        # === Sponge layer: blend toward TARGET equilibrium ===
        f = feq_target + (f - feq_target) * damping

        # === NaN check (periodic) ===
        if step % 100 == 0:
            if torch.isnan(f).any().item() or torch.isinf(f).any().item():
                has_nan = True
                print(f"\n  ✗ 检测到NaN/Inf (步数 {step})!", flush=True)
                break

        # === Measurement: record pressure at monitors ===
        if step >= measure_start:
            rho_m = f.sum(dim=0)  # density only (faster than full macroscopic3d)
            t_meas.append(step)
            for i, (mx, my) in enumerate(monitor_pts):
                p = float((rho_m[0, my, mx] - 1.0) * cs2)
                p_ts[i].append(p)

        # === Logging ===
        if step % log_every == 0 or step == steps:
            rho_m = f.sum(dim=0)
            rho_max = float((rho_m - rho0).abs().max())
            pvals = []
            for i, (mx, my) in enumerate(monitor_pts):
                pvals.append(float((rho_m[0, my, mx] - 1.0) * cs2))
            print(f"  {step:6d}  {rho_max:10.6f}  "
                  + "  ".join(f"{v:+.6f}" for v in pvals), flush=True)

    # --- Early exit on divergence ---
    if has_nan:
        print("=" * 72, flush=True)
        print("  ✗ FAIL — 模拟发散 (NaN/Inf)", flush=True)
        print("=" * 72, flush=True)
        return {"pass": False, "error": "NaN/Inf detected"}

    if len(t_meas) == 0:
        print("=" * 72, flush=True)
        print("  ✗ FAIL — 无测量数据 (步数不足)", flush=True)
        print("=" * 72, flush=True)
        return {"pass": False, "error": "No measurement data"}

    # =================================================================== #
    # Analysis: Fourier projection to extract amplitude at frequency ω
    # =================================================================== #
    t_meas_np = np.array(t_meas, dtype=np.float64)
    sin_wt = np.sin(omega * t_meas_np)
    cos_wt = np.cos(omega * t_meas_np)
    N = len(t_meas_np)

    amplitudes = np.zeros(n_mon)
    for i in range(n_mon):
        p_series = np.array(p_ts[i], dtype=np.float64)
        c_sin = (2.0 / N) * np.sum(p_series * sin_wt)
        c_cos = (2.0 / N) * np.sum(p_series * cos_wt)
        amplitudes[i] = math.sqrt(c_sin ** 2 + c_cos ** 2)

    # Normalize by on-axis amplitude (θ=0)
    idx_0 = angles_deg.index(0)
    amp_0 = amplitudes[idx_0]
    amp_norm = amplitudes / max(amp_0, 1e-15)

    # --- Analytical solutions ---
    # 1. Fraunhofer sinc pattern (far-field approximation)
    sinc_vals = np.array([abs(_sinc(ka * math.sin(t) / 2.0)) for t in angles_rad])
    sinc_norm = sinc_vals / max(sinc_vals.max(), 1e-15)

    # 2. Exact 2D Rayleigh-Sommerfeld diffraction integral
    exact_vals = np.array([
        _exact_diffraction_2d(k, monitor_r, t, slit_width) for t in angles_rad
    ])
    exact_norm = exact_vals / max(exact_vals.max(), 1e-15)

    # =================================================================== #
    # Results
    # =================================================================== #
    print(flush=True)
    print("=" * 72, flush=True)
    print("  衍射图案对比 (归一化压力幅值)", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'θ(°)':>6s}  {'LBM':>7s}  {'精确解':>7s}  {'sinc':>7s}  "
          f"{'误差%':>7s}  {'区域':>6s}", flush=True)
    print("-" * 72, flush=True)

    main_lobe_errs: list[float] = []
    side_lobe_errs: list[float] = []

    for i, a_deg in enumerate(angles_deg):
        lbm_val = amp_norm[i]
        exact_val = exact_norm[i]
        sinc_val = sinc_norm[i]
        # Error relative to exact solution
        if exact_val > 0.02:
            err = abs(lbm_val - exact_val) / exact_val * 100.0
        else:
            err = float("nan")

        abs_a = abs(a_deg)
        if abs_a < theta_zero_deg:
            region = "主瓣"
            if not math.isnan(err):
                main_lobe_errs.append(err)
        else:
            region = "旁瓣"
            if not math.isnan(err):
                side_lobe_errs.append(err)

        err_str = f"{err:7.1f}" if not math.isnan(err) else "   N/A"
        print(f"  {a_deg:+6d}  {lbm_val:7.4f}  {exact_val:7.4f}  "
              f"{sinc_val:7.4f}  {err_str}  {region:>6s}", flush=True)

    # --- Summary ---
    main_avg = sum(main_lobe_errs) / len(main_lobe_errs) if main_lobe_errs else 0.0
    main_max = max(main_lobe_errs) if main_lobe_errs else 0.0
    side_avg = sum(side_lobe_errs) / len(side_lobe_errs) if side_lobe_errs else 0.0
    side_max = max(side_lobe_errs) if side_lobe_errs else 0.0

    print(flush=True)
    print(f"  主瓣: 平均误差={main_avg:.1f}%, 最大误差={main_max:.1f}%", flush=True)
    if side_lobe_errs:
        print(f"  旁瓣: 平均误差={side_avg:.1f}%, 最大误差={side_max:.1f}%",
              flush=True)
    else:
        print(f"  旁瓣: 无旁瓣监测点 (ka={ka:.2f} ≤ 2π={2 * math.pi:.2f})",
              flush=True)

    # --- PASS / FAIL ---
    main_ok = main_max < 15.0
    side_ok = (side_max < 15.0) if side_lobe_errs else True

    print(flush=True)
    print(f"  [{'PASS' if main_ok else 'FAIL'}] "
          f"主瓣误差 < 15%: 最大误差={main_max:.1f}%", flush=True)
    if side_lobe_errs:
        print(f"  [{'PASS' if side_ok else 'FAIL'}] "
              f"旁瓣误差 < 15%: 最大误差={side_max:.1f}%", flush=True)
    else:
        print(f"  [N/A] 旁瓣验证: 无旁瓣 (ka≤2π)", flush=True)

    all_pass = main_ok and side_ok
    print(flush=True)
    if all_pass:
        print("  ✓ PASS — 声学衍射基准测试通过", flush=True)
    else:
        print("  ✗ FAIL — 声学衍射基准测试未通过", flush=True)
    print("=" * 72, flush=True)

    # --- ASCII plot ---
    print(flush=True)
    _ascii_plot_angular(angles_deg, amp_norm, exact_norm, sinc_norm)

    return {
        "amplitudes": amplitudes.tolist(),
        "amp_norm": amp_norm.tolist(),
        "exact_norm": exact_norm.tolist(),
        "sinc_norm": sinc_norm.tolist(),
        "main_max_err": main_max,
        "side_max_err": side_max,
        "pass": all_pass,
    }


# =========================================================================== #
# CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="声学衍射基准测试 (D3Q19 BGK LBM)"
    )
    parser.add_argument("--nx", type=int, default=400,
                        help="x方向网格数 (默认400)")
    parser.add_argument("--ny", type=int, default=300,
                        help="y方向网格数 (默认300)")
    parser.add_argument("--nz", type=int, default=1,
                        help="z方向网格数 (默认1, 2D)")
    parser.add_argument("--tau", type=float, default=0.55,
                        help="BGK松弛时间τ (默认0.55)")
    parser.add_argument("--slit-width", type=int, default=60,
                        help="狭缝宽度a (默认60, 需a>36才有旁瓣)")
    parser.add_argument("--delta-rho", type=float, default=0.001,
                        help="密度扰动幅值δρ (默认0.001)")
    parser.add_argument("--omega", type=float, default=0.1,
                        help="角频率ω (默认0.1)")
    parser.add_argument("--steps", type=int, default=1200,
                        help="时间步数 (默认1200)")
    parser.add_argument("--device", default="cpu",
                        help="设备: 'cpu'或'cuda' (默认cpu)")
    parser.add_argument("--log-every", type=int, default=200,
                        help="日志间隔 (默认200)")
    parser.add_argument("--sponge-width", type=int, default=50,
                        help="海绵层宽度 (默认50)")
    parser.add_argument("--monitor-r", type=int, default=110,
                        help="监测距离r (默认110)")
    args = parser.parse_args()

    run_diffraction_benchmark(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        slit_width=args.slit_width,
        delta_rho=args.delta_rho,
        omega=args.omega,
        steps=args.steps,
        device=args.device,
        log_every=args.log_every,
        sponge_width=args.sponge_width,
        monitor_r=args.monitor_r,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Benchmark: 2D Cavity Resonance — D3Q19 BGK LBM.

Validates the acoustic eigenfrequencies of a closed 2-D cavity with rigid
(bounce-back) walls.  An initial density perturbation matching a specific
eigenmode excites that resonance; the pressure at a monitoring point
oscillates at the eigenfrequency.

Setup
-----
  * 2-D cavity via D3Q19 with nz = 1, nx = ny = 200
  * ALL four walls are bounce-back (rigid, no-slip)
  * Initial density matching mode (m,n):
      Mode (1,1):  ρ(x,y) = 1 + δ·cos(πx/Lx)·cos(πy/Ly)
      Mode (2,0):  ρ(x,y) = 1 + δ·cos(2πx/Lx)
    with δ = 0.001, Lx = nx, Ly = ny
  * Initial velocity u = 0  (excites the standing-wave mode naturally)
  * BGK collision, τ = 0.8  (low viscosity for sharp resonance)
  * No sponge layer (closed cavity)

Analytical eigenfrequencies (2-D rigid-wall cavity)
---------------------------------------------------
  f_mn = (c_s / 2) · √((m/Lx)² + (n/Ly)²)

  Mode (1,1):  f₁₁ = c_s·√2 / (2L)
  Mode (2,0):  f₂₀ = c_s / L

Method
------
  1. Run simulation for ~2000 steps
  2. Record pressure (density) at a mode antinode every step
  3. FFT of pressure time series → dominant frequency
  4. Compare with analytical eigenfrequency

Validation
----------
  PASS if |f_lbm − f_analytical| / f_analytical < 5 %

Run
---
    PYTHONPATH=src python examples/benchmark_cavity_resonance.py \\
        --device cpu --steps 2000

Note on monitoring point
------------------------
For mode (1,1) the cavity centre (L/2, L/2) is a pressure NODE
(cos(π/2)·cos(π/2) = 0).  We therefore monitor at (L/4, L/4), an
antinode with amplitude 0.5·δ.  For mode (2,0) the centre is an
antinode (cos(π) = −1) and is used directly.

Note on bounce-back
-------------------
Half-way bounce-back is used: after periodic streaming (torch.roll),
only the *unknown* populations at each wall (those that wrapped around
from the opposite side) are replaced by their opposites.  This places
the wall half-way between the boundary node and the ghost domain,
giving an effective cavity length L = nx (matching the analytical
formula).  Corner cells where two walls meet are handled by reading
from the pre-bounce distribution, so diagonal populations that are
unknown at both walls are swapped — the standard corner treatment.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Make tensorlbm importable when running from the repo root.
# --------------------------------------------------------------------------- #
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

# On high-core-count machines PyTorch defaults to using *all* cores, which
# adds excessive thread-management overhead for the small 200×200 tensors
# used here.  Capping to 8 threads gives a ~10× speedup.
_DEFAULT_THREADS = min(8, os.cpu_count() or 1)
torch.set_num_threads(_DEFAULT_THREADS)

# =========================================================================== #
# Constants
# =========================================================================== #

CS2 = 1.0 / 3.0          # lattice sound speed squared
CS = math.sqrt(CS2)      # c_s = 1/√3 ≈ 0.5774

# =========================================================================== #
# D3Q19 bounce-back direction tables
# =========================================================================== #
# After periodic streaming (torch.roll), populations at a wall that came
# from "outside" the domain (wrapped around) are unknown.  Half-way
# bounce-back replaces each unknown population with its opposite (which
# streamed from the interior and is known).
#
# D3Q19 lattice (cx, cy, cz):
#   0:(0,0,0)   1:(1,0,0)   2:(-1,0,0)   3:(0,1,0)   4:(0,-1,0)
#   5:(0,0,1)   6:(0,0,-1)  7:(1,1,0)    8:(-1,-1,0)  9:(1,-1,0)
#  10:(-1,1,0) 11:(1,0,1)  12:(-1,0,-1) 13:(1,0,-1) 14:(-1,0,1)
#  15:(0,1,1)  16:(0,-1,-1)17:(0,1,-1)  18:(0,-1,1)
#
# OPPOSITE = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]

# Left wall (x=0): unknown dirs have cx > 0
_LEFT_UNKNOWN = [1, 7, 9, 11, 13]
_LEFT_KNOWN   = [2, 8, 10, 12, 14]   # OPPOSITE[_LEFT_UNKNOWN]

# Right wall (x=nx-1): unknown dirs have cx < 0
_RIGHT_UNKNOWN = [2, 8, 10, 12, 14]
_RIGHT_KNOWN   = [1, 7, 9, 11, 13]   # OPPOSITE[_RIGHT_UNKNOWN]

# Bottom wall (y=0): unknown dirs have cy > 0
_BOTTOM_UNKNOWN = [3, 7, 10, 15, 17]
_BOTTOM_KNOWN   = [4, 8, 9, 16, 18]  # OPPOSITE[_BOTTOM_UNKNOWN]

# Top wall (y=ny-1): unknown dirs have cy < 0
_TOP_UNKNOWN = [4, 8, 9, 16, 18]
_TOP_KNOWN   = [3, 7, 10, 15, 17]    # OPPOSITE[_TOP_UNKNOWN]


def apply_bounce_back_2d(
    f: torch.Tensor, nx: int, ny: int, nz: int = 1
) -> torch.Tensor:
    """Apply half-way bounce-back at the four walls of a 2-D cavity (D3Q19).

    After streaming with periodic BCs (torch.roll), the unknown populations
    at each wall (those that wrapped around from the opposite side) are
    replaced by their opposites — the known populations that streamed from
    the interior.  This implements a no-slip wall located half-way between
    the boundary node and the ghost domain, giving an effective cavity
    length L = nx (matching the analytical formula).

    Corner cells where two walls meet are handled naturally: diagonal
    populations that are unknown at both walls are swapped (read from the
    pre-bounce-back distribution), which is the standard corner treatment.
    """
    f_new = f.clone()
    # Left wall (x = 0)
    f_new[_LEFT_UNKNOWN, :, :, 0] = f[_LEFT_KNOWN, :, :, 0]
    # Right wall (x = nx-1)
    f_new[_RIGHT_UNKNOWN, :, :, -1] = f[_RIGHT_KNOWN, :, :, -1]
    # Bottom wall (y = 0)
    f_new[_BOTTOM_UNKNOWN, :, 0, :] = f[_BOTTOM_KNOWN, :, 0, :]
    # Top wall (y = ny-1)
    f_new[_TOP_UNKNOWN, :, -1, :] = f[_TOP_KNOWN, :, -1, :]
    return f_new


# =========================================================================== #
# FFT analysis
# =========================================================================== #

def find_dominant_frequency(
    signal: np.ndarray, n_steps: int
) -> tuple[float, int, float, np.ndarray]:
    """Find the dominant frequency of a time series via FFT.

    Applies a Hann window, computes the rFFT, and refines the peak
    location with parabolic interpolation for sub-bin accuracy.

    Returns
    -------
    (frequency, peak_bin, interpolation_offset, spectrum)
    """
    sig = np.asarray(signal, dtype=np.float64)
    sig = sig - np.mean(sig)                     # remove DC
    window = np.hanning(len(sig))
    sig_w = sig * window
    spectrum = np.abs(np.fft.rfft(sig_w))
    spectrum[0] = 0.0                             # exclude DC
    k_max = int(np.argmax(spectrum))
    # Parabolic interpolation around the peak
    p = 0.0
    if 0 < k_max < len(spectrum) - 1:
        a = spectrum[k_max - 1]
        b = spectrum[k_max]
        g = spectrum[k_max + 1]
        denom = a - 2.0 * b + g
        if abs(denom) > 1e-15:
            p = 0.5 * (a - g) / denom
    f_dom = (k_max + p) / n_steps
    return f_dom, k_max, p, spectrum


# =========================================================================== #
# ASCII visualisation
# =========================================================================== #

def ascii_plot_1d(
    y_num: np.ndarray,
    y_ana: np.ndarray | None = None,
    width: int = 72,
    height: int = 14,
    title: str = "",
    x_max: float | None = None,
) -> None:
    """Print a 1-D profile as ASCII art with optional analytic overlay.

    ``█`` = numerical, ``·`` = analytic, ``╬`` = both.
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
        print(f"  {y_lbl:12.6f} |" + "".join(line) + "|", flush=True)

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


def ascii_plot_spectrum(
    spectrum: np.ndarray,
    n_steps: int,
    f_ana: float,
    width: int = 72,
    height: int = 12,
    title: str = "",
    f_max: float = 0.01,
) -> None:
    """Print the FFT magnitude spectrum as ASCII art.

    ``█`` = |FFT|, ``A`` = analytical frequency marker.
    """
    n_bins = min(len(spectrum), int(f_max * n_steps) + 1)
    mags = spectrum[:n_bins]

    if title:
        print(f"  {title}", flush=True)

    ymax = float(mags.max())
    if ymax < 1e-15:
        print("  (信号过弱)", flush=True)
        return
    span = ymax

    # Resample to *width* columns
    idx = np.minimum(
        (np.arange(width) * n_bins / width).astype(int), n_bins - 1
    )
    mags_s = mags[idx]

    f_ana_col = min(int(f_ana / f_max * width), width - 1)

    half = span / (2 * height)
    for row in range(height, 0, -1):
        y_val = (row - 0.5) * span / height
        line: list[str] = []
        for col in range(width):
            ch = " "
            if abs(mags_s[col] - y_val) <= half:
                ch = "█"
            if col == f_ana_col:
                ch = "A" if ch == " " else "╬"
            line.append(ch)
        y_lbl = row * span / height
        print(f"  {y_lbl:12.4e} |" + "".join(line) + "|", flush=True)

    print(f"  {'':12} +" + "-" * width + "+", flush=True)
    lbl = [" "] * width
    step = max(width // 5, 1)
    for i in range(0, width, step):
        fv = i * f_max / width
        for j, ch in enumerate(f"{fv:.4f}"):
            if i + j < width:
                lbl[i + j] = ch
    print(f"  {'':12}  " + "".join(lbl), flush=True)
    print(f"  (A = 解析频率 {f_ana:.6f})", flush=True)


# =========================================================================== #
# Main simulation
# =========================================================================== #

def run_mode(
    mode: str = "11",
    nx: int = 200,
    ny: int = 200,
    nz: int = 1,
    tau: float = 0.8,
    delta: float = 0.001,
    n_steps: int = 2000,
    device: str = "cpu",
    log_every: int = 200,
) -> dict:
    """Run cavity resonance simulation for a single mode.

    Parameters
    ----------
    mode : "11" or "20"
        Eigenmode to excite.
    nx, ny, nz : int
        Grid dimensions (nz = 1 for 2-D).
    tau : float
        BGK relaxation time (τ > 0.5).
    delta : float
        Initial density perturbation amplitude.
    n_steps : int
        Number of LBM time steps.
    device : str
        ``"cpu"`` or ``"cuda"``.
    log_every : int
        Print a diagnostics row every this many steps.
    """
    dev = torch.device(device)
    cs = CS
    Lx, Ly = float(nx), float(ny)
    nu = (tau - 0.5) / 3.0

    # ---- Mode parameters ----
    if mode == "11":
        m, n = 1, 1
        f_ana = (cs / 2.0) * math.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
        omega_ana = 2.0 * math.pi * f_ana
        # Centre is a NODE for (1,1); monitor at (nx/4, ny/4) — antinode
        mx, my = nx // 4, ny // 4
        mode_label = "模式 (1,1)"
        ic_label = "ρ = 1 + δ·cos(πx/L)·cos(πy/L)"
    elif mode == "20":
        m, n = 2, 0
        f_ana = (cs / 2.0) * math.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
        omega_ana = 2.0 * math.pi * f_ana
        # Centre is an ANTINODE for (2,0)
        mx, my = nx // 2, ny // 2
        mode_label = "模式 (2,0)"
        ic_label = "ρ = 1 + δ·cos(2πx/L)"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    period_ana = 1.0 / f_ana if f_ana > 0 else float("inf")
    n_oscillations = n_steps * f_ana

    # ---- Grid ----
    x_coords = torch.arange(nx, device=dev, dtype=torch.float32)
    y_coords = torch.arange(ny, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (ny, nx)

    # ---- Initial condition ----
    if mode == "11":
        rho_2d = 1.0 + delta * torch.cos(math.pi * xx / Lx) * torch.cos(
            math.pi * yy / Ly
        )
    else:
        rho_2d = 1.0 + delta * torch.cos(2.0 * math.pi * xx / Lx)

    rho_init = rho_2d.view(1, ny, nx).expand(nz, ny, nx).contiguous()
    ux = torch.zeros_like(rho_init)
    uy = torch.zeros_like(rho_init)
    uz = torch.zeros_like(rho_init)

    # ---- Distribution (start at equilibrium) ----
    f = equilibrium3d(rho_init, ux, uy, uz, device=dev)

    # ---- Header ----
    print("=" * 72, flush=True)
    print(f"  腔体共振基准测试 — {mode_label}", flush=True)
    print("=" * 72, flush=True)
    print(f"  网格            : {nx} × {ny} × {nz}", flush=True)
    print(f"  腔体尺寸 Lx×Ly  : {Lx:.0f} × {Ly:.0f}", flush=True)
    print(f"  初始条件        : {ic_label}", flush=True)
    print(f"  扰动幅度 δ      : {delta}", flush=True)
    print(f"  τ               : {tau}", flush=True)
    print(f"  ν = (τ−½)/3     : {nu:.6f}", flush=True)
    print(f"  c_s = 1/√3      : {cs:.6f}", flush=True)
    print(f"  边界条件        : 四壁反弹回 (half-way bounce-back)", flush=True)
    print(f"  监测点          : ({mx}, {my})", flush=True)
    print(f"  步数            : {n_steps}", flush=True)
    print(f"  设备            : {dev}", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)
    print(f"  解析频率 f_ana   : {f_ana:.6f} /步", flush=True)
    print(f"  解析角频率 ω     : {omega_ana:.6f} rad/步", flush=True)
    print(f"  解析周期 T       : {period_ana:.1f} 步", flush=True)
    print(f"  振荡次数 (估计)  : {n_oscillations:.1f}", flush=True)
    if n_oscillations < 2:
        print("  ⚠ 警告: 振荡次数不足 (< 2)，FFT 可能不准确!", flush=True)
    print(flush=True)

    # ---- Logging header ----
    print(f"  {'步数':>6}  {'ρ_监测':>14}  {'ρ′ = ρ−1':>14}", flush=True)
    print("  " + "-" * 42, flush=True)

    # ---- Data storage ----
    # Record density as a GPU/CPU tensor to avoid per-step .item() sync
    # overhead (14× speedup on CPU: 5 → 71 steps/s for 200×200 grid).
    rho_monitor = torch.zeros(n_steps + 1, dtype=torch.float64, device=dev)

    # ---- Initial measurement (density at monitoring point only) ----
    # rho = sum_q f[q]; computing just the single point is ~10⁵× cheaper
    # than a full macroscopic3d call on a 200×200 grid.
    rho_monitor[0] = f[:, 0, my, mx].sum().double()
    print(
        f"  {0:>6}  {rho_monitor[0].item():14.10f}  "
        f"{(rho_monitor[0] - 1.0).item():14.10f}",
        flush=True,
    )

    # ---- Time loop ----
    has_nan = False
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            # === Collision ===
            f = collide_bgk3d(f, tau)

            # === Streaming (periodic via torch.roll) ===
            f = stream3d(f)

            # === Bounce-back at four walls ===
            f = apply_bounce_back_2d(f, nx, ny, nz)

            # === Measurement (single-point density, no .item() sync) ===
            rho_monitor[step] = f[:, 0, my, mx].sum().double()

            # === Periodic logging + NaN check ===
            if step % log_every == 0 or step == n_steps:
                val = rho_monitor[step].item()
                if torch.isnan(f).any().item() or torch.isinf(f).any().item():
                    has_nan = True
                    print(f"\n  ✗ 检测到 NaN/Inf (步 {step})!", flush=True)
                    break
                print(
                    f"  {step:>6}  {val:14.10f}  {val - 1.0:14.10f}",
                    flush=True,
                )

    # ---- Convert to numpy for analysis ----
    rho_monitor = rho_monitor.cpu().numpy()

    # ---- Early exit on divergence ----
    if has_nan:
        print("=" * 72, flush=True)
        print("  ✗ FAIL — 模拟发散 (NaN/Inf)", flush=True)
        print("=" * 72, flush=True)
        return {"pass": False, "mode": mode, "error": "NaN/Inf"}

    # ======================================================================= #
    # FFT Analysis
    # ======================================================================= #
    # Use the full time series (skip t=0 to avoid the initial transient)
    signal = rho_monitor[1:]  # shape (n_steps,)
    f_dom, k_max, p_interp, spectrum = find_dominant_frequency(
        signal, n_steps
    )
    error_pct = (
        abs(f_dom - f_ana) / f_ana * 100.0 if f_ana > 0 else float("inf")
    )

    # ======================================================================= #
    # Results
    # ======================================================================= #
    print(flush=True)
    print("=" * 72, flush=True)
    print("  结果", flush=True)
    print("=" * 72, flush=True)
    print(f"  解析频率 f_ana    : {f_ana:.6f} /步", flush=True)
    print(f"  FFT频率  f_lbm    : {f_dom:.6f} /步", flush=True)
    print(
        f"  FFT峰值 bin       : {k_max} (插值偏移 {p_interp:+.3f})",
        flush=True,
    )
    print(f"  频率分辨率 Δf     : {1.0 / n_steps:.6f} /步", flush=True)
    print(f"  误差              : {error_pct:.2f} %", flush=True)
    print(f"  阈值              : 5.00 %", flush=True)
    status = "✓ PASS" if error_pct < 5.0 else "✗ FAIL"
    print(f"  状态              : {status}", flush=True)
    print("=" * 72, flush=True)

    # ======================================================================= #
    # ASCII plots
    # ======================================================================= #
    print(flush=True)
    print("  压力时间序列 (█ = ρ−1):", flush=True)
    rho_pert = rho_monitor - 1.0
    ascii_plot_1d(
        rho_pert,
        width=72,
        height=12,
        title=f"ρ′(t) at ({mx}, {my})",
        x_max=n_steps,
    )

    print(flush=True)
    print("  FFT 频谱 (█ = |FFT|, A = 解析频率):", flush=True)
    ascii_plot_spectrum(
        spectrum,
        n_steps,
        f_ana,
        width=72,
        height=12,
        title="|FFT(ρ′)| vs 频率",
        f_max=max(0.01, f_ana * 3),
    )

    return {
        "mode": mode,
        "f_analytical": f_ana,
        "f_measured": f_dom,
        "error_pct": error_pct,
        "pass": error_pct < 5.0,
    }


# =========================================================================== #
# CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D cavity resonance benchmark (D3Q19 BGK LBM)"
    )
    parser.add_argument(
        "--nx", type=int, default=200, help="Grid size in x (default 200)"
    )
    parser.add_argument(
        "--ny", type=int, default=200, help="Grid size in y (default 200)"
    )
    parser.add_argument(
        "--nz", type=int, default=1, help="Grid size in z (default 1, 2-D)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.8,
        help="Relaxation time τ (default 0.8)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.001,
        help="Density perturbation amplitude (default 0.001)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of time steps (default 2000)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: 'cpu' or 'cuda' (default cpu)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print interval (default 200)",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["11", "20", "both"],
        help="Mode to test (default 'both')",
    )
    args = parser.parse_args()

    results: list[dict] = []

    if args.mode in ("11", "both"):
        r = run_mode(
            "11",
            args.nx,
            args.ny,
            args.nz,
            args.tau,
            args.delta,
            args.steps,
            args.device,
            args.log_every,
        )
        results.append(r)

    if args.mode in ("20", "both"):
        print(flush=True)
        r = run_mode(
            "20",
            args.nx,
            args.ny,
            args.nz,
            args.tau,
            args.delta,
            args.steps,
            args.device,
            args.log_every,
        )
        results.append(r)

    # ---- Summary ----
    print(flush=True)
    print("=" * 72, flush=True)
    print("  总结", flush=True)
    print("=" * 72, flush=True)
    all_pass = True
    for r in results:
        if "error_pct" in r:
            status = "✓ PASS" if r["pass"] else "✗ FAIL"
            print(
                f"  模式 {r['mode']}:  f_lbm={r['f_measured']:.6f}  "
                f"f_ana={r['f_analytical']:.6f}  "
                f"误差={r['error_pct']:.2f}%  →  {status}",
                flush=True,
            )
            all_pass = all_pass and r["pass"]
        else:
            print(
                f"  模式 {r['mode']}:  ✗ FAIL "
                f"({r.get('error', 'unknown')})",
                flush=True,
            )
            all_pass = False

    print(flush=True)
    if all_pass:
        print("  ✓✓ 全部通过 — 腔体共振基准测试 PASS", flush=True)
    else:
        print("  ✗ 未通过 — 腔体共振基准测试 FAIL", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Benchmark: 1D Acoustic Wave Propagation — D3Q19 BGK LBM.

Validates the sound speed and viscous attenuation of a plane acoustic wave
propagating through a D3Q19 BGK lattice Boltzmann fluid.  The LBM is
inherently weakly compressible: a small density perturbation evolves as a
sound wave travelling at the lattice sound speed  c_s = 1/√3 ≈ 0.5774.

Setup
-----
  * 1-D plane wave via D3Q19 with nz = 1, ny = 4 (x-propagation)
  * Initial density:  ρ(x) = 1 + δ·sin(2πx/λ),  δ = 0.01,  λ = nx/2
  * Initial velocity:  u_x = c_s·δ·sin(2πx/λ)   (right-travelling wave)
  * Periodic BCs in all directions (torch.roll streaming)
  * BGK collision,  τ = 0.8  →  ν = (τ−½)/3 ≈ 0.1

Note on initial conditions
--------------------------
A pure density perturbation with u = 0 decomposes into two counter-
propagating waves (standing wave).  To obtain a single right-travelling
wave whose analytical solution is  ρ(x,t) = 1 + δ·sin(k(x−c_s t)),  the
velocity must be initialised as  u_x = c_s·δ·sin(kx).  This benchmark
uses the travelling-wave initial condition so that the wave peak can be
tracked as a function of time (the phase φ(t) is equivalent to peak
position via  x_peak = (π/2 − φ) / k).

Analytical solution (linearised, lossless)
------------------------------------------
  ρ(x,t) = 1 + δ·sin(2π(x − c_s·t)/λ)

With LBM numerical viscosity the amplitude decays as
  A(t) = δ·exp(−Γ·t)

The temporal attenuation rate for a plane sound wave in a viscous fluid is
  Γ = ν_L · k² / 2
where ν_L = (4/3)ν_shear + ν_bulk is the longitudinal viscosity and
k = 2π/λ is the wavenumber.  For the BGK LBM:
  ν_shear = (τ − ½) / 3        (shear viscosity)
  ν_bulk  = (2/3)(τ − ½) / 3   (bulk viscosity, non-zero for BGK)
  ν_L     = (4/3 + 2/3)(τ − ½) / 3 = (2/3)(τ − ½)
Hence
  Γ = ν_L · k² / 2 = (τ − ½) · k² / 3 = ν_shear · k²

Validation
----------
  1. Sound speed: track the wave phase φ(t) via Fourier projection;
     linear fit  φ vs t  →  c_lbm = −(dφ/dt)/k.
     PASS if  |c_lbm − c_s| / c_s < 5 %.
  2. Attenuation: fit  A(t)  to  δ·exp(−Γ·t)  →  Γ_lbm  vs  Γ_theory.

Run
---
    PYTHONPATH=src python examples/benchmark_acoustic_wave_1d.py \
        --device cpu --steps 2000

References
----------
Latt, J. & Chopard, B. (2006) *Math. Comput. Simul.* 72 117–132
Krüger, T. et al. (2017) *The Lattice Boltzmann Method*, Springer
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

# =========================================================================== #
# Constants
# =========================================================================== #

CS2 = 1.0 / 3.0          # lattice speed of sound squared
CS = math.sqrt(CS2)      # c_s = 1/√3 ≈ 0.5774


# =========================================================================== #
# Fourier-projection helper
# =========================================================================== #

def extract_amp_phase(
    rho_1d: torch.Tensor, k: float, nx: int
) -> tuple[float, float]:
    """Extract amplitude and phase of the k-mode via discrete Fourier projection.

    Given  ρ'(x) = ρ(x) − 1  and a known wavenumber *k*, project onto
    sin(kx) and cos(kx):

        c_sin = (2/nx) Σ ρ'(x_i) sin(k x_i) = A cos(φ)
        c_cos = (2/nx) Σ ρ'(x_i) cos(k x_i) = A sin(φ)

    so that  ρ'(x) ≈ A sin(kx + φ).

    Returns (amplitude A, phase φ in radians).
    """
    rho_p = rho_1d - 1.0
    x = torch.arange(nx, dtype=rho_1d.dtype, device=rho_1d.device)
    sin_kx = torch.sin(k * x)
    cos_kx = torch.cos(k * x)
    c_sin = (2.0 / nx) * (rho_p * sin_kx).sum().item()
    c_cos = (2.0 / nx) * (rho_p * cos_kx).sum().item()
    amp = math.sqrt(c_sin * c_sin + c_cos * c_cos)
    phase = math.atan2(c_cos, c_sin)
    return amp, phase


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
        print(f"  {y_lbl:9.5f} |" + "".join(line) + "|", flush=True)

    print(f"  {'':9} +" + "-" * width + "+", flush=True)
    if x_max is not None:
        lbl = [" "] * width
        step = max(width // 8, 1)
        for i in range(0, width, step):
            xv = int(i * x_max / width)
            for j, ch in enumerate(str(xv)):
                if i + j < width:
                    lbl[i + j] = ch
        print(f"  {'':9}  " + "".join(lbl), flush=True)


# =========================================================================== #
# Main simulation
# =========================================================================== #

def run_acoustic_wave_benchmark(
    nx: int = 200,
    ny: int = 4,
    nz: int = 1,
    tau: float = 0.8,
    delta: float = 0.01,
    n_steps: int = 2000,
    device: str = "cpu",
    log_every: int = 100,
    measure_every: int = 10,
) -> dict:
    """Run the 1-D acoustic wave benchmark.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.  nz = 1 reduces D3Q19 to a 2-D solver; ny is kept
        small (≥ 3) to avoid trivial-single-row artefacts.
    tau : float
        BGK relaxation time (τ > 0.5).  Kinematic viscosity ν = (τ−½)/3.
    delta : float
        Initial density perturbation amplitude (small for linear acoustics).
    n_steps : int
        Number of LBM time steps.
    device : str
        ``"cpu"`` or ``"cuda"``.
    log_every : int
        Print a diagnostics row every this many steps.
    measure_every : int
        Record (amplitude, phase) every this many steps for post-fit.
    """
    dev = torch.device(device)
    cs = CS
    cs2 = CS2
    nu = (tau - 0.5) / 3.0               # LBM kinematic (shear) viscosity
    lam = nx / 2.0                        # wavelength
    k = 2.0 * math.pi / lam              # wavenumber
    gamma_theory = nu * k * k            # Γ = ν_shear · k²  (see docstring)

    # ---- Grid ----
    xx = torch.arange(nx, device=dev, dtype=torch.float32)

    # ---- Initial condition: right-travelling wave ----
    # ρ(x,0) = 1 + δ sin(kx)
    # u_x(x,0) = c_s δ sin(kx)   ← makes the wave travel rightward
    rho_init = 1.0 + delta * torch.sin(k * xx)
    ux_init = cs * delta * torch.sin(k * xx)

    rho_init = rho_init.view(1, 1, nx).expand(nz, ny, nx).contiguous()
    ux_init = ux_init.view(1, 1, nx).expand(nz, ny, nx).contiguous()
    uy_init = torch.zeros(nz, ny, nx, device=dev)
    uz_init = torch.zeros(nz, ny, nx, device=dev)

    # ---- Distribution (start at equilibrium) ----
    f = equilibrium3d(rho_init, ux_init, uy_init, uz_init, device=dev)

    # ---- Header ----
    print("=" * 72, flush=True)
    print("  1D Acoustic Wave Propagation — D3Q19 BGK LBM", flush=True)
    print("=" * 72, flush=True)
    print(f"  Grid           : {nx} × {ny} × {nz}", flush=True)
    print(f"  Wavelength λ   : {lam}", flush=True)
    print(f"  Wavenumber k   : {k:.6f} rad/lattice", flush=True)
    print(f"  Perturbation δ : {delta}", flush=True)
    print(f"  τ              : {tau}", flush=True)
    print(f"  ν = (τ−½)/3    : {nu:.6f}", flush=True)
    print(f"  c_s = 1/√3     : {cs:.6f}", flush=True)
    print(f"  Γ_theory = νk² : {gamma_theory:.6e} /step", flush=True)
    print(f"  BCs            : periodic (all directions)", flush=True)
    print(f"  Steps          : {n_steps}", flush=True)
    print(f"  Device         : {dev}", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)
    print("  Initial condition (right-travelling wave):", flush=True)
    print("    ρ(x,0) = 1 + δ·sin(kx)", flush=True)
    print("    u_x(x,0) = c_s·δ·sin(kx)", flush=True)
    print(flush=True)
    print("  Analytical solution (lossless + viscous attenuation):", flush=True)
    print("    ρ(x,t) = 1 + δ·exp(−Γt)·sin(k(x − c_s·t))", flush=True)
    print(f"    Expected amplitude at t={n_steps}: "
          f"{delta * math.exp(-gamma_theory * n_steps):.6f} "
          f"({math.exp(-gamma_theory * n_steps)*100:.1f}% of δ)", flush=True)
    print(flush=True)

    # ---- Logging header ----
    print(f"{'step':>6}  {'amp':>10}  {'phase':>9}  {'c_inferred':>10}  "
          f"{'amp/δ':>9}", flush=True)
    print("-" * 72, flush=True)

    # ---- Data storage ----
    times: list[int] = []
    amps: list[float] = []
    phases: list[float] = []

    # ---- Initial measurement ----
    rho, ux, uy, uz = macroscopic3d(f)
    rho_1d = rho[0, 0, :]
    amp0, phase0 = extract_amp_phase(rho_1d, k, nx)
    times.append(0)
    amps.append(amp0)
    phases.append(phase0)
    print(f"{'0':>6}  {amp0:10.6f}  {phase0:9.4f}  {'---':>10}  "
          f"{amp0/delta:9.6f}", flush=True)

    # ---- Time loop ----
    has_nan = False
    for step in range(1, n_steps + 1):
        # === Collision ===
        f = collide_bgk3d(f, tau)

        # === Streaming (periodic via torch.roll) ===
        f = stream3d(f)

        # === Measurement ===
        if step % measure_every == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            rho_1d = rho[0, 0, :]
            amp, phase = extract_amp_phase(rho_1d, k, nx)
            times.append(step)
            amps.append(amp)
            phases.append(phase)

            if torch.isnan(f).any().item() or torch.isinf(f).any().item():
                has_nan = True

            # === Periodic logging ===
            if step % log_every == 0 or step == n_steps:
                # Infer speed from phase change since last log
                if len(times) >= 2:
                    dphi = phases[-1] - phases[-2]
                    dt = times[-1] - times[-2]
                    # Unwrap to [-π, π]
                    while dphi > math.pi:
                        dphi -= 2 * math.pi
                    while dphi < -math.pi:
                        dphi += 2 * math.pi
                    c_inf = -dphi / (k * dt) if dt > 0 else float("nan")
                else:
                    c_inf = float("nan")
                print(f"{step:>6}  {amp:10.6f}  {phase:9.4f}  "
                      f"{c_inf:10.6f}  {amp/delta:9.6f}", flush=True)

        if has_nan:
            print(f"\n  ✗ NaN/Inf detected at step {step}!", flush=True)
            break

    # ---- Early exit on divergence ----
    if has_nan:
        print("=" * 72, flush=True)
        print("  ✗ FAIL — Simulation diverged (NaN/Inf)", flush=True)
        print("=" * 72, flush=True)
        return {"pass": False, "error": "NaN/Inf detected"}

    # ======================================================================= #
    # Analysis
    # ======================================================================= #
    times_np = np.array(times, dtype=np.float64)
    amps_np = np.array(amps, dtype=np.float64)
    phases_np = np.array(phases, dtype=np.float64)

    # ---- Unwrap phases (handle 2π jumps) ----
    phases_unwrapped = np.unwrap(phases_np)

    # ---- Sound speed: linear fit of φ(t) ----
    # φ(t) = φ₀ − k·c_s·t  →  slope = −k·c_s  →  c_s = −slope / k
    A_mat = np.vstack([times_np, np.ones_like(times_np)]).T
    slope_phase, intercept_phase = np.linalg.lstsq(
        A_mat, phases_unwrapped, rcond=None
    )[0]
    c_lbm = -slope_phase / k
    speed_error_pct = abs(c_lbm - cs) / cs * 100.0

    # ---- Attenuation: linear fit of log(A) vs t ----
    # A(t) = δ·exp(−Γ·t)  →  log A = log δ − Γ·t  →  slope = −Γ
    log_amps = np.log(np.maximum(amps_np, 1e-15))
    slope_amp, intercept_amp = np.linalg.lstsq(
        A_mat, log_amps, rcond=None
    )[0]
    gamma_lbm = -slope_amp

    # ---- Alternative theoretical predictions for attenuation ----
    # 1) Γ = ν_shear · k²  (full: 4/3 shear + bulk, our main prediction)
    gamma_full = nu * k * k
    # 2) Γ = (2/3) · ν_shear · k²  (4/3 shear only, no bulk viscosity)
    gamma_shear_43 = (2.0 / 3.0) * nu * k * k
    # 3) Γ = ν_shear · k² / 2  (simple shear, no 4/3, no bulk)
    gamma_shear_half = 0.5 * nu * k * k

    # ---- Residuals for quality-of-fit ----
    phase_fit = slope_phase * times_np + intercept_phase
    phase_residual = np.sqrt(
        np.mean((phases_unwrapped - phase_fit) ** 2)
    )
    amp_fit = np.exp(slope_amp * times_np + intercept_amp)
    amp_residual_rel = np.sqrt(
        np.mean(((amps_np - amp_fit) / np.maximum(amps_np, 1e-15)) ** 2)
    )

    # ======================================================================= #
    # Results
    # ======================================================================= #
    print(flush=True)
    print("=" * 72, flush=True)
    print("  RESULTS", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)

    # ---- Sound speed ----
    print("  ┌─── Sound Speed ───────────────────────────────────────────┐",
          flush=True)
    print(f"  │  c_s  (theory)     = {cs:.6f}                        │",
          flush=True)
    print(f"  │  c_lbm (measured)  = {c_lbm:.6f}                        │",
          flush=True)
    print(f"  │  error             = {speed_error_pct:.2f} %                      │",
          flush=True)
    print(f"  │  threshold         = 5.00 %                       │",
          flush=True)
    print(f"  │  phase fit RMS     = {phase_residual:.4e} rad              │",
          flush=True)
    status = "✓ PASS" if speed_error_pct < 5.0 else "✗ FAIL"
    print(f"  │  status            = {status}                        │",
          flush=True)
    print("  └──────────────────────────────────────────────────────────┘",
          flush=True)
    print(flush=True)

    # ---- Attenuation ----
    print("  ┌─── Attenuation Rate ─────────────────────────────────────┐",
          flush=True)
    print(f"  │  Γ_lbm  (measured)      = {gamma_lbm:.6e} /step       │",
          flush=True)
    print(f"  │  Γ_full (ν·k²)          = {gamma_full:.6e} /step       │",
          flush=True)
    print(f"  │  ratio (meas / full)    = {gamma_lbm/gamma_full:.3f}                       │",
          flush=True)
    print(f"  │  Γ_shear_43 (2νk²/3)    = {gamma_shear_43:.6e} /step       │",
          flush=True)
    print(f"  │  ratio (meas / s43)     = {gamma_lbm/gamma_shear_43:.3f}                       │",
          flush=True)
    print(f"  │  Γ_shear_half (νk²/2)   = {gamma_shear_half:.6e} /step       │",
          flush=True)
    print(f"  │  ratio (meas / s_half)  = {gamma_lbm/gamma_shear_half:.3f}                       │",
          flush=True)
    if gamma_lbm > 0:
        hl = math.log(2) / gamma_lbm
        print(f"  │  half-life (measured)   = {hl:.1f} steps                  │",
              flush=True)
    print(f"  │  amp fit rel. RMS       = {amp_residual_rel:.4e}               │",
          flush=True)
    print("  └──────────────────────────────────────────────────────────┘",
          flush=True)
    print(flush=True)

    # ---- Wave peak tracking (visual confirmation) ----
    # x_peak(t) = (π/2 − φ(t)) / k  (unwrapped, mod nx)
    x_peak = ((math.pi / 2.0 - phases_unwrapped) / k) % nx
    # Theoretical peak position
    x_peak_theory = (math.pi / 2.0 / k + cs * times_np) % nx
    peak_err = np.mean(np.abs(
        np.minimum(np.abs(x_peak - x_peak_theory),
                   nx - np.abs(x_peak - x_peak_theory))
    ))
    print(f"  Wave peak tracking: mean |Δx_peak| = {peak_err:.2f} cells",
          flush=True)
    print(f"  (phase-based speed is the primary measurement)", flush=True)
    print(flush=True)

    # ---- PASS / FAIL ----
    passed = speed_error_pct < 5.0
    if passed:
        print("  ✓✓ PASS — Acoustic wave benchmark PASSED", flush=True)
        print(f"     Sound speed error = {speed_error_pct:.2f}% < 5% threshold",
              flush=True)
        print(f"     c_lbm = {c_lbm:.6f}  vs  c_s = {cs:.6f}", flush=True)
    else:
        print("  ✗ FAIL — Acoustic wave benchmark FAILED", flush=True)
        print(f"     Sound speed error = {speed_error_pct:.2f}% >= 5% threshold",
              flush=True)
    print("=" * 72, flush=True)

    # ======================================================================= #
    # ASCII plots
    # ======================================================================= #

    # ---- Final density profile ----
    rho_final = rho[0, 0, :].cpu().numpy().astype(np.float64)
    x_arr = np.arange(nx, dtype=np.float64)
    # Analytical with theoretical c_s and Γ
    rho_ana = 1.0 + delta * np.exp(-gamma_full * n_steps) * np.sin(
        k * (x_arr - cs * n_steps)
    )
    print(flush=True)
    print("  Final density profile  (█ = numerical,  · = analytical):",
          flush=True)
    ascii_plot_1d(
        rho_final, rho_ana,
        width=72, height=14,
        title=f"ρ(x) at t = {n_steps}",
        x_max=nx,
    )

    # ---- Amplitude decay ----
    print(flush=True)
    print("  Amplitude decay  (█ = measured,  · = A₀·exp(−Γ·t)):",
          flush=True)
    amp_theory_curve = delta * np.exp(-gamma_full * times_np)
    ascii_plot_1d(
        amps_np, amp_theory_curve,
        width=72, height=10,
        title="A(t) vs t",
        x_max=n_steps,
    )

    # ---- Phase evolution ----
    print(flush=True)
    print("  Phase evolution  (█ = measured,  · = linear fit):",
          flush=True)
    ascii_plot_1d(
        phases_unwrapped, phase_fit,
        width=72, height=10,
        title="φ(t) vs t  (unwrapped)",
        x_max=n_steps,
    )

    return {
        "c_lbm": c_lbm,
        "c_theory": cs,
        "speed_error_pct": speed_error_pct,
        "gamma_lbm": gamma_lbm,
        "gamma_theory": gamma_full,
        "gamma_ratio": gamma_lbm / gamma_full,
        "pass": passed,
    }


# =========================================================================== #
# CLI
# =========================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D acoustic wave propagation benchmark (D3Q19 BGK LBM)"
    )
    parser.add_argument("--nx", type=int, default=200,
                        help="Grid size in x (default 200)")
    parser.add_argument("--ny", type=int, default=4,
                        help="Grid size in y (default 4)")
    parser.add_argument("--nz", type=int, default=1,
                        help="Grid size in z (default 1)")
    parser.add_argument("--tau", type=float, default=0.8,
                        help="Relaxation time τ (default 0.8)")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Density perturbation amplitude δ (default 0.01)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of time steps (default 2000)")
    parser.add_argument("--device", default="cpu",
                        help="Device: 'cpu' or 'cuda' (default cpu)")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print interval (default 100)")
    parser.add_argument("--measure-every", type=int, default=10,
                        help="Measurement interval for post-fit (default 10)")
    args = parser.parse_args()

    run_acoustic_wave_benchmark(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        delta=args.delta,
        n_steps=args.steps,
        device=args.device,
        log_every=args.log_every,
        measure_every=args.measure_every,
    )

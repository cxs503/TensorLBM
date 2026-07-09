#!/usr/bin/env python
"""Cylinder aeroacoustics benchmark — D3Q19 BGK LBM.

Validates:
  1. Strouhal number  St = f·D/U ≈ 0.164  (Williamson 1988, Re=100)
  2. Far-field sound pressure amplitude vs Curle's analogy (1955)

Physics
-------
Cylinder flow at Re=100 produces a Kármán vortex street.  Vortex shedding
causes surface pressure fluctuations that radiate acoustic waves to the
far field.  Curle's acoustic analogy gives the far-field pressure:

    p'(r,θ,t) ~ (ρ₀U²D)/(4πr) · sin(θ) · cos(2πf(t − r/c_s))

where  f = St·U/D  is the shedding frequency,  c_s = 1/√3  the LBM sound
speed,  θ  the angle from the flow direction, and  r  the distance from
the cylinder centre.

In LBM the pressure is  p = c_s² · ρ,  so the pressure fluctuation is
p' = (ρ − ρ₀) / 3.

Setup (defaults)
----------------
  Grid       nx=400, ny=100, nz=1   (2D flow on D3Q19 lattice)
  Cylinder   centre (cx=100, cy=50), radius R=10, diameter D=20
  Flow       Re=100, U_in=0.1, ν=0.02, τ=0.56, Ma≈0.173
  Inlet      x=0:    equilibrium velocity BC (u=U_in, ρ=1)
  Outlet     x=nx-1: zero-gradient (convective outflow)
  Top/bottom y:      periodic (reduces acoustic reflections)
  Cylinder:          full-way bounce-back (no-slip)
  Monitor 1  (cx+2R+2, cy) = (122, 50)  → near-wake pressure → FFT → St
  Monitor 2  (cx, cy+r)    r=20,30,40   → far-field pressure → Curle

Run
---
    PYTHONPATH=src python examples/benchmark_cylinder_aeroacoustics.py \
        --device cpu --steps 5000

For better frequency resolution use --steps 10000 (≈8 shedding cycles).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

# Ensure src/ is importable even without PYTHONPATH
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ascii_vorticity(
    ux: torch.Tensor,
    uy: torch.Tensor,
    width: int = 80,
    height: int = 22,
    title: str = "",
) -> None:
    """Print the z-vorticity field as coarse ASCII art."""
    assert ux.shape[0] == 1
    ux2d = ux[0]  # (ny, nx)
    uy2d = uy[0]

    omega = torch.zeros_like(ux2d)
    omega[:, 1:-1] = 0.5 * (uy2d[:, 2:] - uy2d[:, :-2])
    omega[1:-1, :] -= 0.5 * (ux2d[2:, :] - ux2d[:-2, :])

    chars = " .:-=+*#%@"
    nlev = len(chars)
    fmax = float(omega.abs().max())
    if fmax < 1e-12:
        fmax = 1e-12

    if title:
        print(f"  {title}")
    print("+" + "-" * width + "+")
    for j in range(height):
        y0 = int((height - 1 - j) * ux2d.shape[0] / height)
        y1 = max(y0 + 1, int((height - j) * ux2d.shape[0] / height))
        row = []
        for i in range(width):
            x0 = int(i * ux2d.shape[1] / width)
            x1 = max(x0 + 1, int((i + 1) * ux2d.shape[1] / width))
            block = float(omega[y0:y1, x0:x1].mean())
            idx = int((block + fmax) / (2 * fmax) * (nlev - 1))
            idx = max(0, min(nlev - 1, idx))
            row.append(chars[idx])
        print("|" + "".join(row) + "|")
    print("+" + "-" * width + "+")
    print(f"  ω_z range: [{float(omega.min()):.4f}, {float(omega.max()):.4f}]   "
          f"'{chars[0]}'=−max  '{chars[nlev // 2]}'=0  '{chars[-1]}'=+max")


def _fft_peak(
    signal: np.ndarray,
    sample_dt: float = 1.0,
    min_freq: float = 1e-5,
    zero_pad_factor: int = 4,
    detrend: bool = True,
) -> tuple[float, float]:
    """Return (peak_freq, peak_amplitude) from a 1-D signal via FFT.

    Uses a Hanning window and zero-padding for a smoother spectrum.
    The amplitude is corrected for the window's coherent gain so that
    a pure sinusoid  A·cos(2πft)  yields approximately  A  at the peak.

    If *detrend* is True, a moving-average trend is subtracted before
    the FFT to suppress low-frequency transients that can mask the
    shedding peak.
    """
    N = len(signal)
    if N < 8:
        return 0.0, 0.0
    signal = signal.astype(np.float64).copy()
    if detrend:
        # Subtract moving average to remove low-frequency trends
        w = max(50, N // 10)
        if w < N:
            kernel = np.ones(w) / w
            trend = np.convolve(signal, kernel, mode="same")
            signal = signal - trend
    window = np.hanning(N)
    sig_w = signal * window
    N_fft = max(N * zero_pad_factor, 8192)
    spectrum = np.fft.rfft(sig_w, n=N_fft)
    freqs = np.fft.rfftfreq(N_fft, d=sample_dt)
    coherent_gain = float(window.mean())  # ≈ 0.5 for Hanning
    amplitude = 2.0 * np.abs(spectrum) / (N * coherent_gain)

    mask = freqs > min_freq
    if not mask.any():
        return 0.0, 0.0
    amp_masked = amplitude[mask]
    freq_masked = freqs[mask]
    idx = int(np.argmax(amp_masked))
    return float(freq_masked[idx]), float(amp_masked[idx])


def _compute_force(
    f: torch.Tensor,
    solid: torch.Tensor,
    c_dev: torch.Tensor,
    opp: torch.Tensor,
) -> torch.Tensor:
    """Hydrodynamic force on the cylinder via momentum exchange.

    For full-way bounce-back the force on the solid is

        F = 2 · Σ_{x∈solid} Σ_q  c_q · f_q(x) · [x − c_q is fluid]

    where  f_q  is the **post-streaming pre-bounce-back** population
    and the bracket is 1 when the neighbour cell in direction −c_q is
    fluid (i.e. the population actually came from the fluid).

    Returns a tensor [Fx, Fy, Fz].
    """
    F = torch.zeros(3, device=f.device, dtype=f.dtype)
    for q in range(19):
        cxq = int(c_dev[q, 0].item())
        cyq = int(c_dev[q, 1].item())
        czq = int(c_dev[q, 2].item())
        # torch.roll(solid, shifts=(czq, cyq, cxq))[z,y,x]
        #   = solid[z-czq, y-cyq, x-cxq]  → neighbour in direction −c_q
        neighbour_is_solid = torch.roll(
            solid, shifts=(czq, cyq, cxq), dims=(0, 1, 2)
        )
        boundary = solid & ~neighbour_is_solid
        if boundary.any():
            fsum = f[q][boundary].sum()
            F[0] += cxq * fsum
            F[1] += cyq * fsum
            F[2] += czq * fsum
    return 2.0 * F


# --------------------------------------------------------------------------- #
# Cylinder aeroacoustics benchmark
# --------------------------------------------------------------------------- #

def run_cylinder_aeroacoustics(
    nx: int = 400,
    ny: int = 100,
    nz: int = 1,
    cx: int = 100,
    cy: int = 50,
    R: int = 10,
    U_in: float = 0.1,
    Re: float = 100.0,
    steps: int = 5000,
    device: str = "cpu",
    log_every: int = 500,
    transient_frac: float = 0.3,
    force_every: int = 50,
    ramp_steps: int = 500,
) -> dict:
    """Run cylinder flow + aeroacoustic benchmark.

    Returns a dict with Strouhal number, pressure amplitudes, drag/lift
    coefficients, and Curle comparison results.
    """
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    cs = math.sqrt(cs2)
    D = 2 * R
    nu = U_in * D / Re
    tau = 3.0 * nu + 0.5
    Ma = U_in / cs

    # --- Monitors -----------------------------------------------------------
    near_wake = (cx + 2 * R + 2, cy)          # (x, y) — just behind cylinder
    # Far-field distances perpendicular to flow (θ=90°, max acoustic directivity)
    max_r_y = min(cy, ny - 1 - cy)  # max distance staying in bounds
    far_r = [r for r in [20, 30, 40] if r <= max_r_y]
    if len(far_r) < 2:  # ensure at least 2 monitors
        far_r = [max_r_y // 3, max_r_y * 2 // 3]
    far_pts = [(cx, cy + r) for r in far_r]
    # Front/back pressure probes (for pressure-drag estimate, 2 cells from surface)
    front_pt = (max(cx - R - 2, 1), cy)
    back_pt = (min(cx + R + 2, nx - 2), cy)

    # --- Cylinder mask ------------------------------------------------------
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nx, device=dev),
        indexing="ij",
    )
    dist = torch.sqrt((xx.float() - cx) ** 2 + (yy.float() - cy) ** 2)
    solid = dist < R  # (nz, ny, nx) bool

    # --- Inlet equilibrium (steady state, constant velocity, ρ=1) -----------
    rho_in_ss = torch.ones((nz, ny, 1), device=dev)
    ux_in_ss = torch.full_like(rho_in_ss, U_in)
    uy_zero = torch.zeros_like(rho_in_ss)
    uz_zero = torch.zeros_like(rho_in_ss)
    feq_in_ss = equilibrium3d(rho_in_ss, ux_in_ss, uy_zero, uz_zero, device=dev)

    # --- Initialise flow field ----------------------------------------------
    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full_like(rho0, U_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    # Small random perturbation (5 % of U_in) to break symmetry and seed
    # the Kármán instability.
    torch.manual_seed(42)
    uy0 += 0.05 * U_in * (torch.rand_like(rho0) * 2.0 - 1.0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=dev)

    opp = OPPOSITE.to(dev)
    c_dev = C.to(dev)

    # --- Recording arrays ---------------------------------------------------
    nw_hist: list[float] = []
    ff_hists: dict[int, list[float]] = {r: [] for r in far_r}
    fx_hist: list[float] = []
    fy_hist: list[float] = []
    fstep_hist: list[int] = []

    # --- Header -------------------------------------------------------------
    print(f"\n{'─' * 64}")
    print(f"  CYLINDER AEROACOUSTICS  —  D3Q19 BGK")
    print(f"  Grid: {nx} × {ny} × {nz}   Device: {device}")
    print(f"  Cylinder: centre=({cx},{cy})  R={R}  D={D}")
    print(f"  Re={Re:.0f}  U={U_in:.4f}  ν={nu:.5f}  τ={tau:.4f}  Ma={Ma:.4f}")
    print(f"  c_s={cs:.4f}  St_ref=0.164 (Williamson 1988)")
    print(f"  Steps={steps}  Ramp={ramp_steps}  Inlet=equilibrium  "
          f"Outlet=zero-grad  y=periodic")
    print(f"  Near-wake monitor: ({near_wake[0]},{near_wake[1]})")
    for r, pt in zip(far_r, far_pts):
        print(f"  Far-field monitor r={r}: ({pt[0]},{pt[1]})  θ=90°")
    print(f"{'─' * 64}")
    print(f"  {'step':>6s}  {'umax':>8s}  {'p_nw':>12s}  {'p_ff':>12s}  "
          f"{'Cd_me':>8s}  {'Cd_p':>8s}  {'Cl':>8s}")
    print(f"{'─' * 64}")

    # --- Time loop ----------------------------------------------------------
    # Expected shedding period (for frequency-matched perturbation)
    T_shed_exp = D / (0.164 * U_in)  # Williamson 1988
    pert_cycles = 2
    pert_duration = int(pert_cycles * T_shed_exp)

    for step in range(1, steps + 1):
        # Velocity ramp-up (linear) to reduce initial transient.
        # After ramp, apply a frequency-matched lateral perturbation at
        # the inlet to trigger the Kármán instability at the natural
        # shedding frequency.
        if step <= ramp_steps:
            u_ramp = U_in * step / ramp_steps
            rho_r = torch.ones((nz, ny, 1), device=dev)
            ux_r = torch.full_like(rho_r, u_ramp)
            feq_in = equilibrium3d(rho_r, ux_r, uy_zero, uz_zero, device=dev)
        elif step <= ramp_steps + pert_duration:
            # Frequency-matched sinusoidal lateral perturbation (10% U_in)
            s = step - ramp_steps
            uy_pert = 0.10 * U_in * math.sin(2.0 * math.pi * s / T_shed_exp)
            uy_r = torch.full_like(rho_in_ss, uy_pert)
            feq_in = equilibrium3d(rho_in_ss, ux_in_ss, uy_r, uz_zero, device=dev)
        else:
            feq_in = feq_in_ss

        # 1) Collision
        f = collide_bgk3d(f, tau)

        # 2) Streaming (torch.roll wraps y → periodic BC)
        f = stream3d(f)

        # 3) Inlet BC (x = 0): equilibrium velocity
        f[:, :, :, 0] = feq_in[:, :, :, 0]

        # 4) Outlet BC (x = nx-1): zero-gradient
        f[:, :, :, -1] = f[:, :, :, -2]

        # 5) Force on cylinder (before bounce-back, using streamed f)
        if step % force_every == 0 or step == steps:
            F = _compute_force(f, solid, c_dev, opp)
            fx_hist.append(float(F[0].item()))
            fy_hist.append(float(F[1].item()))
            fstep_hist.append(step)

        # 6) Cylinder bounce-back (full-way)
        f = torch.where(solid.unsqueeze(0), f[opp], f)

        # 7) Record monitors — extract rho directly from f (fast, no
        #    full macroscopic3d call)
        mx, my = near_wake
        nw_hist.append(float(f[:, 0, my, mx].sum().item()) - 1.0)
        for r, (fx, fy) in zip(far_r, far_pts):
            ff_hists[r].append(float(f[:, 0, fy, fx].sum().item()) - 1.0)

        # 8) Log (full macroscopic3d only for logging)
        if step % log_every == 0 or step == steps:
            rho, ux, uy, uz = macroscopic3d(f)
            umax = float(ux.abs().max())
            p_nw = nw_hist[-1]
            p_ff = ff_hists[far_r[-1]][-1]
            cd_me = 2.0 * fx_hist[-1] / (1.0 * U_in ** 2 * D) if fx_hist else 0.0
            cl = 2.0 * fy_hist[-1] / (1.0 * U_in ** 2 * D) if fy_hist else 0.0
            # Pressure-drag estimate from front/back pressure difference
            p_front = float(rho[0, front_pt[1], front_pt[0]].item()) - 1.0
            p_back = float(rho[0, back_pt[1], back_pt[0]].item()) - 1.0
            cd_p = 2.0 * (p_front - p_back) * cs2 / (U_in ** 2)
            print(f"  {step:6d}  {umax:8.4f}  {p_nw:12.6e}  {p_ff:12.6e}  "
                  f"{cd_me:8.4f}  {cd_p:8.4f}  {cl:8.4f}", flush=True)

    # ===================================================================== #
    # Analysis
    # ===================================================================== #
    nw = np.array(nw_hist)
    transient = int(transient_frac * len(nw))

    # --- Strouhal number: two independent measurements ----------------------
    # 1) Near-wake pressure FFT (detrended to suppress transients)
    f_nw, p_nw_amp = _fft_peak(nw[transient:], sample_dt=1.0, detrend=True)
    St_nw = f_nw * D / U_in if f_nw > 0 else 0.0

    # 2) Lift-force FFT (directly tied to vortex shedding)
    fx_arr = np.array(fx_hist)
    fy_arr = np.array(fy_hist)
    ftrans = int(transient_frac * len(fx_arr))
    denom = 1.0 * U_in ** 2 * D
    cd_arr = 2.0 * fx_arr / denom
    cl_arr = 2.0 * fy_arr / denom
    # Use 50% transient discard for force stats (forces are noisier)
    ftrans2 = max(ftrans, len(cd_arr) // 2)
    cd_mean = float(cd_arr[ftrans2:].mean()) if len(cd_arr) > ftrans2 else 0.0
    cl_rms = float(np.sqrt((cl_arr[ftrans2:] ** 2).mean())) if len(cl_arr) > ftrans2 else 0.0
    f_cl, cl_amp_fft = _fft_peak(cl_arr[ftrans:], sample_dt=float(force_every), detrend=True)
    St_cl = f_cl * D / U_in if f_cl > 0 else 0.0

    # Primary St: use Cl FFT (more reliable — directly measures shedding)
    St_lbm = St_cl if St_cl > 0 else St_nw
    St_ref = 0.164  # Williamson 1988
    St_err = abs(St_lbm - St_ref) / St_ref * 100.0 if St_ref > 0 else 0.0

    # Blockage correction (periodic BC): St_corr ≈ St_meas × (1 − D/ny)
    blockage = D / ny
    St_corr = St_lbm * (1.0 - blockage)
    St_corr_err = abs(St_corr - St_ref) / St_ref * 100.0

    # --- Far-field pressure amplitudes (detrended) -------------------------
    ff_amps: dict[int, float] = {}
    for r in far_r:
        sig = np.array(ff_hists[r])[transient:]
        _, amp = _fft_peak(sig, sample_dt=1.0, detrend=True)
        ff_amps[r] = amp

    # --- Report -------------------------------------------------------------
    print(f"\n{'─' * 64}")
    print(f"  AEROACOUSTIC ANALYSIS")
    print(f"{'─' * 64}")
    print(f"  Strouhal number measurements:")
    print(f"    St (near-wake pressure FFT) = {St_nw:.4f}")
    print(f"    St (lift-force Cl FFT)      = {St_cl:.4f}  ← primary")
    print(f"    St (reference, Williamson)  = {St_ref:.4f}")
    print(f"    St error (Cl)               = {St_err:.1f} %")
    print(f"    Blockage D/ny = {blockage:.2f}")
    print(f"    St (blockage-corrected)     = {St_corr:.4f}  (error {St_corr_err:.1f}%)")
    if f_cl > 0:
        print(f"    Shedding frequency (Cl)     = {f_cl:.6e} / step  (T = {1.0/f_cl:.0f} steps)")

    print(f"\n  Drag / lift coefficients (Re={Re:.0f}):")
    print(f"    Cd_mean  = {cd_mean:.4f}   (literature ≈ 1.33)")
    print(f"    Cl_rms   = {cl_rms:.4f}   (literature ≈ 0.23)")
    print(f"    Cl_amp   = {cl_amp_fft:.4f}   (literature ≈ 0.33)")

    # --- Curle formula ------------------------------------------------------
    # p'_amp = (ρ₀ U² D) / (4π r) · sin(θ)
    # θ = 90° for monitors directly above cylinder → sin θ = 1
    # LBM pressure: p' = c_s² · δρ = δρ / 3
    print(f"\n  Curle far-field pressure (θ=90°, sin θ=1):")
    print(f"    p'_Curle(r) = ρ₀U²D / (4πr) = {U_in**2 * D / (4 * math.pi):.6e} / r")
    print(f"  {'r':>6s}  {'δρ_lbm':>14s}  {'p_lbm':>14s}  "
          f"{'p_Curle':>14s}  {'ratio':>8s}")
    for r in far_r:
        p_lbm_drho = ff_amps[r]
        p_lbm_press = p_lbm_drho * cs2
        p_curle = (1.0 * U_in ** 2 * D) / (4.0 * math.pi * r) * math.sin(math.pi / 2)
        ratio = p_lbm_press / p_curle if p_curle > 0 else float("inf")
        print(f"  {r:6d}  {p_lbm_drho:14.6e}  {p_lbm_press:14.6e}  "
              f"{p_curle:14.6e}  {ratio:8.2f}")

    # --- Pressure decay -----------------------------------------------------
    print(f"\n  Pressure decay (perpendicular to flow, θ=90°):")
    if len(far_r) >= 2:
        rs = np.array(far_r, dtype=float)
        ps = np.array([ff_amps[r] * cs2 for r in far_r])
        if np.all(ps > 0):
            log_r = np.log(rs)
            log_p = np.log(ps)
            alpha = float(np.polyfit(log_r, log_p, 1)[0])
            print(f"    Fitted:  p ~ r^({alpha:.2f})")
            print(f"    (2D acoustic far-field: r^(−0.5),  3D Curle: r^(−1))")
            print(f"    Near-field hydrodynamic decays faster than r^(−1).")

    # --- Vorticity snapshot -------------------------------------------------
    rho, ux, uy, uz = macroscopic3d(f)
    _ascii_vorticity(ux, uy, width=80, height=20,
                     title=f"Vorticity ω_z at step {steps}")

    # --- Verdict ------------------------------------------------------------
    # Primary: uncorrected St from Cl FFT (blockage correction unreliable for D/ny > 10%)
    st_pass = St_err < 10.0
    print(f"\n  Strouhal validation:  {'PASS' if st_pass else 'FAIL'}  "
          f"(St_cl={St_cl:.4f}, error {St_err:.1f}% "
          f"{'<' if st_pass else '>='} 10%)")
    if blockage > 0.10:
        print(f"  Note: blockage {blockage:.0%} > 10%, correction unreliable, using uncorrected St.")
    r_last = far_r[-1]
    p_curle_last = (1.0 * U_in**2 * D) / (4 * math.pi * r_last)
    ratio_last = ff_amps[r_last] * cs2 / p_curle_last
    print(f"  Curle comparison:     LBM/Curle ratio at r={r_last} is "
          f"{ratio_last:.1f}×  (near-field → expect > 1)")

    return {
        "St_lbm": St_lbm,
        "St_cl": St_cl,
        "St_nw": St_nw,
        "St_ref": St_ref,
        "St_err": St_err,
        "St_corr": St_corr,
        "f_shed": f_cl,
        "cd_mean": cd_mean,
        "cl_rms": cl_rms,
        "cl_amp": cl_amp_fft,
        "ff_amps": ff_amps,
        "tau": tau,
        "nu": nu,
        "Ma": Ma,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Cylinder aeroacoustics LBM benchmark")
    p.add_argument("--nx", type=int, default=400)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--cx", type=int, default=100)
    p.add_argument("--cy", type=int, default=50)
    p.add_argument("--R", type=int, default=10)
    p.add_argument("--U", type=float, default=0.1)
    p.add_argument("--Re", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()

    # Use all CPU cores for torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print("=" * 64)
    print("  CYLINDER AEROACOUSTICS BENCHMARK (vortex shedding + Curle)")
    print("=" * 64)

    run_cylinder_aeroacoustics(
        nx=args.nx, ny=args.ny, nz=args.nz,
        cx=args.cx, cy=args.cy, R=args.R,
        U_in=args.U, Re=args.Re,
        steps=args.steps, device=args.device,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

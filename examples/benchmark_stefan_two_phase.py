#!/usr/bin/env python3
"""Two-phase Stefan solidification benchmark.

Extension of the one-phase Stefan benchmark: both solid and liquid sides
have temperature gradients.  The interface velocity is driven by the heat
flux *difference* between the solid and liquid sides:

    ρ·L·v = k_s·(dT/dx)_solid − k_l·(dT/dx)_liquid

With equal thermal properties (k_s = k_l, α_s = α_l = α, cp_s = cp_l = cp):

    v = α·cp·((dT/dx)_solid − (dT/dx)_liquid) / L

Analytical solution (equal properties):

    s(t) = 2·λ·√(α·t)

    T_s(x,t) = T_left + (T_freeze − T_left)·erf(x/(2√(αt))) / erf(λ)
    T_l(x,t) = T_right − (T_right − T_freeze)·erfc(x/(2√(αt))) / erfc(λ)

Transcendental equation for λ (two-phase Stefan condition):

    Ste_s·exp(−λ²)/(√π·erf(λ)) − Ste_l·exp(−λ²)/(√π·erfc(λ)) = λ

where  Ste_s = cp·(T_freeze − T_left)/L,  Ste_l = cp·(T_right − T_freeze)/L.

When Ste_l = 0 (T_right = T_freeze) this reduces to the one-phase case.

Usage:
    PYTHONPATH=src python examples/benchmark_stefan_two_phase.py --device cpu --steps 2000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Reuse kernels from the one-phase benchmark
from benchmark_stefan_freezing import (
    C_D2Q5,
    W_D2Q5,
    OPPOSITE,
    equilibrium_thermal,
    collide_thermal_bgk,
    stream_thermal,
    macroscopic_thermal,
    apply_temperature_bc_x,
    bounce_back_solid,
    detect_interface_x,
    mean_temperature_profile_x,
)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C
from tensorlbm.solver3d import collide_bgk3d, stream3d


# =========================================================================== #
# Analytical solution (two-phase Stefan, equal thermal properties)
# =========================================================================== #


def solve_stefan_lambda_two_phase(
    Ste_s: float,
    Ste_l: float,
) -> float:
    """Solve the two-phase Stefan transcendental equation for λ.

        Ste_s·exp(−λ²)/(√π·erf(λ)) − Ste_l·exp(−λ²)/(√π·erfc(λ)) = λ

    Uses bisection.  Returns λ > 0.
    """
    from scipy.optimize import brentq

    def f(lam: float) -> float:
        if lam < 1e-12:
            return float("inf")
        e = math.exp(-lam * lam)
        erf_l = math.erf(lam)
        erfc_l = math.erfc(lam)
        if erf_l < 1e-30 or erfc_l < 1e-30:
            return float("inf")
        return (
            Ste_s * e / (math.sqrt(math.pi) * erf_l)
            - Ste_l * e / (math.sqrt(math.pi) * erfc_l)
            - lam
        )

    # Find bracket — erfc(λ) underflows for large λ, so cap hi at 5
    lo, hi = 1e-6, 5.0
    f_lo = f(lo)
    f_hi = f(hi)
    # Expand if needed
    while f_lo * f_hi > 0 and hi < 50:
        hi *= 2
        f_hi = f(hi)

    return float(brentq(f, lo, hi, xtol=1e-12))


def stefan_interface_position(t: float, lam: float, alpha: float) -> float:
    """s(t) = 2·λ·√(α·t)."""
    return 2.0 * lam * math.sqrt(alpha * t)


def stefan_temperature_profile_two_phase(
    x_arr: np.ndarray,
    t: float,
    lam: float,
    alpha: float,
    T_left: float,
    T_freeze: float,
    T_right: float,
) -> np.ndarray:
    """Two-phase Stefan temperature profile (equal α)."""
    s = stefan_interface_position(t, lam, alpha)
    sqrt_at = math.sqrt(alpha * t)
    erf_lam = math.erf(lam)
    erfc_lam = math.erfc(lam)
    T = np.empty_like(x_arr)
    for i, x in enumerate(x_arr):
        if x <= s:
            T[i] = T_left + (T_freeze - T_left) * math.erf(x / (2 * sqrt_at)) / erf_lam
        else:
            T[i] = T_right - (T_right - T_freeze) * math.erfc(x / (2 * sqrt_at)) / erfc_lam
    return T


# =========================================================================== #
# Main simulation
# =========================================================================== #


def run_stefan_two_phase(
    nx: int = 200,
    ny: int = 1,
    nz: int = 1,
    tau: float = 0.8,
    tau_T: float = 0.8,
    T_left: float = -1.0,
    T_right: float = 1.0,
    T_freeze: float = 0.0,
    cp: float = 1.0,
    L_latent: float = 1.0,
    steps: int = 2000,
    device: str = "cpu",
    log_every: int = 200,
    quiet: bool = False,
) -> dict:
    """Run a 1-D two-phase Stefan solidification benchmark."""
    dev = torch.device(device)
    nu = (tau - 0.5) / 3.0
    alpha = (tau_T - 0.5) / 3.0

    Ste_s = cp * (T_freeze - T_left) / L_latent
    Ste_l = cp * (T_right - T_freeze) / L_latent
    lam = solve_stefan_lambda_two_phase(Ste_s, Ste_l)

    # Wall mask
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall_mask[:, :, 0] = True
    wall_mask[:, :, -1] = True

    # Initial conditions: all liquid at T_right
    T_field = torch.full((nz, ny, nx), float(T_right), device=dev, dtype=torch.float32)
    T_field[:, :, 0] = T_left
    T_field[:, :, -1] = T_right

    # Phase field: all liquid, solid at left wall
    phi = torch.ones((nz, ny, nx), device=dev, dtype=torch.float32)
    phi[:, :, 0] = -1.0

    # Momentum: at rest
    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)
    g = equilibrium_thermal(T_field, u0, u0.clone())
    g = apply_temperature_bc_x(g, T_left, T_right)

    w_d2q5_view = W_D2Q5.to(dev).float().view(5, 1, 1, 1)

    s_ana_final = stefan_interface_position(steps, lam, alpha)

    if not quiet:
        print(f"\n{'─' * 64}")
        print(f"  Two-phase Stefan  —  D3Q19 BGK + D2Q5 thermal + Stefan condition")
        print(f"  Grid: {nx} × {ny} × {nz}  (1-D in x)")
        print(f"  τ = {tau:.4f}   τ_T = {tau_T:.4f}   ν = {nu:.6f}   α = {alpha:.6f}")
        print(f"  T_left = {T_left}   T_right = {T_right}   T_freeze = {T_freeze}")
        print(f"  cp = {cp}   L_latent = {L_latent}   Ste_s = {Ste_s:.4f}   Ste_l = {Ste_l:.4f}")
        print(f"  λ (two-phase Stefan root) = {lam:.6f}")
        print(f"  s({steps}) = {s_ana_final:.2f}  (analytical, 2·λ·√(α·t))")
        print(f"  Steps: {steps}   Device: {device}")
        print(f"{'─' * 64}")
        print(f"  {'step':>6s}   {'s_lbm':>8s}   {'s_ana':>8s}   {'err%':>6s}   "
              f"{'T_min':>7s} {'T_max':>7s}")
        print(f"  {'─'*6}   {'─'*8}   {'─'*8}   {'─'*6}   {'─'*7} {'─'*7}")

    history: list[dict] = []
    s_interface: float = 1.0

    for step in range(1, steps + 1):
        # === 1. Macroscopic fields =========================================
        rho, ux, uy, uz = macroscopic3d(f)

        # === 2. Momentum: collide → stream → bounce-back ===================
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        solid_mask = wall_mask | (phi < 0)
        f = bounce_back_solid(f, solid_mask)
        f = f.clamp(min=0.0, max=5.0)

        # === 3. Temperature: collide → stream → BC =========================
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(solid_mask, 0.0)
        uy = uy.masked_fill(solid_mask, 0.0)
        T = macroscopic_thermal(g)
        g = collide_thermal_bgk(g, T, ux, uy, tau_T=tau_T)
        g = stream_thermal(g)
        g = apply_temperature_bc_x(g, T_left, T_right)

        # === 4. Two-phase Stefan condition =================================
        # v = α·cp·((dT/dx)_solid − (dT/dx)_liquid) / L
        #
        # Use the actual temperature field for gradients, but pin the
        # interface cell to T_freeze (latent heat should maintain it there).
        T_cur = macroscopic_thermal(g)
        i_s = int(s_interface)
        if i_s < 1:
            i_s = 1
        if i_s >= nx - 2:
            i_s = nx - 3

        # Pin interface cell to T_freeze for gradient computation
        T_at_interface = T_freeze

        # Solid-side gradient: (T_interface − T[i_s−1]) / dx
        dTdx_solid = T_at_interface - float(T_cur[0, 0, i_s - 1].item())
        # Liquid-side gradient: (T[i_s+1] − T_interface) / dx
        dTdx_liquid = float(T_cur[0, 0, i_s + 1].item()) - T_at_interface

        v_stefan = alpha * cp * (dTdx_solid - dTdx_liquid) / L_latent
        s_new = s_interface + v_stefan

        # === 5. Rebuild φ + latent heat release ============================
        phi_old = phi.clone()
        phi_1d = phi[0, 0, :]
        for i in range(nx):
            d = i - s_new
            if d < -1.0:
                phi_1d[i] = -1.0
            elif d > 1.0:
                phi_1d[i] = 1.0
            else:
                phi_1d[i] = d
        phi[:, :, 0] = -1.0
        phi[:, :, -1] = 1.0

        # Latent heat release
        delta_phi = phi - phi_old
        latent_heat = L_latent * (-delta_phi) / 2.0 * cp
        g = g + w_d2q5_view * latent_heat.unsqueeze(0)
        g = apply_temperature_bc_x(g, T_left, T_right)

        s_interface = s_new

        # === 6. NaN guard ===================================================
        if step % 50 == 0:
            if torch.isnan(f).any() or torch.isinf(f).any():
                print(f"  WARNING: NaN/Inf in f at step {step} — stopping.")
                break
            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"  WARNING: NaN/Inf in g at step {step} — stopping.")
                break

        # === 7. Diagnostics =================================================
        if step % log_every == 0 or step == steps:
            T = macroscopic_thermal(g)
            s_lbm = s_interface
            s_ana = stefan_interface_position(step, lam, alpha)
            err = abs(s_lbm - s_ana) / max(s_ana, 1e-10) * 100
            T_min = float(T.min().item())
            T_max = float(T.max().item())
            history.append({
                "step": step,
                "s_lbm": s_lbm,
                "s_ana": s_ana,
                "err": err,
                "T_min": T_min,
                "T_max": T_max,
            })
            if not quiet:
                print(f"  {step:6d}   {s_lbm:8.2f}   {s_ana:8.2f}   {err:6.2f}   "
                      f"{T_min:7.3f} {T_max:7.3f}", flush=True)

    # --- Final fields -------------------------------------------------------
    T_final = macroscopic_thermal(g)
    T_profile = mean_temperature_profile_x(T_final)
    s_lbm_final = s_interface
    s_ana_final = stefan_interface_position(steps, lam, alpha)
    err_final = abs(s_lbm_final - s_ana_final) / max(s_ana_final, 1e-10) * 100

    # Analytical temperature profile
    x_arr = np.arange(nx, dtype=float)
    T_ana = stefan_temperature_profile_two_phase(
        x_arr, steps, lam, alpha, T_left, T_freeze, T_right
    )

    # Temperature profile error
    deltaT = abs(T_right - T_left)
    T_err_rms = float(np.sqrt(np.mean((T_profile - T_ana) ** 2)) / deltaT * 100)
    T_err_max = float(np.max(np.abs(T_profile - T_ana)) / deltaT * 100)

    # Interface error over time
    late_hist = [h for h in history if h["s_ana"] > 1.0]
    if late_hist:
        avg_err = float(np.mean([h["err"] for h in late_hist]))
        max_err = float(max(h["err"] for h in late_hist))
    else:
        avg_err = err_final
        max_err = err_final

    if not quiet:
        print(f"\n{'─' * 64}")
        print(f"  Final interface (LBM)      : {s_lbm_final:.2f}")
        print(f"  Final interface (analytic) : {s_ana_final:.2f}")
        print(f"  Interface error (final)    : {err_final:.2f}%")
        print(f"  Avg interface error        : {avg_err:.2f}%")
        print(f"  Max interface error        : {max_err:.2f}%")
        print(f"  Temperature RMS error      : {T_err_rms:.2f}%")
        print(f"  Temperature max error      : {T_err_max:.2f}%")
        print(f"{'─' * 64}")

    return {
        "step": steps,
        "s_lbm": s_lbm_final,
        "s_ana": s_ana_final,
        "err_final": err_final,
        "avg_err": avg_err,
        "max_err": max_err,
        "T_err_rms": T_err_rms,
        "T_err_max": T_err_max,
        "T_profile": T_profile,
        "T_ana": T_ana,
        "T_field": T_final.detach().cpu().numpy(),
        "phi_field": phi.detach().cpu().numpy(),
        "history": history,
        "nu": nu,
        "alpha": alpha,
        "Ste_s": Ste_s,
        "Ste_l": Ste_l,
        "lam": lam,
        "T_left": T_left,
        "T_right": T_right,
        "T_freeze": T_freeze,
        "cp": cp,
        "L_latent": L_latent,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }


# =========================================================================== #
# Plotting
# =========================================================================== #


def save_plots(result: dict, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Temperature profile
    ax = axes[0, 0]
    ax.plot(result["T_profile"], "b-o", markersize=3, label="LBM")
    ax.plot(result["T_ana"], "r--", lw=2, label="Analytical")
    ax.axvline(result["s_lbm"], color="b", ls=":", alpha=0.5, label=f"s_LBM={result['s_lbm']:.1f}")
    ax.axvline(result["s_ana"], color="r", ls=":", alpha=0.5, label=f"s_ana={result['s_ana']:.1f}")
    ax.set_xlabel("x (lattice)")
    ax.set_ylabel("T")
    ax.set_title("Temperature profile (final)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Phase field
    ax = axes[0, 1]
    phi_1d = result["phi_field"][0, 0, :]
    ax.plot(phi_1d, "g-", lw=1.5)
    ax.axvline(result["s_lbm"], color="b", ls=":", label=f"s={result['s_lbm']:.1f}")
    ax.set_xlabel("x (lattice)")
    ax.set_ylabel("φ")
    ax.set_title("Phase field (final)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Interface position vs time
    ax = axes[1, 0]
    if result["history"]:
        steps_arr = [h["step"] for h in result["history"]]
        s_lbm_arr = [h["s_lbm"] for h in result["history"]]
        s_ana_arr = [h["s_ana"] for h in result["history"]]
        ax.plot(steps_arr, s_lbm_arr, "b-o", markersize=4, label="LBM")
        ax.plot(steps_arr, s_ana_arr, "r--", lw=2, label="Analytical")
    ax.set_xlabel("step")
    ax.set_ylabel("interface position s(t)")
    ax.set_title("Ice–liquid interface growth (two-phase)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Error vs time
    ax = axes[1, 1]
    if result["history"]:
        steps_arr = [h["step"] for h in result["history"]]
        err_arr = [h["err"] for h in result["history"]]
        ax.plot(steps_arr, err_arr, "r-o", markersize=4, label="interface error")
        ax.axhline(10, color="k", ls="--", alpha=0.5, label="10% threshold")
    ax.set_xlabel("step")
    ax.set_ylabel("error (%)")
    ax.set_title("Interface position error vs analytical")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Two-phase Stefan  —  Ste_s={result['Ste_s']:.3f}, Ste_l={result['Ste_l']:.3f}, "
        f"λ={result['lam']:.4f}, α={result['alpha']:.4f}, "
        f"err={result['err_final']:.1f}%",
        fontsize=11,
    )
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {p}")


# =========================================================================== #
# CLI
# =========================================================================== #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-phase Stefan solidification benchmark "
                    "(D3Q19 BGK + D2Q5 thermal + Stefan condition).",
    )
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=1)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--tau-T", type=float, default=0.8)
    p.add_argument("--T-left", type=float, default=-1.0, help="Cold wall (solid side)")
    p.add_argument("--T-right", type=float, default=1.0, help="Hot wall (liquid side)")
    p.add_argument("--T-freeze", type=float, default=0.0)
    p.add_argument("--cp", type=float, default=1.0)
    p.add_argument("--L-latent", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--output", default="outputs/stefan_two_phase.png")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 64)
    print("  TWO-PHASE STEFAN SOLIDIFICATION BENCHMARK")
    print("  D3Q19 BGK (momentum) + D2Q5 (thermal) + Stefan condition")
    print("=" * 64)

    result = run_stefan_two_phase(
        nx=args.nx, ny=args.ny, nz=args.nz,
        tau=args.tau, tau_T=args.tau_T,
        T_left=args.T_left, T_right=args.T_right, T_freeze=args.T_freeze,
        cp=args.cp, L_latent=args.L_latent,
        steps=args.steps, device=args.device,
        log_every=args.log_every,
    )

    save_plots(result, args.output)

    # Pass/fail
    ok = True
    if result["s_lbm"] > 1.0:
        print(f"\n  ✓ PASS  ice grew from left wall  (s = {result['s_lbm']:.2f})")
    else:
        print(f"\n  ✗ FAIL  ice did not grow  (s = {result['s_lbm']:.2f})")
        ok = False
    if result["err_final"] < 10.0:
        print(f"  ✓ PASS  interface error = {result['err_final']:.2f}%  (< 10%)")
    else:
        print(f"  ✗ FAIL  interface error = {result['err_final']:.2f}%  (≥ 10%)")
        ok = False
    if result["T_err_rms"] < 10.0:
        print(f"  ✓ PASS  temperature RMS error = {result['T_err_rms']:.2f}%  (< 10%)")
    else:
        print(f"  ✗ FAIL  temperature RMS error = {result['T_err_rms']:.2f}%  (≥ 10%)")
        ok = False

    print(
        f"\n  Ste_s={result['Ste_s']:.4f}  Ste_l={result['Ste_l']:.4f}  "
        f"λ={result['lam']:.6f}  Fo={result['step'] * result['alpha']:.4f}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""四极子声源辐射基准测试 (D3Q19 BGK)

A quadrupole acoustic source is created by two opposing dipoles.  It radiates
sound with a cos(2θ) directivity pattern — maxima at θ=0°, 90°, 180°, 270°
and nulls at θ=45°, 135°.

Setup:
 - Grid: nx=300, ny=300, nz=1 (2D D3Q19 BGK)
 - Quadrupole source: two opposing Guo force sources:
     Force A at (cx+3, cy):  F = +F0·sin(ωt)  in +x
     Force B at (cx-3, cy):  F = -F0·sin(ωt)  in +x
                           (opposite force, opposite position)
     This creates a longitudinal quadrupole oriented in x.
     F0=0.01, omega=0.1
 
   NOTE: A longitudinal-only force quadrupole (two opposing forces in x)
   produces a cos²(θ) directivity, not cos(2θ), because its pressure field
   contains a monopole component:  p' ∝ -k²/2·H₀(kr) + k²/2·cos(2θ)·H₂(kr).
   To obtain the pure cos(2θ)·|H₂(kr)| pattern, we superpose two orthogonal
   longitudinal quadrupoles with opposite phase (x and y), so the source
   term becomes 2d·F0·sin(ωt)·[∂²/∂x² − ∂²/∂y²]δ(r), whose pressure field
   is exactly  p' ∝ cos(2θ)·H₂⁽¹⁾(kr).  The four forces are:
     Force A at (cx+3, cy):  F = +F0·sin(ωt)  in +x   (x-dipole, + side)
     Force B at (cx-3, cy):  F = -F0·sin(ωt)  in +x   (x-dipole, − side)
     Force C at (cx, cy+3):  F = -F0·sin(ωt)  in +y   (y-dipole, opposite phase, + side)
     Force D at (cx, cy-3):  F = +F0·sin(ωt)  in +y   (y-dipole, opposite phase, − side)
 - Sponge layer (target equilibrium, width=50) at all boundaries.
 - Monitors: at distance r=40,50,60 from centre, at angles
   θ=0°,45°,90°,135°,180°.

Analytical solution for 2D quadrupole far-field:
  Pressure:  p'(r,θ) ∝ cos(2θ) · |H_2^(1)(kr)|
  where H_2 is the Hankel function of order 2, k=omega/cs.
  Directivity:  |p'(r,θ)| / |p'(r,0°)| = |cos(2θ)|

Validation:
  1. Directivity — pressure amplitude at different angles at fixed r,
     compare with |cos(2θ)|.  Target: <15% error.
  2. Radial decay — pressure amplitude vs r at θ=0°, compare with
     |H_2^(1)(kr)| decay.  Target: <15% error.

Run:
    PYTHONPATH=src python examples/benchmark_quadrupole_source.py --device cpu --steps 1000
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
from scipy.special import hankel1

# Ensure src/ is importable even without PYTHONPATH
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _extract_amplitude(steps_arr, values, omega):
    """Extract the oscillation amplitude at frequency *omega* via a
    single-frequency DFT (projection onto sin/cos at the known frequency).

    Uses the second half of the time series to avoid initial transients.
    Returns a non-negative float.
    """
    t = np.asarray(steps_arr, dtype=np.float64)
    u = np.asarray(values, dtype=np.float64)
    n = len(t)
    if n < 10:
        return 0.0
    half = n // 2
    t2 = t[half:]
    u2 = u[half:] - np.mean(u[half:])  # remove DC offset
    N = len(t2)
    sin_proj = (2.0 / N) * np.sum(u2 * np.sin(omega * t2))
    cos_proj = (2.0 / N) * np.sum(u2 * np.cos(omega * t2))
    return float(np.sqrt(sin_proj ** 2 + cos_proj ** 2))


# --------------------------------------------------------------------------- #
# Quadrupole source benchmark
# --------------------------------------------------------------------------- #

def run_quadrupole_source(
    nx: int = 300,
    ny: int = 300,
    nz: int = 1,
    tau: float = 0.55,
    force_amp: float = 0.01,
    omega: float = 0.1,
    steps: int = 1000,
    device: str = "cpu",
    log_every: int = 200,
    record_every: int = 5,
    source_sep: int = 3,
) -> dict:
    """Run the quadrupole acoustic-source benchmark and print PASS/FAIL verdicts."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    cs = math.sqrt(cs2)

    cx, cy = nx // 2, ny // 2
    k = omega / cs
    lam = 2.0 * math.pi * cs / omega

    # --- Monitor points: r = 40, 50, 60 at θ = 0°, 45°, 90°, 135°, 180° ----
    monitor_r = [40, 50, 60]
    monitor_angles = [0, 45, 90, 135, 180]  # degrees, measured from +x

    margin = 3  # boundary margin for clamping monitor coords
    monitor_pts: dict[tuple[int, int], tuple[int, int, float]] = {}
    for r in monitor_r:
        for ang_deg in monitor_angles:
            ang = math.radians(ang_deg)
            mx = int(round(cx + r * math.cos(ang)))
            my = int(round(cy + r * math.sin(ang)))
            mx = max(margin, min(nx - 1 - margin, mx))
            my = max(margin, min(ny - 1 - margin, my))
            monitor_pts[(r, ang_deg)] = (mx, my, ang)

    # --- Source points (four Guo force sources forming a quadrupole) -------
    # x-dipole (Q_xx component):
    src_a = (cx + source_sep, cy)  # +x side:  F = +F0·sin(ωt) in +x
    src_b = (cx - source_sep, cy)  # -x side:  F = -F0·sin(ωt) in +x
    # y-dipole (-Q_yy component, opposite phase to cancel the monopole):
    src_c = (cx, cy + source_sep)  # +y side:  F = -F0·sin(ωt) in +y
    src_d = (cx, cy - source_sep)  # -y side:  F = +F0·sin(ωt) in +y

    # --- Initialise field ---------------------------------------------------
    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # --- Lattice weights & velocities (for Guo forcing) --------------------
    w_dev = W.to(dev)                # shape (19,)
    c_dev = C.to(dev).float()        # shape (19, 3)
    cx_dev = c_dev[:, 0]             # shape (19,)  — x-component of c_i
    cy_dev = c_dev[:, 1]             # shape (19,)  — y-component of c_i

    # --- Guo forcing coefficient -------------------------------------------
    # delta_f_i = w_i · (1 - 1/(2τ)) · (1/cs²) · (c_i · F)
    # For F = (Fx, 0, 0):  c_i·F = c_ix · Fx,  and 1/cs² = 3
    guo_coeff = (1.0 - 1.0 / (2.0 * tau)) * 3.0  # 1/cs² = 3 for D3Q19

    # --- Sponge layer (sin² damping, width=50) ------------------------------
    sponge_width = 50
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nx, device=dev),
        indexing="ij",
    )
    dist_x = torch.minimum(xx, nx - 1 - xx)
    dist_y = torch.minimum(yy, ny - 1 - yy)
    dist_edge = torch.minimum(dist_x, dist_y).float()
    damping = torch.where(
        dist_edge < sponge_width,
        torch.sin(math.pi * dist_edge / (2.0 * sponge_width)) ** 2,
        torch.ones_like(dist_edge, device=dev),
    )

    # --- Far-field BC: precomputed equilibrium (ρ=1, u=0) -------------------
    feq_bnd = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # --- History recording --------------------------------------------------
    history: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for key in monitor_pts:
        history[key] = []

    # --- Header -------------------------------------------------------------
    print("=" * 64)
    print("  四极子声源辐射基准测试 (D3Q19 BGK)")
    print("=" * 64)
    print(f"  网格: {nx} × {ny} × {nz}   设备: {device}")
    print(f"  声速 cs={cs:.4f}  波长 λ={lam:.1f}  波数 k={k:.4f}")
    print(f"  源类型: 力源 (Guo forcing, 四点 → 纯四极子)")
    print(f"  源A ({src_a[0]},{src_a[1]}):  Fx = +{force_amp}·sin({omega}·t)")
    print(f"  源B ({src_b[0]},{src_b[1]}):  Fx = -{force_amp}·sin({omega}·t)")
    print(f"  源C ({src_c[0]},{src_c[1]}):  Fy = -{force_amp}·sin({omega}·t)")
    print(f"  源D ({src_d[0]},{src_d[1]}):  Fy = +{force_amp}·sin({omega}·t)")
    print(f"  τ={tau}  步数={steps}  吸收层宽度={sponge_width}")
    print(f"  监测点: r={monitor_r}, θ={monitor_angles}°")
    print("-" * 64)
    print(f"  {'step':>6s}  {'ux_max':>10s}  {'drho_max':>10s}")
    print("-" * 64)

    # --- Time loop ----------------------------------------------------------
    for step in range(1, steps + 1):
        # 1) Collision
        f = collide_bgk3d(f, tau)

        # 2) Quadrupole source (Guo forcing, applied after collision)
        #    Superpose two orthogonal longitudinal quadrupoles with opposite
        #    phase so the source term is [∂²/∂x² − ∂²/∂y²]δ(r), giving
        #    p' ∝ cos(2θ)·H₂⁽¹⁾(kr) exactly.
        force_val = force_amp * math.sin(omega * step)
        # x-dipole (Q_xx): +F at +d, −F at −d
        delta_fx = w_dev * guo_coeff * cx_dev * force_val  # shape (19,)
        f[:, 0, src_a[1], src_a[0]] += delta_fx   # +F0·sin(ωt) in +x
        f[:, 0, src_b[1], src_b[0]] -= delta_fx   # −F0·sin(ωt) in +x
        # y-dipole (−Q_yy, opposite phase): −F at +d, +F at −d
        delta_fy = w_dev * guo_coeff * cy_dev * force_val  # shape (19,)
        f[:, 0, src_c[1], src_c[0]] -= delta_fy   # −F0·sin(ωt) in +y
        f[:, 0, src_d[1], src_d[0]] += delta_fy   # +F0·sin(ωt) in +y

        # 3) Streaming
        f = stream3d(f)

        # 4) Far-field BC: fixed equilibrium at all four edges
        f[:, :, 0:margin, :] = feq_bnd[:, :, 0:margin, :]
        f[:, :, -margin:, :] = feq_bnd[:, :, -margin:, :]
        f[:, :, :, 0:margin] = feq_bnd[:, :, :, 0:margin]
        f[:, :, :, -margin:] = feq_bnd[:, :, :, -margin:]

        # 5) Macroscopic (shared by sponge layer and recording)
        rho, ux, uy, uz = macroscopic3d(f)

        # 6) Record monitor points — pressure perturbation p' ∝ (ρ - ρ0)
        #    (interior points are unaffected by the sponge below)
        if step % record_every == 0 or step == steps:
            for key, (mx, my, _ang) in monitor_pts.items():
                history[key].append(
                    (step, float(rho[0, my, mx].item()))
                )

        # 7) Sponge layer: blend toward TARGET equilibrium (absorbs acoustic wave)
        f = feq_bnd + (f - feq_bnd) * damping

        # 8) Logging
        if step % log_every == 0 or step == steps:
            print(
                f"  {step:6d}  {float(ux.abs().max()):10.6f}  "
                f"{float((rho - rho0).abs().max()):10.6f}",
                flush=True,
            )

    # ===================================================================== #
    # Analysis
    # ===================================================================== #
    print("\n" + "=" * 64)
    print("  分析结果")
    print("=" * 64)

    # --- Compute pressure amplitude at each monitor -------------------------
    # p' ∝ (ρ - ρ0);  the DC offset is removed inside _extract_amplitude.
    amplitudes: dict[tuple[int, int], float] = {}
    for key, hist in history.items():
        steps_arr = [h[0] for h in hist]
        rho_arr = [h[1] for h in hist]
        amplitudes[key] = _extract_amplitude(steps_arr, rho_arr, omega)

    # --- Print raw amplitudes -----------------------------------------------
    print("\n  压力振幅 |p'| (各监测点):")
    header = "  r\\θ  "
    for ang_deg in monitor_angles:
        header += f"  θ={ang_deg:3d}°   "
    print(header)
    for r in monitor_r:
        row = f"  {r:3d}  "
        for ang_deg in monitor_angles:
            row += f"  {amplitudes[(r, ang_deg)]:.6f}"
        print(row)

    # ----------------------------------------------------------------------- #
    # Validation 1: Directivity (|cos 2θ| pattern)
    # ----------------------------------------------------------------------- #
    print("\n  --- 验证1: 指向性 (|cos 2θ| 图样) ---")
    directivity_r = 50  # primary directivity measurement radius
    print(f"  固定 r={directivity_r}, 比较 |p'(r,θ)|/|p'(r,0°)| 与 |cos(2θ)|")
    print(f"  {'θ':>6s}  {'|cos2θ|':>8s}  {'测量值':>10s}  {'误差':>8s}")

    ref_amp = amplitudes[(directivity_r, 0)]
    dir_errors: list[float] = []
    for ang_deg in monitor_angles:
        expected = abs(math.cos(2.0 * math.radians(ang_deg)))
        measured = (
            amplitudes[(directivity_r, ang_deg)] / ref_amp
            if ref_amp > 1e-15
            else 0.0
        )
        err = abs(measured - expected)
        dir_errors.append(err)
        print(f"  {ang_deg:5d}°  {expected:8.4f}  {measured:10.4f}  {err * 100:7.2f}%")

    dir_avg_err = float(np.mean(dir_errors)) * 100
    dir_pass = dir_avg_err < 15.0
    print(f"\n  平均误差: {dir_avg_err:.2f}%  (目标 < 15%)")
    print(f"  指向性验证: {'PASS' if dir_pass else 'FAIL'}")

    # Directivity at all radii (supplementary)
    print(f"\n  各半径指向性误差 (补充):")
    for r in monitor_r:
        ref = amplitudes[(r, 0)]
        errs = []
        for ang_deg in monitor_angles:
            expected = abs(math.cos(2.0 * math.radians(ang_deg)))
            measured = amplitudes[(r, ang_deg)] / ref if ref > 1e-15 else 0.0
            errs.append(abs(measured - expected))
        print(f"    r={r}: 平均误差 = {float(np.mean(errs)) * 100:.2f}%")

    # ----------------------------------------------------------------------- #
    # Validation 2: Radial decay (|H_2^(1)(kr)|)
    # ----------------------------------------------------------------------- #
    print("\n  --- 验证2: 径向衰减 (|H₂⁽¹⁾(kr)| 衰减) ---")
    print(f"  固定 θ=0°, 比较 |p'(r,0°)| 衰减与 |H₂⁽¹⁾(kr)| 衰减")
    print(
        f"  {'r':>5s}  {'|H2(kr)|':>12s}  {'归一化H2':>10s}"
        f"  {'归一化测量':>12s}  {'误差':>8s}"
    )

    h2_ref = abs(hankel1(2, k * monitor_r[0]))
    amp_ref = amplitudes[(monitor_r[0], 0)]
    decay_errors: list[float] = []
    for r in monitor_r:
        h2_val = abs(hankel1(2, k * r))
        h2_norm = h2_val / h2_ref if h2_ref > 0 else 0.0
        measured_norm = amplitudes[(r, 0)] / amp_ref if amp_ref > 1e-15 else 0.0
        err = (
            abs(measured_norm - h2_norm) / h2_norm
            if h2_norm > 1e-15
            else 0.0
        )
        decay_errors.append(err * 100)
        print(
            f"  {r:4d}  {h2_val:12.6f}  {h2_norm:10.4f}  "
            f"{measured_norm:12.4f}  {err * 100:7.2f}%"
        )

    decay_avg_err = float(np.mean(decay_errors))
    decay_pass = decay_avg_err < 15.0
    print(f"\n  平均误差: {decay_avg_err:.2f}%  (目标 < 15%)")
    print(f"  径向衰减验证: {'PASS' if decay_pass else 'FAIL'}")

    # ----------------------------------------------------------------------- #
    # Summary
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 64)
    all_pass = dir_pass and decay_pass
    print(f"  总体结果: {'PASS' if all_pass else 'FAIL'}")
    print(
        f"    指向性:   {'PASS' if dir_pass else 'FAIL'}"
        f"  (误差 {dir_avg_err:.2f}%, 目标 < 15%)"
    )
    print(
        f"    径向衰减: {'PASS' if decay_pass else 'FAIL'}"
        f"  (误差 {decay_avg_err:.2f}%, 目标 < 15%)"
    )
    print("=" * 64)

    return {
        "amplitudes": amplitudes,
        "dir_avg_err": dir_avg_err,
        "decay_avg_err": decay_avg_err,
        "dir_pass": dir_pass,
        "decay_pass": decay_pass,
        "all_pass": all_pass,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="四极子声源辐射基准测试 (D3Q19 BGK)")
    p.add_argument("--nx", type=int, default=300)
    p.add_argument("--ny", type=int, default=300)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--force-amp", type=float, default=0.01)
    p.add_argument("--omega", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--source-sep", type=int, default=3,
                   help="Separation of each force source from the centre (lattice units)")
    args = p.parse_args()

    run_quadrupole_source(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        force_amp=args.force_amp,
        omega=args.omega,
        steps=args.steps,
        device=args.device,
        log_every=args.log_every,
        source_sep=args.source_sep,
    )


if __name__ == "__main__":
    main()

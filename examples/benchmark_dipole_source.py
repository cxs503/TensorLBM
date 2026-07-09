#!/usr/bin/env python
"""偶极子声源辐射基准测试 (D3Q19 BGK)

A dipole acoustic source radiates sound with a cos(θ) directivity pattern,
complementary to the monopole (pulsating sphere) which radiates omnidirectionally.

Two source implementations are provided:

  --source-type density  (default, per task spec)
    Two adjacent point sources with OPPOSITE phase at the centre:
      Source A at (cx+2, cy):  rho = 1 + delta_rho * sin(omega*t)
      Source B at (cx-2, cy):  rho = 1 - delta_rho * sin(omega*t)
    A *soft* density source is used: the equilibrium perturbation w·δρ is
    **added** to the existing populations (not overwritten), preserving the
    local velocity field.

  --source-type force
    A single point force  F_x = F0 * sin(omega*t)  at the centre, applied
    via the Guo forcing scheme.  This is physically equivalent to a dipole
    and avoids two-source interaction artefacts.

Analytical 2D far-field (kr >> 1):
  u_r(r,θ) ∝ cos(θ) · |H_1^(1)(kr)|
  Normalised directivity:  |u_r(r,θ)| / |u_r(r,0°)| = |cos(θ)|

Validation:
  1. Directivity — velocity amplitude vs angle at fixed r, compare with cos(θ).
     Target: <10% error.
  2. Radial decay — velocity amplitude vs r at θ=0°, compare with |H_1^(1)(kr)|.
     Target: <15% error.

Run:
    PYTHONPATH=src python examples/benchmark_dipole_source.py --device cpu --steps 1200
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
# Dipole source benchmark
# --------------------------------------------------------------------------- #

def run_dipole_source(
    nx: int = 300,
    ny: int = 300,
    nz: int = 1,
    tau: float = 0.55,
    delta_rho: float = 0.01,
    omega: float = 0.1,
    steps: int = 1200,
    device: str = "cpu",
    log_every: int = 200,
    record_every: int = 5,
    source_type: str = "density",
    force_amp: float = 0.01,
) -> dict:
    """Run the dipole acoustic-source benchmark and print PASS/FAIL verdicts."""
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

    # --- Source points (density source) -------------------------------------
    src_a = (cx + 2, cy)  # +x side
    src_b = (cx - 2, cy)  # -x side

    # --- Initialise field ---------------------------------------------------
    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # --- Lattice weights & velocities (for source) --------------------------
    w_dev = W.to(dev)  # shape (19,)
    c_dev = C.to(dev).float()  # shape (19, 3)
    cx_dev = c_dev[:, 0]  # shape (19,)

    # --- Source masks (density source) --------------------------------------
    source_a = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    source_a[0, src_a[1], src_a[0]] = True
    source_b = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    source_b[0, src_b[1], src_b[0]] = True

    # --- Guo forcing coefficient (force source) -----------------------------
    guo_coeff = (1.0 - 1.0 / (2.0 * tau)) * 9.0  # = 0.818 for tau=0.55

    # --- Sponge layer (sin² damping, width=40) ------------------------------
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
    history: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for key in monitor_pts:
        history[key] = []

    # --- Header -------------------------------------------------------------
    print("=" * 64)
    print("  偶极子声源辐射基准测试 (D3Q19 BGK)")
    print("=" * 64)
    print(f"  网格: {nx} × {ny} × {nz}   设备: {device}")
    print(f"  声速 cs={cs:.4f}  波长 λ={lam:.1f}  波数 k={k:.4f}")
    if source_type == "density":
        print(f"  源类型: 密度源 (软源, 两点反相)")
        print(f"  源A ({src_a[0]},{src_a[1]}):  ρ += {delta_rho}·sin({omega}·t)")
        print(f"  源B ({src_b[0]},{src_b[1]}):  ρ −= {delta_rho}·sin({omega}·t)")
    else:
        print(f"  源类型: 力源 (Guo forcing, 单点)")
        print(f"  力源 ({cx},{cy}):  Fx = {force_amp}·sin({omega}·t)")
    print(f"  τ={tau}  步数={steps}  吸收层宽度={sponge_width}")
    print(f"  监测点: r={monitor_r}, θ={monitor_angles}°")
    print("-" * 64)
    print(f"  {'step':>6s}  {'ux_max':>10s}  {'drho_max':>10s}")
    print("-" * 64)

    # --- Time loop ----------------------------------------------------------
    for step in range(1, steps + 1):
        # 1) Collision
        f = collide_bgk3d(f, tau)

        # 2) Dipole source
        if source_type == "force":
            # Force-based dipole (Guo forcing at centre, applied after collision)
            # delta_f[q] = w[q] * (1-1/(2τ)) * 9 * c_x[q] * F0 * sin(ωt)
            force_val = force_amp * math.sin(omega * step)
            delta_f = w_dev * guo_coeff * cx_dev * force_val  # shape (19,)
            f[:, 0, cy, cx] += delta_f

        # 3) Streaming
        f = stream3d(f)

        if source_type == "density":
            # Soft density source (after streaming)
            # feq(ρ+δρ,0) − feq(ρ,0) = w·δρ  (isotropic, preserves velocity)
            delta_val = delta_rho * math.sin(omega * step)
            f[:, 0, src_a[1], src_a[0]] += w_dev * delta_val
            f[:, 0, src_b[1], src_b[0]] -= w_dev * delta_val

        # 4) Far-field BC: fixed equilibrium at all four edges
        f[:, :, 0:margin, :] = feq_bnd[:, :, 0:margin, :]
        f[:, :, -margin:, :] = feq_bnd[:, :, -margin:, :]
        f[:, :, :, 0:margin] = feq_bnd[:, :, :, 0:margin]
        f[:, :, :, -margin:] = feq_bnd[:, :, :, -margin:]

        # 5) Macroscopic (shared by sponge layer and recording)
        rho, ux, uy, uz = macroscopic3d(f)

        # 6) Record monitor points (before sponge — interior unaffected)
        if step % record_every == 0 or step == steps:
            for key, (mx, my, _ang) in monitor_pts.items():
                history[key].append(
                    (step, float(ux[0, my, mx].item()), float(uy[0, my, mx].item()))
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

    # --- Compute radial-velocity amplitude at each monitor ------------------
    # u_r = ux·cos(θ) + uy·sin(θ)  (projection onto radial direction)
    amplitudes: dict[tuple[int, int], float] = {}
    for key, hist in history.items():
        _r, _ang_deg = key
        _mx, _my, ang = monitor_pts[key]
        steps_arr = [h[0] for h in hist]
        ur_arr = [h[1] * math.cos(ang) + h[2] * math.sin(ang) for h in hist]
        amplitudes[key] = _extract_amplitude(steps_arr, ur_arr, omega)

    # --- Debug: print ux/uy amplitudes at θ=0° and θ=180° for r=50 ----------
    print("\n  调试: r=50 处的 ux/uy 分量振幅")
    for ang_deg in [0, 180]:
        key = (50, ang_deg)
        hist = history[key]
        steps_arr = [h[0] for h in hist]
        ux_arr = [h[1] for h in hist]
        uy_arr = [h[2] for h in hist]
        ux_amp = _extract_amplitude(steps_arr, ux_arr, omega)
        uy_amp = _extract_amplitude(steps_arr, uy_arr, omega)
        print(f"    θ={ang_deg:3d}°: ux_amp={ux_amp:.6f}  uy_amp={uy_amp:.6f}")

    # --- Print raw amplitudes -----------------------------------------------
    print("\n  径向速度振幅 |u_r| (各监测点):")
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
    # Validation 1: Directivity (cos θ pattern)
    # ----------------------------------------------------------------------- #
    print("\n  --- 验证1: 指向性 (cos θ 图样) ---")
    directivity_r = 50  # primary directivity measurement radius
    print(f"  固定 r={directivity_r}, 比较 |u_r(r,θ)|/|u_r(r,0°)| 与 |cos(θ)|")
    print(f"  {'θ':>6s}  {'|cosθ|':>8s}  {'测量值':>10s}  {'误差':>8s}")

    ref_amp = amplitudes[(directivity_r, 0)]
    dir_errors: list[float] = []
    for ang_deg in monitor_angles:
        expected = abs(math.cos(math.radians(ang_deg)))
        measured = (
            amplitudes[(directivity_r, ang_deg)] / ref_amp
            if ref_amp > 1e-15
            else 0.0
        )
        err = abs(measured - expected)
        dir_errors.append(err)
        print(f"  {ang_deg:5d}°  {expected:8.4f}  {measured:10.4f}  {err * 100:7.2f}%")

    dir_avg_err = float(np.mean(dir_errors)) * 100
    dir_pass = dir_avg_err < 10.0
    print(f"\n  平均误差: {dir_avg_err:.2f}%  (目标 < 10%)")
    print(f"  指向性验证: {'PASS' if dir_pass else 'FAIL'}")

    # Directivity at all radii (supplementary)
    print(f"\n  各半径指向性误差 (补充):")
    for r in monitor_r:
        ref = amplitudes[(r, 0)]
        errs = []
        for ang_deg in monitor_angles:
            expected = abs(math.cos(math.radians(ang_deg)))
            measured = amplitudes[(r, ang_deg)] / ref if ref > 1e-15 else 0.0
            errs.append(abs(measured - expected))
        print(f"    r={r}: 平均误差 = {float(np.mean(errs)) * 100:.2f}%")

    # ----------------------------------------------------------------------- #
    # Validation 2: Radial decay (|H_1^(1)(kr)|)
    # ----------------------------------------------------------------------- #
    print("\n  --- 验证2: 径向衰减 (|H₁⁽¹⁾(kr)| 衰减) ---")
    print(f"  固定 θ=0°, 比较 |u_r(r,0°)| 衰减与 |H₁⁽¹⁾(kr)| 衰减")
    print(
        f"  {'r':>5s}  {'|H1(kr)|':>12s}  {'归一化H1':>10s}"
        f"  {'归一化测量':>12s}  {'误差':>8s}"
    )

    h1_ref = abs(hankel1(1, k * monitor_r[0]))
    amp_ref = amplitudes[(monitor_r[0], 0)]
    decay_errors: list[float] = []
    for r in monitor_r:
        h1_val = abs(hankel1(1, k * r))
        h1_norm = h1_val / h1_ref if h1_ref > 0 else 0.0
        measured_norm = amplitudes[(r, 0)] / amp_ref if amp_ref > 1e-15 else 0.0
        err = (
            abs(measured_norm - h1_norm) / h1_norm
            if h1_norm > 1e-15
            else 0.0
        )
        decay_errors.append(err * 100)
        print(
            f"  {r:4d}  {h1_val:12.6f}  {h1_norm:10.4f}  "
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
        f"  (误差 {dir_avg_err:.2f}%, 目标 < 10%)"
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
    p = argparse.ArgumentParser(description="偶极子声源辐射基准测试 (D3Q19 BGK)")
    p.add_argument("--nx", type=int, default=300)
    p.add_argument("--ny", type=int, default=300)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--delta-rho", type=float, default=0.01)
    p.add_argument("--omega", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument(
        "--source-type",
        choices=["density", "force"],
        default="density",
        help="Source implementation: 'density' (two-point, per task spec) or 'force' (Guo forcing)",
    )
    p.add_argument("--force-amp", type=float, default=0.01)
    args = p.parse_args()

    run_dipole_source(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        delta_rho=args.delta_rho,
        omega=args.omega,
        steps=args.steps,
        device=args.device,
        log_every=args.log_every,
        source_type=args.source_type,
        force_amp=args.force_amp,
    )


if __name__ == "__main__":
    main()

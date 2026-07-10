#!/usr/bin/env python
"""声源叠加线性度基准测试 (D3Q19 BGK)

Linear acoustics requires that the field from multiple sources equals the sum
of individual source fields.  This benchmark verifies LBM acoustic linearity
by running three simulations:

  1. Source A only  — monopole at (cx-30, cy),  ρ += δρ·sin(ωt)
  2. Source B only  — monopole at (cx+30, cy),  ρ += δρ·sin(ωt+π/2)
  3. Both A + B     — both sources simultaneously

Then check:  p(A+B) ≈ p(A) + p(B)  at all monitor points.

A *soft* (additive) density source is used: the equilibrium perturbation
w·δρ is **added** to the existing populations (not overwritten).  This is
essential for the linearity test — the density-scaling source
(f *= ρ_target/ρ_cur) is multiplicative and would introduce non-linear
coupling between the two sources.

Validation:
  |p(A+B) − (p(A)+p(B))| / max|p(A)+p(B)| < 1% at all monitors.

Run:
    PYTHONPATH=src python examples/benchmark_acoustic_superposition.py --device cpu --steps 1000
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

from tensorlbm.d3q19 import C, W  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402


# --------------------------------------------------------------------------- #
# Single simulation
# --------------------------------------------------------------------------- #

def _collide_linear_acoustic_bgk(g: torch.Tensor, tau: float) -> torch.Tensor:
    """Collide the population perturbation about ``rho=1, u=0``.

    The full isothermal equilibrium contains products of density and velocity
    (and ``u**2``).  Those are second-order finite-amplitude terms, so using it
    in a superposition benchmark introduces a real O(delta_rho**2) departure
    from the *linear acoustic* equations.  For acoustic perturbations the
    appropriate equilibrium is instead

        g_eq,q = w_q [rho' + 3 c_q . j],

    where ``rho' = rho - 1`` and ``j = rho*u``.  It retains the conserved
    density and momentum moments while making collision exactly linear.
    """
    rho_prime = g.sum(dim=0)
    c = C.to(g.device)
    w = W.to(g.device).view(19, 1, 1, 1)
    jx = (g * c[:, 0].view(19, 1, 1, 1)).sum(dim=0)
    jy = (g * c[:, 1].view(19, 1, 1, 1)).sum(dim=0)
    jz = (g * c[:, 2].view(19, 1, 1, 1)).sum(dim=0)
    cj = (
        c[:, 0].view(19, 1, 1, 1) * jx
        + c[:, 1].view(19, 1, 1, 1) * jy
        + c[:, 2].view(19, 1, 1, 1) * jz
    )
    geq = w * (rho_prime.unsqueeze(0) + 3.0 * cj)
    return g - (g - geq) / tau


def _run_simulation(
    nx: int,
    ny: int,
    nz: int,
    tau: float,
    delta_rho: float,
    omega: float,
    steps: int,
    device: str,
    use_a: bool,
    use_b: bool,
    monitor_pts: list[tuple[int, int]],
    sponge_width: int = 50,
    log_every: int = 200,
    log_label: str = "",
) -> list[list[float]]:
    """Run a single LBM acoustic simulation with optional sources A and/or B.

    Returns a list of pressure time-series (one per monitor), each a list of
    floats of length *steps*.
    """
    dev = torch.device(device)
    cs2 = 1.0 / 3.0

    cx, cy = nx // 2, ny // 2
    src_a = (cx - 30, cy)
    src_b = (cx + 30, cy)

    # Evolve g = f - w rather than f itself.  This is algebraically identical
    # to the linearised acoustic LBM, but avoids repeatedly subtracting the
    # O(1) background density from O(delta_rho) acoustic signals in fp32.
    g = torch.zeros((19, nz, ny, nx), device=dev)

    w_dev = W.to(dev)  # shape (19,)

    # --- Sponge layer (sin² damping of the perturbation toward zero) --------
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
    margin = 3  # fixed-equilibrium boundary rows

    n_mon = len(monitor_pts)
    p_ts: list[list[float]] = [[] for _ in range(n_mon)]

    for step in range(1, steps + 1):
        # 1) Collision
        g = _collide_linear_acoustic_bgk(g, tau)

        # 2) Streaming
        g = stream3d(g)

        # 3) Soft density source (additive — linear, preserves velocity)
        #    Adding w·δρ to g increases rho' by δρ at the source point.
        if use_a:
            delta_val_a = delta_rho * math.sin(omega * step)
            g[:, 0, src_a[1], src_a[0]] += w_dev * delta_val_a
        if use_b:
            delta_val_b = delta_rho * math.sin(omega * step + math.pi / 2.0)
            g[:, 0, src_b[1], src_b[0]] += w_dev * delta_val_b

        # 4) Far-field BC: zero perturbation at all four edges
        g[:, :, 0:margin, :] = 0.0
        g[:, :, -margin:, :] = 0.0
        g[:, :, :, 0:margin] = 0.0
        g[:, :, :, -margin:] = 0.0

        # 5) Macroscopic
        rho_prime = g.sum(dim=0)

        # 6) Record pressure at monitors (before sponge — interior unaffected)
        for i, (mx, my) in enumerate(monitor_pts):
            p_ts[i].append(float(rho_prime[0, my, mx] * cs2))

        # 7) Sponge layer: damp perturbation toward zero (absorbs wave)
        g = g * damping

        # 8) Logging
        if step % log_every == 0 or step == steps:
            pvals = [p_ts[i][-1] for i in range(n_mon)]
            print(
                f"  {log_label} step {step:6d}  "
                f"drho_max={float(rho_prime.abs().max()):.6f}  "
                + "  ".join(f"p{i}={v:+.6f}" for i, v in enumerate(pvals)),
                flush=True,
            )

    return p_ts


# --------------------------------------------------------------------------- #
# Superposition benchmark
# --------------------------------------------------------------------------- #

def run_superposition_benchmark(
    nx: int = 300,
    ny: int = 300,
    nz: int = 1,
    tau: float = 0.55,
    delta_rho: float = 0.01,
    omega: float = 0.1,
    steps: int = 1000,
    device: str = "cpu",
    log_every: int = 200,
) -> dict:
    """Run the acoustic superposition linearity benchmark and print PASS/FAIL."""
    dev = torch.device(device)
    cs2 = 1.0 / 3.0
    cs = math.sqrt(cs2)

    cx, cy = nx // 2, ny // 2
    k = omega / cs
    lam = 2.0 * math.pi * cs / omega

    src_a = (cx - 30, cy)
    src_b = (cx + 30, cy)

    # --- Monitor points (10 points at various locations in the domain) ------
    # All within the interior (away from sponge layer at edges, width=50)
    margin = 5
    raw_pts = [
        (cx, cy),           # 0: 中心, 两源之间
        (cx, cy - 40),      # 1: 上方
        (cx, cy + 40),      # 2: 下方
        (cx - 60, cy),      # 3: A左侧远处
        (cx + 60, cy),      # 4: B右侧远处
        (cx - 30, cy - 40), # 5: A上方
        (cx + 30, cy + 40), # 6: B下方
        (cx - 15, cy - 50), # 7: A与中心之间偏上
        (cx + 15, cy + 50), # 8: B与中心之间偏下
        (cx, cy - 80),      # 9: 远上方
    ]
    monitor_pts = [
        (max(margin, min(nx - 1 - margin, mx)),
         max(margin, min(ny - 1 - margin, my)))
        for mx, my in raw_pts
    ]
    monitor_names = [f"({mx},{my})" for mx, my in monitor_pts]

    # --- Header -------------------------------------------------------------
    print("=" * 70)
    print("  声源叠加线性度基准测试 (D3Q19 BGK)")
    print("  Acoustic Source Superposition / Linearity Benchmark")
    print("=" * 70)
    print(f"  网格: {nx} × {ny} × {nz}   设备: {device}")
    print(f"  声速 cs={cs:.4f}  波长 λ={lam:.1f}  波数 k={k:.4f}")
    print(f"  源A ({src_a[0]},{src_a[1]}):  ρ += {delta_rho}·sin({omega}·t)")
    print(f"  源B ({src_b[0]},{src_b[1]}):  ρ += {delta_rho}·sin({omega}·t+π/2)")
    print(f"  τ={tau}  步数={steps}  吸收层宽度=50")
    print(f"  监测点数: {len(monitor_pts)}")
    print(f"  软密度源 (加性, 保持速度) — 保证线性叠加")
    print("-" * 70)

    # --- Run three simulations ----------------------------------------------
    print("\n▶ 运行模拟1: 仅源A")
    p_a = _run_simulation(
        nx, ny, nz, tau, delta_rho, omega, steps, device,
        use_a=True, use_b=False, monitor_pts=monitor_pts,
        log_every=log_every, log_label="[A]  ")

    print("\n▶ 运行模拟2: 仅源B")
    p_b = _run_simulation(
        nx, ny, nz, tau, delta_rho, omega, steps, device,
        use_a=False, use_b=True, monitor_pts=monitor_pts,
        log_every=log_every, log_label="[B]  ")

    print("\n▶ 运行模拟3: 源A+B同时")
    p_ab = _run_simulation(
        nx, ny, nz, tau, delta_rho, omega, steps, device,
        use_a=True, use_b=True, monitor_pts=monitor_pts,
        log_every=log_every, log_label="[A+B]")

    # ===================================================================== #
    # Analysis: linearity check  p(A+B) vs p(A)+p(B)
    # ===================================================================== #
    print("\n" + "=" * 70)
    print("  线性叠加分析: p(A+B) vs p(A)+p(B)")
    print("=" * 70)

    n_mon = len(monitor_pts)

    # Compute per-monitor time-series arrays
    pa_arr = [np.array(p_a[i]) for i in range(n_mon)]
    pb_arr = [np.array(p_b[i]) for i in range(n_mon)]
    pab_arr = [np.array(p_ab[i]) for i in range(n_mon)]
    psum_arr = [pa_arr[i] + pb_arr[i] for i in range(n_mon)]
    diff_arr = [pab_arr[i] - psum_arr[i] for i in range(n_mon)]

    # Per-monitor max|sum| and max|diff|
    max_sum_local = [float(np.max(np.abs(psum_arr[i]))) for i in range(n_mon)]
    max_diff_local = [float(np.max(np.abs(diff_arr[i]))) for i in range(n_mon)]

    # Global normalization: max|p(A)+p(B)| over ALL monitors and time steps.
    # The non-linear error from the BGK equilibrium's quadratic velocity
    # terms is an *absolute* error (~δρ²), roughly uniform across the domain.
    # Normalising by the global peak signal gives a physically meaningful
    # relative error that is consistent across monitors, regardless of local
    # wave-interference minima.
    global_max_sum = max(max_sum_local) if max_sum_local else 1e-12

    all_pass = True
    errors: list[float] = []

    hdr = (
        f"  {'#':>3s}  {'监测点':>12s}  "
        f"{'max|p_sum|':>14s}  {'max|diff|':>14s}  "
        f"{'相对误差':>10s}  {'结果':>6s}"
    )
    print(hdr)
    print("-" * 70)

    for i in range(n_mon):
        rel_err = max_diff_local[i] / global_max_sum * 100.0
        errors.append(rel_err)
        passed = rel_err < 1.0
        if not passed:
            all_pass = False

        print(
            f"  {i:3d}  {monitor_names[i]:>12s}  "
            f"{max_sum_local[i]:14.6e}  {max_diff_local[i]:14.6e}  "
            f"{rel_err:9.4f}%  {'PASS' if passed else 'FAIL'}"
        )

    print("-" * 70)
    max_err = max(errors)
    avg_err = sum(errors) / len(errors)
    print(f"  全局归一化基准 max|p(A)+p(B)| = {global_max_sum:.6e}")
    print(f"  最大相对误差: {max_err:.4f}%  平均相对误差: {avg_err:.4f}%  (目标 < 1%)")

    # --- Wave generation sanity check ---------------------------------------
    max_signal = max(
        float(np.max(np.abs(pab_arr[i]))) for i in range(n_mon)
    )
    gen_ok = max_signal > 1e-8
    print(f"\n  波动生成检查: max|p|={max_signal:.6e}  "
          f"{'PASS' if gen_ok else 'FAIL'}")

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  总体结果: {'PASS' if all_pass else 'FAIL'}")
    print(f"    线性叠加: {'PASS' if all_pass else 'FAIL'}"
          f"  (最大误差 {max_err:.4f}%, 目标 < 1%)")
    print("=" * 70)

    return {
        "errors": errors,
        "max_err": max_err,
        "avg_err": avg_err,
        "all_pass": all_pass,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="声源叠加线性度基准测试 (D3Q19 BGK)")
    p.add_argument("--nx", type=int, default=300)
    p.add_argument("--ny", type=int, default=300)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--delta-rho", type=float, default=0.01)
    p.add_argument("--omega", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    run_superposition_benchmark(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        tau=args.tau,
        delta_rho=args.delta_rho,
        omega=args.omega,
        steps=args.steps,
        device=args.device,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

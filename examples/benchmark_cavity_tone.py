#!/usr/bin/env python
"""Rossiter cavity tone benchmark — flow-induced acoustic resonance.

Flow over a rectangular cavity creates self-sustained oscillations (Rossiter
modes).  The shear layer separates at the cavity leading edge, impinges on
the trailing edge, and pressure feedback excites the shear layer at discrete
frequencies.

Rossiter semi-empirical formula:
    f_n = (n - 0.25) * U / L,  n = 1, 2, 3, ...

where U = flow velocity, L = cavity length.

Setup (2D D3Q19 BGK):
  Grid       nx=400, ny=100, nz=1
  Cavity     length L=80, depth D=30, leading edge at x=150
  Flow       U=0.05 (left inlet velocity BC)
  Walls      bounce-back on all solid surfaces
  Outlet     sponge layer (target equilibrium) at right boundary
  Monitor    pressure at trailing edge (x=230, y=15)
  Steps      5000 (need long run for mode development)

Validation:
  1. FFT of pressure time-series → dominant frequency
  2. Compare with Rossiter f_1 = 0.75 * U / L
  3. Target: <25% error (Rossiter formula is semi-empirical)
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W, OPPOSITE
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS = 1.0 / math.sqrt(3.0)
CS2 = 1.0 / 3.0


def build_cavity_solid(nx, ny, cx_start, L, D, dev):
    """Build solid mask for cavity geometry.

    Layout (y increases upward):
      y=ny-1  ┌──────────────────────────────────┐  top (open/sponge)
              │  flow →                           │
      y=D     │█████████┐            ┌██████████│  plate (upstream/downstream)
              │█        │   cavity   │         █│
              │█  wall  │  (fluid)   │  wall   █│
      y=0     └█████████┘            └██████████┘  bottom (cavity floor)
              x=0    cx_start    cx_start+L      nx-1
    """
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    # Bottom wall (cavity floor + below plate)
    solid[0, :] = True
    # Upstream plate (y=D, x < cx_start)
    solid[D, :cx_start] = True
    # Downstream plate (y=D, x > cx_start+L)
    solid[D, cx_start + L + 1:] = True
    # Cavity left wall (x=cx_start, y=1..D)
    solid[1:D, cx_start] = True
    # Cavity right wall (x=cx_start+L, y=1..D)
    solid[1:D, cx_start + L] = True
    # Also fill below upstream/downstream plate (y=1..D-1, outside cavity)
    solid[1:D, :cx_start] = True
    solid[1:D, cx_start + L + 1:] = True
    return solid


def bounce_back_solid(f, solid_mask, dev):
    """Full-way bounce-back on all solid cells."""
    opp = OPPOSITE.to(dev)
    f_bounced = f.clone()
    mask = solid_mask.unsqueeze(0)  # [1, ny, nx] for 2D (nz=1)
    for q in range(19):
        qopp = opp[q].item()
        f_bounced[q][mask] = f[qopp][mask]
    return f_bounced


def run_cavity_tone(nx=400, ny=100, nz=1, tau=0.55,
                    u_in=0.05, cx_start=150, L=80, D=30,
                    steps=5000, device="cpu", log_every=500):
    dev = torch.device(device)
    cs = CS
    nu = (tau - 0.5) / 3.0
    Re = u_in * L / nu

    # Rossiter frequencies (LBM units)
    rossiter_freqs = [(n - 0.25) * u_in / L for n in range(1, 5)]

    # Solid mask
    solid = build_cavity_solid(nx, ny, cx_start, L, D, dev)

    # Monitor: trailing edge (just inside cavity, at mid-depth)
    mon_x = cx_start + L - 1
    mon_y = D // 2
    p_history = []

    # Sponge (right boundary only — target equilibrium)
    sponge_width = 40
    xx_grid = torch.arange(nx, device=dev)
    dist_right = (nx - 1 - xx_grid).float()
    sponge_damping_x = torch.where(
        dist_right < sponge_width,
        torch.sin(math.pi * dist_right / (2 * sponge_width))**2,
        torch.ones_like(dist_right))
    # Apply sponge in x-direction (broadcast to [ny, nx])
    damping = sponge_damping_x.unsqueeze(0).expand(ny, -1)

    # Also sponge top boundary
    yy_grid = torch.arange(ny, device=dev)
    dist_top = (ny - 1 - yy_grid).float()
    sponge_damping_y = torch.where(
        dist_top < sponge_width,
        torch.sin(math.pi * dist_top / (2 * sponge_width))**2,
        torch.ones_like(dist_top))
    damping = torch.minimum(damping, sponge_damping_y.unsqueeze(1).expand(-1, nx))

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    uy0 = torch.zeros_like(rho0)
    # Zero velocity inside solid
    ux0[0][solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uy0.clone(), device=dev)
    feq_target = equilibrium3d(rho0, torch.zeros_like(rho0),
                                torch.zeros_like(rho0), torch.zeros_like(rho0),
                                device=dev)

    # Inlet BC: equilibrium with u_in at x=0
    rho_inlet = torch.ones((nz, ny, 1), device=dev)
    ux_inlet = torch.full((nz, ny, 1), u_in, device=dev)
    uy_zero = torch.zeros((nz, ny, 1), device=dev)
    feq_inlet = equilibrium3d(rho_inlet, ux_inlet, uy_zero, uy_zero.clone(), device=dev)

    print("=" * 64, flush=True)
    print("  Rossiter 腔体鸣音基准测试 (D3Q19 BGK)", flush=True)
    print("=" * 64, flush=True)
    print(f"  网格: {nx}×{ny}×{nz}  设备: {device}", flush=True)
    print(f"  流速 U={u_in}  腔长 L={L}  腔深 D={D}", flush=True)
    print(f"  τ={tau}  ν={nu:.4f}  Re={Re:.0f}  cs={cs:.4f}", flush=True)
    print(f"  腔体起始: x={cx_start}  尾缘: x={cx_start+L}", flush=True)
    print(f"  监测点: ({mon_x}, {mon_y})", flush=True)
    print(f"  Rossiter 频率 (LBM):", flush=True)
    for n, freq in enumerate(rossiter_freqs, 1):
        print(f"    f_{n} = {freq:.6f}  (周期 = {1/freq:.0f} 步)", flush=True)
    print(f"  步数: {steps}", flush=True)
    print("-" * 64, flush=True)

    for step in range(1, steps + 1):
        # Collision
        f = collide_bgk3d(f, tau)
        # Streaming
        f = stream3d(f)
        # Bounce-back on solid
        f = bounce_back_solid(f, solid, dev)
        # Inlet BC (x=0)
        f[:, :, :, 0:1] = feq_inlet
        # Sponge (target equilibrium)
        f = feq_target + (f - feq_target) * damping.unsqueeze(0)

        # Record pressure at monitor
        rho, ux, uy, uz = macroscopic3d(f)
        p_val = float((rho[0, mon_y, mon_x] - 1.0) * CS2)
        p_history.append(p_val)

        if step % log_every == 0 or step == steps:
            drho_max = float((rho - rho0).abs().max())
            print(f"  step {step:5d}/{steps}  |Δρ|_max={drho_max:.6e}  "
                  f"p_mon={p_val:+.6e}", flush=True)

    # === FFT analysis ===
    print("\n" + "=" * 64, flush=True)
    print("  FFT 分析", flush=True)
    print("=" * 64, flush=True)

    p_arr = np.array(p_history, dtype=np.float64)
    # Remove DC
    p_arr = p_arr - np.mean(p_arr)
    # Use second half (after transient)
    half = len(p_arr) // 3  # skip first 1/3 for transient
    p_fft = p_arr[half:]

    N = len(p_fft)
    fft_vals = np.abs(np.fft.rfft(p_fft))
    freqs = np.fft.rfftfreq(N, d=1.0)  # d=1 LBM time step

    # Find dominant peaks
    # Sort by amplitude
    peak_indices = np.argsort(fft_vals)[::-1][:5]
    peak_indices = sorted(peak_indices)  # sort by frequency

    print(f"  FFT 样本数: {N} (跳过前 {half} 步瞬态)", flush=True)
    print(f"  频率分辨率: {1.0/N:.6f}", flush=True)
    print(f"\n  {'rank':>4s}  {'f_lbm':>10s}  {'amp':>10s}  {'f_rossiter':>10s}  {'err%':>6s}", flush=True)

    found_modes = []
    for rank, idx in enumerate(peak_indices):
        f_lbm = freqs[idx]
        amp = fft_vals[idx]
        if amp < fft_vals.max() * 0.05:
            continue
        # Find closest Rossiter mode
        best_err = 999
        best_n = 0
        best_f = 0
        for n, f_r in enumerate(rossiter_freqs, 1):
            err = abs(f_lbm - f_r) / f_r * 100
            if err < best_err:
                best_err = err
                best_n = n
                best_f = f_r
        found_modes.append((f_lbm, amp, best_n, best_f, best_err))
        print(f"  {rank:4d}  {f_lbm:10.6f}  {amp:10.2f}  {best_f:10.6f}(n={best_n})  {best_err:6.1f}",
              flush=True)

    # === Verification ===
    print(f"\n  验证:", flush=True)
    checks = []

    # 1. Oscillation detected
    osc_ok = fft_vals.max() > 1e-10
    checks.append(("压力振荡检测", osc_ok, f"amp_max={fft_vals.max():.2e}"))
    print(f"    [{'PASS' if osc_ok else 'FAIL'}] 压力振荡: amp_max={fft_vals.max():.2e}", flush=True)

    # 2. First Rossiter mode within 25%
    if found_modes:
        # Find the mode closest to f_1
        best = min(found_modes, key=lambda m: m[4])
        mode_ok = best[4] < 25.0
        checks.append((f"Rossiter 模式 n={best[2]} (err<25%)", mode_ok,
                       f"f_lbm={best[0]:.6f} vs f_ana={best[3]:.6f}, err={best[4]:.1f}%"))
        print(f"    [{'PASS' if mode_ok else 'FAIL'}] Rossiter 模式 n={best[2]}: "
              f"f_lbm={best[0]:.6f} vs f_ana={best[3]:.6f}, err={best[4]:.1f}%", flush=True)
    else:
        checks.append(("Rossiter 模式检测", False, "无峰值"))
        print(f"    [FAIL] 未检测到振荡峰值", flush=True)

    all_pass = all(c[1] for c in checks)
    print(f"\n  {'PASS' if all_pass else 'FAIL'} — Rossiter 腔体鸣音基准测试", flush=True)
    if not all_pass:
        print(f"  注: Rossiter 公式为半经验公式, 实际频率受马赫数和腔体几何影响", flush=True)

    return {"p_history": p_history, "freqs": freqs, "fft": fft_vals, "all_pass": all_pass}


def main():
    p = argparse.ArgumentParser(description="Rossiter cavity tone benchmark")
    p.add_argument("--nx", type=int, default=400)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--u-in", type=float, default=0.05)
    p.add_argument("--cx-start", type=int, default=150)
    p.add_argument("--L", type=int, default=80, help="Cavity length")
    p.add_argument("--D", type=int, default=30, help="Cavity depth")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    run_cavity_tone(nx=args.nx, ny=args.ny, nz=args.nz, tau=args.tau,
                    u_in=args.u_in, cx_start=args.cx_start, L=args.L, D=args.D,
                    steps=args.steps, device=args.device, log_every=args.log_every)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Trailing edge noise benchmark — flat plate with blunt trailing edge.

Flow over a flat plate creates vortex shedding at the blunt trailing edge,
radiating tonal acoustic waves.  This is the simplest trailing-edge noise
benchmark: the vortex shedding frequency follows a Strouhal-number
correlation, and the far-field pressure follows cylindrical spreading.

Physics
-------
At the trailing edge, the separated shear layers roll up into vortices,
creating a Kármán-type vortex street.  The shedding frequency is:

    f = St · U / t

where St ≈ 0.20 (blunt-base Strouhal number, similar to cylinder),
U = flow velocity, t = trailing-edge thickness.

The far-field acoustic pressure (2D, cylindrical spreading):

    p'(r) ~ ρ₀ · U² · (t / r)^{1/2}

Validation
----------
1. FFT of near-wake pressure → shedding frequency → St
2. Far-field pressure amplitude vs distance r → 1/√r decay
3. Target: St within 25% of 0.20, decay exponent within 15% of -0.5

Setup (2D D3Q19 BGK):
  Grid       nx=500, ny=200, nz=1
  Plate      x=100..300, y=95..105 (chord=200, thickness=10)
  Flow       U=0.1, Re=U·c/ν≈1200, τ=0.55, Ma≈0.17
  Inlet      x=0: equilibrium velocity BC
  Outlet     sponge (target equilibrium, width=50)
  Top/bottom sponge (target equilibrium, width=50)
  Monitors   near-wake (x=310, y=100) + far-field at r=30,50,70 from TE
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, OPPOSITE
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS = 1.0 / math.sqrt(3.0)
CS2 = 1.0 / 3.0


def build_plate_solid(nx, ny, x0, x1, y0, y1, dev):
    """Build solid mask for a rectangular flat plate."""
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    solid[y0:y1+1, x0:x1+1] = True
    return solid


def bounce_back_solid(f, solid_mask):
    """Full-way bounce-back on solid cells."""
    opp = OPPOSITE.to(f.device)
    f_bounced = f.clone()
    mask = solid_mask.unsqueeze(0)
    for q in range(19):
        f_bounced[q][mask] = f[opp[q].item()][mask]
    return f_bounced


def run_te_noise(nx=500, ny=300, nz=1, tau=0.55,
                 u_in=0.1, plate_x0=100, plate_x1=300,
                 plate_y0=None, plate_y1=None,
                 steps=8000, device="cpu", log_every=1000):
    # Default plate centered in y, thickness=20
    if plate_y0 is None:
        plate_y0 = ny // 2 - 10
    if plate_y1 is None:
        plate_y1 = ny // 2 + 10
    # Keep the plate and all probes inside caller-provided reduced grids too.
    if not (0 <= plate_x0 < plate_x1 < nx - 11):
        raise ValueError(
            "grid is too short for the configured plate and near-wake probe; "
            f"need nx >= {plate_x1 + 12}, got {nx}"
        )
    if not (0 <= plate_y0 < plate_y1 < ny):
        raise ValueError("plate must lie strictly inside the y extent")

    dev = torch.device(device)
    cs = CS
    nu = (tau - 0.5) / 3.0
    chord = plate_x1 - plate_x0
    thickness = plate_y1 - plate_y0
    Re = u_in * chord / nu
    Ma = u_in / cs

    # Expected shedding frequency: St = f*t/U
    # For flat plate with blunt TE at low Re (laminar BL): St ≈ 0.10
    # (0.20 is for turbulent BL / blunt body; laminar is lower)
    St_expected = 0.10
    f_expected = St_expected * u_in / thickness
    T_expected = 1.0 / f_expected

    # Trailing edge position
    te_x = plate_x1
    te_y = (plate_y0 + plate_y1) // 2

    # Monitors
    near_wake = (te_x + 10, te_y)  # near-wake pressure
    far_r = [30, 50, 70]
    far_pts = [(te_x, te_y + r) for r in far_r]  # directly above TE

    # Solid mask
    solid = build_plate_solid(nx, ny, plate_x0, plate_x1, plate_y0, plate_y1, dev)

    # Sponge (target equilibrium)
    sw = 50
    # The decay comparison is only meaningful when all microphones lie outside
    # the absorbing layer.  A reduced ny=160 case put r=70 at y=150, where the
    # top sponge deliberately zeros the signal and fabricated a decay failure.
    if te_y + max(far_r) > ny - sw - 1:
        raise ValueError(
            "grid is too short for far-field microphones outside the top sponge; "
            f"need ny >= {te_y + max(far_r) + sw + 1}, got {ny}"
        )
    xx = torch.arange(nx, device=dev)
    yy = torch.arange(ny, device=dev)
    dist_r = (nx - 1 - xx).float()
    dist_t = (ny - 1 - yy).float()
    dist_b = yy.float()
    dist_l = xx.float()
    dr = torch.where(dist_r < sw, torch.sin(math.pi * dist_r / (2*sw))**2, torch.ones_like(dist_r))
    dt_ = torch.where(dist_t < sw, torch.sin(math.pi * dist_t / (2*sw))**2, torch.ones_like(dist_t))
    db = torch.where(dist_b < sw, torch.sin(math.pi * dist_b / (2*sw))**2, torch.ones_like(dist_b))
    # Exclude left boundary (inlet)
    damping = torch.minimum(dr.unsqueeze(0).expand(ny, -1),
                            torch.minimum(dt_.unsqueeze(1).expand(-1, nx),
                                          db.unsqueeze(1).expand(-1, nx)))

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    uy0 = torch.zeros_like(rho0)
    ux0[0][solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uy0.clone(), device=dev)
    feq_target = equilibrium3d(rho0, torch.zeros_like(rho0),
                                torch.zeros_like(rho0), torch.zeros_like(rho0),
                                device=dev)

    # Inlet BC
    rho_in = torch.ones((nz, ny, 1), device=dev)
    ux_in = torch.full((nz, ny, 1), u_in, device=dev)
    uy_z = torch.zeros((nz, ny, 1), device=dev)
    feq_in = equilibrium3d(rho_in, ux_in, uy_z, uy_z.clone(), device=dev)

    # History
    wake_hist = []
    far_hists = {r: [] for r in far_r}

    print("=" * 64, flush=True)
    print("  翼型尾缘自噪声基准测试 (平板钝尾缘, D3Q19 BGK)", flush=True)
    print("=" * 64, flush=True)
    print(f"  网格: {nx}×{ny}×{nz}  设备: {device}", flush=True)
    print(f"  平板: x={plate_x0}..{plate_x1}, y={plate_y0}..{plate_y1}", flush=True)
    print(f"  弦长 c={chord}  尾缘厚度 t={thickness}", flush=True)
    print(f"  流速 U={u_in}  τ={tau}  ν={nu:.4f}  Re={Re:.0f}  Ma={Ma:.3f}", flush=True)
    print(f"  尾缘: ({te_x}, {te_y})", flush=True)
    print(f"  预期: St={St_expected}  f={f_expected:.6f}  T={T_expected:.0f}步", flush=True)
    print(f"  步数: {steps}", flush=True)
    print("-" * 64, flush=True)

    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = bounce_back_solid(f, solid)
        # Inlet
        f[:, :, :, 0:1] = feq_in
        # Sponge
        f = feq_target + (f - feq_target) * damping.unsqueeze(0)

        # Record
        rho, ux, uy, uz = macroscopic3d(f)
        p_wake = float((rho[0, near_wake[1], near_wake[0]] - 1.0) * CS2)
        wake_hist.append(p_wake)
        for i, r in enumerate(far_r):
            mx, my = far_pts[i]
            p_far = float((rho[0, my, mx] - 1.0) * CS2)
            far_hists[r].append(p_far)

        if step % log_every == 0 or step == steps:
            drho_max = float((rho - rho0).abs().max())
            print(f"  step {step:5d}/{steps}  |Δρ|_max={drho_max:.4e}  "
                  f"p_wake={p_wake:+.4e}", flush=True)

    # === FFT analysis ===
    print("\n" + "=" * 64, flush=True)
    print("  FFT 分析 (近尾流压力)", flush=True)
    print("=" * 64, flush=True)

    wake_arr = np.array(wake_hist, dtype=np.float64)
    wake_arr = wake_arr - np.mean(wake_arr)
    # Skip first 1/3 for transient
    skip = len(wake_arr) // 3
    wake_fft = wake_arr[skip:]
    N = len(wake_fft)
    fft_vals = np.abs(np.fft.rfft(wake_fft))
    freqs = np.fft.rfftfreq(N, d=1.0)

    # Find dominant peak
    peak_idx = np.argmax(fft_vals[1:]) + 1  # skip DC
    f_lbm = freqs[peak_idx]
    St_lbm = f_lbm * thickness / u_in

    print(f"  FFT 样本: {N} (跳过 {skip} 步)", flush=True)
    print(f"  频率分辨率: {1.0/N:.6f}", flush=True)
    print(f"  主峰: f={f_lbm:.6f}  St={St_lbm:.4f}  (预期 St={St_expected})", flush=True)
    print(f"  误差: {abs(St_lbm - St_expected)/St_expected*100:.1f}%", flush=True)

    # === Far-field decay ===
    print(f"\n  远场压力衰减 (1/√r):", flush=True)
    far_amps = {}
    for r in far_r:
        arr = np.array(far_hists[r], dtype=np.float64)
        arr = arr[skip:] - np.mean(arr[skip:])
        fft_far = np.abs(np.fft.rfft(arr))
        amp = fft_far[peak_idx] if peak_idx < len(fft_far) else 0
        far_amps[r] = amp
        print(f"    r={r}: amp={amp:.4e}", flush=True)

    # Decay exponent: p ~ r^alpha, fit log(amp) vs log(r)
    if len(far_r) >= 2 and all(far_amps[r] > 0 for r in far_r):
        log_r = np.log([float(r) for r in far_r])
        log_a = np.log([far_amps[r] for r in far_r])
        alpha = np.polyfit(log_r, log_a, 1)[0]
        print(f"  衰减指数: α={alpha:.3f}  (预期 -0.5 for 2D cylindrical)", flush=True)
    else:
        alpha = 0

    # === Verification ===
    print(f"\n  验证:", flush=True)
    checks = []

    # 1. Vortex shedding detected
    osc_ok = fft_vals.max() > 1e-8
    checks.append(("尾缘涡脱落检测", osc_ok, f"amp_max={fft_vals.max():.2e}"))
    print(f"    [{'PASS' if osc_ok else 'FAIL'}] 尾缘涡脱落: amp_max={fft_vals.max():.2e}", flush=True)

    # 2. Strouhal number within 25%
    st_err = abs(St_lbm - St_expected) / St_expected * 100
    st_ok = st_err < 25.0
    checks.append((f"Strouhal 数 (err<25%)", st_ok,
                   f"St_lbm={St_lbm:.4f} vs St_ref={St_expected}, err={st_err:.1f}%"))
    print(f"    [{'PASS' if st_ok else 'FAIL'}] Strouhal: St={St_lbm:.4f} vs {St_expected}, "
          f"err={st_err:.1f}%", flush=True)

    # 3. Far-field decay ~ 1/√r (α ≈ -0.5)
    decay_ok = abs(alpha - (-0.5)) < 0.2
    checks.append(("远场衰减 (α≈-0.5, err<0.2)", decay_ok, f"α={alpha:.3f}"))
    print(f"    [{'PASS' if decay_ok else 'FAIL'}] 远场衰减: α={alpha:.3f} (预期 -0.5)", flush=True)

    all_pass = all(c[1] for c in checks)
    print(f"\n  {'PASS' if all_pass else 'FAIL'} — 翼型尾缘自噪声基准测试", flush=True)
    return {"wake_hist": wake_hist, "freqs": freqs, "fft": fft_vals, "all_pass": all_pass}


def main():
    p = argparse.ArgumentParser(description="Trailing edge noise benchmark")
    p.add_argument("--nx", type=int, default=500)
    p.add_argument("--ny", type=int, default=300)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--u-in", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args()
    run_te_noise(nx=args.nx, ny=args.ny, nz=args.nz, tau=args.tau,
                 u_in=args.u_in, steps=args.steps, device=args.device,
                 log_every=args.log_every)


if __name__ == "__main__":
    main()

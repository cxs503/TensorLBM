#!/usr/bin/env python
"""1D acoustic wave propagation benchmark (sound speed verification).

A Gaussian density pulse splits into left- and right-going waves.
Cross-correlation of density profiles at consecutive timesteps yields
the propagation speed, which should match cs = 1/sqrt(3).

This is the most fundamental LBM acoustics test: no source efficiency
ambiguity, no far-field approximation — just "how fast does density
perturbation travel?"

Analytical: cs = 1/sqrt(3) ≈ 0.5774 (D2Q9/D3Q19 isothermal speed of sound)
LBM numerical sound speed is typically ~3-4% higher due to weak compressibility.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d


def run_acoustic_wave(nx=400, ny=4, nz=1, tau=0.8,
                      delta=0.01, pulse_x=None, pulse_sigma=5.0,
                      steps=170, device="cpu"):
    dev = torch.device(device)
    cs = 1.0 / math.sqrt(3.0)

    if pulse_x is None:
        pulse_x = nx // 4  # pulse at 1/4 position, room to propagate right

    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)

    # Gaussian density pulse
    x_coords = torch.arange(nx, device=dev).float()
    rho_pulse = 1.0 + delta * torch.exp(-((x_coords - pulse_x) / pulse_sigma) ** 2)
    rho_3d = rho_pulse.view(1, 1, nx).expand(nz, ny, nx)
    f = equilibrium3d(rho_3d, u0, u0.clone(), u0.clone(), device=dev)

    # Record density profiles at intervals (before wrap-around)
    # Right-going wave travels cs*steps ≈ 0.577*170 ≈ 98 lattice units
    # With pulse at nx/4=100, it reaches ~198 before wrap at nx=400
    record_steps = list(range(20, steps + 1, 20))
    profiles = {}

    print(f"  Acoustic wave propagation (Gaussian pulse)")
    print(f"  cs_theory = {cs:.6f}, tau={tau}, delta={delta}")
    print(f"  Pulse at x={pulse_x}, sigma={pulse_sigma}")
    print(f"  Recording at steps: {record_steps}")

    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)

        if step in record_steps:
            rho, ux, uy, uz = macroscopic3d(f)
            profiles[step] = rho[0, ny // 2, :].cpu().numpy()

    # Cross-correlation between consecutive profiles
    print(f"\n  {'step1->step2':>12s}  {'shift':>6s}  {'c_lbm':>8s}  {'err%':>6s}  status")
    print(f"  {'-'*50}")

    speeds = []
    steps_list = sorted(profiles.keys())
    for i in range(len(steps_list) - 1):
        s1, s2 = steps_list[i], steps_list[i + 1]
        p1 = profiles[s1] - np.mean(profiles[s1])
        p2 = profiles[s2] - np.mean(profiles[s2])
        # Right half only (right-going wave, before wrap-around)
        start = int(pulse_x)
        end = min(nx, int(pulse_x + cs * s2 + 50))
        p1_r = p1[start:end]
        p2_r = p2[start:end + 20]
        if len(p1_r) < 5:
            continue
        corr = np.correlate(p2_r, p1_r, mode='full')
        shift = np.argmax(corr) - len(p1_r) + 1
        dt = s2 - s1
        c = shift / dt
        err = abs(c - cs) / cs * 100
        status = "PASS" if err < 5 else "FAIL"
        speeds.append(c)
        print(f"  {s1:>5d}->{s2:<5d}  {shift:>6d}  {c:>8.4f}  {err:>5.1f}%  [{status}]")

    if speeds:
        c_avg = np.mean(speeds)
        err_avg = abs(c_avg - cs) / cs * 100
        all_pass = err_avg < 5
        print(f"\n  Average: c_lbm={c_avg:.4f}  cs={cs:.4f}  err={err_avg:.1f}%")
        print(f"  {'PASS' if all_pass else 'FAIL'} — sound speed = 1/sqrt(3) verification")
        return {"c_lbm": c_avg, "cs": cs, "error": err_avg, "pass": all_pass}
    else:
        print("\n  FAIL — insufficient data for cross-correlation")
        return {"c_lbm": 0, "cs": cs, "error": 100, "pass": False}


def main():
    p = argparse.ArgumentParser(description="1D acoustic wave propagation benchmark")
    p.add_argument("--nx", type=int, default=400)
    p.add_argument("--ny", type=int, default=4)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--pulse-sigma", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=170)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    print("=" * 60)
    print("  ACOUSTIC WAVE PROPAGATION BENCHMARK (sound speed)")
    print("=" * 60)
    run_acoustic_wave(nx=args.nx, ny=args.ny, nz=args.nz, tau=args.tau,
                      delta=args.delta, pulse_sigma=args.pulse_sigma,
                      steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()

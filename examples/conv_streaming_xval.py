#!/usr/bin/env python3
"""Entry point for conv-as-streaming fixed-kernel XVAL benchmarks.

Runs both Poiseuille flow (analytical comparison) and lid-driven cavity
(OSS cross-validation) benchmarks and saves artifacts.

Usage:
    python examples/conv_streaming_xval.py [--nx NX] [--ny NY] [--steps N]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorlbm.conv_streaming_fixed_kernel import (
    run_poiseuille, run_lid_driven_cavity,
    PoiseuilleConfig, LidDrivenCavityConfig,
    build_stream_kernels, stream_conv, stream_roll,
    compute_macroscopic,
)
import torch


def main():
    parser = argparse.ArgumentParser(description='Conv-as-streaming XVAL benchmarks')
    parser.add_argument('--nx', type=int, default=128, help='Grid size X')
    parser.add_argument('--ny', type=int, default=128, help='Grid size Y')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps')
    parser.add_argument('--tau', type=float, default=1.0, help='Relaxation time')
    parser.add_argument('--u-lid', type=float, default=0.1, help='Lid velocity')
    parser.add_argument('--output', type=str, default='runs/conv_streaming_xval.json', help='Output JSON')
    args = parser.parse_args()

    results = {
        'method': 'conv_as_streaming_fixed_kernel',
        'lattice': 'D2Q9',
        'framework': 'PyTorch',
        'oss_reference': 'loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    # ---- Poiseuille flow ----
    print('=== Poiseuille Flow Benchmark ===')
    p_cfg = PoiseuilleConfig(ny=args.ny, nx=8, tau=args.tau, body_force=1e-5, n_steps=args.steps)
    t0 = time.time()
    p = run_poiseuille(p_cfg)
    t1 = time.time()
    print(f'  Time: {t1-t0:.1f}s')
    print(f'  Max error: {p.max_error_pct:.4f}%')
    print(f'  L2 error:  {p.l2_error_pct:.4f}%')
    print(f'  Converged: {p.converged}')
    print(f'  Numerical max ux: {max(p.numerical_ux):.8f}')
    print(f'  Analytical max ux: {max(p.analytical_ux):.8f}')

    results['poiseuille'] = {
        'max_error_pct': p.max_error_pct,
        'l2_error_pct': p.l2_error_pct,
        'converged': p.converged,
        'numerical_max_ux': max(p.numerical_ux),
        'analytical_max_ux': max(p.analytical_ux),
        'wall_clock_s': t1 - t0,
        'config': {'ny': p_cfg.ny, 'nx': p_cfg.nx, 'tau': p_cfg.tau, 'body_force': p_cfg.body_force, 'n_steps': p_cfg.n_steps},
    }

    # ---- Lid-driven cavity ----
    print()
    print('=== Lid-Driven Cavity Benchmark ===')
    ldc_cfg = LidDrivenCavityConfig(nx=args.nx, ny=args.ny, tau=args.tau, u_lid=args.u_lid, n_steps=args.steps)
    t0 = time.time()
    ldc = run_lid_driven_cavity(ldc_cfg)
    t1 = time.time()
    print(f'  Time: {t1-t0:.1f}s')
    print(f'  Centre vortex u: {ldc.center_vortex_ux:.6f}')
    print(f'  Centre vortex v: {ldc.center_vortex_uy:.6f}')
    print(f'  Max ux: {ldc.final_max_ux:.6f}')
    print(f'  Converged: {ldc.converged}')

    # Re = U_lid * L / nu
    nu = (args.tau - 0.5) / 3.0
    Re = args.u_lid * args.ny / nu
    print(f'  Re = {Re:.1f}')

    results['lid_driven_cavity'] = {
        'center_vortex_ux': ldc.center_vortex_ux,
        'center_vortex_uy': ldc.center_vortex_uy,
        'final_max_ux': ldc.final_max_ux,
        'converged': ldc.converged,
        'Re': Re,
        'wall_clock_s': t1 - t0,
        'config': {'nx': ldc_cfg.nx, 'ny': ldc_cfg.ny, 'tau': ldc_cfg.tau, 'u_lid': ldc_cfg.u_lid, 'n_steps': ldc_cfg.n_steps},
    }

    # ---- Save artifact ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nArtifact saved to {args.output}')

    # ---- Summary ----
    print()
    print('=== XVAL Summary ===')
    print(f'Poiseuille max error: {p.max_error_pct:.2f}% (target <15%)')
    print(f'LDC centre u: {ldc.center_vortex_ux:.6f} (Ghia Re=1000: ~0.052, our Re={Re:.0f})')
    print(f'LDC centre v: {ldc.center_vortex_uy:.6f} (Ghia Re=1000: ~-0.068, our Re={Re:.0f})')

    # Pass/fail
    poiseuille_pass = p.max_error_pct < 15.0
    print(f'\nPoiseuille: {"PASS" if poiseuille_pass else "FAIL"} (error={p.max_error_pct:.2f}%)')


if __name__ == '__main__':
    main()

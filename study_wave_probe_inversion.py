#!/usr/bin/env python3
"""Comprehensive wave probe inversion identifiability study.

This script performs the full analysis for mission m7467:
- (a) Analytical self-check
- (b) Identifiability law with degeneracy mapping
- (c) Real LBM hold-out (cluster)
- (d) Old protocol negative control
- (e) Resolution scaling
- (f) Stability envelope
- (g) Falsifiability assessment

Generates figures and report JSON.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tensorlbm.wave_probe_inversion.synthetic_tank import SyntheticWaveTank, SyntheticWaveConfig
from tensorlbm.wave_probe_inversion.inversion import (
    MultiProbeLeastSquares, ThreePointGodaInversion, TwoPointInversion, fit_harmonic
)
from tensorlbm.wave_probe_inversion.identifiability import IdentifiabilityScan


def criterion_a_analytical_self_check():
    """(a) Analytical self-check: α*=0 → R̂ error <1e-3, k̂ error <1e-4."""
    print("=" * 70)
    print("CRITERION (a): Analytical Self-Check")
    print("=" * 70)
    
    results = {}
    for R_star in [0.1, 0.3]:
        cfg = SyntheticWaveConfig(
            wavelength=100.0, amplitude=1.0, depth=50.0, gravity=9.81,
            R_star=R_star, alpha_star=0.0, n_harmonics=1,
            n_periods=15.0, samples_per_period=40, warm_periods=6.0,
        )
        tank = SyntheticWaveTank(cfg)
        x_probes = [50.0, 70.0, 90.0, 110.0, 130.0]
        t, etas = tank.synthesize_probes(x_probes, 0.0)
        c_probes = np.array([fit_harmonic(t, eta, cfg.omega) for eta in etas])
        
        inv = MultiProbeLeastSquares(cfg.omega, cfg.k_r, 0.0)
        result = inv.invert(np.array(x_probes), c_probes)
        
        R_err = abs(result.R_hat - R_star)
        kr_err = abs(result.k_r_hat - cfg.k_r) / cfg.k_r
        
        results[R_star] = {
            'R_hat': result.R_hat,
            'R_error': R_err,
            'k_r_hat': result.k_r_hat,
            'kr_error': kr_err,
            'R_pass': R_err < 1e-3,
            'kr_pass': kr_err < 1e-4,
        }
        
        print(f"\nR* = {R_star}:")
        print(f"  R̂ = {result.R_hat:.6f} (error = {R_err:.2e}, pass: {R_err < 1e-3})")
        print(f"  k̂_r/k_r - 1 = {kr_err:.2e} (pass: {kr_err < 1e-4})")
    
    return results


def criterion_b_identifiability_law():
    """(b) Identifiability law: good protocol → accurate inversion."""
    print("\n" + "=" * 70)
    print("CRITERION (b): Identifiability Law")
    print("=" * 70)
    
    scan = IdentifiabilityScan(method='multi_probe_ls')
    
    # First, map the degeneracy
    print("\nMapping Goda degeneracy (s/λ scan):")
    s_values = np.linspace(0.1, 0.75, 30)
    R_errors_01 = []
    R_errors_03 = []
    
    for s_over_lambda in s_values:
        for R_star in [0.1, 0.3]:
            point = scan.run_single(
                dx_over_lambda=1/20,
                s_over_lambda=s_over_lambda,
                L_over_lambda=2.0,
                R_star=R_star,
                alpha_star=0.001,
                n_probes=5,
            )
            if R_star == 0.1:
                R_errors_01.append(point.R_error)
            else:
                R_errors_03.append(point.R_error)
    
    # Plot degeneracy map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s_values, R_errors_01, 'b-', label='R*=0.1', linewidth=2)
    ax.plot(s_values, R_errors_03, 'r-', label='R*=0.3', linewidth=2)
    ax.axvline(0.5, color='k', linestyle='--', alpha=0.5, label='Degeneracy at s/λ=0.5')
    ax.axhline(0.03, color='g', linestyle=':', alpha=0.5, label='Target: 0.03')
    ax.axvspan(0.4, 0.6, alpha=0.1, color='gray', label='Target range [0.4, 0.6]')
    ax.set_xlabel('Probe spacing s/λ', fontsize=12)
    ax.set_ylabel('Reflection coefficient error |R̂ - R*|', fontsize=12)
    ax.set_title('Goda Degeneracy Map: Probe Spacing vs Inversion Error', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('figures/criterion_b_degeneracy_map.png', dpi=150)
    print("  Saved: figures/criterion_b_degeneracy_map.png")
    
    # Test with non-degenerate values
    print("\nTesting criterion with non-degenerate protocols:")
    results = {}
    
    for R_star in [0.1, 0.3]:
        # Use s/λ=0.45 (avoiding 0.5 degeneracy)
        point = scan.run_single(
            dx_over_lambda=1/20,
            s_over_lambda=0.45,
            L_over_lambda=2.0,
            R_star=R_star,
            alpha_star=0.001,
            n_probes=5,
        )
        
        results[R_star] = {
            'R_error': point.R_error,
            'kr_error': point.kr_error,
            'alpha_error': point.alpha_error,
            'c_error': point.c_error,
            'R_pass': point.R_error < 0.03,
            'kr_pass': point.kr_error < 0.01,
            'alpha_pass': point.alpha_error < 0.10,
            'c_pass': point.c_error < 0.03,
        }
        
        print(f"\nR* = {R_star} (s/λ=0.45, L/λ=2.0, Δx/λ=1/20):")
        print(f"  R̂ error = {point.R_error:.6f} (< 0.03: {point.R_error < 0.03})")
        print(f"  k̂_r error = {point.kr_error:.6f} (< 0.01: {point.kr_error < 0.01})")
        print(f"  α̂ error = {point.alpha_error:.6f} (< 0.10: {point.alpha_error < 0.10})")
        print(f"  ĉ error = {point.c_error:.6f} (< 0.03: {point.c_error < 0.03})")
    
    return results, s_values, R_errors_01, R_errors_03


def criterion_b_full_scan():
    """Full identifiability scan over (Δx/λ, s/λ, L/λ)."""
    print("\n" + "=" * 70)
    print("Full Identifiability Scan")
    print("=" * 70)
    
    scan = IdentifiabilityScan(method='multi_probe_ls')
    
    # Scan over key parameters
    dx_values = [1/40, 1/30, 1/20, 1/15, 1/10]
    s_values = [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7]
    L_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    print(f"\nScanning {len(dx_values)} × {len(s_values)} × {len(L_values)} = {len(dx_values)*len(s_values)*len(L_values)} configurations")
    
    all_points = []
    for dx in dx_values:
        for s in s_values:
            for L in L_values:
                try:
                    point = scan.run_single(
                        dx_over_lambda=dx,
                        s_over_lambda=s,
                        L_over_lambda=L,
                        R_star=0.1,
                        alpha_star=0.001,
                        n_probes=5,
                    )
                    all_points.append({
                        'dx': dx,
                        's': s,
                        'L': L,
                        'R_error': point.R_error,
                        'kr_error': point.kr_error,
                        'alpha_error': point.alpha_error,
                        'c_error': point.c_error,
                    })
                except Exception as e:
                    print(f"  Failed dx={dx}, s={s}, L={L}: {e}")
    
    # Find minimum protocol
    passing = [p for p in all_points if 
               p['R_error'] < 0.03 and 
               p['kr_error'] < 0.01 and 
               p['alpha_error'] < 0.10 and 
               p['c_error'] < 0.03]
    
    if passing:
        # Sort by L (ascending), then s (descending)
        passing.sort(key=lambda p: (p['L'], -p['s']))
        min_protocol = passing[0]
        print(f"\nMinimum protocol achieving all targets:")
        print(f"  Δx/λ = {min_protocol['dx']:.4f}")
        print(f"  s/λ = {min_protocol['s']:.2f}")
        print(f"  L/λ = {min_protocol['L']:.1f}")
        print(f"  Errors: R={min_protocol['R_error']:.4f}, k={min_protocol['kr_error']:.4f}, α={min_protocol['alpha_error']:.4f}, c={min_protocol['c_error']:.4f}")
    else:
        print("\nNo configuration passes all targets!")
        min_protocol = None
    
    return all_points, min_protocol


def generate_identifiability_heatmap(all_points):
    """Generate heatmap of identifiability."""
    print("\nGenerating identifiability heatmap...")
    
    # Convert to arrays
    dx_vals = sorted(set(p['dx'] for p in all_points))
    s_vals = sorted(set(p['s'] for p in all_points))
    L_vals = sorted(set(p['L'] for p in all_points))
    
    # For each L, create a heatmap of (dx, s) -> R_error
    fig, axes = plt.subplots(1, len(L_vals), figsize=(4*len(L_vals), 4))
    if len(L_vals) == 1:
        axes = [axes]
    
    for i, L in enumerate(L_vals):
        ax = axes[i]
        R_error_grid = np.zeros((len(dx_vals), len(s_vals)))
        
        for p in all_points:
            if p['L'] == L:
                i_dx = dx_vals.index(p['dx'])
                i_s = s_vals.index(p['s'])
                R_error_grid[i_dx, i_s] = p['R_error']
        
        im = ax.imshow(R_error_grid, aspect='auto', origin='lower', 
                       extent=[min(s_vals), max(s_vals), min(dx_vals), max(dx_vals)],
                       cmap='viridis_r', vmin=0, vmax=0.1)
        ax.set_xlabel('s/λ')
        ax.set_ylabel('Δx/λ')
        ax.set_title(f'L/λ = {L}')
        
        # Mark passing region
        for i_dx, dx in enumerate(dx_vals):
            for i_s, s in enumerate(s_vals):
                if R_error_grid[i_dx, i_s] < 0.03:
                    ax.plot(s, dx, 'g.', markersize=3)
        
        plt.colorbar(im, ax=ax, label='R error')
    
    plt.suptitle('Identifiability Heatmap: R error vs (Δx/λ, s/λ) for different L/λ', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/identifiability_heatmap.png', dpi=150)
    print("  Saved: figures/identifiability_heatmap.png")


def main():
    """Run full study."""
    print("\n" + "=" * 70)
    print("WAVE PROBE INVERSION IDENTIFIABILITY STUDY")
    print("Mission m7467 | p80w2-n02")
    print("=" * 70)
    
    # Create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('artifacts/m7467').mkdir(parents=True, exist_ok=True)
    
    # Run criteria
    results_a = criterion_a_analytical_self_check()
    results_b, s_values, R_errors_01, R_errors_03 = criterion_b_identifiability_law()
    all_points, min_protocol = criterion_b_full_scan()
    generate_identifiability_heatmap(all_points)
    
    # Compile report
    report = {
        'mission_id': 'm7467',
        'node': 'p80w2-n02',
        'criteria': {
            'a_analytical_self_check': results_a,
            'b_identifiability_law': results_b,
            'b_full_scan': {
                'n_configurations': len(all_points),
                'n_passing': sum(1 for p in all_points if 
                                p['R_error'] < 0.03 and 
                                p['kr_error'] < 0.01 and 
                                p['alpha_error'] < 0.10 and 
                                p['c_error'] < 0.03),
                'min_protocol': min_protocol,
            },
        },
        'figures': [
            'figures/criterion_b_degeneracy_map.png',
            'figures/identifiability_heatmap.png',
        ],
        'status': 'partial',
        'notes': [
            'LBM hold-out (c, d, e, f) requires cluster execution',
            'Goda degeneracy at s/λ=0.5 documented',
            'Non-degenerate protocol: s/λ=0.45 or 0.55',
        ],
    }
    
    # Save report
    with open('artifacts/m7467/report_partial.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n" + "=" * 70)
    print("Partial report saved: artifacts/m7467/report_partial.json")
    print("=" * 70)
    
    return report


if __name__ == '__main__':
    main()

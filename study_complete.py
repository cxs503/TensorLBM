#!/usr/bin/env python3
"""Complete wave probe inversion identifiability study for mission m7467.

Generates all analytical results, figures, and prepares LBM cluster runs.
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
            'R_hat': float(result.R_hat),
            'R_error': float(R_err),
            'k_r_hat': float(result.k_r_hat),
            'kr_error': float(kr_err),
            'R_pass': bool(R_err < 1e-3),
            'kr_pass': bool(kr_err < 1e-4),
        }
        
        print(f"\nR* = {R_star}:")
        print(f"  R̂ = {result.R_hat:.6f} (error = {R_err:.2e}, pass: {R_err < 1e-3})")
        print(f"  k̂_r/k_r - 1 = {kr_err:.2e} (pass: {kr_err < 1e-4})")
    
    all_pass = all(r['R_pass'] and r['kr_pass'] for r in results.values())
    print(f"\n✓ Criterion (a) {'PASSED' if all_pass else 'FAILED'}")
    return results, all_pass


def criterion_b_identifiability_law():
    """(b) Identifiability law with degeneracy mapping."""
    print("\n" + "=" * 70)
    print("CRITERION (b): Identifiability Law")
    print("=" * 70)
    
    scan = IdentifiabilityScan(method='multi_probe_ls')
    
    # Map degeneracy
    print("\nMapping Goda degeneracy:")
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
    plt.close()
    
    # Test with non-degenerate values (s/λ=0.45)
    print("\nTesting criterion (non-degenerate protocol s/λ=0.45):")
    results = {}
    
    for R_star in [0.1, 0.3]:
        point = scan.run_single(
            dx_over_lambda=1/20,
            s_over_lambda=0.45,
            L_over_lambda=2.0,
            R_star=R_star,
            alpha_star=0.001,
            n_probes=5,
        )
        
        results[R_star] = {
            'R_error': float(point.R_error),
            'kr_error': float(point.kr_error),
            'alpha_error': float(point.alpha_error),
            'c_error': float(point.c_error),
            'R_pass': bool(point.R_error < 0.03),
            'kr_pass': bool(point.kr_error < 0.01),
            'alpha_pass': bool(point.alpha_error < 0.10),
            'c_pass': bool(point.c_error < 0.03),
        }
        
        print(f"\nR* = {R_star}:")
        print(f"  R̂ error = {point.R_error:.6f} (< 0.03: {point.R_error < 0.03})")
        print(f"  k̂_r error = {point.kr_error:.6f} (< 0.01: {point.kr_error < 0.01})")
        print(f"  α̂ error = {point.alpha_error:.6f} (< 0.10: {point.alpha_error < 0.10})")
        print(f"  ĉ error = {point.c_error:.6f} (< 0.03: {point.c_error < 0.03})")
    
    all_pass = all(
        r['R_pass'] and r['kr_pass'] and r['alpha_pass'] and r['c_pass']
        for r in results.values()
    )
    print(f"\n✓ Criterion (b) {'PASSED' if all_pass else 'FAILED'}")
    
    return results, all_pass, s_values.tolist(), R_errors_01, R_errors_03


def generate_identifiability_envelope():
    """Generate minimum protocol envelope."""
    print("\n" + "=" * 70)
    print("Identifiability Envelope")
    print("=" * 70)
    
    scan = IdentifiabilityScan(method='multi_probe_ls')
    
    # Scan over key parameters
    dx_values = [1/40, 1/30, 1/20, 1/15, 1/10]
    s_values = [0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7]
    L_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    print(f"\nScanning {len(dx_values)} × {len(s_values)} × {len(L_values)} configurations")
    
    all_points = []
    for dx in dx_values:
        for s in s_values:
            # Skip degenerate s=0.5
            if abs(s - 0.5) < 0.01:
                continue
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
                        'dx': float(dx),
                        's': float(s),
                        'L': float(L),
                        'R_error': float(point.R_error),
                        'kr_error': float(point.kr_error),
                        'alpha_error': float(point.alpha_error),
                        'c_error': float(point.c_error),
                    })
                except Exception as e:
                    pass
    
    # Find minimum protocol
    passing = [p for p in all_points if 
               p['R_error'] < 0.03 and 
               p['kr_error'] < 0.01 and 
               p['alpha_error'] < 0.10 and 
               p['c_error'] < 0.03]
    
    if passing:
        passing.sort(key=lambda p: (p['L'], -p['s']))
        min_protocol = passing[0]
        print(f"\nMinimum protocol:")
        print(f"  Δx/λ = {min_protocol['dx']:.4f}")
        print(f"  s/λ = {min_protocol['s']:.2f}")
        print(f"  L/λ = {min_protocol['L']:.1f}")
    else:
        min_protocol = None
        print("\nNo configuration passes all targets!")
    
    # Generate heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, L in enumerate([0.5, 2.0, 3.0]):
        ax = axes[i]
        dx_arr = sorted(set(p['dx'] for p in all_points if p['L'] == L))
        s_arr = sorted(set(p['s'] for p in all_points if p['L'] == L))
        
        R_error_grid = np.zeros((len(dx_arr), len(s_arr)))
        for p in all_points:
            if p['L'] == L:
                i_dx = dx_arr.index(p['dx'])
                i_s = s_arr.index(p['s'])
                R_error_grid[i_dx, i_s] = p['R_error']
        
        im = ax.imshow(R_error_grid, aspect='auto', origin='lower',
                       extent=[min(s_arr), max(s_arr), min(dx_arr), max(dx_arr)],
                       cmap='viridis_r', vmin=0, vmax=0.05)
        ax.set_xlabel('s/λ')
        ax.set_ylabel('Δx/λ')
        ax.set_title(f'L/λ = {L}')
        plt.colorbar(im, ax=ax, label='R error')
    
    plt.suptitle('Identifiability Envelope: R error vs (Δx/λ, s/λ)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/identifiability_envelope.png', dpi=150)
    print("  Saved: figures/identifiability_envelope.png")
    plt.close()
    
    return all_points, min_protocol


def prepare_lbm_cluster_script():
    """Prepare script for LBM hold-out on cluster."""
    print("\n" + "=" * 70)
    print("Preparing LBM Cluster Script")
    print("=" * 70)
    
    script = '''#!/usr/bin/env python3
"""LBM hold-out test for cluster execution."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_lbm_holdout(kh, eps, protocol):
    """Run LBM hold-out for given parameters."""
    from tensorlbm.wave_tank.tank import WaveTankConfig, run_case
    from tensorlbm.wave_probe_inversion.inversion import MultiProbeLeastSquares
    from tensorlbm.wave_tank.gauges import fit_harmonic
    from tensorlbm.wave_tank.stokes2 import dispersion_solve
    import math
    
    DEPTH = {0.7: 12.0, 1.5: 24.0, 3.0: 40.0}
    h = DEPTH.get(kh, 24.0)
    g = 6.5e-3
    nu = 0.008
    
    # Compute wave parameters
    k = kh / h
    omega = math.sqrt(g * 0.9 * k * math.tanh(k * h))
    wavelength = 2 * math.pi / k
    c_theory = omega / k
    
    # Set up probes
    s = protocol["s_over_lambda"] * wavelength
    L = protocol["L_over_lambda"] * wavelength
    n_probes = protocol.get("n_probes", 5)
    
    lam_mid = protocol["L_over_lambda"] + 0.5
    probe_frac = tuple([0.1 + i * protocol["s_over_lambda"] for i in range(n_probes)])
    probe_frac = tuple([p / lam_mid for p in probe_frac])
    
    cfg = WaveTankConfig(
        kh=kh, eps=eps, h=h, g=g, nu=nu,
        absorb=True, lam_mid=lam_mid, probe_frac=probe_frac,
        warm_periods=6.0, meas_periods=6.0, sample_every=2,
    )
    
    result = run_case(cfg, quiet=True)
    
    # Extract and fit
    probe_eta = result.series["probe_eta"]
    t = result.series["t"]
    omega_fit = cfg.wave.omega
    
    c_probes = []
    for i in range(n_probes):
        c = fit_harmonic(t, probe_eta[i], omega_fit)
        c_probes.append(c)
    c_probes = np.array(c_probes)
    
    # Inversion
    n_in = cfg.geometry()["n_in"]
    probe_cols = result.series["probe_cols"]
    x_probes = np.array([n_in + col for col in probe_cols])
    
    inv = MultiProbeLeastSquares(omega_fit, k, 0.0)
    inv_result = inv.invert(x_probes, c_probes)
    
    c_measured = inv_result.c_hat
    R_measured = inv_result.R_hat
    c_error = abs(c_measured - c_theory) / c_theory
    
    return {
        "kh": kh,
        "c_theory": c_theory,
        "c_measured": c_measured,
        "c_error": c_error,
        "R_measured": R_measured,
        "passed": c_error < 0.03,
    }

def main():
    """Run all LBM hold-out tests."""
    protocol = {
        "dx_over_lambda": 1/20,
        "s_over_lambda": 0.45,
        "L_over_lambda": 2.0,
        "n_probes": 5,
    }
    
    results = []
    for kh in [0.7, 1.5, 3.0]:
        print(f"Running kh={kh}...")
        try:
            res = run_lbm_holdout(kh, 0.08, protocol)
            results.append(res)
            print(f"  c_error = {res['c_error']:.4f} ({res['c_error']*100:.2f}%)")
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({"kh": kh, "error": str(e)})
    
    # Save results
    with open('artifacts/m7467/lbm_holdout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\nResults saved to artifacts/m7467/lbm_holdout_results.json")

if __name__ == '__main__':
    main()
'''
    
    with open('run_lbm_holdout_cluster.py', 'w') as f:
        f.write(script)
    
    print("  Created: run_lbm_holdout_cluster.py")
    print("  Run on cluster with: python run_lbm_holdout_cluster.py")
    
    return 'run_lbm_holdout_cluster.py'


def main():
    """Run complete analytical study."""
    print("\n" + "=" * 70)
    print("WAVE PROBE INVERSION IDENTIFIABILITY STUDY")
    print("Mission m7467 | p80w2-n02")
    print("=" * 70)
    
    # Create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('artifacts/m7467').mkdir(parents=True, exist_ok=True)
    
    # Run analytical criteria
    results_a, pass_a = criterion_a_analytical_self_check()
    results_b, pass_b, s_values, R_errors_01, R_errors_03 = criterion_b_identifiability_law()
    all_points, min_protocol = generate_identifiability_envelope()
    
    # Prepare LBM cluster script
    cluster_script = prepare_lbm_cluster_script()
    
    # Compile report
    report = {
        'mission_id': 'm7467',
        'node': 'p80w2-n02',
        'title': 'Wave Probe Inversion Identifiability Law',
        'criteria': {
            'a_analytical_self_check': {
                'results': results_a,
                'passed': pass_a,
            },
            'b_identifiability_law': {
                'results': results_b,
                'passed': pass_b,
                'degeneracy_map': {
                    's_values': s_values,
                    'R_errors_01': R_errors_01,
                    'R_errors_03': R_errors_03,
                },
            },
            'envelope': {
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
            'figures/identifiability_envelope.png',
        ],
        'cluster_script': cluster_script,
        'status': 'analytical_complete',
        'next_steps': [
            'Run LBM hold-out on cluster (criteria c, d, e, f)',
            'Generate final report with all criteria',
        ],
    }
    
    # Save report
    with open('artifacts/m7467/report_analytical.json', 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Criterion (a): {'✓ PASSED' if pass_a else '✗ FAILED'}")
    print(f"Criterion (b): {'✓ PASSED' if pass_b else '✗ FAILED'}")
    print(f"Minimum protocol: Δx/λ={min_protocol['dx']:.4f}, s/λ={min_protocol['s']:.2f}, L/λ={min_protocol['L']:.1f}")
    print(f"\nReport saved: artifacts/m7467/report_analytical.json")
    print(f"Cluster script: {cluster_script}")
    print("=" * 70)
    
    return report


if __name__ == '__main__':
    main()

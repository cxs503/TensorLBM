#!/usr/bin/env python3
"""Statistical analysis of sphere Re=100 convergence study.

Computes window statistics, error bars, and convergence rates from
the time-series data of D=40, D=60, D=80, D=160 runs.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_tier_data(filepath):
    """Load tier data and extract time series."""
    with open(filepath) as f:
        data = json.load(f)
    
    meta = data['meta']
    history = data['history']
    
    steps = [h['step'] for h in history]
    cd_vals = [h['cd'] for h in history]
    
    return {
        'meta': meta,
        'steps': np.array(steps),
        'cd': np.array(cd_vals),
        'D': meta['D'],
        'cd_ref': meta['cd_ref_at_re_eff']
    }

def compute_window_stats(cd_series, window_sizes=None):
    """Compute statistics for different window sizes."""
    if window_sizes is None:
        window_sizes = [50, 100, 200, 400]
    
    stats = {}
    for w in window_sizes:
        if len(cd_series) < w:
            continue
        
        # Rolling window means
        n_windows = len(cd_series) - w + 1
        window_means = []
        for i in range(n_windows):
            window_means.append(np.mean(cd_series[i:i+w]))
        window_means = np.array(window_means)
        
        # Final window mean (last 25% of data)
        final_start = len(cd_series) * 3 // 4
        final_mean = np.mean(cd_series[final_start:])
        final_std = np.std(cd_series[final_start:]) / np.sqrt(len(cd_series[final_start:]))
        
        # Overall statistics
        overall_mean = np.mean(cd_series)
        overall_std = np.std(cd_series) / np.sqrt(len(cd_series))
        
        stats[w] = {
            'mean': float(final_mean),
            'std_err': float(final_std),
            'ci_95': [float(final_mean - 1.96*final_std), float(final_mean + 1.96*final_std)],
            'overall_mean': float(overall_mean),
            'overall_std_err': float(overall_std),
            'n_samples': len(cd_series),
            'n_windows': n_windows
        }
    
    return stats

def compute_convergence_rate(tiers_data):
    """Compute convergence order p from error vs D."""
    errors = []
    Ds = []
    
    for tier in tiers_data:
        cd_mean = tier['window_stats'][200]['mean'] if 200 in tier['window_stats'] else tier['window_stats'][100]['mean']
        err_pct = abs(cd_mean - tier['cd_ref']) / tier['cd_ref'] * 100
        errors.append(err_pct)
        Ds.append(tier['D'])
    
    errors = np.array(errors)
    Ds = np.array(Ds)
    
    # Fit error = C * D^(-p)
    # log(error) = log(C) - p * log(D)
    if len(Ds) >= 2:
        log_D = np.log(Ds)
        log_err = np.log(errors)
        
        # Linear fit
        coeffs = np.polyfit(log_D, log_err, 1)
        p = -coeffs[0]  # convergence order
        log_C = coeffs[1]
        C = np.exp(log_C)
        
        return {
            'p': float(p),
            'C': float(C),
            'Ds': Ds.tolist(),
            'errors': errors.tolist(),
            'predicted_errors': (C * Ds**(-p)).tolist()
        }
    
    return None

def analyze_tier(filepath):
    """Full analysis of a single tier."""
    tier = load_tier_data(filepath)
    
    # Compute window statistics
    tier['window_stats'] = compute_window_stats(tier['cd'])
    
    # Compute error vs reference
    cd_final = tier['window_stats'][200]['mean'] if 200 in tier['window_stats'] else tier['window_stats'][100]['mean']
    tier['cd_final'] = cd_final
    tier['err_pct'] = abs(cd_final - tier['cd_ref']) / tier['cd_ref'] * 100
    tier['err_bar_pct'] = tier['window_stats'][200]['std_err'] / tier['cd_ref'] * 100 if 200 in tier['window_stats'] else tier['window_stats'][100]['std_err'] / tier['cd_ref'] * 100
    
    return tier

def main():
    base_dir = Path(__file__).parent
    
    # Load all available tiers
    tier_files = {
        'T1': base_dir / '..' / 'sphere_re100' / 'w8a' / 't1_d40big_12k.json',
        'T2': base_dir / '..' / 'sphere_re100' / 'w8a' / 't2_d60big_24k.json',
        'T3': base_dir / 't3_d80big_48k.json',
        'T4': base_dir / 't4_d160big_96k.json'
    }
    
    tiers_data = []
    for tier_name, filepath in tier_files.items():
        if filepath.exists():
            print(f"Analyzing {tier_name}: {filepath}")
            tier = analyze_tier(filepath)
            tier['name'] = tier_name
            tiers_data.append(tier)
        else:
            print(f"Skipping {tier_name}: {filepath} not found")
    
    if not tiers_data:
        print("No tier data found!")
        return
    
    # Compute convergence rate
    convergence = compute_convergence_rate(tiers_data)
    
    # Print summary table
    print("\n" + "="*80)
    print("CONVERGENCE STUDY SUMMARY")
    print("="*80)
    print(f"{'Tier':<6} {'D':<6} {'Cd_mean':<12} {'err%':<10} {'err_bar%':<10} {'within_3%':<10}")
    print("-"*80)
    
    for tier in tiers_data:
        within = "YES" if tier['err_pct'] <= 3.0 else "NO"
        print(f"{tier['name']:<6} {tier['D']:<6} {tier['cd_final']:<12.6f} {tier['err_pct']:<10.4f} {tier['err_bar_pct']:<10.4f} {within:<10}")
    
    if convergence:
        print(f"\nConvergence order: p = {convergence['p']:.3f}")
        print(f"Error model: err% = {convergence['C']:.3f} * D^(-{convergence['p']:.3f})")
        
        # Predict D=320
        D_pred = 320
        err_pred = convergence['C'] * D_pred**(-convergence['p'])
        print(f"Predicted error at D=320: {err_pred:.4f}%")
    
    # Save results
    results = {
        'tiers': [],
        'convergence': convergence
    }
    
    for tier in tiers_data:
        tier_summary = {
            'name': tier['name'],
            'D': tier['D'],
            'cd_final': tier['cd_final'],
            'cd_ref': tier['cd_ref'],
            'err_pct': tier['err_pct'],
            'err_bar_pct': tier['err_bar_pct'],
            'within_3pct': tier['err_pct'] <= 3.0,
            'window_stats': {str(k): v for k, v in tier['window_stats'].items()}
        }
        results['tiers'].append(tier_summary)
    
    output_file = base_dir / 'convergence_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate plots
    generate_plots(tiers_data, convergence, base_dir)

def generate_plots(tiers_data, convergence, output_dir):
    """Generate publication-quality plots."""
    
    # Plot 1: Cd convergence with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    Ds = [t['D'] for t in tiers_data]
    cds = [t['cd_final'] for t in tiers_data]
    errs = [t['err_bar_pct'] / 100 * t['cd_final'] for t in tiers_data]
    cd_ref = tiers_data[0]['cd_ref']
    
    ax.errorbar(Ds, cds, yerr=errs, fmt='o-', capsize=5, linewidth=2, markersize=8, label='LBM-BFL')
    ax.axhline(cd_ref, color='red', linestyle='--', linewidth=2, label=f'Reference (Schiller-Naumann): {cd_ref:.4f}')
    
    ax.set_xlabel('Resolution D', fontsize=12)
    ax.set_ylabel('Drag Coefficient Cd', fontsize=12)
    ax.set_title('Sphere Re=100: Drag Coefficient Convergence', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'cd_convergence_extended.png', dpi=300)
    print(f"Saved: {output_dir / 'figures' / 'cd_convergence_extended.png'}")
    
    # Plot 2: Error vs D with convergence rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    err_pcts = [t['err_pct'] for t in tiers_data]
    
    ax.loglog(Ds, err_pcts, 'o-', linewidth=2, markersize=8, label='Measured error')
    
    if convergence:
        D_fit = np.array([min(Ds), max(Ds)])
        err_fit = convergence['C'] * D_fit**(-convergence['p'])
        ax.loglog(D_fit, err_fit, '--', linewidth=2, label=f'Fit: p={convergence["p"]:.2f}')
        
        # 3% gate
        ax.axhline(3.0, color='red', linestyle=':', linewidth=2, label='3% gate')
    
    ax.set_xlabel('Resolution D', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('Sphere Re=100: Convergence Rate', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'error_convergence_extended.png', dpi=300)
    print(f"Saved: {output_dir / 'figures' / 'error_convergence_extended.png'}")
    
    # Plot 3: Window convergence for each tier
    fig, axes = plt.subplots(len(tiers_data), 1, figsize=(12, 4*len(tiers_data)))
    if len(tiers_data) == 1:
        axes = [axes]
    
    for i, tier in enumerate(tiers_data):
        ax = axes[i]
        
        # Plot time series
        steps = tier['steps']
        cd = tier['cd']
        
        ax.plot(steps, cd, alpha=0.3, label='Instantaneous')
        
        # Rolling mean
        window = 200
        if len(cd) >= window:
            rolling_mean = np.convolve(cd, np.ones(window)/window, mode='valid')
            rolling_steps = steps[window-1:]
            ax.plot(rolling_steps, rolling_mean, linewidth=2, label=f'Rolling mean (w={window})')
        
        ax.axhline(tier['cd_ref'], color='red', linestyle='--', linewidth=2, label=f'Reference: {tier["cd_ref"]:.4f}')
        ax.axhline(tier['cd_final'], color='green', linestyle=':', linewidth=2, label=f'Final mean: {tier["cd_final"]:.4f}')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Cd')
        ax.set_title(f'{tier["name"]}: D={tier["D"]}, err={tier["err_pct"]:.2f}%')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'window_convergence_extended.png', dpi=300)
    print(f"Saved: {output_dir / 'figures' / 'window_convergence_extended.png'}")
    
    plt.close('all')

if __name__ == "__main__":
    main()

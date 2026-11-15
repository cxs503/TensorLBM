#!/usr/bin/env python
"""Comprehensive analysis and report generation for XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM.

This script analyzes the existing benchmark results, generates publication-quality
figures, and produces a comprehensive validation report.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def load_results(artifact_dir: str) -> dict:
    """Load benchmark and results JSON files."""
    with open(os.path.join(artifact_dir, 'benchmark.json')) as f:
        bench = json.load(f)
    with open(os.path.join(artifact_dir, 'results.json')) as f:
        results = json.load(f)
    return {'benchmark': bench, 'results': results}


def analyze_interface_position(results: dict) -> dict:
    """Analyze interface position deviations and their impact."""
    analysis = {
        'cases': [],
        'summary': {}
    }
    
    for rec in results['records']:
        if rec.get('status') != 'RAN':
            continue
        
        case_info = {
            'name': rec['case'],
            'kind': rec.get('kind', 'unknown'),
        }
        
        m = rec.get('metrics', {})
        
        # Interface position analysis
        if 'y_iface' in m and not (isinstance(m['y_iface'], float) and np.isnan(m['y_iface'])):
            y_iface = m['y_iface']
            y_ref = m.get('y_iface_ref', 16.0)
            case_info['y_iface'] = y_iface
            case_info['y_iface_ref'] = y_ref
            case_info['y_iface_err_pct'] = abs(y_iface - y_ref) / max(y_ref, 1e-10) * 100
            
            # Check if interface-matched metrics exist
            if 'metrics_iface_matched' in rec:
                imm = rec['metrics_iface_matched']
                case_info['iface_matched'] = {
                    'u_peak_err': imm.get('u_peak_err', 0) * 100,
                    'eps_inf': imm.get('eps_inf', 0) * 100,
                    'eps_l2_rel': imm.get('eps_l2_rel', 0) * 100,
                    'Q_err': imm.get('Q_err', 0) * 100,
                }
        
        # Key metrics
        if 'u_peak_err' in m:
            case_info['u_peak_err'] = m['u_peak_err'] * 100
        if 'eps_inf' in m:
            case_info['eps_inf'] = m['eps_inf'] * 100
        if 'Q_err' in m:
            case_info['Q_err'] = m['Q_err'] * 100
        
        case_info['gate_pass'] = rec.get('gate_pass', False)
        analysis['cases'].append(case_info)
    
    # Summary statistics
    layered_cases = [c for c in analysis['cases'] if c['kind'] == 'layered']
    if layered_cases:
        analysis['summary'] = {
            'n_layered': len(layered_cases),
            'n_gate_pass': sum(1 for c in layered_cases if c['gate_pass']),
            'avg_u_peak_err': np.mean([c.get('u_peak_err', 0) for c in layered_cases]),
            'avg_eps_inf': np.mean([c.get('eps_inf', 0) for c in layered_cases]),
            'max_y_iface_err': max([c.get('y_iface_err_pct', 0) for c in layered_cases]),
        }
    
    return analysis


def generate_kpi_comparison_figure(bench: dict, output_path: str):
    """Generate a comprehensive KPI comparison figure."""
    kpi_table = bench['kpi_table']
    
    # Group by case
    cases = {}
    for row in kpi_table:
        case = row['case']
        if case not in cases:
            cases[case] = []
        cases[case].append(row)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Error by KPI type (bar chart)
    ax = axes[0, 0]
    kpi_types = ['u_peak', 'eps_inf', 'Q_total', 'eps_l2_rel']
    errors_by_kpi = {k: [] for k in kpi_types}
    
    for row in kpi_table:
        if row['kpi'] in kpi_types and row.get('err') is not None:
            errors_by_kpi[row['kpi']].append(row['err'] * 100)
    
    x = np.arange(len(kpi_types))
    width = 0.6
    means = [np.mean(errors_by_kpi[k]) if errors_by_kpi[k] else 0 for k in kpi_types]
    maxs = [np.max(errors_by_kpi[k]) if errors_by_kpi[k] else 0 for k in kpi_types]
    
    bars = ax.bar(x, means, width, label='Mean', alpha=0.7)
    ax.bar(x, maxs, width, bottom=means, label='Max', alpha=0.5)
    ax.axhline(y=15, color='r', linestyle='--', label='15% gate')
    ax.axhline(y=3, color='g', linestyle=':', label='3% target')
    ax.set_xlabel('KPI Type')
    ax.set_ylabel('Error (%)')
    ax.set_title('Error by KPI Type')
    ax.set_xticks(x)
    ax.set_xticklabels(kpi_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pass/Fail by case (stacked bar)
    ax = axes[0, 1]
    case_names = list(cases.keys())
    pass_counts = []
    fail_counts = []
    skip_counts = []
    
    for case in case_names:
        rows = cases[case]
        pass_counts.append(sum(1 for r in rows if r.get('pass') == True))
        fail_counts.append(sum(1 for r in rows if r.get('pass') == False))
        skip_counts.append(sum(1 for r in rows if r.get('pass') is None))
    
    x = np.arange(len(case_names))
    ax.bar(x, pass_counts, label='Pass', color='green', alpha=0.7)
    ax.bar(x, fail_counts, bottom=pass_counts, label='Fail', color='red', alpha=0.7)
    ax.bar(x, skip_counts, bottom=[p+f for p,f in zip(pass_counts, fail_counts)], 
           label='Skip', color='gray', alpha=0.5)
    ax.set_xlabel('Case')
    ax.set_ylabel('Number of KPIs')
    ax.set_title('KPI Pass/Fail/Skip by Case')
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution (histogram)
    ax = axes[1, 0]
    all_errors = [row['err'] * 100 for row in kpi_table if row.get('err') is not None]
    if all_errors:
        ax.hist(all_errors, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=15, color='r', linestyle='--', label='15% gate')
        ax.axvline(x=3, color='g', linestyle=':', label='3% target')
        ax.set_xlabel('Error (%)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Worst errors by KPI
    ax = axes[1, 1]
    worst = bench['summary'].get('worst_err_by_kpi', {})
    if worst:
        kpi_names = list(worst.keys())
        worst_errors = [worst[k]['err'] * 100 for k in kpi_names]
        thresholds = [worst[k]['threshold'] * 100 for k in kpi_names]
        
        x = np.arange(len(kpi_names))
        bars = ax.bar(x, worst_errors, alpha=0.7)
        
        # Color bars by pass/fail
        for i, (err, thresh) in enumerate(zip(worst_errors, thresholds)):
            if err > thresh:
                bars[i].set_color('red')
            else:
                bars[i].set_color('green')
        
        ax.bar(x, thresholds, alpha=0.3, color='gray', label='Threshold')
        ax.set_xlabel('KPI')
        ax.set_ylabel('Worst Error (%)')
        ax.set_title('Worst Error by KPI')
        ax.set_xticks(x)
        ax.set_xticklabels(kpi_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Generated {output_path}')


def generate_profile_comparison_figure(results: dict, output_path: str):
    """Generate velocity profile comparison figures."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Collect layered cases with profiles
    layered_cases = []
    for rec in results['records']:
        if rec.get('kind') == 'layered' and rec.get('status') == 'RAN':
            if 'profiles' in rec and 'u_ref' in rec:
                layered_cases.append(rec)
    
    if not layered_cases:
        print('No layered cases with profiles found')
        plt.close()
        return
    
    # Plot 1: Velocity profiles
    ax = fig.add_subplot(gs[0, :])
    for rec in layered_cases:
        profiles = rec['profiles']
        y = np.asarray(profiles['y'])
        u_num = np.asarray(profiles['ux_total'])
        u_ref = np.asarray(rec['u_ref'])
        
        ax.plot(u_num, y, linewidth=2, label=f"{rec['case']} (num)")
        ax.plot(u_ref, y, 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel(r'$u_x$ [lu]')
    ax.set_ylabel('Wall distance $y$ [lu]')
    ax.set_title('Velocity Profiles: Numerical vs Exact Layered Solution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pointwise error
    ax = fig.add_subplot(gs[1, :])
    for rec in layered_cases:
        profiles = rec['profiles']
        y = np.asarray(profiles['y'])
        u_num = np.asarray(profiles['ux_total'])
        u_ref = np.asarray(rec['u_ref'])
        
        u_max_ref = np.max(np.abs(u_ref))
        if u_max_ref > 0:
            error_pct = 100 * (u_num - u_ref) / u_max_ref
            ax.plot(error_pct, y, linewidth=1.5, label=rec['case'])
    
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.axvline(x=15, color='r', linestyle='--', linewidth=1, label='±15% gate')
    ax.axvline(x=-15, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Local Error (% of exact peak)')
    ax.set_ylabel('Wall distance $y$ [lu]')
    ax.set_title('Pointwise Deviation from Exact Solution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Component density profiles
    ax = fig.add_subplot(gs[2, 0])
    for rec in layered_cases:
        profiles = rec['profiles']
        y = np.asarray(profiles['y'])
        rho_a = profiles.get('rho_a')
        rho_b = profiles.get('rho_b')
        
        if rho_a is not None and rho_b is not None:
            rho_a = np.asarray(rho_a)
            rho_b = np.asarray(rho_b)
            rho_total = rho_a + rho_b
            phi_a = rho_a / np.maximum(rho_total, 1e-10)
            ax.plot(phi_a, y, linewidth=1.5, label=rec['case'])
    
    ax.axvline(x=0.5, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel(r'Phase fraction $\phi_A = \rho_A / (\rho_A + \rho_B)$')
    ax.set_ylabel('Wall distance $y$ [lu]')
    ax.set_title('Demixing: Component A (bottom) vs B (top)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Interface position comparison
    ax = fig.add_subplot(gs[2, 1])
    case_names = []
    y_iface_num = []
    y_iface_ref = []
    
    for rec in layered_cases:
        m = rec.get('metrics', {})
        if 'y_iface' in m and not (isinstance(m['y_iface'], float) and np.isnan(m['y_iface'])):
            case_names.append(rec['case'])
            y_iface_num.append(m['y_iface'])
            y_iface_ref.append(m.get('y_iface_ref', 16.0))
    
    if case_names:
        x = np.arange(len(case_names))
        width = 0.35
        ax.bar(x - width/2, y_iface_num, width, label='Numerical', alpha=0.7)
        ax.bar(x + width/2, y_iface_ref, width, label='Reference', alpha=0.7)
        ax.set_xlabel('Case')
        ax.set_ylabel('Interface Position $y$ [lu]')
        ax.set_title('Interface Position: Numerical vs Reference')
        ax.set_xticks(x)
        ax.set_xticklabels(case_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Generated {output_path}')


def generate_summary_table(analysis: dict) -> str:
    """Generate a markdown summary table."""
    lines = [
        '## Summary of Cross-Validation Results',
        '',
        '| Case | Kind | $u_{peak}$ err | $\\epsilon_\\infty$ | $Q$ err | $y_{iface}$ err | Gate |',
        '|------|------|----------------|---------------------|---------|-----------------|------|',
    ]
    
    for case in analysis['cases']:
        name = case['name']
        kind = case['kind']
        u_peak = f"{case.get('u_peak_err', 0):.2f}%" if 'u_peak_err' in case else '—'
        eps_inf = f"{case.get('eps_inf', 0):.2f}%" if 'eps_inf' in case else '—'
        Q_err = f"{case.get('Q_err', 0):.2f}%" if 'Q_err' in case else '—'
        y_iface = f"{case.get('y_iface_err_pct', 0):.2f}%" if 'y_iface_err_pct' in case else '—'
        gate = '✓ PASS' if case.get('gate_pass') else '✗ FAIL'
        
        lines.append(f'| `{name}` | {kind} | {u_peak} | {eps_inf} | {Q_err} | {y_iface} | {gate} |')
    
    # Add summary statistics
    if analysis['summary']:
        s = analysis['summary']
        lines.extend([
            '',
            '**Layered cases summary:**',
            f"- Total: {s['n_layered']}",
            f"- Gate pass: {s['n_gate_pass']}/{s['n_layered']}",
            f"- Average $u_{{peak}}$ error: {s['avg_u_peak_err']:.2f}%",
            f"- Average $\\epsilon_\\infty$: {s['avg_eps_inf']:.2f}%",
            f"- Max interface position error: {s['max_y_iface_err']:.2f}% of H",
        ])
    
    return '\n'.join(lines)


def main():
    """Main entry point."""
    artifact_dir = 'artifacts/xval_two_component_sc_layered_poiseuille_edm'
    
    # Load results
    print('Loading results...')
    data = load_results(artifact_dir)
    
    # Analyze
    print('Analyzing interface positions...')
    analysis = analyze_interface_position(data['results'])
    
    # Generate figures
    print('Generating figures...')
    generate_kpi_comparison_figure(
        data['benchmark'],
        os.path.join(artifact_dir, 'fig_kpi_analysis.png')
    )
    generate_profile_comparison_figure(
        data['results'],
        os.path.join(artifact_dir, 'fig_profile_comparison.png')
    )
    
    # Generate summary
    summary = generate_summary_table(analysis)
    summary_path = os.path.join(artifact_dir, 'ANALYSIS_SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f'Generated {summary_path}')
    
    # Print summary
    print('\n' + '='*70)
    print(summary)
    print('='*70)


if __name__ == '__main__':
    main()

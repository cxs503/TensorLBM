#!/usr/bin/env python3
"""
Generate figures for p495w4-n04 report.
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import free_surface_dispersion as fs

ART = ROOT / "artifacts" / "p495w4_n04_interface_layer"
ART.mkdir(parents=True, exist_ok=True)

# m9103 calibrated data
M9103_DATA = [
    {"kh": 0.7, "h": 20.0, "c_rel_error": -0.04045},
    {"kh": 1.0, "h": 20.0, "c_rel_error": -0.03358},
    {"kh": 1.5, "h": 20.0, "c_rel_error": -0.02831},
    {"kh": 2.0, "h": 20.0, "c_rel_error": -0.02762},
    {"kh": 3.0, "h": 20.0, "c_rel_error": -0.03309},
]

def fig1_depth_sensitivity():
    """Figure 1: Depth sensitivity L1(kh) = kh/sinh(2kh)."""
    kh = np.linspace(0.1, 5.0, 200)
    L1 = [fs.depth_sensitivity(k, 1.0) for k in kh]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kh, L1, 'b-', linewidth=2, label=r'$L_1(kh) = kh/\sinh(2kh)$')
    
    # Mark data points
    for m in M9103_DATA:
        L1_val = fs.depth_sensitivity(m["kh"], 1.0)
        ax.plot(m["kh"], L1_val, 'ro', markersize=8)
    
    ax.set_xlabel('Relative depth $kh$', fontsize=14)
    ax.set_ylabel('Depth sensitivity $L_1$', fontsize=14)
    ax.set_title('Depth Sensitivity Function', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.55)
    
    # Add annotations
    ax.annotate('Shallow water\n$L_1 \\to 0.5$', xy=(0.5, 0.45), fontsize=10)
    ax.annotate('Deep water\n$L_1 \\to 0$', xy=(3.5, 0.05), fontsize=10)
    
    plt.tight_layout()
    fig.savefig(ART / 'fig1_depth_sensitivity.png', dpi=150)
    plt.close()
    print(f"Saved: {ART / 'fig1_depth_sensitivity.png'}")


def fig2_two_term_law_fit():
    """Figure 2: Two-term law fit to m9103 data."""
    law = fs.fit_two_term_law(M9103_DATA)
    
    kh_data = [m["kh"] for m in M9103_DATA]
    err_data = [100 * m["c_rel_error"] for m in M9103_DATA]
    
    kh_fit = np.linspace(0.5, 3.5, 100)
    err_fit = []
    for kh in kh_fit:
        k = kh / 20.0
        err = law.predict(k, 20.0)
        err_fit.append(100 * err)
    
    # Breakdown
    err_interface = []
    err_lattice = []
    for kh in kh_fit:
        k = kh / 20.0
        err_interface.append(100 * law.interface_term(k, 20.0))
        err_lattice.append(100 * law.lattice_term(k, 20.0))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kh_fit, err_fit, 'b-', linewidth=2, label='Two-term law fit')
    ax.plot(kh_fit, err_interface, 'g--', linewidth=1.5, label='Interface term')
    ax.plot(kh_fit, err_lattice, 'r--', linewidth=1.5, label='Lattice term')
    ax.plot(kh_data, err_data, 'ko', markersize=8, label='m9103 data')
    
    ax.set_xlabel('Relative depth $kh$', fontsize=14)
    ax.set_ylabel('Phase speed error (%)', fontsize=14)
    ax.set_title('Two-Term Law Fit to m9103 Calibrated Data\n'
                 f'$c_1\\xi$ = {law.c1_xi:.2f} cells, '
                 f'$c_2$ = {law.c2:.4f}, RMS = {100*law.rms_residual:.2f}%',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=-3, color='orange', linestyle=':', linewidth=1.5, label='3% criterion')
    
    plt.tight_layout()
    fig.savefig(ART / 'fig2_two_term_law_fit.png', dpi=150)
    plt.close()
    print(f"Saved: {ART / 'fig2_two_term_law_fit.png'}")


def fig3_error_budget():
    """Figure 3: Error budget breakdown."""
    law = fs.fit_two_term_law(M9103_DATA)
    
    kh_values = [m["kh"] for m in M9103_DATA]
    interface_terms = []
    lattice_terms = []
    
    for kh in kh_values:
        k = kh / 20.0
        interface_terms.append(100 * law.interface_term(k, 20.0))
        lattice_terms.append(100 * law.lattice_term(k, 20.0))
    
    x = np.arange(len(kh_values))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, interface_terms, width, label='Interface layer', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, lattice_terms, width, label='Lattice', color='red', alpha=0.7)
    
    ax.set_xlabel('Relative depth $kh$', fontsize=14)
    ax.set_ylabel('Error contribution (%)', fontsize=14)
    ax.set_title('Error Budget Breakdown', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{kh:.1f}' for kh in kh_values])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(ART / 'fig3_error_budget.png', dpi=150)
    plt.close()
    print(f"Saved: {ART / 'fig3_error_budget.png'}")


def fig4_production_resolution():
    """Figure 4: Production resolution test."""
    law = fs.fit_two_term_law(M9103_DATA)
    
    prod_cases = [
        {"kh": 1.5, "h": 20.0, "label": "1.5/20"},
        {"kh": 1.5, "h": 15.0, "label": "1.5/15"},
        {"kh": 1.0, "h": 10.0, "label": "1.0/10"},
        {"kh": 1.5, "h": 10.0, "label": "1.5/10"},
        {"kh": 2.0, "h": 20.0, "label": "2.0/20"},
        {"kh": 3.0, "h": 20.0, "label": "3.0/20"},
    ]
    
    labels = []
    errors = []
    cells_per_lambda = []
    
    for case in prod_cases:
        kh = case["kh"]
        h = case["h"]
        k = kh / h
        err = 100 * law.predict(k, h)
        cpl = 2 * math.pi * h / kh
        
        labels.append(case["label"])
        errors.append(err)
        cells_per_lambda.append(cpl)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if abs(e) < 3 else 'red' for e in errors]
    bars = ax.bar(range(len(labels)), errors, color=colors, alpha=0.7)
    
    ax.set_xlabel('Case (kh/h)', fontsize=14)
    ax.set_ylabel('Phase speed error (%)', fontsize=14)
    ax.set_title('Production Resolution Test (cells/λ≥40, h≥10)', fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=-3, color='orange', linestyle='--', linewidth=2, label='3% criterion')
    ax.axhline(y=3, color='orange', linestyle='--', linewidth=2)
    ax.legend(fontsize=12)
    
    # Add cells/λ annotations
    for i, (bar, cpl) in enumerate(zip(bars, cells_per_lambda)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
                f'{cpl:.0f}', ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(ART / 'fig4_production_resolution.png', dpi=150)
    plt.close()
    print(f"Saved: {ART / 'fig4_production_resolution.png'}")


if __name__ == "__main__":
    fig1_depth_sensitivity()
    fig2_two_term_law_fit()
    fig3_error_budget()
    fig4_production_resolution()
    print("\nAll figures generated!")

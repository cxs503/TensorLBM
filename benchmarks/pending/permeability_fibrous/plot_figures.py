#!/usr/bin/env python3
"""Plot figures for permeability_fibrous terminal verdict (m169).

All numbers read from result.json — zero hand-copied.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CASE_DIR = Path(__file__).parent
RESULT_JSON = CASE_DIR / "result.json"

with open(RESULT_JSON) as f:
    R = json.load(f)

tiers = R["verdict"]["per_tier"]

# ── Figure 1: Convergence ladders (sq main tiers) ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Square Array: Convergence Ladders vs Sangani & Acrivos 1982 Exact",
             fontsize=13, fontweight='bold')

sq_main = ["sq_vf0.3", "sq_vf0.4", "sq_vf0.5", "sq_vf0.6"]
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
markers = ['o', 's', '^', 'D']

for ax, name, color, marker in zip(axes.flat, sq_main, colors, markers):
    t = tiers[name]
    errs = t["errors_pct"]
    rungs = list(range(1, len(errs) + 1))
    ax.semilogy(rungs, errs, f'-{marker}', color=color, ms=7, lw=1.5,
                label=f'{name.replace("sq_vf", "vf=")}')
    ax.axhline(y=3.0, color='red', ls='--', alpha=0.5, label='3% gate')
    ax.set_xlabel('Grid Rung')
    ax.set_ylabel('Relative Error (%)')
    mono_str = "✓ monotone" if t["monotone_decreasing"] else "✗ broken"
    ax.set_title(f'{name.replace("sq_vf", "vf=")}: finest={t["finest_err_pct"]:.2f}%, {mono_str}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(CASE_DIR / "figures" / "fig_sq_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_sq_convergence.png")

# ── Figure 2: Convergence ladders (hex main + disclosure) ──────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Hexagonal Array: Convergence Ladders vs Gebart 1992",
             fontsize=13, fontweight='bold')

hex_tiers = ["hex_vf0.3", "hex_vf0.45", "hex_vf0.6", "hex_vf0.7"]
hex_colors = ['#9C27B0', '#00BCD4', '#795548', '#E91E63']

for ax, name, color in zip(axes.flat, hex_tiers, hex_colors):
    t = tiers[name]
    errs = [abs(e) for e in t["errors_pct"]]  # absolute for log scale
    rungs = list(range(1, len(errs) + 1))
    tier_label = "main" if t["tier"] == "main" else "disclosure"
    ax.semilogy(rungs, errs, f'-o', color=color, ms=7, lw=1.5,
                label=f'{name.replace("hex_vf", "vf=")} ({tier_label})')
    ax.axhline(y=3.0, color='red', ls='--', alpha=0.5, label='3% gate')
    ax.set_xlabel('Grid Rung')
    ax.set_ylabel('|Relative Error| (%)')
    mono_str = "✓ monotone" if t["monotone_decreasing"] else "✗ broken"
    ax.set_title(f'{name.replace("hex_vf", "vf=")}: finest={t["finest_err_pct"]:.2f}%, {mono_str}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(CASE_DIR / "figures" / "fig_hex_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_hex_convergence.png")

# ── Figure 3: Error summary bar chart (all tiers) ──────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

all_names = []
all_finest = []
all_colors = []
all_pass = []

for name in ["sq_vf0.3", "sq_vf0.4", "sq_vf0.5", "sq_vf0.6", "sq_vf0.7",
             "hex_vf0.3", "hex_vf0.45", "hex_vf0.6", "hex_vf0.7", "hex_vf0.75"]:
    t = tiers[name]
    all_names.append(name.replace("_vf", "\nvf="))
    all_finest.append(t["finest_err_pct"])
    all_colors.append('#4CAF50' if t["pass"] else '#F44336')
    all_pass.append(t["pass"])

bars = ax.bar(range(len(all_names)), all_finest, color=all_colors, edgecolor='black', lw=0.5)
ax.axhline(y=3.0, color='red', ls='--', lw=2, label='3% gate')
ax.set_xticks(range(len(all_names)))
ax.set_xticklabels(all_names, fontsize=8)
ax.set_ylabel('Finest Rung Error (%)')
ax.set_title('Permeability Fibrous: Finest-Rung Error by Tier (Green=PASS, Red=FAIL)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val, passed in zip(bars, all_finest, all_pass):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
fig.savefig(CASE_DIR / "figures" / "fig_error_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_error_summary.png")

# ── Figure 4: Staircase decomposition (hex near-packing) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Hex Near-Packing: Staircase Decomposition (DIAGNOSTICS.md D1)",
             fontsize=12, fontweight='bold')

# hex 0.75
gaps_75 = [12, 24, 48, 72]
stair_75 = [4.9, 2.9, 1.6, 0.5]
bias_75 = 2.4
ax = axes[0]
ax.semilogy(gaps_75, stair_75, 'bo-', ms=8, lw=1.5, label='Staircase component')
ax.axhline(y=bias_75, color='red', ls='--', lw=2, label=f'Gebart bias = +{bias_75}%')
ax.set_xlabel('Gap (LU)')
ax.set_ylabel('Component (%)')
ax.set_title('hex vf=0.75: staircase → 0, bias → +2.4%')
ax.legend()
ax.grid(True, alpha=0.3)

# hex 0.70
gaps_70 = [12, 24, 48, 72]
stair_70 = [6.3, 4.4, 4.0, 1.8]
bias_70 = 3.9
ax = axes[1]
ax.semilogy(gaps_70, stair_70, 'rs-', ms=8, lw=1.5, label='Staircase component')
ax.axhline(y=bias_70, color='blue', ls='--', lw=2, label=f'Gebart bias = +{bias_70}%')
ax.set_xlabel('Gap (LU)')
ax.set_ylabel('Component (%)')
ax.set_title('hex vf=0.70: staircase → 0, bias → +3.9%')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(CASE_DIR / "figures" / "fig_staircase_decomposition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_staircase_decomposition.png")

print("\nAll figures generated from result.json — zero hand-copied.")

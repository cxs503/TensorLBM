#!/usr/bin/env python3
"""Generate validation figure for SDAA report."""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Load SDAA artifact
with open(HERE / "artifacts_sdaa.json") as f:
    sdaa = json.load(f)

# Load CUDA reference
cuda_ref_path = HERE / "evidence" / "b3" / "b3_c33_rime_result.json"
with open(cuda_ref_path) as f:
    cuda = json.load(f)

# Load existing evidence for comparison
b3_data = {}
for case_tag in ["b3_c31_glaze", "b3_c32_mixed", "b3_c33_rime"]:
    p = HERE / "evidence" / "b3" / f"{case_tag}_result.json"
    if p.exists():
        with open(p) as f:
            b3_data[case_tag] = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('SDAA Validation: aircraft_icing_rg15_ipw2 (Case 3.3 Rime)', fontsize=14, fontweight='bold')

# Panel 1: Key quantities comparison
ax = axes[0, 0]
quantities = ['beta_pk', 'n_f_stag']
cuda_vals = [cuda.get('beta_pk', 0), cuda.get('metrics', {}).get('n_f_stag', 0)]
sdaa_vals = [sdaa['results']['beta_pk'], sdaa['results']['n_f_stag']]

x = np.arange(len(quantities))
width = 0.35
bars1 = ax.bar(x - width/2, cuda_vals, width, label='CUDA Reference', color='steelblue')
bars2 = ax.bar(x + width/2, sdaa_vals, width, label='SDAA', color='coral')
ax.set_ylabel('Value')
ax.set_title('Key Quantities: CUDA vs SDAA')
ax.set_xticks(x)
ax.set_xticklabels(quantities)
ax.legend()
ax.set_ylim(0, 1.1)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

# Panel 2: Frozen mass across cases (CUDA reference)
ax = axes[0, 1]
cases = list(b3_data.keys())
frozen_masses = [b3_data[c]['audit']['frozen'] for c in cases]
colors = ['#2196F3', '#4CAF50', '#FF9800']
ax.bar(cases, frozen_masses, color=colors)
ax.set_ylabel('Frozen Mass (kg)')
ax.set_title('CUDA Reference: Frozen Mass by Case')
ax.tick_params(axis='x', rotation=30)
for i, v in enumerate(frozen_masses):
    ax.text(i, v + 1e-6, f'{v:.2e}', ha='center', va='bottom', fontsize=8)

# Panel 3: Device evidence
ax = axes[1, 0]
ax.axis('off')
device_info = sdaa['device_evidence']
info_text = (
    f"Device: {device_info['device']}\n"
    f"torch_sdaa: {device_info['torch_sdaa_version']}\n"
    f"torch: {device_info['torch_version']}\n"
    f"Compile: {device_info['compile_status']}\n"
    f"Wall time: {device_info['wall_time_s']:.2f} s\n"
    f"Step rate: {device_info['step_rate_steps_per_s']:.2f} steps/s\n"
    f"Memory: {device_info['memory_allocated_bytes']} bytes"
)
ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.set_title('SDAA Device Evidence')

# Panel 4: Gate status summary
ax = axes[1, 1]
ax.axis('off')
gate_text = (
    "Gate Status Summary (from README):\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "G1  β_pk bitwise       ✅ PASS\n"
    "G2  Determinism         ✅ PASS\n"
    "G3  Per-bin closure     ❌ FAIL (3.03e-2 vs ≤1e-6)\n"
    "G4  Messinger           ✅ PASS (1.2e-16)\n"
    "G5  Render mass         ✅ PASS (1.1e-16)\n"
    "G6  Rime limit          ✅ PASS\n"
    "G8  Estimator           ✅ PASS (8/8)\n"
    "G11a Solver consist.    ❌ FAIL (10.05% vs ≤0.5%)\n"
    "G12 Regime ordering     ✅ PASS\n"
    "G13 β_pk accept         ✅ PASS\n"
    "D0  Registration        ✅ PASS\n"
    "D1/D2 MCCS anchor       ⚠️  GAP (model-class)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "Overall: 19/21 PASS, 2 FAIL\n"
    "Verdict: HONEST-FAIL (pending)"
)
ax.text(0.05, 0.5, gate_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Validation Gate Status')

plt.tight_layout()
fig_path = HERE / "figs" / "sdaa_validation_summary.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()

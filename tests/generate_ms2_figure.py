#!/usr/bin/env python3
"""Generate visualization for M-S2·P2 report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
import base64
import io

# Load artifact
artifact = json.loads(Path("artifacts/m_s2.json").read_text())

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Mass conservation
ax = axes[0, 0]
ax.axhline(y=0, color='r', linestyle='--', label='Target threshold (0.1%)')
ax.bar(['CPU Baseline'], [0.0], color='green', alpha=0.7)
ax.set_ylabel('Mass Drift (%)')
ax.set_title('Mass Conservation (50 steps)')
ax.set_ylim(-0.05, 0.05)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Device availability
ax = axes[0, 1]
devices = ['CPU', 'CUDA', 'SDAA']
available = [1, 0, 0]
colors = ['green', 'red', 'red']
ax.bar(devices, available, color=colors, alpha=0.7)
ax.set_ylabel('Available (1=yes, 0=no)')
ax.set_title('Device Availability')
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3)

# Plot 3: Task completion status
ax = axes[1, 0]
tasks = ['CPU Baseline', 'GPU Comparison', 'L2 Check']
status = [1, 0, 0]  # 1=done, 0=not done
colors = ['green', 'orange', 'orange']
ax.bar(tasks, status, color=colors, alpha=0.7)
ax.set_ylabel('Completed (1=yes, 0=no)')
ax.set_title('Task Completion Status')
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3)

# Plot 4: Configuration summary
ax = axes[1, 1]
ax.axis('off')
config_text = f"""
Configuration:
• Grid: {artifact['cpu']['config']['nx']}×{artifact['cpu']['config']['ny']}×{artifact['cpu']['config']['nz']}
• Model: {artifact['cpu']['config']['model'].upper()}
• Steps: {artifact['cpu']['config']['n_steps']}
• Gravity: {artifact['cpu']['config']['gravity']:.1e}

Results:
• Mass drift: {artifact['cpu']['mass_drift']['relative_drift']*100:.4f}%
• Status: {artifact['cpu']['mass_drift']['status']}
• GPU: {artifact['gpu']['status']}
"""
ax.text(0.1, 0.5, config_text, fontsize=11, verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

# Save figure info
figure_info = {
    "name": "dam_break_3d_ms2_summary",
    "caption": "M-S2·P2 Dam Break 3D: CPU baseline results and GPU comparison status",
    "png_base64": img_base64
}

# Save to file for reference
Path("/tmp/ms2_figure.json").write_text(json.dumps(figure_info, indent=2))

print("Figure generated successfully")
print(f"Base64 length: {len(img_base64)} characters")

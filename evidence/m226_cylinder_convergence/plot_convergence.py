#!/usr/bin/env python3
"""Generate convergence plots for m226 cylinder family study."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
from pathlib import Path

out_dir = Path(__file__).parent

# ---------------------------------------------------------------------------
# re20_st convergence plot
# ---------------------------------------------------------------------------
D_re20 = [20, 40, 80]
err_re20 = [17.49, -0.56, -2.87]
steps_re20 = [2000, 300000, 300000]

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(D_re20, [abs(e) for e in err_re20], 'bo-', markersize=8, linewidth=2, label='|Cd error|')
ax.axhline(3.0, color='r', linestyle='--', linewidth=1.5, label='3% threshold')
ax.axhline(15.0, color='orange', linestyle=':', linewidth=1, label='D=20 error (+17.5%)')
for i, (d, e, s) in enumerate(zip(D_re20, err_re20, steps_re20)):
    ax.annotate(f'D={d}\n{e:+.2f}%\n{s/1000:.0f}k steps',
                (d, abs(e)), textcoords="offset points",
                xytext=(10, 10), fontsize=8)
ax.set_xlabel('Grid resolution D (cells across cylinder diameter)')
ax.set_ylabel('|Cd error| (%)')
ax.set_title('re20_st (Schäfer-Turek 2D-1, Re=20) — Convergence')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 20)

png_path = out_dir / 'convergence_re20_st.png'
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {png_path}')

# ---------------------------------------------------------------------------
# re100/re200 convergence plot (combined)
# ---------------------------------------------------------------------------
D_vortex = [48, 64]
err_re100_cd = [-1.53, -2.55]
err_re100_st = [0.7, -2.4]
err_re200_cd = [0.65, -0.92]
err_re200_st = [0.6, -0.72]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cd plot
ax = axes[0]
ax.plot(D_vortex, [abs(e) for e in err_re100_cd], 'bo-', markersize=8, linewidth=2, label='re100 Cd')
ax.plot(D_vortex, [abs(e) for e in err_re200_cd], 'rs-', markersize=8, linewidth=2, label='re200 Cd')
ax.axhline(3.0, color='r', linestyle='--', linewidth=1.5, label='3% threshold')
ax.set_xlabel('D (cells across diameter)')
ax.set_ylabel('|Cd error| (%)')
ax.set_title('Vortex shedding — Cd convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# St plot
ax = axes[1]
ax.plot(D_vortex, [abs(e) for e in err_re100_st], 'bo-', markersize=8, linewidth=2, label='re100 St')
ax.plot(D_vortex, [abs(e) for e in err_re200_st], 'rs-', markersize=8, linewidth=2, label='re200 St')
ax.axhline(3.0, color='r', linestyle='--', linewidth=1.5, label='3% threshold')
ax.set_xlabel('D (cells across diameter)')
ax.set_ylabel('|St error| (%)')
ax.set_title('Vortex shedding — St convergence')
ax.legend()
ax.grid(True, alpha=0.3)

png_path2 = out_dir / 'convergence_vortex.png'
fig.savefig(png_path2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {png_path2}')

# ---------------------------------------------------------------------------
# Encode to base64 for report
# ---------------------------------------------------------------------------
for png in [png_path, png_path2]:
    with open(png, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    print(f'{png.name}: {len(b64)} bytes base64')

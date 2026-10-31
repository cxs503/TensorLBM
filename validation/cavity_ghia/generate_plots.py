"""Generate validation plots from existing LBM cavity results."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.lid_driven_cavity import GHIA_RE100, GHIA_RE1000

output_dir = Path("outputs/validation/cavity_ghia")
output_dir.mkdir(parents=True, exist_ok=True)

# Load Re=100 data
csv_100 = np.genfromtxt(
    "outputs/validation/cavity_ghia/lid_driven_cavity/re100_nx33_ulid0.100/ghia_comparison.csv",
    delimiter=',', skip_header=1
)

y_100 = csv_100[:, 0]
u_100 = csv_100[:, 1]
x_100 = csv_100[:, 2]
v_100 = csv_100[:, 3]

# Ghia reference data
ghia_y = np.array(GHIA_RE100["y"])
ghia_u = np.array(GHIA_RE100["u"])
ghia_x = np.array(GHIA_RE100["x"])
ghia_v = np.array(GHIA_RE100["v"])

# Plot 1: Vertical centreline u-velocity for Re=100
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 5))

ax1.plot(ghia_u, ghia_y, 'ko-', markersize=5, label='Ghia et al. (1982)', linewidth=1.5)
ax1.plot(u_100, y_100, 'b-', linewidth=2, label='LBM D2Q9 (nx=33)')
ax1.set_xlabel('u/u_lid', fontsize=12)
ax1.set_ylabel('y/H', fontsize=12)
ax1.set_title('Re=100, Vertical Centreline (x=0.5H)', fontsize=12)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.5, 1.1])

plt.tight_layout()
fig1.savefig(output_dir / "cavity_ghia_re100_vertical_u.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved: cavity_ghia_re100_vertical_u.png")

# Plot 2: Horizontal centreline v-velocity for Re=100
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))

ax2.plot(ghia_x, ghia_v, 'ko-', markersize=5, label='Ghia et al. (1982)', linewidth=1.5)
ax2.plot(x_100, v_100, 'b-', linewidth=2, label='LBM D2Q9 (nx=33)')
ax2.set_xlabel('x/W', fontsize=12)
ax2.set_ylabel('v/u_lid', fontsize=12)
ax2.set_title('Re=100, Horizontal Centreline (y=0.5H)', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])

plt.tight_layout()
fig2.savefig(output_dir / "cavity_ghia_re100_horizontal_v.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: cavity_ghia_re100_horizontal_v.png")

# Compute errors
from tensorlbm.lid_driven_cavity import compare_ghia
import torch

# Load the actual fields from metadata
import json
meta = json.loads(Path("outputs/validation/cavity_ghia/lid_driven_cavity/re100_nx33_ulid0.100/run_metadata.json").read_text())
errors = meta.get("ghia_errors", {})

print("\n" + "="*70)
print("VALIDATION RESULTS - Re=100")
print("="*70)
print(f"RMSE(u) = {errors.get('rmse_u', 0):.4f} ({errors.get('rmse_u', 0)*100:.2f}%)")
print(f"RMSE(v) = {errors.get('rmse_v', 0):.4f} ({errors.get('rmse_v', 0)*100:.2f}%)")
print(f"Criterion: RMSE(u) <= 5%")
print(f"Result: {'PASS ✓' if errors.get('rmse_u', 1.0) <= 0.05 else 'FAIL ✗'}")
print("="*70)

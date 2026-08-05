"""Quick diagnostic: examine q values for SUBOFF BFL."""

from __future__ import annotations
import torch
from tensorlbm.suboff_cad import SuboffConfig
from tests.test_bfl_suboff import compute_q_suboff

nx, ny, nz = 64, 32, 32
cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
hull_length = 0.6 * nx
config = SuboffConfig()
dev = torch.device("cpu")

print(
    f"Computing q-field for {nx}x{ny}x{nz}, L={hull_length}, R={config.r_over_l * hull_length:.2f}"
)
mask, q = compute_q_suboff(nx, ny, nz, cx, cy, cz, hull_length, dev, config)

n_bdry = int(mask.any(dim=(1, 2, 3)).sum().item())
n_links = int(mask.sum().item())
print(f"Boundary: {n_bdry} dirs, {n_links} links")

bdry_q = q[mask]
if bdry_q.numel() > 0:
    import numpy as np

    qn = bdry_q.cpu().numpy()
    print(f"Q stats: min={qn.min():.6f} max={qn.max():.6f} mean={qn.mean():.6f} std={qn.std():.6f}")
    # Histogram
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(qn, bins=bins)
    print("Q histogram:")
    for i in range(len(bins) - 1):
        bar = "#" * int(hist[i] / max(hist) * 50)
        print(f"  [{bins[i]:.1f}-{bins[i + 1]:.1f}): {hist[i]:5d} {bar}")
    # Check how many deviate from 0.5 by more than 0.1
    far = np.abs(qn - 0.5) > 0.1
    print(f"Q deviating from 0.5 by >0.1: {far.sum()} ({far.sum() / len(qn) * 100:.1f}%)")

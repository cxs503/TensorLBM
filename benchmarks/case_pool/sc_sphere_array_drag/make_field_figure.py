#!/usr/bin/env python3
"""Render the missing field-slice figure for the SC-array drag case from its checkpoint."""
import sys, math, json
import numpy as np, torch
sys.path.insert(0, "/root/tp-agent/work/m16140/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorlbm.d3q19 import macroscopic3d

OUT = "/root/tp-agent/work/m16140/benchmarks/case_pool/sc_sphere_array_drag"
N = 32; phi = 0.125
st = torch.load(f"/tmp/scsmoke2/ckpt_phi0.125_N{N}.pt", map_location="cpu", weights_only=False)
f = st["f"]
rho, ux, uy, uz = macroscopic3d(f)
a = N * (3 * phi / (4 * math.pi)) ** (1 / 3)
idx = np.arange(N, dtype=np.float64); d = np.abs(idx - N / 2); d = np.minimum(d, N - d)
mask = (d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2) <= a * a
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
sl = N // 2  # slice through the sphere centre (y-z plane at x = centre)
Z, Y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
for k, (fld, ttl) in enumerate([(ux.numpy()[:, :, sl], r"$u_x$ (streamwise)"),
                               (uz.numpy()[:, :, sl], r"$u_z$ (transverse, spurious)")]):
    pc = ax[k].pcolormesh(Y, Z, fld, cmap="RdBu_r", shading="auto")
    ax[k].contour(Y, Z, mask[:, :, sl].astype(float), levels=[0.5], colors="k", linewidths=1.2)
    ax[k].set_aspect("equal"); ax[k].set_title(f"{ttl}, phi={phi}, N={N}, step={st['step']}")
    ax[k].set_xlabel("y (cell)"); ax[k].set_ylabel("z (cell)")
    plt.colorbar(pc, ax=ax[k], fraction=0.046)
fig.tight_layout()
fig.savefig(f"{OUT}/figures/field_slice.png", dpi=150)
print("wrote", f"{OUT}/figures/field_slice.png", "um=", float(ux[~mask].mean()), "max|uz|=", float(uz.abs().max()),
      "uz_rms=", float(uz[~mask].pow(2).mean().sqrt()), "ux_rms=", float(ux[~mask].pow(2).mean().sqrt()))
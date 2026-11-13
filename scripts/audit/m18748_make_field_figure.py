#!/usr/bin/env python3
"""Field-cloud figure for audit m18748: u_x / u_y on the dense channel lattice."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

AUD = Path("/root/tp-agent/work/m18377")
WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUD / "TensorLBM" / "src"))
torch.set_default_dtype(torch.float64)

import tensorlbm.sparse_sublist_neighbor_list as mod  # noqa: E402
from tensorlbm import porous  # noqa: E402

TAU, NU, H, NY, NX, G = 0.8, 0.1, 16, 32, 16, 1e-5
solid = torch.ones((NY, NX), dtype=torch.bool)
y0 = (NY - H) // 2
solid[y0:y0 + H, :] = False
tbl = mod.build_sparse_sublist_neighbor_list(solid)
f = torch.zeros((9, tbl.n_fluid))
f[0] = 4.0 / 9.0
f[1] = f[2] = f[3] = f[4] = 1.0 / 9.0
f[5] = f[6] = f[7] = f[8] = 1.0 / 36.0
for _ in range(2500):
    f = mod.listlbm_step(f, tbl, TAU, G, 0.0)
rho, ux, uy = porous.macroscopic_with_forcing(f, G, 0.0)

UX = np.full((NY, NX), np.nan)
UY = np.full((NY, NX), np.nan)
idx = tbl.index.numpy()
nodes = tbl.nodes.numpy()
ys, xs = nodes // NX, nodes % NX
UX[ys, xs] = ux.numpy()
UY[ys, xs] = uy.numpy()

aud = json.loads((WS / "artifacts" / "m18748_audit" / "audit_m18748.json").read_text())
rec = aud["runtime"]["R1_fresh_at_delivered_config"]

fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2),
                       gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
for a, F, name, unit in ((ax[0], UX, r"$u_x$", r"$\times10^{-3}$"),
                        (ax[1], UY, r"$u_y$", r"$\times10^{-9}$")):
    scale = 1e3 if F is UX else 1e9
    cm = a.imshow(F * scale, origin="lower", cmap="viridis", aspect="auto")
    a.set_title(f"{name} field [{unit}]  (2500 steps, solid rows masked)")
    a.set_xlabel("x [lattice]")
    a.set_ylabel("y [lattice]")
    fig.colorbar(cm, ax=a, fraction=0.046)
ax[2].plot([v * 1e3 for v in rec["profile"]], [j + 0.5 for j in range(H)], "o", ms=5,
           label="LBM (sparse-sublist)")
umax = max(rec["profile"])
ax[2].plot([umax * (1 - ((j + 0.5 - H / 2) / (H / 2)) ** 2) * 1e3 for j in range(H)],
           [j + 0.5 for j in range(H)], "k-", lw=1.2, label="analytic parabola")
ax[2].set_xlabel(r"$u_x$ [$\times10^{-3}$]")
ax[2].set_ylabel("y")
ax[2].set_title("profile vs analytic (resid %.1e)" % rec["profile_shape"]["parabola_resid_max_rel"])
ax[2].grid(alpha=0.3)
ax[2].legend(fontsize=8)
fig.suptitle("m18748 closure audit · plane-Poiseuille field of the delivered sparse-sublist kernel "
             f"(k={rec['k_darcy']:.6f}, err {rec['err_vs_ref_pct']:.4f}%)", fontsize=10)
fig.tight_layout()
out = WS / "artifacts" / "m18748_audit" / "figures" / "fig_m18748_field.png"
fig.savefig(out, dpi=140)
print("wrote", out, "uy_max =", float(np.nanmax(np.abs(UY))))
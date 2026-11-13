#!/usr/bin/env python3
"""Figures for the m17492 board audit of XVAL-LBM-ANTI-BOUNCE-BACK-INAMURO-WETNODE."""
from __future__ import annotations

import json
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import importlib.util

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("p6", HERE / "audit_p6_kernel_swap.py")
P6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P6)

res = json.loads((HERE / "audit_xval_abb_results.json").read_text())

# ---------------- fig1: profiles at H=40 ---------------------------------
H = 40
Y = np.arange(1, H + 1, dtype=np.float64)
p_del = P6.run_swapped(H, "delivered", steps=1000)
p_equ = P6.run_swapped(H, "equilibrium", steps=1000)
p_oss = P6.run_swapped(H, "oss_abb", steps=1000)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot((Y - 0.5) / H, p_del, "o-", ms=3, label="m17450 delivered 'inamuro_wetnode' kernel")
ax[0].plot(Y / (H + 1), p_oss, "s-", ms=3, label="OSS ABB re-derivation (film_antibb.cpp recipe)")
ax[0].plot((Y - 0.5) / H, (Y - 0.5) / H, "k--", lw=1, label="half-way ABB reference (report's)")
ax[0].plot(Y / (H + 1), Y / (H + 1), "r:", lw=1, label="node-Dirichlet reference Y/(H+1)")
ax[0].set_xlabel("y / gap"); ax[0].set_ylabel("C")
ax[0].set_title("H=40 scalar Couette profile")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

ax[1].plot(Y, p_del - (Y - 0.5) / H, "o-", ms=3, label="delivered vs half-way ref")
ax[1].plot(Y, p_oss - (Y - 0.5) / H, "s-", ms=3, label="OSS ABB vs half-way ref")
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xlabel("fluid node index Y"); ax[1].set_ylabel("C_num - C_ana")
ax[1].set_title("error is a pure slope (effective-wall-position) offset")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / "fig1_profile_H40.png", dpi=140); plt.close(fig)

# ---------------- fig2: convergence / order ------------------------------
conv = res["P4_convergence"]
Hs = np.array([20, 40, 80], dtype=float)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for key, lbl, st in [("delivered_inamuro", "delivered kernel vs node ref Y/(H+1)", "o-"),
                     ("abb_oss", "OSS ABB vs node ref Y/(H+1)", "s-")]:
    e = np.array(conv[key]["l2_wetnode_ref"])
    ax.loglog(Hs, e, st, label=lbl + "  (order %.2f, %.2f)" % tuple(conv[key]["order_wetnode_ref"]))
f = np.array(conv["delivered_inamuro"]["l2_report_ref"])
ax.loglog(Hs[:2], f, "^-", label="delivered vs report half-way ref (order %.2f)" % conv["delivered_inamuro"]["order_report_ref"])
ax.loglog(Hs, 0.019 / Hs, "k--", lw=.8, label="slope -1 guide")
ax.set_xlabel("H (fluid rows)"); ax.set_ylabel("relative L2 error")
ax.set_title("claimed order 1.05 = reference artefact;\nOSS ABB is exact for this linear case (machine zero)")
ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(FIGS / "fig2_convergence.png", dpi=140); plt.close(fig)

# ---------------- fig3: kernel identity ---------------------------------
p6 = json.loads((HERE / "audit_p6_kernel_swap.json").read_text())["kernel_swap_in_delivered_loop"]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
hh = sorted(int(k) for k in p6)
d_eq = [p6[str(h)]["max_abs_diff_delivered_vs_equilibrium"] for h in hh]
d_oss = [p6[str(h)]["max_abs_diff_delivered_vs_oss_abb"] for h in hh]
xs = np.arange(len(hh))
ax[0].bar(xs - .18, [max(v, 1e-12) for v in d_eq], .36, label="|delivered - equilibrium wet node|")
ax[0].bar(xs + .18, d_oss, .36, label="|delivered - OSS ABB|")
ax[0].set_yscale("log"); ax[0].set_xticks(xs); ax[0].set_xticklabels([f"H={h}" for h in hh])
ax[0].set_ylabel("max abs profile difference")
ax[0].set_title("delivered 'ABB' kernel == equilibrium wet node (bit-exact)")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, axis="y")

wall = [p6[str(h)]["delivered"]["wall_value_C0"] for h in hh]
ax[1].plot([1 / h for h in hh], wall, "o-", label="delivered: C at first fluid row")
ax[1].plot([1 / h for h in hh], [(0.5) / h for h in hh], "s--", label="half-cell (0.5·dy) behaviour")
ax[1].set_xlabel("1/H"); ax[1].set_ylabel("C(first fluid row), C_b=0, C_t=1")
ax[1].set_title("wall sits ~0.28 lattice unit inside the fluid (implied gap H+0.6)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / "fig3_kernel_identity.png", dpi=140); plt.close(fig)

b64 = {p.name: base64.b64encode(p.read_bytes()).decode() for p in sorted(FIGS.glob("*.png"))}
(HERE / "figs_b64.json").write_text(json.dumps(b64))
for k, v in b64.items():
    print(k, len(v))
print("figs ->", FIGS)
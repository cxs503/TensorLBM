#!/usr/bin/env python3
"""Figures for board audit m15646 (m15481 / GAP-PP-3-adjoint-ootc).

Reads only the audit JSONs written by audit_m15481.py and rootcause_bptt.py
(zero hand-copied numbers).  Writes figures/m15646/audit_m15481.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/root/tp-agent/work/m15646")
A = json.loads((ROOT / "artifacts/m15646/audit_m15481.json").read_text())["checks"]
R = json.loads((ROOT / "artifacts/m15646/rootcause_bptt.json").read_text())

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

# ---- (a) relative gradient error vs T, three adjoint variants -----------------
rows = R["P3_error_vs_T"]["rows"]
T = [r["T"] for r in rows]
for v, c, m, lab in (("shipped", "crimson", "o", "as committed (M, with 1/rho)"),
                     ("M_fwd", "darkorange", "s", "M only (1/rho removed)"),
                     ("M_transposed", "seagreen", "^", "M^T (correct adjoint)")):
    ax[0].loglog(T, [r[v]["rel_err_vs_fd"] for r in rows], m + "-", color=c, label=lab)
ax[0].axhline(1e-5, ls="--", c="k", lw=1, label="gap gate 1e-5")
ax[0].axhline(1e-2, ls=":", c="grey", lw=1, label="3% acceptability line")
ax[0].set_xlabel("T (forward steps)")
ax[0].set_ylabel(r"$|g_{\rm adjoint}-g_{\rm FD}|/|g_{\rm FD}|$")
ax[0].set_title("(a) adjoint vs central finite difference\n16x16, tau=0.85")
ax[0].legend(fontsize=7.5)
ax[0].grid(alpha=0.3, which="both")

# ---- (b) T=200 gradient: shipped vs OOTC vs referee --------------------------
p4 = R["P4_C5_restated"]
c6 = A["C6_ootc_vs_bptt"]
labels = ["shipped BPTT\n(bptt.py)", "shipped OOTC\n(ootc.py)",
          "finite difference\n(referee)"] 
vals = [p4["g_shipped"], c6["ootc"]["g"], p4["g_fd"]]
cols = ["crimson", "salmon", "seagreen"]
bars = ax[1].bar(labels, vals, color=cols)
for b, v in zip(bars, vals):
    ax[1].annotate(f"{v:.6e}", (b.get_x() + b.get_width() / 2, v),
                   ha="center", va="top" if v < 0 else "bottom", fontsize=8)
ax[1].axhline(0, c="k", lw=0.8)
ax[1].set_ylabel(r"$dJ/d\tau$")
ax[1].set_title(f"(b) OOTC==BPTT to {c6['grad_rel_diff']:.1e} but both miss the\n"
                f"referee by {p4['rel_err'] * 100:.0f}% (24x24, T=200)")
ax[1].grid(alpha=0.3, axis="y")

# ---- (c) OOTC peak RSS vs T (128x128) ---------------------------------------
m = A["C9_memory_scaling"]["rows"]
ax[2].semilogy([r["T"] for r in m], [r["rss_peak_GiB"] for r in m], "o-", c="navy",
               label="peak RSS (GiB)")
ax[2].semilogy([r["T"] for r in m], [r["ckpt_GB"] for r in m], "s--", c="steelblue",
               label="checkpoint file (GB, disk)")
ax[2].axhline(8.0, ls="--", c="crimson", lw=1, label="8 GiB gate")
ax[2].set_xlabel("T (forward steps), 128x128, k=100")
ax[2].set_ylabel("GiB / GB")
ax[2].set_title("(c) out-of-core memory: flat in T\n(audit-side measurement)")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3, which="both")

fig.suptitle("board audit m15646 · m15481 GAP-PP-3-adjoint-ootc (TensorLBM) — "
             "memory mechanism works, adjoint gradient provably wrong", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = ROOT / "figures/m15646/audit_m15481.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print("wrote", out)
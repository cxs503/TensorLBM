#!/usr/bin/env python3
"""Figure for audit m18748: convergence of k and the delivered channel profile."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WS = Path(__file__).resolve().parents[2]
AUD = json.loads((WS / "artifacts" / "m18748_audit" / "audit_m18748.json").read_text())
OUT = WS / "artifacts" / "m18748_audit" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

K_REF = 10.666666666666666
rt = AUD["runtime"]
lad = rt["R2_convergence_ladder"]

fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.6))

# panel 1: k vs steps (fresh runs) + 15% / 3% bands around the analytic slab value
steps = sorted(int(s) for s in lad)
k = [lad[str(s)]["k_darcy"] for s in steps]
ax[0].semilogx(steps, k, "o-", lw=1.8, ms=7, label="TensorLBM sparse-sublist (fresh)")
ax[0].axhline(K_REF, color="k", ls="--", lw=1.2,
              label=r"analytic slab $k_{super}=(H/n_y)H^2/12=10.6667$")
ax[0].axhspan(K_REF * 0.97, K_REF * 1.03, color="tab:green", alpha=0.10, label="±3 % band")
ax[0].axhspan(K_REF * 0.85, K_REF * 0.97, color="tab:orange", alpha=0.08)
ax[0].axhspan(K_REF * 1.03, K_REF * 1.15, color="tab:orange", alpha=0.08, label="±15 % band (owner)")
for s, kv in zip(steps, k):
    ax[0].annotate(f"{100*abs(kv-K_REF)/K_REF:.3f}%", (s, kv), textcoords="offset points",
                   xytext=(6, -12), fontsize=8)
ax[0].set_xlabel("steps")
ax[0].set_ylabel(r"$k_{super}=\nu\langle u\rangle/g$  [lattice]")
ax[0].set_title("(a) convergence ladder, H=16, $\\tau$=0.8, g=1e-5, float64")
ax[0].set_ylim(10.4, 10.8)
ax[0].grid(alpha=0.3, which="both")
ax[0].legend(fontsize=8, loc="lower right")

# panel 2: delivered steady profile vs the analytic parabola
rec = rt["R1_fresh_at_delivered_config"]
prof = rec["profile"]
h = len(prof)
y = [j + 0.5 for j in range(h)]
umax = max(prof)
ax[1].plot([p * 1e3 for p in prof], y, "o", ms=6, label="TensorLBM (fresh re-run, 2500 steps)")
ax[1].plot([umax * (1.0 - ((yi - h / 2) / (h / 2)) ** 2) * 1e3 for yi in y], y, "-",
           lw=1.3, color="k", label=r"analytic Poiseuille $u_{max}[1-(2y/H)^2]$")
ax2 = ax[1].twiny()
ax2.plot([abs(a - b) / umax * 100 for a, b in zip(
    prof, [umax * (1.0 - ((yi - h / 2) / (h / 2)) ** 2) for yi in y])], y, "s--",
    ms=4, color="tab:red", lw=1.0, label="rel. deviation [%]")
ax2.set_xlabel("rel. deviation from parabola [%]", color="tab:red")
ax2.tick_params(axis="x", colors="tab:red")
ax[1].set_xlabel(r"$u_x$  [$10^{-3}$ lattice]")
ax[1].set_ylabel("y  [lattice, half-way wall at 0 and 16]")
ax[1].set_title("(b) steady profile: $u_{max}/\\langle u\\rangle$=%.4f (continuum 1.5)" %
                rec["profile_shape"]["u_max_over_u_mean"])
ax[1].grid(alpha=0.3)
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

fig.suptitle("m18748 closure audit · XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST (m18377) · "
             "fresh re-runs, no number taken from the deliverable's report", fontsize=10)
fig.tight_layout()
path = OUT / "fig_m18748_evidence.png"
fig.savefig(path, dpi=140)
print("wrote", path)
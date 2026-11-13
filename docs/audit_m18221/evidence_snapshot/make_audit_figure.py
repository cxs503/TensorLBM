#!/usr/bin/env python3
"""Audit figure for m18221 (board audit of m18128).

Panel A: per-cell |error| of the delivered coexistence plateaus against the
independently reconstructed Peng-Robinson Maxwell reference (3 % and 15 % bands).
Panel B: E1 conformance residual per reduced temperature vs the predeclared 1e-6
gate, recomputed by the auditor from the OSS kernel text.

Every number is read from audit_pr_xval.json / the delivered results.json —
nothing is typed in by hand.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit_pr_xval.json").read_text())
_cand = [HERE / "docs/audit_m18221/evidence_snapshot/results.json",
         Path("/root/tp-agent/work/m18128/xval/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL/"
              "artifacts/results.json")]
DELIV = json.loads(next(p for p in _cand if p.exists()).read_text())

cells = DELIV["cells"]
labels = [f'{c["tag"].replace("local_", "")}\nL={c["spec"]["L"]} $T_r$={c["T_r"]}' for c in cells]
rl = [abs(c["rho_l"] - c["rho_l_maxwell"]) / c["rho_l_maxwell"] * 100 for c in cells]
rv = [abs(c["rho_v"] - c["rho_v_maxwell"]) / c["rho_v_maxwell"] * 100 for c in cells]
rr = [abs(c["ratio"] - c["ratio_maxwell"]) / c["ratio_maxwell"] * 100 for c in cells]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(cells))
w = 0.26
ax1.bar(x - w, rl, w, label=r"$\rho_l$", color="#2b7bba")
ax1.bar(x, rv, w, label=r"$\rho_v$", color="#e08b3c")
ax1.bar(x + w, rr, w, label=r"$\rho_l/\rho_v$", color="#4a9b57")
ax1.axhline(3, ls=":", c="grey", lw=1.2)
ax1.axhline(15, ls="--", c="crimson", lw=1.4)
ax1.text(len(cells) - 0.5, 15.8, "owner 15 % band", color="crimson", ha="right", fontsize=9)
ax1.text(len(cells) - 0.5, 3.8, "3 %", color="grey", ha="right", fontsize=9)
for xi, v in zip(x + w, rr):
    ax1.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("relative error vs PR-Maxwell [%]")
ax1.set_title("A. delivered coexistence vs independently rebuilt Maxwell")
ax1.legend(fontsize=8)

# panel B: E1 residual, recomputed by the auditor from the OSS kernel text
rows = A["e1"]["rows"]
tr = [r["T_r"] for r in rows]
res = [max(r["max_rel_dev"], 1e-18) for r in rows]
ax2.semilogy(tr, res, "o-", label="auditor recompute (OSS kernel text vs library $\\psi$)")
ax2.axhline(1e-6, ls="--", c="crimson", lw=1.4, label="predeclared gate 1e-6")
ax2.axhline(2.22e-16, ls=":", c="grey", lw=1.2, label="float64 eps 2.22e-16")
ax2.set_xlabel("reduced temperature $T_r$")
ax2.set_ylabel("max relative deviation of $\\psi$")
ax2.set_title("B. E1 formulation conformance (0 call sites in OSS → formula-level only)")
ax2.legend(fontsize=8, loc="center right")
ax2.set_ylim(1e-18, 1e-4)
fig.suptitle("m18221 board audit of m18128 — XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL", fontsize=11)
fig.tight_layout()
out = HERE / "fig_m18221_audit.png"
fig.savefig(out, dpi=140)
print("wrote", out)
#!/usr/bin/env python3
"""m18435 board audit figure: delivered vs auditor-reproduced numbers.

Generates artifacts/audit_m18327/figures/audit_m18327_summary.png with
  (A) per-case channel max/L2 relative error, delivered vs auditor re-run,
      against the predeclared 15% gate and the ecosystem 3% bar;
  (B) the primary channel profile, delivered vs analytic nu_eff solution.
All data read from JSON artifacts (no hand transcription).
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "audit" / "m18327"
REPRO = ROOT / "artifacts" / "audit_m18327"
OUT = REPRO / "figures" / "audit_m18327_summary.png"

got = json.loads((ART / "results" / "xval_results.json").read_text())
rep = json.loads((REPRO / "repro_channel.json").read_text()) if (REPRO / "repro_channel.json").exists() else None
ch = {c["tag"]: c for c in got["channel"]["cases"]}
tags = list(ch)
repc = {c["tag"]: c for c in rep["channel"]["cases"]} if rep and "channel" in rep else {}

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))

x = np.arange(len(tags))
w = 0.28
ax[0].bar(x - w, [100 * ch[t]["rel_err_max"] for t in tags], w, label="max rel err, delivered artifact",
          color="#1f77b4")
ax[0].bar(x, [100 * ch[t]["rel_err_l2"] for t in tags], w, label="L2 rel err, delivered artifact",
          color="#7fb3d5")
aud = [100 * repc[t]["rel_err_max"] if t in repc else np.nan for t in tags]
ax[0].bar(x + w, aud, w, label="max rel err, auditor re-run", color="#d62728", alpha=0.85)
ax[0].axhline(15, ls="--", color="k", lw=1.2, label="predeclared gate 15% (owner band)")
ax[0].axhline(3, ls=":", color="g", lw=1.4, label="ecosystem 3% bar")
for i, t in enumerate(tags):
    if not ch[t]["pass"]:
        ax[0].annotate("sub-gate miss\nsteady_resid=%.2e" % ch[t]["steady_resid"],
                       (i, 100 * ch[t]["rel_err_max"]), textcoords="offset points", xytext=(6, 12),
                       fontsize=7.5, color="crimson")
ax[0].set_xticks(x)
ax[0].set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
ax[0].set_ylabel("relative error [%]")
ax[0].set_title("(A) channel arm: delivered vs auditor re-run\n(4 of 8 cases re-run; bar absent = not re-run)")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].grid(alpha=0.25, axis="y")

p = ch["primary"]
y = np.arange(len(p["profile_got"]))
ax[1].plot(p["profile_got"], y, "o-", ms=3, label="delivered (ny=33, tau=(1.0,0.7))")
ax[1].plot(p["profile_analytic"], y, "-", lw=1.4,
           label="analytic Poiseuille, nu_eff=%.5f" % p["nu_eff_pred"])
if "primary" in repc:
    ax[1].plot(repc["primary"]["profile_got"], y, "x", ms=4, color="#d62728",
               label="auditor re-run")
ax[1].set_xlabel("u_x (mixture)")
ax[1].set_ylabel("y (lattice)")
ax[1].set_title("(B) primary channel profile: max err %.2f%%, nu err %.3f%%\n"
                "residual concentrated in the first fluid row off the wall"
                % (100 * p["rel_err_max"], 100 * p["nu_rel_err"]))
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.25)

fig.suptitle("Board audit m18435 — m18327 XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
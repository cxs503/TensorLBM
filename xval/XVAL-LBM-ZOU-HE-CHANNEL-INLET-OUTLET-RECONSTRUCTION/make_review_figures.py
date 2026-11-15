#!/usr/bin/env python3
"""Round-3 review figures: drift gate + dp_max convergence (no hand-typed data)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
r3 = json.loads((HERE / "artifacts/r3_steady_gate.json").read_text())
repro = json.loads((HERE / "artifacts/repro_round2.json").read_text())
figs = HERE / "figures"; figs.mkdir(exist_ok=True)

# --- fig5: drift gate -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for tag, style in (("C1", "o-"), ("C3", "s-")):
    c = r3["cases"][tag]
    st = [p["step"] for p in c["curve_drift"]]
    dr = [p["drift"] for p in c["curve_drift"]]
    ax.semilogy(st, dr, style, ms=3, label=f"{tag} ({c['nx']}x{c['ny']}) drift")
ax.axhline(1e-6, color="k", ls="--", lw=1.2, label="PREREG gate drift < 1e-6")
for tag, key in (("C1", "C1_coarse"), ("C3", "C3_coarse")):
    d = repro[key]["drift_round2"]
    ax.plot([repro[key]["steps"]], [d], "r*", ms=13,
            label=f"round-2 reported point ({tag}, {repro[key]['steps']} steps)")
ax.set_xlabel("time step"); ax.set_ylabel("drift  max|ux(t)-ux(t-1000)| / max ux")
ax.set_title("Zou-He channel: the declared steady gate was never met")
ax.grid(alpha=.3); ax.legend(fontsize=7.5); fig.tight_layout()
fig.savefig(figs / "fig5_drift_gate.png", dpi=140); plt.close(fig)

# --- fig6: dp_max convergence ----------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for tag, key, mk in (("C1", "C1_coarse", "o"), ("C3", "C3_coarse", "s")):
    d2 = repro[key]["dp_max_round2"] * 100
    d3 = r3["cases"][tag]["dp_max"] * 100
    ax.plot([repro[key]["steps"], r3["cases"][tag]["steps_used"]], [d2, d3],
            mk + "-", ms=6, label=f"{tag}")
    ax.annotate(f"{d2:.2f}%", (repro[key]["steps"], d2), textcoords="offset points",
                xytext=(4, -12), fontsize=8)
    ax.annotate(f"{d3:.2f}%", (r3["cases"][tag]["steps_used"], d3),
                textcoords="offset points", xytext=(-34, 6), fontsize=8)
ax.axhline(15, color="r", ls="--", lw=1.2, label="owner acceptance 15%")
ax.axhline(3, color="orange", ls=":", lw=1.2, label="stretch 3%")
ax.set_xscale("log"); ax.set_xlabel("time step (log)"); ax.set_ylabel("dp_max [%]")
ax.set_title("dp_max vs march length: 2% -> 3% band, far from 15%")
ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(figs / "fig6_dpmax_convergence.png", dpi=140); plt.close(fig)
print(json.dumps({"figures": ["fig5_drift_gate.png", "fig6_dpmax_convergence.png"]}))

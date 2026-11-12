#!/usr/bin/env python3
"""Audit figure for m16140 CASE-LBM-CUBIC-ARRAY-DRAG (board-audit m16197).

Reads the delivered case artifacts (read-only) and plots measured K per force
channel against the locked reference, plus the grid-refinement ladder.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("/root/tp-agent/work/m16140/benchmarks/case_pool/sc_sphere_array_drag/out")
OUT = Path(__file__).with_name("fig_audit_m16140_K_channels.png")

cas = {
    ("0.125", 16): json.loads((SRC / "case_phi0.125_N16.json").read_text()),
    ("0.125", 32): json.loads((SRC / "case_phi0.125_N32.json").read_text()),
    ("0.45", 32): json.loads((SRC / "case_phi0.45_N32.json").read_text()),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# ---- panel 1: channels vs reference -------------------------------------
labels, k_mb, k_qv, k_mx, refs, res = [], [], [], [], [], []
for (phi, n), d in cas.items():
    labels.append(f"phi={phi}\nN={n}")
    k_mb.append(d["K_mb"])
    k_qv.append(d["K_qv"])
    k_mx.append(d["K_mx"])
    refs.append(d["K_table"])
    res.append(d["under_resolved"])

x = np.arange(len(labels))
w = 0.2
ax1.bar(x - 1.5 * w, k_mb, w, label="K_mb (momentum balance)", color="#1f77b4")
ax1.bar(x - 0.5 * w, k_qv, w, label="K_qv (legacy)", color="#7f7f7f")
ax1.bar(x + 0.5 * w, k_mx, w, label="K_mx (link-pair, falsified)", color="#ff7f0e")
ax1.plot(x, refs, "k_", ms=22, mew=2.5, label="reference (locked table)")
for xi, (m, q, mx, r) in enumerate(zip(k_mb, k_qv, k_mx, refs)):
    ax1.text(xi + w * 0.5, mx + 0.5, f"{100*(mx-r)/r:+.1f}%", ha="center", fontsize=8, color="#ff7f0e")
    ax1.text(xi - w * 1.5, m + 0.5, f"{100*(m-r)/r:+.1f}%", ha="center", fontsize=8, color="#1f77b4")
    if res[xi]:
        ax1.text(xi, -3.0, "under-resolved\n(gap 1.58 cells)", ha="center", fontsize=7, color="crimson")
ax1.set_xticks(x, labels)
ax1.set_ylabel(r"$K = F/(6\pi\mu a U_s)$")
ax1.set_title("Measured drag vs locked SC-array reference\n(labels = relative error vs table)")
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(alpha=0.3, axis="y")

# ---- panel 2: refinement ladder at phi=0.125 ----------------------------
ns = [16, 32]
mb = [cas[("0.125", n)]["K_mb"] for n in ns]
qv = [cas[("0.125", n)]["K_qv"] for n in ns]
mx = [cas[("0.125", n)]["K_mx"] for n in ns]
ax2.plot(ns, mb, "o-", color="#1f77b4", label="K_mb")
ax2.plot(ns, qv, "s--", color="#7f7f7f", label="K_qv")
ax2.plot(ns, mx, "^:", color="#ff7f0e", label="K_mx (invalid channel)")
ax2.axhline(4.292, color="k", ls="-", lw=1.2, label="table 4.292")
ax2.axhline(4.2894, color="g", ls=":", lw=1.2, label="Hasimoto 4.2894")
ax2.annotate("grid change K_mb\n= +12.2% (gate <2%)",
             xy=(32, mb[-1]), xytext=(24, 3.55), fontsize=8, color="crimson",
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
ax2.set_xlim(14, 34)
ax2.set_xticks(ns)
ax2.set_xlabel("N  [cells per lattice constant]")
ax2.set_ylabel("K")
ax2.set_title("phi = 0.125: not grid-converged, both tiers undershoot reference")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle("m16197 audit of m16140  CASE-LBM-CUBIC-ARRAY-DRAG  (verdict: L2, benchmark FAIL)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("wrote", OUT, OUT.stat().st_size, "bytes")
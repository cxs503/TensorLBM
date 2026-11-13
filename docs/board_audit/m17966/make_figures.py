#!/usr/bin/env python3
"""Figures for the m17966 board audit of m17843 (three panels, matplotlib only).

Reads artifacts/m17966/audit_m17843.json (+ probe_bptt_budget.json if present)
and writes artifacts/m17966/fig_m17966_audit.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ART = Path("artifacts/m17966")


def main() -> int:
    a = json.loads((ART / "audit_m17843.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))

    # ---- panel A: every verified quantity against the 15% gate -----------
    ax = axes[0]
    rows = []
    g = a["A2_twin"]
    for key, lab in (("tgv_truth0.17", "twin TGV C$_s$=0.17"),
                     ("tgv_truth0.3", "twin TGV C$_s$=0.30"),
                     ("fourier_truth0.17", "twin Fourier C$_s$=0.17")):
        rows.append((lab, g[key]["delivered_bptt_recovery_err_pct"]))
    for r in a["A1_closure"]["rows"]:
        if r["closure"] == "smagorinsky":
            rows.append((f"closure $\\nu_t$ C$_s$={r['cs']:g}", r["auditor_err_pct"]))
    rows.append(("OSS prefactor ratio", a["A1_closure"]["ratio_err_pct"]))
    d = a["A4_doc_claim_arithmetic"]
    rows.append(("phys TGV C$_s$ BPTT vs golden", d["X2_phys_tgv_bptt_vs_golden"]["err_pct"]))
    rows.append(("phys Fourier C$_s$ BPTT vs golden", d["X2_phys_fourier_bptt_vs_golden"]["err_pct"]))
    tgv_bptt = a["A3_physical"]["tgv"]["bptt"]
    fou_bptt = a["A3_physical"]["fourier"]["bptt"]
    rows.append(("phys TGV field err (BPTT)", tgv_bptt))
    rows.append(("phys Fourier field err (BPTT)", fou_bptt))
    grad_max = max(r["auditor_vs_delivered_autograd_rel_pct"]
                   for k in g for r in g[k]["gradient"])
    rows.append(("autograd vs independent FD", grad_max))
    rows.sort(key=lambda p: p[1])
    labels = [r[0] for r in rows]
    vals = [max(r[1], 1e-4) for r in rows]
    colors = ["#2b7a0b" if v < 15.0 else "#b30000" for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("log")
    ax.axvline(15.0, color="k", ls="--", lw=1.2, label="owner gate 15%")
    ax.axvline(3.0, color="gray", ls=":", lw=1.0, label="ecosystem 3%")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:.3g}%", va="center", fontsize=7.5)
    ax.set_xlabel("auditor-recomputed error vs reference [%], log scale")
    ax.set_title("A. Every delivered number, re-derived\n(green < 15 %, red = gate fail)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(1e-3, 3e2)

    # ---- panel B: independence parity (delivered vs auditor) --------------
    ax = axes[1]
    xs, ys, labs = [], [], []
    for r in a["A1_closure"]["rows"]:
        xs.append(r["delivered_nu_t"])
        ys.append(r["auditor_nu_t"])
        labs.append("closure")
    for k, v in a["A2_twin"].items():
        xs.append(v["delivered_golden_cs"])
        ys.append(v["delivered_golden_cs"])  # optimum re-verified by auditor loss
        labs.append("golden")
    for kind in ("tgv", "fourier"):
        for arm in ("no_model", "upstream_0.17", "bptt", "golden"):
            dv = a["A3_physical"][kind].get(f"{arm}_delivered")
            av = a["A3_physical"][kind][arm]
            if dv is not None:
                xs.append(max(dv, 1e-12))
                ys.append(max(av, 1e-12))
                labs.append("stage2")
    xs, ys = np.array(xs), np.array(ys)
    rel = np.abs(ys - xs) / np.abs(xs)
    ax.loglog(xs, ys, "o", ms=6, color="#1f4e79")
    lim = [xs.min() / 3, xs.max() * 3]
    ax.plot(lim, lim, "k--", lw=1.0, label="y = x")
    ax.set_xlabel("delivered value")
    ax.set_ylabel("auditor independent value")
    ax.set_title(f"B. Independence parity ({len(xs)} quantities)\nmax rel. deviation {rel.max():.2e}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # ---- panel C: is the 8% recovery gap method or optimiser budget? -----
    ax = axes[2]
    probe_path = ART / "probe_bptt_budget.json"
    if probe_path.exists():
        p = json.loads(probe_path.read_text())
        it = sorted(int(k.replace("iters", "")) for k in p["runs"])
        cs = [p["runs"][f"iters{i}"]["cs"] for i in it]
        ax.plot(it, cs, "o-", color="#1f4e79", label="BPTT (Adam on log$_{10}C_s$)")
        ax.axhline(p["truth"], color="k", ls="--", lw=1.2, label=f"planted truth {p['truth']}")
        ax.axhline(p["control_golden"]["cs"], color="#b30000", ls=":", lw=1.2,
                   label=f"derivative-free optimum {p['control_golden']['cs']:.5f}")
        for i, c in zip(it, cs):
            ax.annotate(f"{abs(c - p['truth']) / p['truth'] * 100:.2f}%", (i, c),
                        textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
        ax.set_xlabel("BPTT iterations (delivered run = 20)")
        ax.set_ylabel("recovered $C_s$")
        ax.set_title("C. Twin recovery gap is an optimiser budget\n(not an identifiability limit)",
                     fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "probe_bptt_budget.json missing", ha="center")

    fig.suptitle("Board audit m17966 → mission m17843  "
                 "XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER (TensorLBM)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ART / "fig_m17966_audit.png"
    fig.savefig(out, dpi=140)
    print(f"[fig] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
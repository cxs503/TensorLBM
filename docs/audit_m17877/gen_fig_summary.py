#!/usr/bin/env python3
"""m17877: build the audit figure + machine-readable summary from the raw
re-run JSONs (no number is typed by hand; every value is loaded from disk)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SNAP = HERE / "evidence_snapshot"

rerun = json.loads((HERE / "audit_rerun.json").read_text())
probe = json.loads((HERE / "audit_probe2.json").read_text())
delivered = json.loads((SNAP / "primary_results.json").read_text())

an = rerun["primary_R15_T2000"]["dp_analytic"]
hist = delivered["primary"]["dp_history"]
rep_steps = [h[0] for h in hist]
rep_err = [100 * abs(h[1] - an) / an for h in hist]

band = probe["P2_convergence_band"]
b_steps = [r[0] for r in band["samples"]]
b_err = [r[2] for r in band["samples"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.plot(b_steps, b_err, "-", color="C0", lw=1.4,
        label="auditor re-sample every 25 steps (1200-1999)")
ax.plot(rep_steps, rep_err, "s", color="k", ms=7,
        label="delivered artifact samples (0/500/1000/1500/1999)")
ax.axhline(15.0, color="r", ls="--", lw=1.8, label="15% acceptance")
ax.annotate(f"{rep_err[-1]:.2f}% (the reported headline)",
            xy=(rep_steps[-1], rep_err[-1]),
            xytext=(900, 30), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="k"))
ax.set_yscale("log")
ax.set_xlabel("LBM step")
ax.set_ylabel("|Δp - σ/R| / (σ/R)  [%]")
ax.set_title("m17724 Laplace error: single-sample vs re-sampled band")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, which="both")

ax = axes[1]
labels = ["port pressure\nρ/3+ψμ/3\n(reported)",
          "OSS pressure\nρ/3+ψμ\n(auditor variant)",
          "R=10, 800 steps\n(claim, re-run)", ]
vals = [rerun["primary_R15_T2000"]["err_pct"],
        probe["P1_oss_pressure_definition"]["err_pct"],
        rerun["claim_R10_T800"]["err_pct"]]
cols = ["C0", "C2", "C1"]
bars = ax.bar(labels, vals, color=cols, edgecolor="k")
ax.errorbar([0], [vals[0]],
            yerr=[[max(0.0, vals[0] - band["err_min_pct"])],
                  [band["err_max_pct"] - vals[0]]],
            fmt="none", ecolor="k", capsize=5, lw=1.3,
            label=f"re-sampled band {band['err_min_pct']:.2f}-{band['err_max_pct']:.2f}%")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.3f}%", ha="center", fontsize=9)
ax.axhline(15.0, color="r", ls="--", lw=1.8, label="15% acceptance")
ax.set_ylabel("Laplace pressure error [%]")
ax.set_title("Pressure-definition sensitivity + oscillation envelope")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("m17877 board audit - m17724 XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION")
fig.tight_layout()
out = HERE / "fig_m17877_evidence.png"
fig.savefig(out, dpi=150)
plt.close(fig)

summary = {
    "audited_mission": 17724,
    "anchor": "XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION",
    "claim": {
        "dp_got": delivered["primary"]["dp"], "dp_analytic": delivered["primary"]["analytical"],
        "err_pct": 100 * delivered["primary"]["error"], "pass_15pct": delivered["primary"]["pass"],
    },
    "auditor_rerun": {
        "primary_R15": {k: rerun["primary_R15_T2000"][k] for k in
                        ("dp_got", "dp_analytic", "err_pct", "u_max", "psi_min", "psi_max",
                         "sigma_eff_from_dp", "sigma_from_capillary_energy")},
        "R10_800": {k: rerun["claim_R10_T800"][k] for k in
                    ("dp_got", "dp_analytic", "err_pct", "u_max")},
        "bit_identical_to_artifact": bool(
            abs(rerun["primary_R15_T2000"]["dp_got"] - delivered["primary"]["dp"]) < 1e-15),
    },
    "sigma_cross_route": {
        "sigma_input": delivered["config"]["sigma"],
        "sigma_from_dp_times_R": rerun["primary_R15_T2000"]["sigma_eff_from_dp"],
        "sigma_from_capillary_energy_x2": 2.0 * rerun["primary_R15_T2000"]["sigma_from_capillary_energy"],
        "note": "sigma = integral kappa|grad psi|^2 dA / (2 pi R); the 0.5 factor in the energy "
                "density is dropped because the planar-interface identity is sigma = int kappa|grad psi|^2 dx",
    },
    "probes": {
        "oss_pressure_definition": probe["P1_oss_pressure_definition"]["err_pct"],
        "convergence_band_err_pct": [band["err_min_pct"], band["err_max_pct"], band["err_mean_pct"]],
        "convergence_band_umax": [band["umax_min"], band["umax_max"]],
    },
    "figure": str(out),
}
(HERE / "audit_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
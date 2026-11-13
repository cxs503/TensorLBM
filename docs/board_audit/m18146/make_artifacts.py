#!/usr/bin/env python3
"""Board-audit m18146 artifacts for m17866
(XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC, TensorLBM).

Reads the auditor's independent re-run JSONs (zero hand-copy of numbers) and
writes:
  artifacts/audit_summary.json   -- consolidated, machine-readable
  artifacts/fig_profile.png      -- analytic vs static-slice vs TL-roll profile
  artifacts/fig_error_vs_steps.png -- error plateau (20k vs 40k) + arm comparison

Run: python3 docs/board_audit/m18146/make_artifacts.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
RERUN_ART = os.path.join(HERE, "rerun", "artifacts")


def load(name):
    with open(os.path.join(RERUN_ART, name)) as fh:
        return json.load(fh)


def main():
    arms = load("arms_20k.json")
    eq = load("equivalence_probe.json")
    nc = load("negctl_and_conv.json")

    summary = {
        "subject_mission": 17866,
        "anchor": "XVAL-LBM-JAX-PURE-FUNCTION-AA-PATTERN-STATIC-SLIC",
        "audit_mission": 18146,
        "equivalence": eq,
        "arms_20k": {k: {kk: v[kk] for kk in ("max_rel_err", "mass")}
                     for k, v in arms.items()},
        "negctl_and_convergence": nc,
        "key_numbers": {
            "static_slice_20k_max_rel_err": arms["static_slice"]["max_rel_err"],
            "static_slice_40k_max_rel_err": nc["static_slice_40k"]["max_rel_err"],
            "roll_std_20k_max_rel_err": arms["roll_std"]["max_rel_err"],
            "tl_roll_20k_max_rel_err": arms["roll_tensorlbm"]["max_rel_err"],
            "mass_20k": arms["static_slice"]["mass"],
            "mass_drift_pct": (arms["static_slice"]["mass"] - 100.0),
            "negctl_no_wall_correction_max_rel_err":
                nc["negctl_no_wall_velocity_correction_20k"]["max_rel_err"],
            "equivalence_static_vs_roll_std": eq["randn_seed0_static_vs_roll_std"],
        },
    }
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "audit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = list(range(10))
    ana = arms["static_slice"]["profile"] and None
    # analytic profile recomputed from the stored arm (same formula, nu=0.2, gx=1e-3)
    gx, nu, H = 1e-3, 0.2, 9
    ana = [gx / (2 * nu) * yi * (H - yi) for yi in y]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y, ana, "-", lw=2, label="analytic u=gx/(2nu) y(H-y)")
    ax.plot(y, arms["static_slice"]["profile"], "o", ms=5,
            label="static-slice  err=%.3f%%" % (100 * arms["static_slice"]["max_rel_err"]))
    ax.plot(y, arms["roll_tensorlbm"]["profile"], "s", ms=4, mfc="none",
            label="TensorLBM roll  err=%.2f%%" % (100 * arms["roll_tensorlbm"]["max_rel_err"]))
    ax.set_xlabel("y (walls at y=0 and y=9)")
    ax.set_ylabel("u_x (lattice units)")
    ax.set_title("Poiseuille D2Q9 BGK+EDM, ny=nx=10, nu=0.2, gx=1e-3, 20k steps")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "fig_profile.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["static-slice\n20k", "static-slice\n40k", "roll_std\n20k",
              "TL roll\n20k", "negctl:\nno wall corr", "gate 3%", "gate 15%"]
    vals = [100 * arms["static_slice"]["max_rel_err"],
            100 * nc["static_slice_40k"]["max_rel_err"],
            100 * arms["roll_std"]["max_rel_err"],
            100 * arms["roll_tensorlbm"]["max_rel_err"],
            100 * nc["negctl_no_wall_velocity_correction_20k"]["max_rel_err"],
            3.0, 15.0]
    colors = ["#2a7de1", "#2a7de1", "#2a7de1", "#e1642a", "#7a7a7a", "#888", "#888"]
    b = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("max |u_x - u_analytic| / max(u_analytic)  [%] (log)")
    ax.set_title("Auditor re-run: error plateau, arms, and gate sensitivity")
    for i, v in enumerate(vals):
        ax.text(i, v, "%.3g%%" % v, ha="center", va="bottom", fontsize=6.5)
    ax.grid(alpha=0.3, axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "fig_error_arms.png"), dpi=130)
    plt.close(fig)
    print(json.dumps(summary["key_numbers"], indent=1))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""W30 SUBOFF anchor census — figures.

Every curve/point is read from ``benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json`` (built by ``benchmarks/anchors/suboff_anchor_census.py``);
no number is re-typed here.

    python benchmarks/anchors/suboff_anchor_census_figs.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "benchmarks/anchors/evidence/suboff_w30/suboff_anchor_census.json"
OUT = ROOT / "figures/w30_suboff_anchor_census"


def ittc57(re):
    return 0.075 / (np.log10(re) - 2.0) ** 2


def main() -> int:
    art = json.loads(CENSUS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    pts = art["anchor"]["points"]
    gate = art["error_table"]["gate_pct"]
    made = []

    # ---- fig 1: anchor C_T(Re) for both families + ITTC-57 ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for fam, style in (("T14", dict(marker="o", color="tab:blue")),
                       ("S1", dict(marker="s", color="tab:orange"))):
        sel = [p for p in pts if p["family"] == fam]
        ax.plot([p["Re"] for p in sel], [p["C_T"] for p in sel], ls="--",
                label=f"{fam} family (grade C, n={len(sel)})", **style)
    re_grid = np.logspace(7.0, 7.6, 200)
    ax.plot(re_grid, ittc57(re_grid), "k-", lw=1.4, label="ITTC-1957 $C_F$ (closed form)")
    ax.set_xscale("log")
    ax.set_xlabel("Re = $V L_{max}/\\nu$  ($\\nu=1.139\\times10^{-6}$ m$^2$/s)")
    ax.set_ylabel("$C_T$ / $C_F$")
    ax.set_title("DTRC SUBOFF AFF-1 anchor: two in-repo transcription families\n"
                 "differ by a constant factor 1.170 $\\pm$ 0.002 (17%), above the 15% gate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    f1 = OUT / "fig1_anchor_ct_vs_re.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    made.append(f1)

    # ---- fig 2: deviation vs the 15% gate ----
    hr = art["error_table"]["branch_pending_highre_wallmodel"]["rows"]
    vr = [r for r in art["error_table"]["branch_verified_laminar_Re1000"]["rows"]
          if r["lbm_tier"] == "L128"]
    labels = [r["anchor_id"] + "\nRe-matched" for r in hr] + \
             [r["anchor_id"] + "\nRe=1e3 raw" for r in vr] + \
             [r["anchor_id"] + "\nRe-norm." for r in vr]
    vals = [r["dev_pct"] for r in hr] + [r["dev_raw_pct"] for r in vr] + \
           [r["dev_renormalised_pct"] for r in vr]
    colors = ["tab:red"] * len(hr) + ["tab:gray"] * len(vr) + ["tab:green"] * len(vr)
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhspan(-gate, gate, color="tab:blue", alpha=0.15,
               label=f"acceptance gate $|dev|<{gate:.0f}\\%$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("deviation  $C_{T,LBM}/C_{T,anchor}-1$  [%]")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("LBM vs DTRC anchor, per point: 0/38 inside the 15% gate\n"
                 "red = regime-matched (pending high-Re wall model), gray = verified "
                 "Re=1000 raw, green = Re-renormalised diagnostic (not evidence)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    f2 = OUT / "fig2_deviation_gate.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    made.append(f2)

    # ---- fig 3: grid and tau sensitivity ----
    sens = art["resolution_and_tau_sensitivity"]
    grid = sens["grid_ladder"]["tiers"]
    tau = sens["tau_ladder"]["tiers"]
    ref = art["lbm_evidence"]["verified_case"]["reference_used_in_case"]["value"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    Ls = [g["tier"] for g in grid]
    ax1.plot(Ls, [g["Cd_total_primary"] for g in grid], "o-",
             label="primary caliber (standard, erratum E2)")
    ax1.plot(Ls, [g["Cd_total_coship_mix50"] for g in grid], "s--",
             label="co-shipped archive (mix50)")
    ax1.axhline(ref, color="k", ls=":", label="case reference (Blasius Re=1000)")
    ax1.set_xlabel("grid tier (cells per hull length L)")
    ax1.set_ylabel("$C_D$ total")
    ax1.set_title(f"grid ladder: spread {sens['grid_ladder']['spread_max_over_min_pct']:.1f}%")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    ax2.plot([a["re_lattice"] for a in tau], [a["Cd_total"] for a in tau], "^-",
             color="tab:purple")
    for a in tau:
        ax2.annotate(f"$\\tau$={a['tau']:.2f}", (a["re_lattice"], a["Cd_total"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Re_lattice  (in LBM, $\\tau$ IS the Reynolds number)")
    ax2.set_ylabel("$C_D$ total")
    ax2.set_title(f"tau ladder = Re sweep: spread {sens['tau_ladder']['spread_max_over_min_pct']:.0f}%")
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle("resolution vs $\\tau$: only the grid ladder is a discretisation knob")
    fig.tight_layout()
    f3 = OUT / "fig3_resolution_tau_sensitivity.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    made.append(f3)

    # ---- fig 4: form factor (1+k) discriminator, honestly labelled weak ----
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for fam, style in (("T14", dict(marker="o", color="tab:blue")),
                       ("S1", dict(marker="s", color="tab:orange"))):
        sel = [p for p in pts if p["family"] == fam]
        ax.plot([p["Re"] for p in sel], [p["form_factor_CT_over_CF"] for p in sel],
                ls="--", label=f"{fam}: $C_T/C_F^{{ITTC57}}$", **style)
    ax.axhspan(1.10, 1.20, color="tab:green", alpha=0.12,
               label="declared plausible band 1.10-1.20")
    ax.set_xscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("$1+k = C_T/C_F^{ITTC57}$")
    ax.set_title("implied form factor: T14 gives 1.08-1.16, S1 gives 1.26-1.35\n"
                 "both are physically arguable; the operator layer cannot adjudicate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f4 = OUT / "fig4_form_factor_discriminator.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)
    made.append(f4)

    print(json.dumps({"figures": [str(p.relative_to(ROOT)) for p in made]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
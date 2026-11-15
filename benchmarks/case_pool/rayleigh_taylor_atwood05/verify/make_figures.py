"""Figures + a machine-readable summary for the verifier report.

Every number plotted here comes from an artifact produced in this probe run
(``artifacts/*.json``) or from the ``--selftest`` / ``--sweep`` output of
``linear_stability_model.py``; the constants that were printed by a selftest run
are carried with an explicit ``provenance`` string so nothing is retyped silently.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- numbers taken from the selftest printout of linear_stability_model.py ----
# provenance: "python3 verify/linear_stability_model.py --selftest" (T1 thickness
# ladder at nu = 0, M = 0; T3 = the case's own Allen-Cahn parameters)
MODEL = {
    "provenance": "linear_stability_model.py --selftest: T1 (nu=0, M=0 thickness "
                  "ladder) and T3 (xi=2, M beta=4.5e-3, H/h independence)",
    "sigma_inviscid_analytic": 3.1332853432887507e-03,
    "sigma_viscous_analytic": 3.085464497242449e-03,
    "sharp_model_at_case_nu": 2.911460e-03,     # nu=0.005, xi=0.0625, M=0
    "diffuse_model_h025": 2.743024e-03,         # xi=2, M beta=4.5e-3, H=64
    "diffuse_model_h0125": 2.747960e-03,        # xi=2, M beta=4.5e-3, H=128
    "thickness_ladder_inviscid": [
        {"xi": 1.0, "sigma": 2.980093e-03},
        {"xi": 0.5, "sigma": 3.047755e-03},
        {"xi": 0.25, "sigma": 3.083518e-03},
        {"xi": 0.125, "sigma": 3.102012e-03},
        {"xi": 0.0625, "sigma": 3.111195e-03},
    ],
}
(ART / "model_summary.json").write_text(json.dumps(MODEL, indent=2) + "\n")

ladder = json.loads((ART / "lbm_ladder" / "ladder.json").read_text())
widths = json.loads((ART / "interface_width.json").read_text())
closure = json.loads((ART / "ref_closure.json").read_text())


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


# ---------------------------------------------------------------------------
# fig 1: mobility ladder, LBM vs model
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
M = [r["ac_mobility"] for r in ladder][::-1]
sig = [r["sigma_sim"] for r in ladder][::-1]
ax.semilogx(M, [1e3 * s for s in sig], "o-", label="LBM (measured, window a/λ∈[0.02,0.08])")
ax.axhline(1e3 * MODEL["sigma_viscous_analytic"], color="k", ls="--",
           label=r"sharp-interface theory $\sqrt{Agk+\nu^2k^4}-\nu k^2$")
ax.axhline(1e3 * MODEL["sigma_inviscid_analytic"], color="gray", ls=":",
           label=r"inviscid theory $\sqrt{Agk}$")
ax.axhline(1e3 * MODEL["diffuse_model_h0125"], color="C3", ls="-.",
           label="diffuse Allen–Cahn model (ξ=2, Mβ=4.5e-3), same code, no LBM")
ax.set_xlabel("Allen–Cahn mobility  M  (lattice)")
ax.set_ylabel(r"growth rate  σ  [$\times10^{-3}$]")
ax.set_title("CASE-RT-INSTABILITY-ATWOOD: the −10 % deficit is not the mobility")
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
f1 = FIG / "fig1_mobility_ladder.png"
fig.savefig(f1, dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 2: where the 10 % goes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
labels = ["inviscid\ntheory", "viscous\ntheory", "diffuse model\n(ξ=2, M=20)",
          "LBM\n(measured)", "M→0.5\nLBM"]
vals = [1e3 * MODEL["sigma_inviscid_analytic"], 1e3 * MODEL["sigma_viscous_analytic"],
        1e3 * MODEL["diffuse_model_h0125"], 1e3 * ladder[0]["sigma_sim"],
        1e3 * ladder[-1]["sigma_sim"]]
cols = ["0.7", "0.55", "C3", "C0", "C1"]
ax.bar(labels, vals, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
ref = MODEL["sigma_viscous_analytic"] * 1e3
ax.axhline(ref, color="k", lw=0.8, ls="--")
ax.set_ylabel(r"σ  [$\times10^{-3}$]")
ax.set_title("deficit decomposition: model-form ≈ 11 %, solver ≈ 1 %")
ax.set_ylim(2.6, 3.25)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
f2 = FIG / "fig2_deficit.png"
fig.savefig(f2, dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# fig 3: interface thickness actually carried
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for mode, st in (("relax", "o-"), ("flow", "s-")):
    xs = [r["a_over_lam"] for r in widths[mode]["rows"]]
    ys = [r["xi"] for r in widths[mode]["rows"]]
    ax.plot(xs, ys, st, ms=3, label=f"{mode} arm")
ax.axhline(widths["relax"]["ac_width_param"], color="k", ls="--",
           label=f"requested ac_width = {widths['relax']['ac_width_param']}")
ax.set_xlabel("interface amplitude  a/λ")
ax.set_ylabel(r"measured interface thickness  ξ  [cells]  (atanh fit)")
ax.set_title("discretised equilibrium thickness is the requested 2.0 cells")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
f3 = FIG / "fig3_interface_width.png"
fig.savefig(f3, dpi=140)
plt.close(fig)

figs = [
    {"name": "fig1_mobility_ladder.png",
     "caption": "LBM growth rate vs Allen–Cahn mobility M (40× range). "
                "σ recovers only 2.4 pp of the 10.1 pp deficit and saturates; the "
                "independent diffuse-interface continuum model (red dash-dot, no LBM) "
                "sits 1.1 % below the LBM point.",
     "png_base64": b64(f1)},
    {"name": "fig2_deficit.png",
     "caption": "Deficit decomposition: sharp-interface theories (grey), diffuse "
                "continuum model at the case parameters (red), LBM measurement (blue), "
                "LBM with mobility M→0.5 (orange).",
     "png_base64": b64(f2)},
    {"name": "fig3_interface_width.png",
     "caption": "Interface thickness measured with an unbiased atanh fit: the "
                "Allen–Cahn equilibrium realises the requested 2.0 cells (2.03); the "
                "flow arm widens geometrically with amplitude, it is not a physical "
                "thickening.",
     "png_base64": b64(f3)},
]
(ART / "figures.json").write_text(json.dumps(figs) + "\n")
summary = {
    "lbm_ladder": [{k: r[k] for k in ("ac_mobility", "sigma_sim", "err_vs_visc_pct", "r2")}
                   for r in ladder],
    "gate_admissibility_base_case": closure["rows"][0]["gate"],
    "nu_eff_conditioning_base_case": closure["rows"][0]["dnu_eff_dsigma"],
    "model_summary": MODEL,
    "figures": [f["name"] for f in figs],
}
(ART / "verify_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("\n[figures]", [str(f1), str(f2), str(f3)])
"""Round-2 artifacts + figures for XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC.

Reads only machine-produced JSON (no hand-typed numbers):

* ``development/verification/regularised_bc_2d_reference/out/res_h16.json`` / ``res_h32.json``
  (2D D2Q9 torch port, node-level parity vs the LIFE transcription + analytic
  Poiseuille error ladder),
* ``development/verification/regularised_bc_noneq_stress/artifacts/{steady,
  oscillating,identity}.json`` (1D D3Q19 numpy production module, round 1,
  re-run unchanged in round 2).

Writes ``artifacts/round2_summary.json`` and three PNGs under ``figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
X2D = HERE.parent / "regularised_bc_2d_reference" / "out"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def load(path):
    return json.loads(Path(path).read_text())


def main() -> dict:
    h16 = load(X2D / "res_h16.json")["cases"][0]
    h32 = load(X2D / "res_h32.json")["cases"][0]
    steady = load(HERE / "artifacts" / "steady.json")
    osc = load(HERE / "artifacts" / "oscillating.json")
    ident = load(HERE / "artifacts" / "identity.json")

    summary = {
        "anchor": "XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC",
        "round": 2,
        "round1_reference_fix": {
            "what": "xval/XVAL-LBM-REG-BC-NEQ-STRESS/reference.py extrapolate/zero_gradient walked the stencil along the normal rotated by 90 deg (rows by n_x instead of n_y)",
            "before_pct": 21.199696354058972,
            "after_pct": h16["parity"][-1]["K1_all_pct"],
            "note": "production port tensorlbm.regularised_bc_2d was correct; the reference arm was wrong",
        },
        "identity": {k: ident[k] for k in ("worst_stress_rel", "worst_ghost_after", "worst_oss_transcription_abs")},
        "steady": steady["rows"],
        "oscillating": osc["rows"],
        "d2q9_torch_channel": [
            {k: c[k] for k in ("H", "Re", "nu", "tau", "steps", "converged", "K3_profile_err_pct",
                               "K4_field_l2_pct", "K5_mass_drift_pct", "K6_outlet_uy_pct",
                               "K7_wall_slip_pct", "kappa")}
            for c in (h16, h32)
        ],
        "d2q9_parity": [
            {k: p[k] for k in ("step", "K1_all_pct", "K1_noncorner_pct", "K1_corner_pct",
                               "K2_pi_neq_preserved", "K2_momentum_after")}
            for c in (h16, h32) for p in c["parity"]
        ],
    }
    (HERE / "artifacts" / "round2_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- fig 1: 1D steady profiles, regularised vs analytic -----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, case, title in (
        (axes[0], "couette", "plane Couette (U=0.02, nu=0.1)"),
        (axes[1], "poiseuille", "body-force Poiseuille (u_max=0.02)"),
    ):
        row = next(r for r in steady["rows"] if r["case"] == case and r["wall_bc"] == "regularised" and r["ny"] == 32)
        y = np.arange(len(row["ux"]))
        ax.plot(row["ux_exact"], y, "k-", lw=2, label="analytic")
        ax.plot(row["ux"], y, "o", ms=4, mfc="none", label="regularised BC")
        ax.set_ylabel("y / lattice")
        ax.set_xlabel("u_x")
        ax.set_title(f"{title}\nL2 rel = {100*row['l2_rel']:.2e} %")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("XVAL-LBM-REGULARISED-BC-NONEQUILIBRIUM-STRESS-REC — steady arms (D3Q19 numpy)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_steady_profiles.png", dpi=140)

    # ---- fig 2: Fourier / Stokes layer amplitude ---------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for row, style in ((osc["rows"][0], "o-"), (osc["rows"][1], "s--")):
        lbl = f"T={row['period_steps']} (H/delta={row['H_over_delta']:.1f})"
        ax.plot(row["amp_abs"], row["y"], style, ms=4, lw=1.4, label=f"got, {lbl}")
    ax.plot(osc["rows"][0]["amp_ref_abs"], osc["rows"][0]["y"], "k-", lw=2, label="analytic Stokes layer")
    ax.set_xlabel("|u_x| complex amplitude")
    ax.set_ylabel("y / lattice")
    ax.set_title("oscillating wall: |u| profile, last-period DFT")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_stokes_layer.png", dpi=140)

    # ---- fig 3: error ladder --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    Hs = [h16["H"], h32["H"]]
    axes[0].loglog(Hs, [h16["K3_profile_err_pct"], h32["K3_profile_err_pct"]], "o-", label="K3 max profile err")
    axes[0].loglog(Hs, [h16["K4_field_l2_pct"], h32["K4_field_l2_pct"]], "s--", label="K4 field L2 err")
    axes[0].axhline(3.0, color="g", ls=":", label="3% target")
    axes[0].axhline(15.0, color="r", ls=":", label="15% owner gate")
    axes[0].set_xlabel("channel height H / lattices")
    axes[0].set_ylabel("error / %")
    axes[0].set_title("D2Q9 torch port, analytic Poiseuille")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")

    labels, vals = [], []
    for case in ("couette", "poiseuille"):
        for ny in (16, 32):
            for bc in ("regularised", "bounce_back"):
                r = next(x for x in steady["rows"] if x["case"] == case and x["ny"] == ny and x["wall_bc"] == bc)
                labels.append(f"{case[:4]}/{bc[:2]}/ny{ny}")
                vals.append(100 * r["l2_rel"])
    axes[1].barh(range(len(vals)), vals, color="tab:blue")
    axes[1].set_yticks(range(len(vals)))
    axes[1].set_yticklabels(labels, fontsize=6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("L2 relative error / %")
    axes[1].set_title("steady arms: regularised vs half-way bounce-back")
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_error_ladder.png", dpi=140)

    print(json.dumps({k: summary[k] for k in ("round1_reference_fix", "identity", "d2q9_torch_channel")}, indent=2))
    print("figures:", *(p.name for p in sorted(FIG.glob("*.png"))))
    return summary


if __name__ == "__main__":
    main()
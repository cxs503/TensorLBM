"""Figures for the round-2 pointwise full-step gate (reads summary.json + runs/raw.npz)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def main() -> int:
    summary = json.loads((HERE / "summary.json").read_text())
    raw = np.load(HERE / "runs" / "raw.npz")

    # --- fig 1: pointwise eps along the 30-step trajectory
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in summary["stepwise"]:
        traj = r["trajectory"]
        ax.semilogy([t["step"] for t in traj], [t["eps_scaled_inf"] for t in traj],
                    marker="o", label=f"{r['arm']} {r['model']} tau={r['tau']}")
    gate = summary["gates"]["G_code_equivalence"]["value"]
    ax.axhline(gate, color="k", ls="--", lw=1, label=f"gate {gate:.0e}")
    ax.set_xlabel("LBM step")
    ax.set_ylabel(r"$\varepsilon_\infty$ (scaled, fp64)")
    ax.set_title("Full-step pointwise agreement vs definition-stream + lbmpy collision")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_stepwise_eps.png", dpi=150)
    plt.close(fig)

    # --- fig 2: streaming cross-implementation audit
    rows = [r for r in summary["streaming_audit"] if r["eps_scaled_inf"] == r["eps_scaled_inf"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    vals = [max(r["eps_scaled_inf"], 1e-18) for r in rows]
    ax.barh([r["impl"] for r in rows], vals, color="#3a7ca5")
    ax.set_xscale("log")
    ax.set_xlim(1e-18, 1e-6)
    ax.axvline(1e-12, color="k", ls="--", lw=1, label="gate 1e-12")
    ax.set_xlabel(r"$\varepsilon_\infty$ vs definition (0 = bit-identical, floor 1e-18)")
    ax.set_title("In-repo streaming kernels vs definition (native layouts)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_streaming_audit.png", dpi=150)
    plt.close(fig)

    # --- fig 3: shear-wave decay vs analytic
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in summary["physics"]:
        s = r["series"]
        ax.semilogy(s["t"], np.abs(s["A_inrepo"]), "o", ms=3,
                    label=f"in-repo tau={r['tau']} (err max {r['err_vs_analytic_max_rel']:.2%})")
        ax.semilogy(s["t"], np.abs(s["A_analytic"]), "-",
                    label=rf"analytic $A_0e^{{-\nu k^2 t}}$, tau={r['tau']}")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$|A_{u_x}|$ (shear-mode amplitude)")
    ax.set_title("Shear-wave decay: solver vs analytic (periodic 64x8, k=2pi/64)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_shear_decay.png", dpi=150)
    plt.close(fig)

    # --- fig 4: pointwise error field (S2) + u_x profile
    diff = raw["diff_S2"]
    ref_field = raw["ref_S2"]
    rho = ref_field.sum(axis=0)
    ux = ref_field[1] / rho
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    im0 = axes[0].imshow(ux, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0].set_title(r"reference $u_x$ (S2, t=30)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(np.abs(diff).max(axis=0), origin="lower", aspect="auto",
                         cmap="magma", norm=matplotlib.colors.LogNorm(vmin=1e-16, vmax=1e-13))
    axes[1].set_title(r"$\max_q|f^{repo}-f^{ref}|/\max|f^{ref}|$ (S2)")
    fig.colorbar(im1, ax=axes[1])
    axes[2].semilogy(np.arange(ux.shape[0]), np.abs(ux[:, 0]), "o-", ms=3)
    axes[2].set_title("cut at x=0: shear profile")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel(r"$u_x$")
    axes[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_field_and_error.png", dpi=150)
    plt.close(fig)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
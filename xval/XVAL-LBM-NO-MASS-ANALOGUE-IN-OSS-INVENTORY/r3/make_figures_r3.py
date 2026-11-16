"""Round-3 figures (matplotlib, headless).  Consumes summary_r3.json only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.oscillatory_cylinder_inertia import stokes_cylinder_coefficients

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
S = json.loads((HERE / "summary_r3.json").read_text())


def fig1_reference():
    """Shipped vs corrected unsteady-Stokes reference with the measurements on top."""
    beta = np.logspace(np.log10(10), np.log10(600), 200)
    gs = np.array([1.6414 if False else _shipped(b) for b in beta])
    gc = np.array([stokes_cylinder_coefficients(b)["Gamma_re"] for b in beta])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(beta, gs, "C3--", label="shipped  $k=1/(\\delta/R)$, $\\delta/R=2\\sqrt{2/\\pi\\beta}$")
    ax[0].loglog(beta, gc, "C0-", label="corrected $k=\\sqrt{\\pi\\beta/2}$, $\\delta/R=2/\\sqrt{\\pi\\beta}$")
    for row in S["table"]:
        if row["healthy"]:
            ax[0].plot(row["beta"], row["gamma_re_shipped"], "C3o", ms=5)
            ax[0].plot(row["beta"], row["gamma_re_corrected"], "C0o", ms=5)
    ax[0].set_xlabel(r"$\beta = D^2/(\nu T)$")
    ax[0].set_ylabel(r"$\mathrm{Re}\,\Gamma$ (in-phase reference)")
    ax[0].set_title("exact unsteady-Stokes reference: shipped vs corrected")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    # right: quadrature channel
    qs = np.array([_shipped_q(b) for b in beta])
    qc = np.array([stokes_cylinder_coefficients(b)["C_Q_viscous"] for b in beta])
    ax[1].loglog(beta, qs, "C3--", label="shipped $-\\mathrm{Im}\\Gamma$")
    ax[1].loglog(beta, qc, "C0-", label="corrected $-\\mathrm{Im}\\Gamma$")
    for row in S["table"]:
        if row["healthy"]:
            ax[1].plot(row["beta"], row["cq_ref_shipped"], "C3o", ms=5)
            ax[1].plot(row["beta"], row["cq_ref_corrected"], "C0o", ms=5)
            ax[1].plot(row["beta"], row["CQ_E1"], "k^", ms=5)
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel(r"$-\mathrm{Im}\,\Gamma$ (quadrature reference)")
    ax[1].set_title("quadrature channel: measurements (black) vs both references")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_reference_audit.png", dpi=160)
    plt.close(fig)


def _shipped(beta: float) -> float:
    """Reproduce the shipped (defective) reference straight from the committed raw JSON."""
    for row in S["table"]:
        if abs(row["beta"] - beta) < 1e-9:
            return row["gamma_re_shipped"]
    d = 2.0 * math.sqrt(2.0 / (beta * math.pi))
    k = 1.0 / d
    from scipy.special import kv
    z = complex(1.0, 1.0) * k
    return float((1.0 + 4.0 * kv(1, z) / (z * kv(0, z))).real)


def _shipped_q(beta: float) -> float:
    for row in S["table"]:
        if abs(row["beta"] - beta) < 1e-9:
            return row["cq_ref_shipped"]
    d = 2.0 * math.sqrt(2.0 / (beta * math.pi))
    k = 1.0 / d
    from scipy.special import kv
    z = complex(1.0, 1.0) * k
    return float(-(1.0 + 4.0 * kv(1, z) / (z * kv(0, z))).imag)


def fig2_error():
    """Re-scored in-phase error: shipped vs corrected reference, vs delta and vs D."""
    rows = [r for r in S["table"] if r["healthy"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, col, lab, mk in (("err_shipped_pct", "0.55", "vs shipped reference", "s"),
                              ("err_corrected_pct", "C3", "vs corrected reference", "o")):
        ax[0].semilogx([r["delta_true_cells"] for r in rows], [r[key] for r in rows],
                       mk, color=col, label=lab, ms=7)
    ax[0].axhline(15.0, color="k", ls=":", lw=1, label="owner gate 15 %")
    ax[0].axhline(3.0, color="k", ls="--", lw=1, label="platform caliber 3 %")
    ax[0].set_xlabel(r"true Stokes layer $\delta = D/\sqrt{\pi\beta}$ [cells]")
    ax[0].set_ylabel("in-phase error [%]")
    ax[0].set_title("error re-scored with the corrected reference")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    Ds = sorted({r["D"] for r in rows})
    for D in Ds:
        sub = [r for r in rows if r["D"] == D]
        ax[1].plot([r["beta"] for r in sub], [r["err_corrected_pct"] for r in sub], "o-",
                   label=f"D = {int(D)}")
    ax[1].axhline(15.0, color="k", ls=":", lw=1)
    ax[1].axhline(3.0, color="k", ls="--", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel("in-phase error vs corrected reference [%]")
    ax[1].set_title("flat in $\\beta$ at fixed D: not a Stokes-layer effect")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_error_rescored.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig1_reference()
    fig2_error()
    print("wrote", [p.name for p in sorted(FIG.glob("*.png"))])
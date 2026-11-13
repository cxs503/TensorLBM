"""Figures for XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT round 2.

Reads ``artifacts/xval_pressure_tensor_equilibrium/*.json`` only (every number is
injected programmatically, none hand-typed) and writes

``docs/figures/xval_pt_pressure_profile.png``   measured P_NN / P_TT / phi across the band
``docs/figures/xval_pt_sigma_resolution.png``   sigma error and xi_fit error vs xi/h
``docs/figures/xval_pt_laplace.png``            radial pressure of the R=16 drop + jump vs sigma/R
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
ART = os.path.join(_ROOT, "artifacts", "xval_pressure_tensor_equilibrium")
FIG = os.path.join(_ROOT, "docs", "figures")


def load(arm: str) -> dict:
    return json.load(open(os.path.join(ART, f"{arm}.json")))


def fig_profiles() -> str:
    d = load("flat_xi3")
    pr = d["profiles"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax[0].plot(pr["x"], pr["Pxx_measured"], "o-", ms=2.5, label=r"$P_{NN}$ measured")
    ax[0].plot(pr["x"], pr["Pyy_measured"], "s-", ms=2.5, label=r"$P_{TT}$ measured")
    ax[0].plot(pr["x"], pr["Pxx_analytic"], "--", lw=1.0, label=r"$P_{NN}$ analytic")
    ax[0].set_xlabel("x [lattice]")
    ax[0].set_ylabel("pressure [lattice]")
    ax[0].set_title(r"flat band, $\xi/h=3$: $P_{NN}$=const (mech. equilibrium)")
    ax[0].legend(fontsize=7)
    ax2 = ax[1]
    ax2.plot(pr["x"], pr["stress_measured"], "o-", ms=2.5, label=r"$P_{NN}-P_{TT}$ measured")
    ax2.set_xlabel("x [lattice]")
    ax2.set_ylabel(r"$P_{NN}-P_{TT}$")
    ax2.set_title(
        rf"$\sigma$={d['sigma_measured']:.4f} vs analytic {d['sigma_analytic']:.4f} "
        rf"({100 * d['sigma_rel_error']:.2f} %)"
    )
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "xval_pt_pressure_profile.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_resolution() -> str:
    res = json.load(open(os.path.join(ART, "results.json")))
    pts = res["resolution_series"]
    x = [p["abscissa"] for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax[0].plot(x, [100 * p["sigma_tension"] for p in pts], "o-", label=r"$\sigma$ error")
    ax[0].plot(x, [100 * p["xi_fit"] for p in pts], "s-", label=r"$\xi$ fit error")
    ax[0].axhline(15.0, color="r", ls="--", lw=1.0, label="owner band 15 %")
    ax[0].set_xlabel(r"interface resolution $\xi/h$")
    ax[0].set_ylabel("relative error [%]")
    ax[0].set_title("resolution series (flat band)")
    ax[0].legend(fontsize=7)
    ax[1].plot(x, [p["Pnn_uniformity"] for p in pts], "d-")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\xi/h$")
    ax[1].set_ylabel(r"max$|P_{NN}-\langle P_{NN}\rangle|/\langle P_{NN}\rangle$")
    ax[1].set_title(r"mechanical equilibrium residual $dP_{NN}/dx$")
    fig.tight_layout()
    out = os.path.join(FIG, "xval_pt_sigma_resolution.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_laplace() -> str:
    d = load("dropR16")
    rad = d["radial"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax[0].plot(rad["r"], rad["p"], "o-", ms=2.5, label="p = (Pxx+Pyy)/2")
    ax[0].plot(rad["r"], rad["P_rr"], "-", lw=1.0, label=r"$P_{rr}$")
    ax[0].plot(rad["r"], rad["P_tt"], "-", lw=1.0, label=r"$P_{tt}$")
    ax[0].set_xlabel("r [lattice]")
    ax[0].set_ylabel("pressure")
    ax[0].set_title(rf"drop R={d['R']}, $\xi=2$: $p_{{in}}-p_{{out}}$={d['dp']:.4e}")
    ax[0].legend(fontsize=7)
    ax[1].bar([0], [100 * d["laplace_rel_error_analytic"]], color="tab:green")
    ax[1].axhline(15.0, color="r", ls="--", lw=1.0, label="owner band 15 %")
    ax[1].set_xticks([0])
    ax[1].set_xticklabels([rf"$R/\xi$=8"])
    ax[1].set_ylabel(r"error of $\Delta p$ vs $\sigma/R$ [%]")
    ax[1].set_title(rf"Laplace: got {d['dp']:.4e} vs $\sigma/R$={d['dp_analytic']:.4e}")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "xval_pt_laplace.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_profiles, fig_resolution, fig_laplace):
        print("wrote", f())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
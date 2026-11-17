#!/usr/bin/env python3
"""Figures for the full-scale (N=128, T=10^4) OOTC acceptance evidence.

Reads (produced on the compute node, fetched inside the repo):
  artifacts/adjoint_ootc_acceptance/results.json             (verify driver)
  artifacts/adjoint_ootc_acceptance/inversion_fullscale.json (this mission)
  artifacts/adjoint_ootc_acceptance/field_snapshot_fullscale.npz

Writes PNGs to artifacts/adjoint_ootc_acceptance/figures/.  Pure plotting:
runs anywhere in <10 s, no solver calls.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts", "adjoint_ootc_acceptance")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)


def fig_gates(res: dict) -> None:
    v = res["variants"]
    names = ["bptt", "ootc_k100", "ootc_k1000", "fd"]
    labels = ["BPTT\n(full storage)", "OOTC k=100", "OOTC k=1000",
              "central diff\n(independent)"]
    dJ = [v[n]["dJ"] for n in names]
    mem = [v[n]["hwm_gib"] for n in names]
    g = res["gates"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(labels, dJ,
              color=["#c44", "#36c", "#59c", "#888"])
    ax[0].set_ylabel(r"$\mathrm{d}J/\mathrm{d}\tau$  (T=10^4, 128^2)")
    ax[0].set_title(f"gradient gates: OOTC vs BPTT rel="
                    f"{g['ootc_vs_bptt_rel']:.1e} (<1e-5:"
                    f" {g['ootc_vs_bptt_pass']}), BPTT vs FD rel="
                    f"{g['bptt_vs_fd_rel']:.1e}")
    ax[0].tick_params(axis="x", labelsize=8)

    bars = ax[1].bar(labels, mem, color=["#c44", "#36c", "#59c", "#888"])
    ax[1].axhline(8.0, color="r", ls="--",
                  label="acceptance gate 8 GiB")
    ax[1].set_ylabel("peak RSS VmHWM [GiB]")
    ax[1].set_title(f"memory: OOTC {g['ootc_hwm_gib']:.2f} GiB vs BPTT "
                    f"{g['bptt_hwm_gib']:.2f} GiB "
                    f"({g['mem_ratio_bptt_over_ootc']:.1f}x)")
    ax[1].tick_params(axis="x", labelsize=8)
    ax[1].legend(fontsize=8)
    for b, m in zip(bars, mem):
        ax[1].text(b.get_x() + b.get_width() / 2, m + 0.15, f"{m:.2f}",
                   ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_fullscale_gates.png"), dpi=150)
    plt.close(fig)


def fig_inversion(inv: dict, tau_true: float) -> None:
    hist = inv["history"]
    n = [h["iter"] for h in hist]
    tau = [h["tau"] for h in hist]
    J = [abs(h["J"]) for h in hist]
    g = [abs(h["dJ_dtau"]) for h in hist]
    methods = [h["method"] for h in hist]

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].semilogy(n, J, "o-", ms=4)
    ax[0].set_xlabel("gradient evaluation #")
    ax[0].set_ylabel(r"$|J(\tau_n)|$")
    ax[0].set_title(f"objective: {inv['nfev']} evals, "
                    f"converged={inv['converged']} ({inv['reason']})")

    ax[1].semilogy(n, g, "s-", ms=4, label=r"$|\mathrm{d}J/\mathrm{d}\tau|$")
    ax[1].set_xlabel("gradient evaluation #")
    ax[1].set_ylabel(r"$|\mathrm{d}J/\mathrm{d}\tau|$")
    ax[1].set_title("adjoint gradient magnitude per iterate")

    colors = {"initial": "k", "bound-lo": "gray", "bound-hi": "gray",
              "newton": "#36c", "grad-step": "#c83", "bisect": "#838"}
    for m in set(methods):
        xs = [x for x, mm in zip(n, methods) if mm == m]
        ys = [y for y, mm in zip(tau, methods) if mm == m]
        ax[2].plot(xs, ys, "o", ms=5, label=m, color=colors.get(m, "b"))
    ax[2].axhline(tau_true, color="r", ls="--",
                  label=r"$\tau_{true}$" + f"={tau_true}")
    ax[2].set_xlabel("gradient evaluation #")
    ax[2].set_ylabel(r"$\tau_n$")
    ax[2].set_title(f"recovered $\\tau^*={inv['tau']:.10f}$, "
                    f"rel err {inv['rel_err_tau']:.2e}")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_inversion_fullscale.png"), dpi=150)
    plt.close(fig)


def fig_landscape(lan: dict, inv: dict, tau_true: float) -> None:
    taus = np.array(lan["tau"])
    Js = np.array(lan["J"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(taus, Js, "o-", label="forward-only $J(\\tau)$ (5 pts)")
    hist = inv["history"]
    ax.semilogy([h["tau"] for h in hist],
                [max(abs(h["J"]), 1e-300) for h in hist],
                "x", ms=6, color="#c44",
                label="inversion iterates (OOTC gradient)")
    ax.axvline(tau_true, color="r", ls="--", alpha=0.6,
               label=r"$\tau_{true}$")
    ax.axvline(inv["tau"], color="#36c", ls=":", alpha=0.8,
               label=r"$\tau^*$ recovered")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$J(\tau)=\frac{1}{2}\|u_T(\tau)-u_{ref}\|^2$")
    ax.set_title(f"N=128, T=10^4 unbiased objective "
                 f"(landscape {lan['wall_s']:.0f}s forward-only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_landscape_fullscale.png"), dpi=150)
    plt.close(fig)


def fig_fields(npz_path: str) -> None:
    d = np.load(npz_path)
    ux, uy, costate = d["ux_terminal"], d["uy_terminal"], d["costate"]
    speed = np.hypot(ux, uy)
    costate_sum = np.sqrt((costate ** 2).sum(axis=0))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    im0 = ax[0].imshow(speed, origin="lower", cmap="viridis")
    ax[0].set_title(f"terminal speed |u_T|  (tau*={float(d['tau_recovered']):.6f},"
                    f" T={int(d['T'])}, {int(d['N'])}^2)")
    fig.colorbar(im0, ax=ax[0], shrink=0.85)
    im1 = ax[1].imshow(costate_sum, origin="lower", cmap="magma")
    ax[1].set_title("costate magnitude $|\\mathrm{d}J/\\mathrm{d}f^0|$"
                    " (sum over 9 directions, same backward sweep)")
    fig.colorbar(im1, ax=ax[1], shrink=0.85)
    for a in ax:
        a.set_xlabel("x"); a.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_fields_fullscale.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    with open(os.path.join(ART, "results.json")) as fh:
        res = json.load(fh)
    with open(os.path.join(ART, "inversion_fullscale.json")) as fh:
        inv_res = json.load(fh)
    inv = inv_res["inversion"]
    tau_true = inv_res["tau_true"]

    fig_gates(res)
    fig_inversion(inv, tau_true)
    fig_landscape(inv_res["landscape"], inv, tau_true)
    npz = os.path.join(ART, "field_snapshot_fullscale.npz")
    if os.path.exists(npz):
        fig_fields(npz)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()

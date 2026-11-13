"""Figures for XVAL-LBM-PLUGGABLE-PSEUDOPOTENTIAL-GUO-FORCING (from results.json only).

Three panels, every number read from ``results.json`` (zero hand-copy):
(a) Young-Laplace: ``dp`` vs ``1/R_eq`` for each arm/scheme with the fitted line.
(b) Coexistence densities of the simulated bulk vs the Maxwell-construction
    anchor (dashed lines).
(c) Spurious-current level ``max|u|`` vs step for the three forcing schemes
    (tau = 1) -- the ``sigma = 0`` null test of a two-phase model.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    res = json.load(open(os.path.join(HERE, "results.json")))
    recs = {}
    for fn in sorted(os.listdir(os.path.join(HERE, "runs"))):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(HERE, "runs", fn)))
            recs[r["case"]["name"]] = r

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) Laplace law
    colors = {"guo": "C0", "edm": "C2", "velocity_shift": "C3"}
    for key, row in sorted(res["laplace"].items()):
        scheme = key.split(":")[1]
        arm = key.split(":")[0]
        x = np.array([1.0 / p["R_eq"] for p in row["points"]])
        y = np.array([p["dp"] for p in row["points"]])
        style = "-o" if arm == "ref" else "--s"
        ax[0].plot(x, y, style, color=colors.get(scheme, "C7"), ms=4,
                   label=f"{key}  sigma={row['sigma_from_slope']:.4f} R2={row['r2']:.4f}")
    ax[0].set_xlabel("1 / R_eq")
    ax[0].set_ylabel(r"$\Delta p$ (lattice units)")
    ax[0].set_title("(a) Young-Laplace: $\\Delta p = \\sigma/R$")
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)

    # (b) coexistence vs Maxwell
    mw = res["anchor"]["maxwell"]
    names = sorted(res["maxwell_check"])
    ii = np.arange(len(names))
    ax[1].bar(ii - 0.2, [res["maxwell_check"][n]["rho_in"] for n in names], 0.4,
              label="simulated liquid")
    ax[1].bar(ii + 0.2, [res["maxwell_check"][n]["rho_out"] for n in names], 0.4,
              label="simulated vapour")
    ax[1].axhline(mw["rho_l"], ls="--", c="k", lw=1, label="Maxwell $\\rho_l$")
    ax[1].axhline(mw["rho_v"], ls=":", c="k", lw=1, label="Maxwell $\\rho_v$")
    ax[1].set_xticks(ii)
    ax[1].set_xticklabels([n.replace("_tau1.0", "") for n in names], rotation=90, fontsize=5)
    ax[1].set_ylabel("bulk density")
    ax[1].set_title("(b) coexistence vs Maxwell construction")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3)

    # (c) spurious currents
    for scheme in ("guo", "edm", "velocity_shift"):
        r = recs.get(f"ref_{scheme}_R14_tau1.0")
        if r is None:
            continue
        steps = [s["step"] for s in r["result"]["series"]]
        umax = [s["u_max"] for s in r["result"]["series"]]
        ax[2].semilogy(steps, umax, "-o", ms=3, color=colors[scheme], label=scheme)
    ax[2].set_xlabel("step")
    ax[2].set_ylabel(r"$\max|u|$ (lattice units)")
    ax[2].set_title("(c) spurious currents, static droplet (tau=1)")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "xval_pp_guo_forcing.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
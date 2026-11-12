#!/usr/bin/env python3
"""Figures for the m15917 capillary-rise report (all panels read from artifacts).

    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/make_figures.py

Reads ``artifacts/m15917/*.json`` and writes ``docs/reports/m15917/figs/*.png``.
No number is typed into the plots: every curve comes from an artifact trace.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "artifacts/m15917"
FIG = REPO / "docs/reports/m15917/figs"


def load(name):
    return json.loads((ART / name).read_text())


def fig_ab(ab: dict) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = [v["variant"] for v in ab["variants"]]
    mass = [v["mass_drift_pct"] for v in ab["variants"]]
    h = [v["h_equilibrium"] for v in ab["variants"]]
    colors = ["C3", "C0", "C1", "C2"]
    ax[0].bar(labels, mass, color=colors)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("liquid mass drift [%]")
    ax[0].set_title(f"theta={ab['theta_deg']:g} deg, {ab['variants'][0]['steps']} steps")
    ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(labels, h, color=colors)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("meniscus rise h [lattice]")
    ax[1].set_title("neutral-wall artefact (Jurin ref = 0)")
    ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    out = FIG / "fig1_ab_wall_fix.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_neutral(res: dict) -> Path:
    tr = res["traj_decimated"]
    st = [t["step"] for t in tr]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(st, [t["h"] for t in tr], "C2", label="h(t) = meniscus - pool")
    ax[0].axhline(res["ref"], color="k", ls=":", label="Jurin ref (theta=90) = 0")
    ax[0].axhline(res["got"], color="C3", ls="--", label=f"measured mean = {res['got']:.3f}")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("h [lattice]")
    ax[0].set_title(f"neutral-wall control, sigma_meas={res['measured']['sigma_meas']:.5f}")
    ax[0].legend(fontsize=7)
    ax[1].plot(st, [t["u_max"] for t in tr], "C4")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("max |u| (all nodes)")
    ax[1].set_title("velocity floor of the fixed solver")
    fig.tight_layout()
    out = FIG / "fig2_neutral_control.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_theta60(res: dict) -> Path:
    v = res["variant"]
    tr = [t for t in v["traj"] if math.isfinite(t["h"])]
    st = [t["step"] for t in tr]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(st, [t["h"] for t in tr], "o-", color="C3", label="h(t) (finite samples)")
    ax[0].axhline(res["ref_jurin"], color="k", ls=":", label=f"Jurin ref = {res['ref_jurin']:.2f}")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("h [lattice]")
    ax[0].set_title(f"theta={res['theta_deg']:g} deg: rise then blow-up")
    ax[0].legend(fontsize=7)
    ax[1].plot(st, [t["phi_integral_fluid"] for t in tr], "o-", color="C1")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("liquid phase integral $\\int\\phi$")
    ax[1].set_title(f"wetting-wall mass sink ({v['mass_drift_pct']:+.1f} %)")
    fig.tight_layout()
    out = FIG / "fig3_theta60_fail.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    outs = [
        fig_ab(load("ab_wall_fix_theta90.json")),
        fig_neutral(load("jurin_neutral_theta90.json")),
        fig_theta60(load("jurin_theta60_wetting.json")),
    ]
    for path in outs:
        print(path.relative_to(REPO), path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
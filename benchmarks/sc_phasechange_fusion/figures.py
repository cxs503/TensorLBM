#!/usr/bin/env python
"""Report figures for the fused phase-change verification surface (node p348w3-n04).

Reads the digitised outputs written by ``run.py`` (nothing is re-typed) and
emits four PNGs into the same directory:

``fig1_neumann.png``        melting front vs the Neumann similarity solution,
                            with the per-point relative error panel.
``fig2_stefan_energy.png``  Stefan-condition residual per record window and the
                            windowed energy budget.
``fig3_laplace.png``        Young-Laplace ladder at 10:1 / 50:1 / 100:1, radial
                            density profiles and a density cloud of the drop.
``fig4_rt.png``             Rayleigh-Taylor growth: ln a(t) fits and the
                            measured/theory growth-rate ratio.

Usage::

    PYTHONPATH=src python benchmarks/sc_phasechange_fusion/figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "sc_phasechange_fusion"


def _load(name: str):
    return json.loads((OUT / name).read_text())


def fig_neumann() -> Path:
    enth = _load("enthalpy.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {0.1: "#1f77b4", 0.5: "#ff7f0e", 1.0: "#2ca02c"}
    by_key = {(c["ste"], c["dTsub"]): c for c in enth["cases"]}
    for ste in (0.1, 0.5, 1.0):
        case = by_key[(ste, 0.0)]
        t = np.array(case["times"])
        s = np.array(case["fronts"])
        sref = np.array(case["fronts_ref"])
        axes[0].plot(t, sref, "--", color=colors[ste], lw=1.4, label=f"Neumann Ste={ste:g}")
        axes[0].plot(t, s, "o", ms=3.2, color=colors[ste], label=f"enthalpy LBM-ish FV Ste={ste:g}")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("melt front s(t) [m]")
    axes[0].set_title("(a) melting front vs Neumann similarity solution")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for case in enth["cases"]:
        if abs(case["dTsub"]) > 1e-12:
            continue
        axes[1].semilogx(
            case["times"],
            case["errs_pct"],
            "o-",
            ms=3,
            color=colors[case["ste"]],
            label=f"Ste={case['ste']:g}, dT_sub=0",
        )
    for case in enth["cases"]:
        if abs(case["dTsub"]) < 1e-12:
            continue
        axes[1].semilogx(
            case["times"],
            case["errs_pct"],
            ":",
            lw=1.0,
            color=colors[case["ste"]],
            alpha=0.7,
            label=f"Ste={case['ste']:g}, dT_sub={case['dTsub']:g} K",
        )
    axes[1].axhline(3.0, color="k", ls="--", lw=1.0, label="C1 gate 3%")
    axes[1].axhline(-3.0, color="k", ls="--", lw=1.0)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("(s_sim - s_Neumann)/s_Neumann [%]")
    axes[1].set_title("(b) front error, all 9 cases")
    axes[1].legend(fontsize=6, ncol=2)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = OUT / "fig1_neumann.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def fig_stefan_energy() -> Path:
    enth = _load("enthalpy.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("viridis")
    cases = enth["cases"]
    for i, case in enumerate(cases):
        wins = case["stefan_windows"]
        x = [w["front_m"] * 1e3 for w in wins]
        y = [w["residual_pct"] for w in wins]
        axes[0].plot(
            x,
            y,
            "o-",
            ms=3,
            lw=1.0,
            color=cmap(i / max(len(cases) - 1, 1)),
            label=f"Ste={case['ste']:g}, dT_sub={case['dTsub']:g}",
        )
    axes[0].axhline(2.0, color="k", ls="--", lw=1.0, label="C2 gate 2%")
    axes[0].axhline(-2.0, color="k", ls="--", lw=1.0)
    axes[0].set_xlabel("front position s [mm]")
    axes[0].set_ylabel("Stefan residual [%]")
    axes[0].set_title("(a) latent heat vs heat-flux jump")
    axes[0].legend(fontsize=6.5)
    axes[0].grid(alpha=0.3)

    for i, case in enumerate(cases):
        axes[1].plot(
            range(1, len(case["energy_windows"]) + 1),
            case["energy_windows"],
            "s-",
            ms=3,
            lw=1.0,
            color=cmap(i / max(len(cases) - 1, 1)),
            label=f"Ste={case['ste']:g}, dT_sub={case['dTsub']:g}",
        )
    axes[1].axhline(2.0, color="k", ls="--", lw=1.0, label="C6 gate 2%")
    axes[1].axhline(-2.0, color="k", ls="--", lw=1.0)
    axes[1].set_xlabel("record window index")
    axes[1].set_ylabel("energy budget imbalance [%]")
    axes[1].set_title("(b) windowed sensible+latent energy budget")
    axes[1].legend(fontsize=6.5)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "fig2_stefan_energy.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def fig_laplace() -> Path:
    lap = _load("laplace.json")
    fields = {f["target_ratio"]: f for f in _load("laplace_fields.json")}
    tiers = sorted(lap["tiers"], key=lambda t: t["target_ratio"])
    fig = plt.figure(figsize=(12, 4.2))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    for i, tier in enumerate(tiers):
        inv_r = np.array(tier["fit"]["inv_r"])
        dp = np.array(tier["fit"]["dp"])
        order = np.argsort(inv_r)
        sigma = tier["fit"]["sigma"]
        ax0.plot(
            inv_r[order],
            (sigma * inv_r)[order],
            "-",
            color=colors[i],
            lw=1.2,
            label=f"{tier['target_ratio']:g}:1 fit sigma={sigma:.4f}",
        )
        ax0.plot(inv_r, dp, "o", ms=4.5, color=colors[i])
    ax0.set_xlabel("1/R_eff")
    ax0.set_ylabel(r"$\Delta p$ (lattice units)")
    ax0.set_title("(a) Young-Laplace $\\Delta p=\\sigma/R$")
    ax0.legend(fontsize=7)
    ax0.grid(alpha=0.3)

    for i, tier in enumerate(tiers):
        rep = fields[tier["target_ratio"]]
        r = np.array(rep["r_bins"])
        rho = np.array(rep["rho_radial"])
        ax1.semilogy(
            r / rep["radius_eff"],
            rho,
            "-",
            lw=1.3,
            color=colors[i],
            label=f"{tier['target_ratio']:g}:1 (Tr={tier['t_reduced']:.3f})",
        )
    ax1.set_xlabel("r / R_eff")
    ax1.set_ylabel(r"$\rho$")
    ax1.set_title("(b) radial density profile")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")

    rep = fields[max(t["target_ratio"] for t in tiers)]
    snap = np.array(rep["snapshot"])
    im = ax2.imshow(snap, origin="lower", cmap="viridis")
    ax2.set_title(f"(c) density field, {max(t['target_ratio'] for t in tiers):g}:1 droplet")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, fraction=0.046, label=r"$\rho$")
    fig.tight_layout()
    path = OUT / "fig3_laplace.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def fig_rt() -> Path:
    rt = _load("rt.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    atwoods, ratios = [], []
    for i, case in enumerate(rt["cases"]):
        samples = np.array(case["samples"])
        if samples.size == 0:
            continue
        t, a = samples[:, 0], samples[:, 1]
        mask = a > 0
        axes[0].semilogy(
            t[mask], a[mask], "-", lw=1.0, color=colors[i], label=f"A={case['atwood']:.3f} sim"
        )
        t0, t1 = case["fit_window"]
        tt = np.linspace(t0, t1, 10)
        a0 = math.exp(math.log(max(a[mask][0], 1e-30)))
        axes[0].semilogy(
            tt,
            a0 * np.exp(case["sigma_sim"] * (tt - t[mask][0])),
            "--",
            lw=1.0,
            color=colors[i],
            label=f"A={case['atwood']:.3f} fit sigma={case['sigma_sim']:.4f}",
        )
        atwoods.append(case["atwood"])
        ratios.append(case["sigma_sim"] / case["sigma_analytic"])
    axes[0].set_xlabel("t (lattice steps)")
    axes[0].set_ylabel("interface mode amplitude a(t)")
    axes[0].set_title("(a) exponential growth of the first mode")
    axes[0].legend(fontsize=6.5)
    axes[0].grid(alpha=0.3, which="both")

    x = np.arange(len(atwoods))
    axes[1].bar(x - 0.18, ratios, width=0.36, color="#1f77b4", label=r"$\sigma_{sim}/\sqrt{Agk}$")
    visc = [c["sigma_sim"] / c["sigma_viscous"] for c in rt["cases"]]
    axes[1].bar(
        x + 0.18, visc, width=0.36, color="#ff7f0e", label=r"$\sigma_{sim}/\sigma_{viscous}$"
    )
    axes[1].axhline(1.0, color="k", lw=1.0)
    for off in (0.9, 1.1):
        axes[1].axhline(off, color="k", ls="--", lw=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"A={a:.3f}" for a in atwoods])
    axes[1].set_ylim(0.6, 1.4)
    axes[1].set_title("(b) measured / theory (C4 gate: within 10%)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUT / "fig4_rt.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_neumann, fig_stefan_energy, fig_laplace, fig_rt):
        try:
            made.append(fn())
            print(f"[fig] {made[-1]}", flush=True)
        except FileNotFoundError as exc:  # pragma: no cover - depends on run order
            print(f"[fig] skipped {fn.__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

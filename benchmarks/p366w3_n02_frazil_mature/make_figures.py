"""Figures for the mature frazil power-law rheology campaign.

Reads only ``artifacts/frazil_powerlaw_mature/campaign.json`` and writes PNGs to
``artifacts/frazil_powerlaw_mature/``.  Every number on every axis comes from the
artifact (no hand-typed data), so a figure cannot drift away from the measured
record.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ART = REPO / "artifacts" / "frazil_powerlaw_mature"
CAMPAIGN = ART / "campaign.json"
RUNS = HERE / "case_records"

C_DISPERSE = "#1f4e79"
C_FLOC = "#c0504d"
C_REF = "#7f7f7f"


def load():
    return json.loads(CAMPAIGN.read_text())


def _fits(art, family):
    out = {}
    for key, f in art["fits"].items():
        if f["family"] != family:
            continue
        out[(f["phi"], f["box"])] = f
    return out


def fig_flow_curves(art):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, family, colour, title in (
        (axes[0], "dispersed", C_DISPERSE, "dispersed (hard crystals)"),
        (axes[1], "flocculated", C_FLOC, "flocculated (shear-limited aggregates)"),
    ):
        data = _fits(art, family)
        for (phi, box), f in sorted(data.items()):
            if box not in ("coarse", "fine"):
                continue
            rates = list(f["eta_by_rate"].values())
            g = np.array([r["gamma_nominal"] for r in rates])
            gb = np.array([r["gamma_bulk"] for r in rates])
            tau = np.array([r["eta_r_mean"] * r["gamma_bulk"] * art["protocol"]["nu_lattice"]
                            for r in rates])
            style = "-o" if box == "coarse" else "--s"
            ax.loglog(gb, tau, style, color=colour, alpha=0.85,
                      label=f"c={phi:g} ({box})")
            if "error" not in f["power_law"]:
                K, n = f["power_law"]["K"], f["power_law"]["n"]
                gg = np.geomspace(gb.min() * 0.7, gb.max() * 1.4, 50)
                ax.loglog(gg, K * gg ** n, "-", color=colour, lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"measured bulk shear rate $\dot\gamma$ (lattice)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$\tau$ (lattice units)")
    fig.suptitle(r"Flow curves $\tau=K\dot\gamma^{\,n}$ from the resolved shear cell", fontsize=11)
    fig.tight_layout()
    out = ART / "fig1_flow_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_relative_viscosity(art):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    phi = np.linspace(0, 0.34, 200)
    ax.plot(phi, 1 + 2.5 * phi, ":", color=C_REF, label=r"Einstein $1+2.5c$")
    ax.plot(phi, 1 + 2.5 * phi + 6.2 * phi ** 2, "--", color=C_REF,
            label=r"Einstein–Batchelor–Green $1+2.5c+6.2c^2$")
    ax.plot(phi, (1 - phi / 0.6) ** -1.5, "-", color="k", lw=1.0,
            label=r"Krieger–Dougherty ($\phi_m=0.6$)")
    for family, colour, marker in (("dispersed", C_DISPERSE, "o"),
                                   ("flocculated", C_FLOC, "^")):
        for (p, box), f in sorted(_fits(art, family).items()):
            if box not in ("coarse", "fine"):
                continue
            filled = box == "coarse"
            ax.errorbar(f["phi_lattice_mean"], f["eta_r"], fmt=marker, color=colour,
                        ms=6 if filled else 5, mfc=colour if filled else "white",
                        mew=1.0, capsize=0,
                        label=f"{family} {box}")
    if "c0.2" in art.get("finite_size", {}):
        fs = art["finite_size"]["c0.2"]
        ax.axhline(fs["intercept_bulk"], color=C_DISPERSE, lw=0.8, ls="-.",
                   label=r"dispersed $H/D\to\infty$ extrapolation")
        for p in fs["points"]:
            ax.plot([p["eta_r"]], [0], alpha=0.0)
    ax.set_xlabel(r"ice volume fraction $c$")
    ax.set_ylabel(r"relative viscosity $\mu_{eff}/\mu_0$")
    ax.set_xlim(0, 0.34)
    ax.set_ylim(0.95, 3.2)
    ax.legend(fontsize=7, ncol=1)
    ax.grid(alpha=0.25)
    ax.set_title("Concentration–viscosity curve vs the three closures", fontsize=10)
    fig.tight_layout()
    out = ART / "fig2_relative_viscosity.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_shear_thinning(art):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for family, colour, marker in (("dispersed", C_DISPERSE, "o"),
                                   ("flocculated", C_FLOC, "^")):
        xs, ys, lo, hi = [], [], [], []
        for (p, box), f in sorted(_fits(art, family).items()):
            if box != "coarse" or "error" in f["power_law"]:
                continue
            xs.append(f["phi_lattice_mean"])
            ys.append(f["power_law"]["n"])
            ci = f["power_law"].get("n_ci")
            if ci:
                lo.append(f["power_law"]["n"] - ci[0])
                hi.append(ci[1] - f["power_law"]["n"])
            else:
                lo.append(0.0)
                hi.append(0.0)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker + "-", color=colour,
                    capsize=3, label=family)
    ax.axhline(1.0, color=C_REF, ls=":", label="Newtonian $n=1$")
    ax.axhline(0.95, color="k", ls="--", lw=0.8, label="shear-thinning criterion $n=0.95$")
    ax.set_xlabel(r"ice volume fraction $c$")
    ax.set_ylabel(r"power-law index $n$ (95 % bootstrap interval)")
    ax.set_title("Shear thinning: only the structure-following family thins", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ART / "fig3_shear_thinning.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_finite_size(art):
    fs = art.get("finite_size", {}).get("c0.2")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if not fs:
        ax.text(0.5, 0.5, "finite-size scan not run", ha="center")
    else:
        pts = fs["points"]
        x = np.array([1.0 / p["h_over_d"] for p in pts])
        y = np.array([p["eta_r"] for p in pts])
        ax.plot(x, y, "o", color=C_DISPERSE, ms=7, label="measured (confined cell)")
        xx = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(xx, fs["intercept_bulk"] + fs["slope_per_HD"] * xx, "-",
                color=C_DISPERSE, lw=0.9, label="linear fit in $1/(H/D)$")
        ax.plot([0], [fs["intercept_bulk"]], "*", color="k", ms=12,
                label=fr"bulk extrapolation $\eta_r={fs['intercept_bulk']:.3f}$")
        for p in pts:
            ax.annotate(f"box {p['n_xy']}", (1.0 / p["h_over_d"], p["eta_r"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=7)
        ax.axhline(float((1 - 0.2 / 0.6) ** -1.5), color=C_REF, ls="--",
                   label="Krieger–Dougherty at $c=0.2$")
        ax.set_xlabel(r"$1/(H/D)$")
        ax.set_ylabel(r"$\eta_r$ at $c=0.20$")
        ax.set_title("Wall-depletion (finite-size) scan at $c=0.20$", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ART / "fig4_finite_size.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_grid_convergence(art):
    conv = art.get("grid_convergence", {})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    names = list(conv)
    for ax, field, label in ((axes[0], "n", r"power-law index $n$"),
                             (axes[1], "K", r"consistency $K$")):
        if names:
            vals_c = [conv[k][f"{field}_coarse"] for k in names]
            vals_f = [conv[k][f"{field}_fine"] for k in names]
            xx = np.arange(len(names))
            ax.bar(xx - 0.18, vals_c, width=0.36, color=C_DISPERSE, label="coarse (32)")
            ax.bar(xx + 0.18, vals_f, width=0.36, color="#8fb4d8", label="fine (48)")
            for i, k in enumerate(names):
                ax.text(i, max(vals_c[i], vals_f[i]) * 1.02,
                        f"{100 * conv[k][f'{field}_rel_diff']:.1f} %", ha="center", fontsize=8)
            ax.set_xticks(xx)
            ax.set_xticklabels([k.replace("_", " ") for k in names], fontsize=8)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "not run", ha="center")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Grid convergence of the power-law fit (gate: 5 %)", fontsize=11)
    fig.tight_layout()
    out = ART / "fig5_grid_convergence.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_stress_control(art):
    pure = art.get("pure", [])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if pure:
        flux = [abs(p["flux_rel_err"]) * 100 for p in pure]
        wall = [abs(p["wall_rel_err"]) * 100 for p in pure]
        xx = np.arange(len(pure))
        ax.semilogy(xx, flux, "o", color=C_DISPERSE, label="plane momentum flux")
        ax.semilogy(xx, wall, "s", color=C_FLOC, mfc="white", label="per-wall momentum exchange")
        ax.axhline(2.0, color="k", ls="--", lw=0.8, label="2 % gate")
        ax.set_xticks(xx)
        ax.set_xticklabels([f"{p['box']}\nr{p['rate_index']}" for p in pure], fontsize=7)
        ax.set_ylabel(r"$|\tau/(\rho\nu\dot\gamma)-1|$  [%]")
        ax.set_title(r"Stress estimators against the exact Couette solution ($c=0$)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, which="both")
    else:
        ax.text(0.5, 0.5, "no pure controls", ha="center")
    fig.tight_layout()
    out = ART / "fig6_stress_control.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_profiles(art):
    """Velocity profiles of the c = 0.20 runs (depletion structure, audit trail)."""
    files = sorted(p for p in RUNS.glob("dispersed_0.2_coarse_*.json")
                   if "profile" in json.loads(p.read_text()))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if not files:
        ax.text(0.5, 0.5, "no runs", ha="center")
    else:
        for f in files:
            rec = json.loads(f.read_text())
            prof = rec.get("profile") or {}
            if not prof:
                continue
            ax.plot(prof["u"], prof["z"], "-o", ms=2,
                    label=(fr"seed {rec['case']['seed']}, "
                           fr"$\dot\gamma$ = {rec['gamma_dot_nominal']:.2e}"))
        ax.set_xlabel(r"$\langle u_x\rangle(z)$ (lattice)")
        ax.set_ylabel("z (lattice)")
        ax.set_title(r"Mean velocity profiles at $c=0.20$ (dispersed, coarse box)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ART / "fig7_profiles.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ART.mkdir(parents=True, exist_ok=True)
    art = load()
    made = [fig_flow_curves(art), fig_relative_viscosity(art), fig_shear_thinning(art),
            fig_finite_size(art), fig_grid_convergence(art), fig_stress_control(art),
            fig_profiles(art)]
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
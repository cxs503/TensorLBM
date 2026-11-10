#!/usr/bin/env python3
"""Figures for p223w3-n02.  Every curve is read from ``case_data/*.json``; nothing
is hard-coded except the DNS reference tables and the analytic laws."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import analysis as an  # noqa: E402
import gates as gt  # noqa: E402

RESULTS = _HERE / "case_data"
FIGURES = _HERE / "figures"
CANONICAL = ["R180_dy3", "R180_dy4", "R180_dy5", "R180_dy6", "R395_dy8", "R590_dy12"]
CASES = []
COLORS = {"R180_dy3": "#1f77b4", "R180_dy4": "#4c9be8", "R180_dy5": "#9dc7ef",
          "R180_dy6": "#b7d7f2", "R395_dy8": "#d62728", "R590_dy12": "#2ca02c"}


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def fig_mean_velocity(records: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for name, rec in records.items():
        yp = np.asarray(rec["profiles"]["y_plus"])
        up = np.asarray(rec["profiles"]["u_plus"])
        ax.semilogx(yp, up, "o-", ms=2.5, lw=1.2, color=COLORS[name], label=name)
        re_tau = int(rec["geometry"]["re_tau"])
        dns = gt.load_dns(re_tau)
        ax.semilogx(dns["y_plus"], dns["u_plus"], "k--", lw=1.0,
                    label=f"MKM DNS Re_tau={re_tau}")
    grid = np.geomspace(1.0, 700.0, 200)
    ax.semilogx(grid, an.log_law(grid), color="0.55", lw=1.0, ls=":",
                label=r"$\ln y^+/\kappa + B$ ($\kappa$=0.41, B=5.2)")
    ax.semilogx(grid, an.reichardt(grid), color="0.75", lw=1.0, ls="-.",
                label="Reichardt (1951)")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("mean velocity profile")
    ax.set_xlim(0.7, 700)
    ax.set_ylim(0, 26)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, rec in records.items():
        yp = np.asarray(rec["profiles"]["y_plus"])
        up = np.asarray(rec["profiles"]["u_plus"])
        rel = (up - an.log_law(yp)) / an.log_law(yp)
        ax.semilogx(yp, 100 * rel, "-", lw=1.3, color=COLORS[name], label=name)
    ax.axhspan(-5, 5, color="tab:green", alpha=0.12, label=r"5% gate")
    ax.axvspan(30, 200, color="0.85", alpha=0.5, label=r"$30 \leq y^+ \leq 200$")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$(U^+ - U^+_{\log})/U^+_{\log}$  [%]")
    ax.set_title("log-law error (gate a)")
    ax.set_xlim(5, 700)
    ax.set_ylim(-45, 45)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out = FIGURES / "fig1_mean_velocity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_reynolds_stresses(records: dict) -> Path:
    rec = records[CASES[0]]
    dns = gt.load_dns(180)
    yp = np.asarray(rec["profiles"]["y_plus"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(yp, rec["profiles"]["uu_plus"], "o-", ms=2.5, lw=1.2, color="#1f77b4",
            label=r"LBM $u'u'^+$")
    ax.plot(dns["y_plus"], dns["uu_plus"], "k--", lw=1.1, label=r"MKM DNS $u'u'^+$")
    ax.plot(yp, rec["profiles"]["vv_plus"], "s-", ms=2.2, lw=1.0, color="#d62728",
            label=r"LBM $v'v'^+$")
    ax.plot(dns["y_plus"], dns["vv_plus"], "k:", lw=1.0, label=r"DNS $v'v'^+$")
    ax.plot(yp, rec["profiles"]["ww_plus"], "^-", ms=2.2, lw=1.0, color="#2ca02c",
            label=r"LBM $w'w'^+$")
    ax.plot(dns["y_plus"], dns["ww_plus"], "k-.", lw=1.0, label=r"DNS $w'w'^+$")
    peak = rec["diagnostics"]["uu_peak"]
    ax.plot([peak["y_plus"]], [peak["value"]], "k*", ms=13,
            label=rf"sim peak $y^+$={peak['y_plus']:.1f}")
    i = int(np.argmax(dns["uu_plus"]))
    ax.plot([dns["y_plus"][i]], [dns["uu_plus"][i]], "k+", ms=12,
            label=rf"DNS peak $y^+$={dns['y_plus'][i]:.1f}")
    ax.axvspan(10, 20, color="tab:orange", alpha=0.12, label="peak window")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$R_{ii}^+$")
    ax.set_title(r"Reynolds normal stresses, $Re_\tau=180$ (gate b)")
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(yp, -np.asarray(rec["profiles"]["uv_plus"]), "o-", ms=2.5, lw=1.2,
            color="#9467bd", label=r"LBM $-\langle u'v'\rangle^+$")
    ax.plot(dns["y_plus"], -dns["uv_plus"], "k--", lw=1.1,
            label=r"MKM DNS $-\langle u'v'\rangle^+$")
    ax.plot(yp, 1.0 - np.asarray(rec["profiles"]["y_over_h"]),
            color="0.6", lw=1.0, ls=":", label=r"$1-y/h$ (total stress)")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"shear stress")
    ax.set_title("Reynolds shear stress and momentum balance")
    ax.set_xlim(0, 190)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out = FIGURES / "fig2_reynolds_stresses.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_spectrum(records: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for name in [c for c in ("R180_dy3", "R180_dy4", "R395_dy8", "R590_dy12") if c in records]:
        rec = records[name]
        k = np.asarray(rec["spectrum"]["k_plus"])
        e = np.asarray(rec["spectrum"]["e"])
        ax.loglog(k, e, "-", lw=1.3, color=COLORS[name], label=name)
        fit = rec["spectrum"]["fit"]
        band = (k >= fit["k_min"]) & (k <= fit["k_max"]) & (e > 0)
        if band.sum() > 2:
            ax.loglog(k[band], np.exp(fit["intercept"]) * k[band] ** fit["slope"], "--",
                      lw=1.0, color=COLORS[name],
                      label=f"fit {fit['slope']:.2f}")
    kk = np.asarray(records[CASES[0]]["spectrum"]["k_plus"])
    anchor = np.interp(0.12, kk, records[CASES[0]]["spectrum"]["e"])
    ax.loglog(kk, anchor * (kk / 0.12) ** (-5.0 / 3.0), "k:", lw=1.4,
              label=r"$k^{-5/3}$ reference")
    ax.set_xlabel(r"$k^+$ (streamwise, mid-plane)")
    ax.set_ylabel(r"$E_{uu}(k^+)$")
    ax.set_title("streamwise energy spectrum (gate c)")
    ax.set_xlim(0.02, 3.0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    out = FIGURES / "fig3_energy_spectrum.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_friction(records: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    re = np.geomspace(1500, 12000, 100)
    ax.loglog(re, an.dean_cf(re), "k--", lw=1.2, label=r"Dean: $0.073\,Re_b^{-1/4}$")
    for name, rec in records.items():
        mf = rec["mean_flow"]
        ax.plot([mf["re_bulk"]], [mf["cf"]], "o", ms=8, color=COLORS[name],
                label=f"{name}: {mf['cf_error_vs_dean']:+.1%} vs Dean")
    # DNS Markers: Cf = 2/(U_b+)^2 with U_b+ from the MKM mean profile
    for r, col in ((180, "#1f77b4"), (395, "#d62728"), (590, "#2ca02c")):
        dns = gt.load_dns(r)
        u_bulk_plus = float(np.trapezoid(dns["u_plus"], dns["y"]))
        ax.plot([u_bulk_plus * r], [2.0 / u_bulk_plus**2], "x", ms=9, color=col,
                label=f"MKM DNS Re_tau={r}")
    ax.axhspan(0.00842 * 0.95, 0.00842 * 1.05, color="tab:green", alpha=0.10)
    ax.set_xlabel(r"$Re_b = U_b\,2h/\nu$")
    ax.set_ylabel(r"$C_f$")
    ax.set_title("skin friction vs Dean correlation (gate d)")
    ax.set_xlim(2000, 9000)
    ax.set_ylim(0.004, 0.011)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    out = FIGURES / "fig4_friction.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_grid_convergence(records: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    names = [c for c in ("R180_dy6", "R180_dy5", "R180_dy4", "R180_dy3") if c in records]
    dy = [records[n]["geometry"]["dy_plus"] for n in names]
    fit = [records[n]["mean_flow"]["u_tau_routes"]["stress_fit"] for n in names]
    imposed = records[names[0]]["mean_flow"]["u_tau_routes"]["body_force"]
    ax.plot(dy, fit, "o-", color="#1f77b4", label=r"stress-fit $u_\tau$")
    ax.axhline(imposed, color="k", ls="--", label=r"imposed $\sqrt{a\,h}$")
    ax.fill_between([min(dy) - 0.3, max(dy) + 0.3], imposed * 0.97, imposed * 1.03,
                    color="tab:green", alpha=0.12, label=r"$\pm3\%$ band")
    ax.set_xlabel(r"$\Delta y^+$")
    ax.set_ylabel(r"$u_\tau$ (lattice)")
    ax.set_title("grid convergence (gate f)")
    ax.set_xlim(min(dy) - 0.3, max(dy) + 0.3)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for n in names:
        rec = records[n]
        ax.plot(rec["profiles"]["y_plus"], rec["profiles"]["u_plus"], "o-", ms=2.4,
                lw=1.1, label=f"{n} (dy+={rec['geometry']['dy_plus']:.0f})")
    dns = gt.load_dns(180)
    ax.plot(dns["y_plus"], dns["u_plus"], "k--", lw=1.1, label="MKM DNS")
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title(r"wall-normal ladder, $Re_\tau=180$")
    ax.set_xlim(0, 190)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out = FIGURES / "fig5_grid_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_laminar() -> Path:
    rec = load("LAM_poiseuille")
    y = np.asarray(rec["profiles"]["y"])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot(y - 0.5, rec["profiles"]["u_mean"], "o-", ms=4, lw=1.2,
            label=f"LBM (L2={rec['norm']['l2_rel_error']:.3%})")
    ax.plot(y - 0.5, rec["profiles"]["u_exact"], "k--", lw=1.4, label="analytic parabola")
    ax.set_xlabel("wall distance [lattice]")
    ax.set_ylabel(r"$u_x$")
    ax.set_title("laminar negative control (gate e)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = FIGURES / "fig6_laminar_control.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_flow_field() -> Path:
    """Instantaneous streamwise velocity on the mid-plane (x-z) of the primary case."""
    path = RESULTS / f"{CASES[0]}_xzm.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    nx = int(data["x"].max()) + 1
    nz = int(data["z"].max()) + 1
    ux = np.full((nz, nx), np.nan)
    ux[data["z"].astype(int), data["x"].astype(int)] = data["ux"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    im = ax.imshow(ux, origin="lower", aspect="auto", cmap="RdBu_r",
                   extent=[0, nx, 0, nz])
    fig.colorbar(im, ax=ax, label=r"$u_x$ [lattice]")
    ax.set_xlabel(r"$x$ (streamwise, $L_x^+=%.0f$)" % (2.0 * 180.0))
    ax.set_ylabel(r"$z$ (spanwise)")
    ax.set_title(r"instantaneous $u_x$ on the mid-plane, $Re_\tau=180$")
    out = FIGURES / "fig7_flow_field.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    global CASES
    FIGURES.mkdir(exist_ok=True)
    CASES = [c for c in CANONICAL if (RESULTS / f"{c}.json").exists()]
    records = {name: load(name) for name in CASES}
    if not records:
        print("no completed turbulent case: only the laminar control is plotted")
        made = []
        if (RESULTS / "LAM_poiseuille.json").exists():
            made.append(fig_laminar())
        for path in made:
            print(path)
        return 0 if made else 1
    made = [fig_mean_velocity(records)]
    if CASES:
        made.append(fig_reynolds_stresses(records))
        made.append(fig_flow_field())
    made.append(fig_spectrum(records))
    made.append(fig_friction(records))
    if len([n for n in ("R180_dy3", "R180_dy4", "R180_dy5", "R180_dy6") if n in records]) >= 2:
        made.append(fig_grid_convergence(records))
    if (RESULTS / "LAM_poiseuille.json").exists():
        made.append(fig_laminar())
    for p in made:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
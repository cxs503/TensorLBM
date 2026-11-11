#!/usr/bin/env python3
"""Figures for the four-SGS LBM-LES verification report.

Reads the campaign artefacts in ``runs/sgs_les`` and writes PNGs into
``runs/sgs_les/figures``.  Nothing is hand-typed: every curve comes from a JSON
produced by ``scripts/run_sgs_les_campaign.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorlbm.turbulence import spectral_dit as sd  # noqa: E402

RUNS = REPO / "runs" / "sgs_les"
FIGS = RUNS / "figures"

COLORS = {
    "smagorinsky": "#1f77b4",
    "dynamic_smagorinsky": "#2ca02c",
    "wale": "#d62728",
    "vreman": "#9467bd",
    "none": "#7f7f7f",
}
LABELS = {
    "smagorinsky": "Smagorinsky (Cs=0.16)",
    "dynamic_smagorinsky": "Dynamic Smagorinsky",
    "wale": "WALE (Cw=0.325)",
    "vreman": "Vreman (Cv=0.025)",
    "none": "SGS off (BGK)",
}
STATIONS = (42.0, 98.0, 172.0)


def load(name: str) -> dict | None:
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_cbc_spectra() -> str | None:
    got = [(m, load(f"dit_{m}_n64")) for m in COLORS if m != "none"]
    got = [(m, r) for m, r in got if r]
    if not got:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, st in zip(axes, STATIONS):
        k_ref, e_ref = sd.cbc_spectra(st)
        ax.loglog(k_ref, e_ref, "ko", ms=4, label=f"CBC 1971 (tU0/M={st:g})")
        for model, res in got:
            entry = next(s for s in res["stations"] if abs(s["station"] - st) < 1e-9)
            ax.loglog(entry["k1"], entry["E11"], "-", lw=1.3, color=COLORS[model],
                      label=LABELS[model])
        ax.set_xlabel(r"$k_1$ [1/m]")
        ax.set_ylabel(r"$E_{11}(k_1)$ [m$^3$/s$^2$]")
        ax.set_title(f"station tU0/M = {st:g}")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Decaying isotropic turbulence (64$^3$): longitudinal spectra vs CBC 1971")
    fig.tight_layout()
    out = FIGS / "cbc_spectra_n64.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def fig_dit_decay() -> str | None:
    curves = []
    for model in COLORS:
        r = load(f"dit_{model}_n64")
        if r:
            curves.append((model, r["time_series"]))
    if not curves:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ed = sd.cbc_derived()
    t_cbc = np.asarray(ed["t_seconds"])
    tk_cbc = 1.5 * np.asarray(ed["u1_sq"])
    for model, ts in curves:
        axes[0].plot(ts["t_star"], ts["tke"], "-", color=COLORS[model], label=LABELS[model])
        axes[1].semilogy(ts["t_star"], ts["nut_over_nu"], "-", color=COLORS[model],
                         label=LABELS[model])
    axes[0].semilogy(t_cbc / (sd.CBC_PHYSICAL["M"] / sd.CBC_PHYSICAL["U0"]), tk_cbc, "ko",
                     label="CBC 1971 (spectral integral)")
    axes[0].set_xlabel(r"$tU_0/M$")
    axes[0].set_ylabel(r"$k$ [m$^2$/s$^2$]")
    axes[0].set_ylim(1e-3, 3e-1)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    axes[0].set_title("turbulent kinetic energy decay")
    axes[1].set_xlabel(r"$tU_0/M$")
    axes[1].set_ylabel(r"$\langle\nu_t\rangle/\nu$")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].set_title("eddy-viscosity ratio (64$^3$)")
    fig.tight_layout()
    out = FIGS / "dit_decay_n64.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def fig_grid_convergence() -> str | None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    any_data = False
    for model in COLORS:
        if model == "none":
            continue
        for n, ls in ((32, ":"), (64, "--"), (128, "-")):
            r = load(f"dit_{model}_n{n}")
            if not r:
                continue
            any_data = True
            entry = next(s for s in r["stations"] if abs(s["station"] - 98.0) < 1e-9)
            k = np.asarray(entry["k3"])
            e = np.asarray(entry["E3"])
            eta = r["config"]["eta_ref"]
            axes[0].loglog(k * eta, e, ls, color=COLORS[model], lw=1.1,
                           label=f"{model} n={n}" if model == "wale" else None)
            axes[1].loglog(k * eta, e, ls, color=COLORS[model], lw=1.1)
    if not any_data:
        plt.close(fig)
        return None
    axes[0].set_title("all four closures, station 98")
    axes[1].set_title("grid levels 32/64/128$^3$ (colour = closure)")
    for ax in axes:
        ax.set_xlabel(r"$k\eta$")
        ax.set_ylabel(r"$E(k)$ [m$^3$/s$^2$]")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = FIGS / "dit_grid_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def fig_tgv() -> str | None:
    dns = None
    for cand in ("dns_n128", "dns_n96", "dns_n64", "dns_n48", "dns_n32"):
        dns = load(cand)
        if dns:
            break
    les = []
    for model in COLORS:
        r = load(f"tgv_{model}_n128") or load(f"tgv_{model}_n64")
        if r:
            les.append((model, r))
    if not dns or not les:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(dns["t"], dns["eps"], "k-", lw=2.0,
                 label=f"spectral DNS {dns['n']}$^3$ (Re={dns['re']:.0f})")
    for model, r in les:
        axes[0].plot(r["t"], r["eps"], "-", color=COLORS[model],
                     label=f"{LABELS[model]} n={r['n']}")
        axes[1].plot(r["t"], np.asarray(r["eps_sgs"]) / np.maximum(np.asarray(r["eps"]), 1e-30),
                     "-", color=COLORS[model], label=LABELS[model])
    axes[0].plot(dns["t"], dns["eps"], "k.", ms=2)
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\varepsilon(t)$")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("dissipation rate: LBM-LES vs DNS")
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\varepsilon_{SGS}/\varepsilon$")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    axes[1].set_title("SGS share of the dissipation")
    fig.tight_layout()
    out = FIGS / "tgv_dissipation.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def fig_laminar() -> str | None:
    deg = load("laminar_degeneracy")
    if not deg:
        return None
    fields = ["uniform", "rigid_rotation", "linear_shear", "poiseuille"]
    models = [m for m in COLORS if m != "none"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    width = 0.8 / len(models)
    floor = 1e-20
    for i, model in enumerate(models):
        vals = [max(deg["fields"][f]["models"][model]["nut_max"], floor) for f in fields]
        ax.bar(np.arange(len(fields)) + i * width, vals, width,
               color=COLORS[model], label=LABELS[model])
    ax.set_yscale("log")
    ax.axhline(1e-9, color="k", ls="--", lw=1.0, label="gate 1e-9")
    ax.set_xticks(np.arange(len(fields)) + 0.4 - width / 2)
    ax.set_xticklabels(fields)
    ax.set_ylabel(r"max $|\nu_t|$ (lattice units)")
    ax.set_title("laminar degeneracy of the four closures (frozen analytic fields)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGS / "laminar_degeneracy.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def fig_stokes() -> str | None:
    any_data = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in COLORS:
        r = load(f"stokes_{model}")
        if not r:
            continue
        any_data = True
        ax.semilogy(r["t"], r["amplitude"], "-", color=COLORS[model], label=LABELS[model])
    if not any_data:
        plt.close(fig)
        return None
    r0 = load("stokes_none")
    if r0:
        nu = r0["nu_molecular"]
        k = r0["k"]
        t = np.asarray(r0["t"])
        a0 = r0["amplitude"][0]
        ax.semilogy(t, a0 * np.exp(-nu * k ** 2 * t), "k--", lw=1.0,
                    label=r"exact $A e^{-\nu k^2 t}$")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("shear-wave amplitude")
    ax.set_title("laminar Stokes shear wave: no spurious SGS viscosity")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGS / "stokes_decay.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def fig_delta_scaling() -> str | None:
    rep = load("report")
    if not rep or not rep.get("delta_scaling", {}).get("points"):
        return None
    pts = rep["delta_scaling"]["points"]
    n = np.array([p["n"] for p in pts], dtype=float)
    y = np.array([p["nut_over_nu_mean"] for p in pts], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(n, y, "o-", color="#1f77b4", label=r"$\langle\nu_t\rangle/\nu$ (Smagorinsky)")
    if rep["delta_scaling"].get("fitted_exponent"):
        s = rep["delta_scaling"]["fitted_exponent"]
        ax.loglog(n, y[0] * (n / n[0]) ** s, "k--", lw=1.0, label=f"fit slope {s:.2f}")
    ax.set_xlabel("grid n (Delta = dx -> 0)")
    ax.set_ylabel(r"$\langle\nu_t\rangle/\nu$")
    ax.set_title("SGS viscosity vanishes under grid refinement")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "delta_scaling.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = {}
    for fn in (fig_cbc_spectra, fig_dit_decay, fig_grid_convergence, fig_tgv,
               fig_laminar, fig_stokes, fig_delta_scaling):
        p = fn()
        if p:
            made[fn.__name__] = p
            print(f"[figure] {p}")
    (FIGS / "index.json").write_text(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

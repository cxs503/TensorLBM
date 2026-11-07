#!/usr/bin/env python3
"""Figures for the P49-N02 added-mass instability report.

Reads ``artifacts/fsi_added_mass/campaign.json`` (+ ``free_decay_series.json``)
and writes compact PNGs to ``artifacts/fsi_added_mass/figs``:

    fig1_rho_dd.png          measured vs analytic interface factor
    fig2_error_sequences.png sub-iteration error sequences (3 ratios x 3 schemes)
    fig3_stability_map.png   (rho_s/rho_f, omega) convergence domain
    fig4_free_decay.png      free decay + identified added mass vs blockage
    fig5_field.png           velocity-magnitude field around the heaving body

The two error-sequence figures are recomputed here (deterministic, same
settings as the campaign) and the increments are stored in
``error_sequences.json`` so the figure is reproducible from artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorlbm.fsi_added_mass import (  # noqa: E402
    HeavingDiskPartitioned,
)

RATIOS = (0.1, 0.5, 1.0)
SCHEMES = (("none", "explicit", "C0"), ("aitken", "Aitken", "C1"), ("iqn-ils", "IQN-ILS(5)", "C2"))
BASE = dict(nx=48, ny=48, radius=6.0, tau=0.55, window=10)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fig1(camp, outdir):
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=115)
    pts = []
    for r in camp.get("coupled_2d", []):
        if r["scheme"] == "none" and r["rho_dd_analytic"] > 1e-3:
            num = r["rho_dd_preclip"] if r["velocity_clipped"] else r["rho_dd_numeric"]
            pts.append((r["rho_dd_analytic"], num, r["density_ratio"], "2D LBM"))
    for r in camp.get("anchor_1d", []):
        if r["scheme"] == "none" and r["rho_dd_analytic"] > 1e-3:
            pts.append((r["rho_dd_analytic"], r["rho_dd_numeric"], r["density_ratio"], "1D model"))
    lim = (0.02, 12.0)
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.fill_between(
        lim,
        [0.95 * v for v in lim],
        [1.05 * v for v in lim],
        color="0.85",
        alpha=0.6,
        label="±5% band",
    )
    for x, y, ratio, tag in pts:
        color = "C0" if tag == "2D LBM" else "C3"
        marker = "o" if tag == "2D LBM" else "s"
        ax.loglog([x], [y], marker=marker, color=color, ms=6, ls="none")
        ax.annotate(f"{ratio:g}", (x, y), textcoords="offset points", xytext=(5, -9), fontsize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"analytic $|1-c_a\rho_f/\rho_s|$")
    ax.set_ylabel(r"measured $\rho_{dd}$ (error sequence)")
    ax.set_title("Interface factor: analysis vs numerics (unrelaxed)", fontsize=9)
    ax.plot([], [], "o", color="C0", label="2D LBM+body")
    ax.plot([], [], "s", color="C3", label="1D anchor")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_rho_dd.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def fig2(outdir, workers_case=None):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), dpi=115, sharey=True)
    sequences = {}
    for k, ratio in enumerate(RATIOS):
        ax = axes[k]
        case = HeavingDiskPartitioned(**BASE, density_ratio=ratio, initial_displacement=1e-3)
        for scheme, label, color in SCHEMES:
            rec = case.sub_iterate(scheme=scheme, max_iter=40, tol=1e-6)
            inc = np.abs(np.asarray(rec["increments"], dtype=float))
            inc = np.maximum(inc, 1e-18)
            sequences[f"{ratio}:{scheme}"] = rec["increments"]
            ax.semilogy(
                np.arange(1, inc.size + 1),
                inc / inc[0],
                color=color,
                lw=1.1,
                label=f"{label} ({rec['iterations']}it)",
            )
        ax.axhline(1e-6, color="k", lw=0.6, ls=":")
        ax.set_title(rf"$\rho_s/\rho_f={ratio:g}$", fontsize=8)
        ax.set_xlabel("sub-iteration k")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(fontsize=6)
    axes[0].set_ylabel(r"$|d_k|/|d_0|$")
    fig.suptitle("Sub-iteration error sequences (2D LBM, omega = 1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(outdir, "fig2_error_sequences.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path, sequences


def fig3(camp, outdir):
    sm = camp["stability_map"]
    dens = sm["densities"]
    oms = sm["omegas"]
    grid = np.full((len(dens), len(oms)), np.nan)
    conv = np.zeros_like(grid)
    for r in sm["records"]:
        i = dens.index(r["density_ratio"])
        j = oms.index(r["omega"])
        grid[i, j] = r["rho_dd_preclip"]
        conv[i, j] = 1.0 if r["converged"] else 0.0
    c_a = 0.9515
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=115)
    logg = np.log10(np.clip(grid, 1e-3, 50.0))
    im = ax.imshow(
        logg,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        extent=(oms[0], oms[-1], 0, len(dens) - 1),
    )
    ax.set_yticks(range(len(dens)))
    ax.set_yticklabels([f"{d:g}" for d in dens], fontsize=7)
    ax.set_xlabel(r"relaxation $\omega$")
    ax.set_ylabel(r"$\rho_s/\rho_f$")
    om_line = np.linspace(oms[0], oms[-1], 200)
    ax.plot(om_line, 2.0 * om_line / c_a, "k-", lw=1.2, label=r"$\rho_s/\rho_f=\omega c_a/2$")
    ax.set_ylim(-0.5, len(dens) - 0.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\rho_{dd}$ (measured)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title(r"Convergence domain, $\rho_{dd}<1$ above the line", fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_stability_map.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def fig4(camp, series, outdir, inv=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=115)
    ax = axes[0]
    for k, s in enumerate(series):
        if s.get("t") and s.get("y"):
            dom = s.get("domain") or {}
            lab = f"L={dom.get('nx')} ({s.get('scheme')})"
            ax.plot(s["t"], s["y"], lw=1.0, label=lab, color=f"C{min(k, 4)}")
    ax.set_xlabel("t [lattice steps]")
    ax.set_ylabel("heave displacement y")
    ax.set_title(r"Coupled free decay, $\rho_s/\rho_f=1.0$", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax = axes[1]
    if inv:
        ops = inv["B_lbm_operator"]["rows"]
        w = np.array([o["window"] for o in ops])
        ca = np.array([o["c_a"] for o in ops])
        ax.plot(w, ca, "o-", color="C0", ms=4, lw=1.0, label="LBM window operator")
        pot = inv["A_potential_flow"]["rows"]
        pref = min(pot, key=lambda r: abs(r["nx"] - 48))["c_a"]
        ax.axhline(
            pref, color="C2", ls="-", lw=1.0, label=f"potential flow, same mask ({pref:.3f})"
        )
        ax.axhspan(0.95, 1.05, color="0.85", alpha=0.6, label="±5% of $A_\\infty/M=1$")
        ax.axhline(1.0, color="k", ls="--", lw=0.7)
        decay = inv.get("C_coupled_free_decay", {}).get("rows", [])
        for d in decay:
            ax.plot([12.0], [d["c_a_from_frequency"]], "x", color="C3", ms=6)
        if decay:
            ax.plot([], [], "x", color="C3", label="free-decay ID (L=96..192)")
        ax.set_xscale("log")
        ax.set_xlabel("coupling window [LBM steps]")
        ax.set_ylabel(r"$c_a=A/(\rho_f V)$")
        ax.set_title("Added-mass measurements", fontsize=8)
        ax.legend(fontsize=5.5, loc="upper left")
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_invariance.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path


def fig5(outdir):
    case = HeavingDiskPartitioned(
        nx=96,
        ny=96,
        radius=8.0,
        density_ratio=1.0,
        tau=0.55,
        window=5,
        spring_c=1.0,
        initial_displacement=0.25,
    )
    res = case.run_time_accurate(n_windows=12, scheme="aitken", tol=1e-8, max_iter=12)
    f = case._base.f
    rho = f.sum(0)
    import torch

    from tensorlbm.d2q9 import C as CC

    mob = torch.einsum("iyx,ia->ayx", f, CC.double()) / rho.clamp_min(1e-30)
    speed = torch.sqrt((mob**2).sum(0)).numpy()
    solid = case._base.solid.numpy()
    disp = np.where(solid, np.nan, speed)
    fig, ax = plt.subplots(figsize=(4.0, 3.4), dpi=115)
    im = ax.imshow(
        disp, origin="lower", cmap="magma", vmin=0.0, vmax=float(np.nanpercentile(disp, 99))
    )
    ax.contour(solid, levels=[0.5], colors="cyan", linewidths=0.8)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("|u| [lattice]", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Velocity magnitude, heaving body (window %d)" % res["n_windows"], fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_field.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return path, float(np.nanmax(disp)), res["n_windows"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/fsi_added_mass")
    args = ap.parse_args(argv)
    outdir = os.path.abspath(args.artifacts)
    figdir = os.path.join(outdir, "figs")
    os.makedirs(figdir, exist_ok=True)
    camp = load(os.path.join(outdir, "campaign.json"))
    series_path = os.path.join(outdir, "free_decay_series.json")
    series = load(series_path) if os.path.exists(series_path) else []
    made = {}
    made["fig1"] = fig1(camp, figdir)
    p2, seqs = fig2(figdir)
    made["fig2"] = p2
    with open(os.path.join(outdir, "error_sequences.json"), "w", encoding="utf-8") as fh:
        json.dump(seqs, fh)
    made["fig3"] = fig3(camp, figdir)
    inv_path = os.path.join(outdir, "invariance.json")
    inv = load(inv_path) if os.path.exists(inv_path) else None
    made["fig4"] = fig4(camp, series, figdir, inv)
    p5, smax, nwin = fig5(figdir)
    made["fig5"] = p5
    info = {"figures": made, "field_max_speed": smax, "field_window": nwin}
    with open(os.path.join(outdir, "figures.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    # keep the report payload small: quantise line-art PNGs to a 64-colour palette
    try:
        from PIL import Image

        for k, v in made.items():
            im = Image.open(v).convert("RGB")
            im = im.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
            im.save(v, optimize=True)
    except Exception as exc:  # pragma: no cover - optional shrink
        print("quantise skipped:", exc)
    for k, v in made.items():
        print(k, v, os.path.getsize(v) // 1024, "KB")


if __name__ == "__main__":
    main()

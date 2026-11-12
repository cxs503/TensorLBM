#!/usr/bin/env python3
"""Figures for CASE-LBM-RB-ONSET-RA1708 — generated only from run artifacts.

    PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/make_figures.py \
        [--run-dir runs/case_rayleigh_benard_onset] [--asym]

``--asym`` additionally time-marches a short supercritical run and stores the
temperature / vertical-velocity fields (``fields_ny32_ra1.2.npz``) plus the
convection-roll snapshot used in the report.  Everything else is read from
``summary.json`` and the per-route JSON files, so figures never contain
hand-transcribed numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASE_DIR = Path(__file__).resolve().parent


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def fig_sigma_vs_ra(summary: dict, out: Path) -> Path:
    pre = summary["predeclared"]
    ra_ref = pre["ra_ref"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for k, chain in enumerate(summary["route_a"]):
        if not chain.get("Ra_c"):
            continue
        h = chain["history"]
        x = np.array([a["ratio"] for a in h])
        y = np.array([a["sigma_star"] for a in h])
        v = [a["verdict"] for a in h]
        order = np.argsort(x)
        x, y, v = x[order], y[order], [v[i] for i in order]
        ax.plot(x, y, "-", color=colors[k % 3], lw=1.0, alpha=0.5,
                label=f"ny={chain['ny']} (Ra$_c$={chain['Ra_c']:.1f})")
        for xi, yi, vi in zip(x, y, v):
            marker = {"stable": "v", "unstable": "^", "marginal": "o",
                      "indeterminate": "x"}[vi]
            ax.plot(xi, yi, marker, color=colors[k % 3], ms=7, mfc="none" if vi == "marginal" else colors[k % 3],
                    mew=1.4 if vi == "indeterminate" else 1.0)
        ax.axvline(chain["Ra_c"] / ra_ref, color=colors[k % 3], ls=":", lw=1.0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.axhspan(-pre["sigma_marginal"], pre["sigma_marginal"], color="0.85", zorder=0)
    ax.set_xlabel(r"$Ra/Ra_{ref}$   ($Ra_{ref}=1707.7597$, Chandrasekhar 1961 §II)")
    ax.set_ylabel(r"$\sigma^{*}=\sigma H^{2}/\alpha$   (quiescent-state growth rate)")
    ax.set_title("RB onset: bisection scan on the sign of the perturbation growth rate")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig1_sigma_vs_ra.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_grid_convergence(summary: dict, out: Path) -> Path:
    pre = summary["predeclared"]
    ra_ref = pre["ra_ref"]
    tol = pre["tol_pct"]
    chain = [c for c in summary["route_a"] if c.get("Ra_c")]
    ny = np.array([c["ny"] for c in chain], dtype=float)
    ra = np.array([c["Ra_c"] for c in chain], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(ra_ref * (1 - tol / 100), ra_ref * (1 + tol / 100), color="0.85",
               zorder=0, label=f"±{tol:g}% case-pool band")
    ax.axhline(ra_ref, color="k", ls="--", lw=0.9, label="Chandrasekhar 1961 §II: 1707.76")
    g = summary["grid_convergence"]
    if "Ra_inf" in g:
        xs = np.linspace(0.0, max(1.0 / ny) * 1.15, 50)
        ax.plot(xs, g["Ra_inf"] + g["slope"] * xs, "-", color="#d62728", lw=1.0,
                label=fr"fit $Ra_c(\infty)+C/ny$: {g['Ra_inf']:.1f}")
        ax.plot(0.0, g["Ra_inf"], "o", color="#d62728", ms=7)
    ax.plot(1.0 / ny, ra, "s-", color="#1f77b4", ms=6, label="Route A (time-marched LBM)")
    for x, y, c in zip(1.0 / ny, ra, chain):
        ax.annotate(f"ny={c['ny']}\n{c['err_pct']:+.2f}%", (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    b = summary.get("route_b", {})
    if b.get("verdict") == "ok" and b.get("Ra_c_routeA_convention"):
        ax.plot(1.0 / b["ny"], b["Ra_c_routeA_convention"], "D", color="#ff7f0e", ms=6,
                label=f"Route B (eigen, ny={b['ny']})")
    ax.set_xlabel("$1/ny$")
    ax.set_ylabel(r"$Ra_c$  (nominal $Ra=g_\beta\Delta T H^{3}/\nu\alpha$)")
    ax.set_title("Grid convergence of the LBM onset Rayleigh number")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig2_grid_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_wavenumber(summary: dict, out: Path) -> Path:
    w = summary.get("wavenumber_sensitivity", {})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    pre = summary["predeclared"]
    if w.get("arms"):
        a = np.array([r["a"] for r in w["arms"]])
        s = np.array([r["sigma_star"] for r in w["arms"]])
        ax.plot(a, s, "o", color="#1f77b4", ms=7, label=fr"LBM $\sigma^*$ at Ra={w['ra_ratio']}$Ra_{{ref}}$")
        aa = np.linspace(a.min() * 0.97, a.max() * 1.03, 100)
        par = w["parabola"]
        ax.plot(aa, par["a2"] * aa ** 2 + par["a1"] * aa + par["a0"], "-",
                color="#1f77b4", lw=1.0, alpha=0.7)
        ax.axvline(w["a_peak"], color="#1f77b4", ls=":", lw=1.0,
                   label=fr"parabola peak $a$={w['a_peak']:.4f}")
    ax.axvline(pre["a_ref"], color="k", ls="--", lw=0.9,
               label=r"$a_c=3.11632$ (Chandrasekhar 1961)")
    ax.axvline(math.pi, color="#d62728", ls="-.", lw=1.0,
               label=r"case wavenumber $k H=\pi$ ($\Gamma=2$, m=1)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$a = kH$")
    ax.set_ylabel(r"$\sigma^{*}$")
    ax.set_title("Wavenumber sensitivity of the growth rate (ny=32)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig3_wavenumber_scan.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_control(summary: dict, out: Path) -> Path:
    c = summary.get("route_c", {})
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    if c.get("grids"):
        M = np.array([g["M"] for g in c["grids"]], dtype=float)
        ra = np.array([g["Ra_c"] for g in c["grids"]], dtype=float)
        axs[0].plot(1.0 / M, ra, "o-", color="#2ca02c", ms=6, label="continuum FD bisection")
        axs[0].axhline(c["Ra_c_richardson"], color="#2ca02c", ls=":", lw=1.0,
                       label=fr"Richardson {c['Ra_c_richardson']:.3f}")
        axs[0].axhline(c["Ra_c_ref"], color="k", ls="--", lw=0.9, label="ref 1707.7597")
        axs[0].set_xlabel("$1/M$ (FD interior nodes)")
        axs[0].set_ylabel("$Ra_c$")
        axs[0].set_title("Route C control: continuum marginal stability")
        axs[0].legend(fontsize=8)
        axs[0].grid(alpha=0.3)
    curve = c.get("marginal_curve")
    if curve:
        if isinstance(curve, dict):
            keys = sorted(curve.keys(), key=lambda k: float(k))
            a = np.array([float(k) for k in keys])
            ra = np.array([float(curve[k]) for k in keys])
        else:  # list of {"a":..., "Ra_c":...}
            a = np.array([row["a"] for row in curve], dtype=float)
            ra = np.array([row["Ra_c"] for row in curve], dtype=float)
        axs[1].plot(a, ra, "o-", color="#2ca02c", ms=5)
        axs[1].axvline(summary["predeclared"]["a_ref"], color="k", ls="--", lw=0.9)
        axs[1].axvline(math.pi, color="#d62728", ls="-.", lw=1.0)
        axs[1].set_xlabel(r"$a=kH$")
        axs[1].set_ylabel("marginal $Ra$")
        axs[1].set_title("Continuum marginal curve (control)")
        axs[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig4_control.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def snapshot_fields(ny: int = 32, ratio: float = 1.2, steps: int = 30000) -> dict:
    """Short supercritical run: store T, u_y and the roll pattern for the report."""
    import torch

    from tensorlbm.rb_convection_validation import make_convection

    torch.set_num_threads(int(os.environ.get("RB_LBM_THREADS", "4")))
    ra_ref = 1707.7597
    sim = make_convection(ny=ny, nx=2 * ny, ra=ratio * ra_ref, pr=0.71, case="rb",
                          alpha=1.0 / 6.0)
    T0 = sim.base_profile()
    y = (torch.arange(ny, dtype=sim.dtype) + 0.5) / sim.H
    x = torch.arange(sim.nx, dtype=sim.dtype)
    k = 2.0 * math.pi / sim.nx
    sim.set_temperature(T0 + 1e-2 * torch.sin(math.pi * y).unsqueeze(1)
                        * torch.cos(k * x).unsqueeze(0))
    sim.step(steps)
    rho, ux, uy, T = sim.fields()
    return {
        "ny": ny, "ratio": ratio, "steps": steps,
        "T": T.numpy(), "uy": uy.numpy(), "ux": ux.numpy(),
        "nusselt": sim.nusselt(T), "ma": sim.mach(),
    }


def fig_fields(run_dir: Path, out: Path, snap: dict) -> Path:
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.0))
    T = snap["T"]
    im0 = axs[0].imshow(T, origin="lower", cmap="inferno", aspect="auto")
    axs[0].set_title(f"temperature T (ny={snap['ny']}, Ra={snap['ratio']:.2f}$Ra_{{ref}}$, "
                     f"{snap['steps']} steps)")
    axs[0].set_xlabel("x (lattice)"); axs[0].set_ylabel("y (lattice)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    uy = snap["uy"]
    im1 = axs[1].imshow(uy, origin="lower", cmap="RdBu_r", aspect="auto",
                        vmin=-np.abs(uy).max(), vmax=np.abs(uy).max())
    axs[1].set_title(r"vertical velocity $u_y$ (convection rolls)")
    axs[1].set_xlabel("x (lattice)"); axs[1].set_ylabel("y (lattice)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    fig.tight_layout()
    p = out / "fig5_fields_rolls.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    np.savez_compressed(run_dir / "fields_snapshot.npz",
                        T=snap["T"], uy=snap["uy"], ux=snap["ux"],
                        ny=snap["ny"], ratio=snap["ratio"])
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path,
                    default=REPO_ROOT / "runs" / "case_rayleigh_benard_onset")
    ap.add_argument("--asym", action="store_true",
                    help="also run the supercritical field snapshot (adds ~2 min)")
    args = ap.parse_args(argv)
    run_dir: Path = args.run_dir
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    summary = load(run_dir)
    made = [fig_sigma_vs_ra(summary, out), fig_grid_convergence(summary, out),
            fig_wavenumber(summary, out), fig_control(summary, out)]
    if args.asym:
        snap = snapshot_fields()
        (run_dir / "fields_meta.json").write_text(json.dumps(
            {k: v for k, v in snap.items() if not isinstance(v, np.ndarray)}, indent=2,
            default=float))
        made.append(fig_fields(run_dir, out, snap))
    print(json.dumps([str(p) for p in made], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
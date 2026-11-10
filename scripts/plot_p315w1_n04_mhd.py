#!/usr/bin/env python
"""Figures for node p315w1-n04-lbm-mhd (read-only: every curve comes from the artefact JSONs).

Usage::

    python scripts/plot_p315w1_n04_mhd.py [--indir artifacts/mhd_p315w1_n04]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _save(fig, path: Path, retries: int = 30, wait: float = 10.0) -> None:
    """Save a figure, retrying while the shared disk is full (ENOSPC)."""
    for attempt in range(retries):
        try:
            fig.savefig(path, dpi=150)
            return
        except OSError as exc:
            if attempt == retries - 1:
                raise
            print(f"[disk-full] retry {attempt + 1}/{retries} for {path} ({exc})", flush=True)
            time.sleep(wait)


def _load(indir: Path, name: str) -> dict:
    return json.loads((indir / name).read_text())


def fig_hartmann(indir: Path, out: Path) -> Path:
    data = _load(indir, "hartmann_ladder.json")
    ny = data["ny_list"][-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(data["ha_list"])))
    for colour, ha in zip(colours, data["ha_list"]):
        run = data["runs"][f"{float(ha):g}_{ny}"]
        y = np.asarray(run["y_hat"])
        axes[0].plot(np.asarray(run["u_lbm"]) / run["u_mean_lbm"], y, "o", ms=3.5,
                     mfc="none", color=colour, label=f"Ha={ha:g} LBM")
        axes[0].plot(np.asarray(run["u_analytic"]) / run["u_mean_lbm"], y, "-",
                     lw=1.2, color=colour, label=f"Ha={ha:g} exact")
        axes[1].semilogx(np.abs(np.asarray(run["u_lbm"]) - np.asarray(run["u_analytic"]))
                         / run["u_mean_lbm"] + 1e-16, y, "-", lw=1.2, color=colour,
                         label=f"Ha={ha:g}  (L∞={run['rel_linf']:.1e})")
    axes[0].set_xlabel(r"$u_x / u_{\rm mean}$")
    axes[0].set_ylabel(r"$y/h$")
    axes[0].set_title(f"Hartmann (conducting walls), ny={ny}")
    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r"$|u_{\rm LBM}-u_{\rm exact}|/u_{\rm mean}$")
    axes[1].set_ylabel(r"$y/h$")
    axes[1].set_title("pointwise profile error (mean-normalised)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "fig1_hartmann_profiles.png"
    _save(fig, path)
    plt.close(fig)
    return path


def fig_shercliff(indir: Path, out: Path) -> Path:
    data = _load(indir, "shercliff_centreline.json")
    ny = data["ny_list"][-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colours = plt.cm.plasma(np.linspace(0.15, 0.75, len(data["ha_list"])))
    for colour, ha in zip(colours, data["ha_list"]):
        run = data["runs"][f"{float(ha):g}_{ny}"]
        y = np.asarray(run["y_hat"])
        ax.plot(np.asarray(run["u_lbm"]) * 1e3, y, "o", ms=3.5, mfc="none",
                color=colour, label=f"Ha={ha:g} LBM (centre err {run['rel_centre']:.1e})")
        ax.plot(np.asarray(run["u_analytic"]) * 1e3, y, "-", lw=1.2, color=colour)
    ax.set_xlabel(r"$u_x$ [$10^{-3}$ lattice units]")
    ax.set_ylabel(r"$y/h$")
    ax.set_title(f"Shercliff (insulating walls, zero net current), ny={ny}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "fig2_shercliff_profiles.png"
    _save(fig, path)
    plt.close(fig)
    return path


def fig_damping(indir: Path, out: Path) -> Path:
    data = _load(indir, "damping.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = plt.cm.cividis(np.linspace(0.1, 0.8, len(data["ha_values"])))
    for colour, ha in zip(colours, data["ha_values"]):
        run = data["runs"][f"{float(ha):g}"]
        t = np.asarray([h["t"] for h in run["history"]])
        a = np.asarray([h["amplitude"] for h in run["history"]])
        ax = np.asarray([h["exact"] for h in run["history"]])
        axes[0].semilogy(t, a, "o", ms=3, mfc="none", color=colour,
                         label=f"Ha={ha:g} measured")
        axes[0].semilogy(t, ax, "-", lw=1.1, color=colour)
    axes[0].set_xlabel("lattice time $t$")
    axes[0].set_ylabel(r"shear-wave amplitude $A(t)$")
    axes[0].set_title("magnetic damping (symbols: LBM, lines: exact)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    ha = np.asarray(data["ha_values"], dtype=float)
    rates = np.asarray(data["magnetic_rates_fit"], dtype=float)
    nu, h = data["nu"], 0.5 * data["ny"]
    axes[1].loglog(ha, nu * ha**2 / h**2, "-", lw=1.2, color="k",
                   label=r"theory $\nu Ha^2/h^2$")
    axes[1].loglog(ha, rates, "s", ms=7, mfc="none", color="tab:red",
                   label=f"LBM (slope {data['scaling_exponent']:.3f})")
    axes[1].set_xlabel("Hartmann number $Ha$")
    axes[1].set_ylabel("magnetic damping rate")
    axes[1].set_title(r"$\gamma_{\rm mag}\propto Ha^2$ scaling")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out / "fig3_damping_decay.png"
    _save(fig, path)
    plt.close(fig)
    return path


def fig_convergence(indir: Path, out: Path) -> Path:
    data = _load(indir, "hartmann_ladder.json")
    pois = _load(indir, "degeneration.json")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ny = np.asarray(data["ny_list"], dtype=float)
    for ha in data["ha_list"]:
        errs = np.asarray(
            [data["runs"][f"{float(ha):g}_{int(n)}"]["rel_l2"] for n in ny]
        )
        p = np.log(errs[0] / errs[-1]) / np.log(ny[-1] / ny[0])
        ax.loglog(ny, errs, "o-", label=f"Ha={ha:g} (fit slope {p:.2f})")
    errs = np.asarray([pois["runs"][k]["rel_linf"] for k in pois["runs"]])
    ax.loglog(ny[: len(errs)], errs, "s--", color="k", label="Ha=0 Poiseuille gate (1 %)")
    ax.axhline(0.03, color="tab:red", ls=":", lw=1.2, label="3 % gate")
    ax.set_xlabel("grid rows across the channel $n_y$")
    ax.set_ylabel(r"velocity $L_2$ error / $u_{\rm mean}$")
    ax.set_title("grid convergence of the Hartmann branch")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig4_convergence.png"
    _save(fig, path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default=str(ROOT / "artifacts" / "mhd_p315w1_n04"))
    args = ap.parse_args(argv)
    indir = Path(args.indir)
    paths = [
        fig_hartmann(indir, indir),
        fig_shercliff(indir, indir),
        fig_damping(indir, indir),
        fig_convergence(indir, indir),
    ]
    for path in paths:
        print(path, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
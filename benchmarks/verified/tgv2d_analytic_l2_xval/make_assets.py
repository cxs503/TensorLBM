#!/usr/bin/env python
"""Figures for the XVAL analytic-L2 harness benchmark (reads ``result.json``).

Produces
--------
``figures/l2_vs_time.png``        L2 diagnostics vs t/td, both arms, OSS ladder
``figures/acoustic_transient.png`` pressure-harmonic oscillation + its convergence
``figures/fields_nx64.png``        u_x field: numerical, analytic, difference

Run:  python make_assets.py     (after ``run.py``)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (REPO_ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run as xval  # noqa: E402
from tensorlbm.verification.analytic_l2 import tgv2d_analytic, tgv2d_decay_time  # noqa: E402

FIG = HERE / "figures"


def fig_l2_vs_time(report: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    styles = {1: "o", 2: "s", 4: "^"}
    for ax, key, label in ((axes[0], "L2ux", "velocity $L^2$ ($u_x$, $u_y$)"),
                           (axes[1], "L2rho", "density $L^2$ (OSS fluctuation norm)")):
        for case in report["oss_ladder"]:
            sc = case["case"]["scale"]
            t = [s["t_over_td"] for s in case["tensorlbm"]["samples"]]
            v = [s[key] for s in case["tensorlbm"]["samples"]]
            ax.plot(t, v, marker=styles[sc], ls="-", label=f"TensorLBM NX={32*sc}")
            vo = [s[key] for s in case["oss_port"]["samples"]]
            ax.plot(t, vo, ls="none", marker="x", ms=9, mfc="none", label=f"OSS port NX={32*sc}")
        ax.set_yscale("log")
        ax.set_xlabel("$t/t_d$")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative $L^2$ error")
    axes[0].axhline(0.03, color="tab:red", ls="--", lw=1.2, label="3% target")
    axes[0].axhline(0.15, color="k", ls=":", lw=1.2, label="15% owner limit")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    out = FIG / "l2_vs_time.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_acoustic_transient(report: dict) -> Path:
    scan = report["harmonic_transient"]
    t = np.array([r["t"] for r in scan["samples"]])
    ratio = np.array([r["ratio"] for r in scan["samples"]])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(t, ratio, "-", lw=1.0, color="tab:blue", label="measured / analytic pressure harmonic")
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    period = scan["predicted_acoustic_period"]
    ax.axvspan(0, period, color="tab:orange", alpha=0.12,
               label=f"predicted acoustic period $2\\pi/(c_s 2k)$ = {period:.2f} steps")
    ax.set_xlabel("lattice step $t$")
    ax.set_ylabel("amplitude ratio")
    ax.set_title(f"IC-excited acoustic transient (NX={scan['nx']}), measured period "
                 f"{scan['measured_deviation_period']:.2f} steps")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rows = report["transient_averaged"]
    nx = np.array([r["nx"] for r in rows], float)
    err = np.array([abs(r["amp_rel_err_pct"]) for r in rows])
    ax.loglog(nx, err, "o-", label="period-averaged harmonic amplitude error")
    ref = err[0] * (nx / nx[0]) ** -2.0
    ax.loglog(nx, ref, "k--", lw=1.0, label=r"slope $-2$ reference")
    ax.set_xlabel("NX")
    ax.set_ylabel("|error| vs analytic [%]")
    ax.set_title("transient averaging restores monotone, $\\approx$2nd-order convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    out = FIG / "acoustic_transient.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def fig_fields(nx: int = 64, tau: float = 1.0, u_max: float = 0.04, tstar: float = 0.5) -> Path:
    nu = (2.0 * tau - 1.0) / 6.0
    td = tgv2d_decay_time(nx, nx, nu)
    nsteps = max(1, int(round(tstar * td)))
    res = xval.run_tensorlbm(nx, tau=tau, u_max=u_max, nsteps=nsteps, nmsg=nsteps)
    t = res.samples[-1]["t"]
    ref = tgv2d_analytic(t, nx, nx, nu, u_max=u_max)
    ux = res.final_fields["ux"]
    diff = ux - ref.ux
    lim = max(abs(ux).max(), abs(ref.ux).max())
    dlim = max(abs(diff).max(), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, field, title, lim_ in (
        (axes[0], ux, "TensorLBM $u_x$", lim),
        (axes[1], ref.ux, "analytic $u_x$", lim),
        (axes[2], diff, "difference (x$10^{6}$)", dlim * 1e6),
    ):
        im = ax.imshow(field * (1e6 if "x$10^{6}$" in title else 1.0), origin="lower",
                       cmap="RdBu_r", vmin=-lim_ * (1e6 if "x$10^{6}$" in title else 1.0),
                       vmax=lim_ * (1e6 if "x$10^{6}$" in title else 1.0), extent=[0, nx, 0, nx])
        ax.set_title(title)
        ax.set_xlabel("$x$")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("$y$")
    fig.suptitle(
        f"Taylor-Green NX={nx}, t/t_d={t/td:.3f}: relative $L^2(u_x)$ = "
        f"{res.samples[-1]['L2ux']:.3e}, $L^2(\\rho)$ = {res.samples[-1]['L2rho']:.3e}",
        fontsize=10,
    )
    out = FIG / "fields_nx64.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    report = json.loads((HERE / "result.json").read_text())
    outs = [fig_l2_vs_time(report), fig_acoustic_transient(report), fig_fields()]
    for o in outs:
        print(o.relative_to(REPO_ROOT), f"{o.stat().st_size/1024:.1f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Figures for p223w2b-n01.  Every curve is read from the audit/ledger JSON or
from the in-repo MKM-1999 tables -- nothing is typed in by hand.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/p223w2b_n01_channel_turb"))

import audit  # noqa: E402
from tensorlbm.turbulence.spectral_channel import (  # noqa: E402
    SpectralChannel,
    SpectralChannelConfig,
    log_law,
    poiseuille_exact,
)

OUT = ROOT / "artifacts/p223w2b_n01_channel_turb"


def f1_log_law() -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = {180: "#1f77b4", 395: "#d62728", 590: "#2ca02c"}
    for nominal in audit.available_cases():
        p = audit.load_profile(nominal)
        half = p.y_plus <= p.re_tau
        ax.semilogx(p.y_plus[half], p.u_plus[half], color=colors[nominal], lw=1.8,
                    label=fr"MKM DNS $Re_\tau={p.re_tau:.0f}$")
        err = audit.log_law_error(p.y_plus, p.u_plus, band=(30.0, 100.0))
        ax.fill_between([30, 100], [0, 0], [0, 0])  # keeps the legend order stable
        ax.annotate(fr"max dev {100*err['max_rel']:.1f}\%",
                    xy=(60, float(np.interp(60, p.y_plus, p.u_plus)) + 0.6),
                    fontsize=7.5, color=colors[nominal])
    yp = np.logspace(0, np.log10(600), 200)
    ax.semilogx(yp, log_law(yp), "k--", lw=1.2,
                label=r"$\frac{1}{0.41}\ln y^+ + 5.2$")
    ax.semilogx(yp, 1.05 * log_law(yp), "k:", lw=0.8)
    ax.semilogx(yp, 0.95 * log_law(yp), "k:", lw=0.8, label=r"$\pm5\%$ band")
    ax.axvspan(30, 100, color="grey", alpha=0.08)
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    ax.set_title("Criterion (a): log law is attainable on the reference itself")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0.5, 600)
    path = OUT / "fig1_log_law_reference.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def f2_stresses() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    colors = {180: "#1f77b4", 395: "#d62728", 590: "#2ca02c"}
    for nominal in audit.available_cases():
        p = audit.load_profile(nominal)
        half = (p.y_plus <= p.re_tau) & (p.y_plus > 0)
        ax = axes[0]
        ax.plot(p.y_plus[half], p.r_uu[half], color=colors[nominal], lw=1.7,
                label=fr"$u'u'^+$ $Re_\tau={p.re_tau:.0f}$")
        ax.plot(p.y_plus[half], p.r_vv[half], color=colors[nominal], lw=1.0, ls="--")
        ax.plot(p.y_plus[half], p.r_ww[half], color=colors[nominal], lw=1.0, ls=":")
        peak = p.peak("u_rms")
        ax.plot([peak["y_plus"]], [peak["value"] ** 2], "k.", ms=6)
        ax.annotate(fr"{peak['value']**2:.2f} @ {peak['y_plus']:.0f}",
                    xy=(peak["y_plus"], peak["value"] ** 2), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5)
        axes[1].plot(p.y_plus[half], -p.r_uv[half], color=colors[nominal], lw=1.7,
                     label=fr"$-u'v'^+$ $Re_\tau={p.re_tau:.0f}$")
    axes[0].set_xlim(0, 120)
    axes[0].set_ylim(0, 9)
    axes[0].set_xlabel(r"$y^+$")
    axes[0].set_ylabel(r"stress$^+$")
    axes[0].set_title(r"streamwise $u'u'^+$ (solid), $v'v'^+$ (--) , $w'w'^+$ (:), peaks dotted")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlim(0, 200)
    axes[1].set_xlabel(r"$y^+$")
    axes[1].set_ylabel(r"$-u'v'^+$")
    axes[1].set_title("shear stress; peak 1 = the momentum balance")
    axes[1].legend(fontsize=7.5)
    fig.suptitle("Criterion (b) reference targets (MKM-1999)", fontsize=10)
    path = OUT / "fig2_reference_stresses.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def f3_cf() -> Path:
    """Criterion (d): the Dean correlation against exact channel DNS."""
    payload = json.loads((ROOT / audit.REFERENCE_JSON).read_text())
    cases = payload["cases"]
    re_b = np.array([c["re_tau"] * c["ub_plus"] for c in cases])
    cf_dns = np.array([c["cf_dns"] for c in cases])
    plt.figure(figsize=(6.2, 4.4))
    rg = np.logspace(np.log10(re_b.min() * 0.8), np.log10(re_b.max() * 1.2), 100)
    plt.loglog(rg, 0.073 * rg ** -0.25, "k--", lw=1.4,
               label=r"Dean 1978: $0.073\,Re_b^{-0.25}$")
    fit = payload["power_law"]
    plt.loglog(rg, fit["A"] * rg ** (-fit["n"]), "-", color="#2ca02c", lw=1.4,
               label=fr"fit through DNS: ${fit['A']:.4f}\,Re_b^{{-{fit['n']:.4f}}}$")
    plt.loglog(re_b, cf_dns, "o", ms=7, color="#1f77b4", label="MKM-1999 DNS")
    for c in cases:
        plt.annotate(fr"+{100*c['cf_dean_rel_dev']:.1f}\%",
                     xy=(c["re_tau"] * c["ub_plus"], c["cf_dns"]), xytext=(6, -12),
                     textcoords="offset points", fontsize=8)
    plt.xlabel(r"$Re_b = U_b\,2h/\nu$")
    plt.ylabel(r"$C_f = 2\tau_w/(\rho U_b^2)$")
    plt.title("Criterion (d) is mis-specified: exact DNS fails Dean by ~24%")
    plt.legend(fontsize=8)
    path = OUT / "fig3_cf_vs_dean.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def f4_laminar() -> Path:
    """Engine anchor: the forced channel against the exact parabola."""
    cfg = SpectralChannelConfig(nx=16, ny=33, nz=16, re_tau=10.0, dt=5.0e-3,
                                init_profile="parabola", noise=0.0, stat_every=10**9)
    solver = SpectralChannel(cfg)
    hist = []
    for _ in range(8):
        solver.run(250)
        exact = poiseuille_exact(solver.y, cfg.height, cfg.accel, cfg.nu)
        hist.append((solver.step_index * cfg.dt,
                     float(np.abs(solver.mean_profile() - exact).max() / exact.max()),
                     solver.wall_shear()))
    exact = poiseuille_exact(solver.y, cfg.height, cfg.accel, cfg.nu)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(solver.y, solver.mean_profile(), "o", ms=3, label="spectral/FD engine")
    axes[0].plot(solver.y, exact, "-", lw=1.2, label="exact Poiseuille")
    axes[0].set_xlabel("y/h")
    axes[0].set_ylabel("u")
    axes[0].set_title("laminar twin after 8 viscous-time units")
    axes[0].legend(fontsize=8)
    h = np.array(hist)
    axes[1].semilogy(h[:, 0], h[:, 1], "o-", label=r"$\|u-u_{exact}\|_\infty/u_{max}$")
    axes[1].axhline(1e-2, color="k", ls=":", lw=1, label="1% line")
    axes[1].set_xlabel("t / (h^2/nu)")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("viscous relaxation of the anchor")
    axes[1].legend(fontsize=8)
    path = OUT / "fig4_laminar_anchor.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [f1_log_law(), f2_stresses(), f3_cf(), f4_laminar()]
    print(json.dumps([str(p.relative_to(ROOT)) for p in paths], indent=1))
    return [str(p) for p in paths]


if __name__ == "__main__":  # pragma: no cover
    main()

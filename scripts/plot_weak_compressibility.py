#!/usr/bin/env python3
"""Figures for the IN130-N03 weak-compressibility verification report.

Reads the campaign artefacts (``artifacts/IN130-N03/raw/*.json`` and
``verification.json``) and writes PNGs into ``artifacts/IN130-N03/``.  Every
number plotted comes from the JSON files (no hand-typed data).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p: Path):
    return json.loads(p.read_text())


def _series(recs, key):
    """Numeric series with NaN for null/missing entries (diverged shards)."""
    out = []
    for r in recs:
        v = r.get(key)
        out.append(float("nan") if v is None else float(v))
    return np.asarray(out)


def _powerlaw_line(ax, machs, exp=2.0, y0=None, **kw):
    m = np.asarray(machs, dtype=float)
    if y0 is None:
        y0 = kw.pop("anchor")
    ax.plot(m, y0 * (m / m[0]) ** exp, **kw)


def fig_error_scaling(raw: Path, out: Path) -> Path:
    pois = sorted(raw.glob("poiseuille_ma*.json"))
    pres = sorted(raw.glob("pressure_ma*.json"))
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for files, label, marker in ((pois, "body-force, periodic (rho uniform)", "o"),
                                (pres, "pressure-driven (physical $\\nabla p$)", "s")):
        if not files:
            continue
        recs = [_load(f) for f in files]
        ma = _series(recs, "mach_measured")
        err = _series(recs, "err_inf")
        ok = np.isfinite(ma) & np.isfinite(err) & (err > 0)
        if not ok.any():
            continue
        order = np.argsort(ma[ok])
        ax.loglog(ma[ok][order], err[ok][order], marker + "-", label=label, lw=1.4, ms=5)
    if pres:
        recs = sorted((_load(f) for f in pres), key=lambda r: r["mach_measured"])
        err0 = recs[0]["err_inf"]
        machs = [r["mach_measured"] for r in recs]
        _powerlaw_line(ax, machs, 2.0, anchor=err0,
                       color="k", ls="--", lw=1.0, label=r"$\propto Ma^{2}$ reference")
    ax.set_xlabel("centre-line Mach number $Ma = u_{max}/c_s$")
    ax.set_ylabel(r"shape error $\max|u_{num}-u_{exact}|/u_{max}$")
    ax.set_title("Weak-compressibility error of the steady channel profile")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = out / "fig1_profile_error_vs_ma.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_density_order(raw: Path, out: Path) -> Path:
    pres = sorted(raw.glob("pressure_ma*.json"))
    cyl = sorted(raw.glob("cylinder_ma*.json"))
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for files, label, marker in ((pres, "pressure-driven channel", "s"),
                                (cyl, "Re=100 cylinder", "o")):
        if not files:
            continue
        recs = [_load(f) for f in files]
        ma = _series(recs, "mach_measured")
        d = _series(recs, "drho_over_rho0")
        ok = np.isfinite(d) & (d > 0)
        if not ok.any():
            continue
        order = np.argsort(ma[ok])
        m, dd = ma[ok][order], d[ok][order]
        ax.loglog(m, dd, marker + "-", label=label, lw=1.4, ms=5)
        if m.size > 1:
            ax.annotate(f"slope {np.polyfit(np.log(m), np.log(dd), 1)[0]:.2f}",
                        xy=(m[-1], dd[-1]), xytext=(6, -12),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mach number $Ma$")
    ax.set_ylabel(r"$\max|\rho-\rho_0|/\rho_0$")
    ax.set_title(r"Density-fluctuation order (target $Ma^{2}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig2_density_fluctuation_order.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_cylinder_mach(raw: Path, out: Path) -> Path:
    files = sorted(raw.glob("cylinder_ma*.json"))
    if not files:
        return out / "fig3_cylinder_mach_independence.png"
    recs = sorted((_load(f) for f in files), key=lambda r: r["mach_measured"])
    ma = _series(recs, "mach_measured")
    st = _series(recs, "st")
    cd = _series(recs, "cd_mean")
    cl = _series(recs, "cl_amplitude")
    if not np.isfinite(st).any():
        # cylinder sub-face not verified (all shards diverged): emit a panel that
        # states the measured divergence instead of plotting nothing
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        blow = [r.get("blowup_step") for r in recs]
        tau = _series(recs, "tau")
        ax.plot(tau, [b if b and b > 0 else r["steps"] for b, r in zip(blow, recs)],
                "x-", ms=8, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\tau$ at fixed $D$ (Re=100)")
        ax.set_ylabel("steps until divergence")
        ax.set_title("Re=100 cylinder: low-$Ma$ corner is $\\tau\\to1/2$ stiff\n"
                     "(no St/C$_D$ could be measured -> NOT VERIFIED)")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        p = out / "fig3_cylinder_mach_independence.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        return p
    st = np.where(np.isfinite(st), st, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, v, ref, name, limit in (
        (axes[0], st, st[0], "Strouhal number $St$", 0.01),
        (axes[1], cd, cd[0], "mean drag $C_D$", 0.005),
        (axes[2], cl, cl[0], "lift amplitude $C_L'$", 0.005),
    ):
        ax.plot(ma, v, "o-", lw=1.4, ms=5)
        ax.axhline(ref, color="k", ls=":", lw=0.9, label="reference ($Ma\\to0$ proxy)")
        ax.set_xlabel("$Ma$")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.semilogy(ma, np.abs(v - ref) / abs(ref) + 1e-12, "s--", color="C1", ms=4,
                     lw=1.0, label="relative deviation")
        ax2.axhline(limit, color="C1", ls=":", lw=0.9)
        ax2.set_ylabel("relative deviation", color="C1", fontsize=8)
        ax2.tick_params(axis="y", colors="C1", labelsize=8)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Re=100 circular cylinder: Mach independence of the hydrodynamic coefficients",
                 fontsize=11)
    fig.tight_layout()
    p = out / "fig3_cylinder_mach_independence.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_profiles(raw: Path, out: Path) -> Path:
    files = sorted(raw.glob("pressure_ma*.json"))
    if not files:
        return out / "fig4_profiles.png"
    recs = sorted((_load(f) for f in files), key=lambda r: r["mach_measured"])
    pick = [recs[0], recs[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for r in pick:
        y = np.asarray(r["y"])
        u = np.asarray(r["u_num"])
        ax = axes[0]
        ax.plot(u / max(u.max(), 1e-30), y / y.max(), "o-", ms=3, lw=1.2,
                label=f"$Ma$={r['mach_measured']:.3f}")
        ax = axes[1]
        ue = np.asarray(r["u_exact"])
        ax.plot((u - ue) / max(u.max(), 1e-30) * 1e3, y / y.max(), "o-", ms=3, lw=1.2,
                label=f"$Ma$={r['mach_measured']:.3f}")
    axes[0].plot(np.asarray(pick[0]["u_exact"]) / max(np.max(pick[0]["u_exact"]), 1e-30),
                 np.asarray(pick[0]["y"]) / np.max(pick[0]["y"]), "k--", lw=1.0,
                 label="analytic Poiseuille")
    axes[0].set_xlabel("$u/u_{max}$")
    axes[0].set_ylabel("$y/H$")
    axes[0].set_title("Developed velocity profile")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(r"$(u_{num}-u_{exact})/u_{max}\times 10^{3}$")
    axes[1].set_title("Profile error (amplified)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig4_profiles_and_error.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_acoustic(raw: Path, out: Path) -> Path:
    files = sorted(raw.glob("acoustic_ma*.json"))
    if not files:
        return out / "fig5_acoustic_filter.png"
    rec = _load(files[0])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for key, label, color in (("standard", "standard LBM", "C0"),
                              ("filtered", f"density low pass ($\\alpha$={rec['alpha']})", "C2")):
        run = rec["runs"][key]
        sig = np.asarray(run["probe_series"])
        t = np.arange(sig.size) * run["sample_every"]
        axes[0].plot(t, sig - sig.mean(), color=color, lw=1.0, label=label)
        axes[1].semilogy(t, np.maximum(np.abs(sig - sig.mean()), 1e-16), color=color,
                         lw=0.8, alpha=0.8, label=label)
    axes[0].set_xlabel("lattice step")
    axes[0].set_ylabel(r"$\rho(t)-\bar\rho$ at probe")
    axes[0].set_title("Acoustic response to a $10^{-3}$ density pulse")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(rec.get("target_filtered_drho") or 1e-5, color="k", ls=":",
                    lw=1.0, label="target $10^{-5}$")
    axes[1].set_xlabel("lattice step")
    axes[1].set_ylabel(r"$|\rho-\bar\rho|$")
    axes[1].set_title("Acoustic amplitude (log scale)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig5_acoustic_filter.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_cylinder_field(raw: Path, out: Path) -> Path:
    files = sorted(raw.glob("cylinder_ma*.json"))
    if not files:
        return out / "fig6_cylinder_field.png"
    rec = min((_load(f) for f in files), key=lambda r: r["mach_measured"])
    fld = rec["field"]
    try:
        ux_probe = np.asarray(fld["ux"], dtype=float)
        field_ok = np.isfinite(ux_probe).all()
    except (TypeError, ValueError, KeyError):
        field_ok = False
    if not field_ok:
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Re=100 cylinder field not available: the shards diverged before a\n"
                "physical field could be recorded (see fig3 and report section 3.3).",
                ha="center", va="center", fontsize=10, wrap=True)
        p = out / "fig6_cylinder_field.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        return p
    ux = np.asarray(fld["ux"], dtype=float)
    uy = np.asarray(fld["uy"], dtype=float)
    rho = np.asarray(fld["rho"], dtype=float)
    solid = np.asarray(fld["solid"])
    ny, nx = ux.shape
    vort = np.zeros_like(ux)
    vort[1:-1, 1:-1] = (uy[1:-1, 2:] - uy[1:-1, :-2]) - (ux[2:, 1:-1] - ux[:-2, 1:-1])
    vort[solid] = np.nan
    drho = (rho - 1.0)
    drho[solid] = np.nan
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 4.6), sharex=True)
    vmax = np.nanpercentile(np.abs(vort), 99)
    im0 = axes[0].imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    fig.colorbar(im0, ax=axes[0], label=r"$\omega_z$ [lattice]")
    axes[0].set_title(f"Re={rec['re']:.0f} cylinder, Ma={rec['mach_measured']:.3f}, "
                      f"St={rec['st']:.4f}, $C_D$={rec['cd_mean']:.3f}")
    dmax = np.nanpercentile(np.abs(drho), 99.9)
    im1 = axes[1].imshow(drho, origin="lower", cmap="viridis", vmin=0, vmax=dmax,
                         aspect="auto")
    fig.colorbar(im1, ax=axes[1], label=r"$\rho-\rho_0$")
    axes[1].set_title(f"density fluctuation, "
                      f"$\\max|\\rho-\\rho_0|/\\rho_0$={rec['drho_over_rho0']:.2e}")
    axes[1].set_xlabel("x [lattice]")
    for ax in axes:
        ax.set_ylabel("y [lattice]")
    fig.tight_layout()
    p = out / "fig6_cylinder_field.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts/IN130-N03"))
    args = ap.parse_args(argv)
    out = args.artifacts
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    made = [
        fig_error_scaling(raw, out),
        fig_density_order(raw, out),
        fig_cylinder_mach(raw, out),
        fig_profiles(raw, out),
        fig_acoustic(raw, out),
        fig_cylinder_field(raw, out),
    ]
    for p in made:
        print(p, p.stat().st_size if p.exists() else "MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
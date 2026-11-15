#!/usr/bin/env python
"""JSON -> figures + tables for CASE-TAYLOR-GREEN-3D round 3 (series-exact reference).

    python benchmarks/case_taylor_green_3d/make_report_series.py

Reads ``runs/case_taylor_green_3d_series/*.json`` and writes
``runs/case_taylor_green_3d_series/figures/*.png`` plus ``summary.md``.
The field-cloud figure recomputes one snapshot (N=32, T=1) because JSON is a
poor container for 3-D fields; every number in the tables comes from the JSON.
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

RUNS = ROOT / "runs" / "case_taylor_green_3d_series"
FIGS = RUNS / "figures"

from tensorlbm.tg3d_lbm_check import (  # noqa: E402
    compare_to_series,
    run_tg3d_lbm,
    series_reference,
    tg3d_parameters,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(name: str) -> dict:
    return json.loads((RUNS / f"{name}.json").read_text())


def fig_decay(main: dict) -> str:
    rows = main["rows"]
    T = [r["eddy_time"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(T, [r["energy_ratio_series"] for r in rows], "k-", lw=1.8, label="exact series")
    ax[0].plot(T, [r["energy_ratio_lbm"] for r in rows], "o", ms=3.4, label="D3Q27 LBM (N=32)")
    ax[1].plot(T, [r["enstrophy_ratio_series"] for r in rows], "k-", lw=1.8, label="exact series")
    ax[1].plot(T, [r["enstrophy_ratio_lbm"] for r in rows], "o", ms=3.4, label="D3Q27 LBM (N=32)")
    disc = main["discrimination"]
    for a, key, title in (
        (0, "energy", r"kinetic energy $E(T)/E(0)$"),
        (1, "enstrophy", r"enstrophy $\Omega(T)/\Omega(0)$"),
    ):
        a.axhline(
            disc[f"{key}_ratio_pure_viscous"], color="tab:red", ls="--", lw=1.2,
            label="single-mode viscous decay",
        )
        a.set_xlabel(r"eddy time $T = A k t$")
        a.set_ylabel(title)
        a.legend(loc="lower left", fontsize=7)
    fig.suptitle("Classic 3-D Taylor–Green vortex: nonlinear decay vs exact time-Taylor series")
    fig.tight_layout()
    p = FIGS / "decay_vs_series.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_errors(main: dict) -> str:
    rows = main["rows"]
    T = [r["eddy_time"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(T, [r["energy_rel_err_pct"] for r in rows], "o-", ms=3, label=r"$E$ error")
    ax.plot(T, [r["enstrophy_rel_err_pct"] for r in rows], "s-", ms=3, label=r"$\Omega$ error")
    ax.axhline(8.0, color="tab:red", ls="--", lw=1.0, label="case band 8%")
    ax.axhline(-8.0, color="tab:red", ls="--", lw=1.0)
    ax.axhline(3.0, color="tab:orange", ls=":", lw=1.0, label="owner stretch target 3%")
    ax.axhline(-3.0, color="tab:orange", ls=":", lw=1.0)
    ax.set_xlabel(r"eddy time $T$")
    ax.set_ylabel("relative error vs exact series  [%]")
    ax.set_title("LBM vs exact reference (N=32, Ma=0.087)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGS / "error_vs_series.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_dns(ex: dict) -> str:
    rows = ex["dns_crosscheck"]
    T = [r["eddy_time"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.semilogy(T, [abs(r["energy_rel"]) + 1e-18 for r in rows], "o-", ms=3, label=r"$E$")
    ax.semilogy(T, [abs(r["enstrophy_rel"]) + 1e-18 for r in rows], "s-", ms=3, label=r"$\Omega$")
    ax.set_xlabel(r"eddy time $T$")
    ax.set_ylabel("|series / spectral DNS − 1|")
    ax.set_title("Reference cross-check: series vs pseudo-spectral DNS (N=32)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "reference_crosscheck.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_truncation(ex: dict) -> str:
    rows = ex["truncation"]
    T = [r["eddy_time"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.semilogy(T, [max(r["energy_step_rel"], 1e-18) for r in rows], "o-", ms=3, label=r"$E$ tail")
    ax.semilogy(T, [max(r["enstrophy_step_rel"], 1e-18) for r in rows], "s-", ms=3,
                label=r"$\Omega$ tail")
    ax.axhline(1e-3, color="tab:red", ls="--", lw=1.0, label="1e-3 acceptance")
    ax.set_xlabel(r"eddy time $T$")
    ax.set_ylabel("relative contribution of last order")
    ax.set_title(f"Series truncation (order {ex['series_order']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "truncation.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_mach(mach: dict) -> str:
    rows = mach["rows"]
    ma2 = np.array([r["ma2"] for r in rows])
    e = np.array([r["energy_rel_err_pct"] for r in rows])
    fit = mach["fit_energy_err_pct_vs_ma2"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(ma2, e, "o", ms=5, label="measured (N=32, T=1)")
    xs = np.linspace(0, ma2.max() * 1.05, 50)
    ax.plot(xs, fit["slope_pct_per_ma2"] * xs + fit["intercept_pct"], "k--", lw=1.0,
            label=f"fit {fit['slope_pct_per_ma2']:.1f} %/Ma² + {fit['intercept_pct']:.2f}%")
    ax.set_xlabel(r"$Ma^2 = (A\sqrt{3})^2$")
    ax.set_ylabel(r"$E$ relative error vs exact  [%]")
    ax.set_title("Compressibility bias is O(Ma²), not a reference defect")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "mach_attribution.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def fig_grid(grid: dict) -> str:
    rows = grid["rows"]
    n = np.array([r["n"] for r in rows], dtype=float)
    e = np.array([abs(r["u_l2_rel_err_pct"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.loglog(n, e, "o-", ms=5, label="velocity L2 error")
    p = grid["observed_order_velocity_l2"]
    ref = e[0] * (n / n[0]) ** (-2.0)
    ax.loglog(n, ref, "k:", lw=1.0, label=r"slope $N^{-2}$")
    ax.set_xlabel("N (fixed A, Re, T_end)")
    ax.set_ylabel("relative L2 error  [%]")
    ax.set_title(f"Grid ladder, observed order {p:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    pth = FIGS / "grid_ladder.png"
    fig.savefig(pth)
    plt.close(fig)
    return pth.name


def fig_fields(main: dict) -> str:
    """Field cloud at T=1: the w component (zero initially ⇒ pure 3-D production)."""
    n = int(main["n"])
    a = main["parameters"]["amplitude"]
    re_ = main["parameters"]["reynolds"]
    ser = series_reference(n, amplitude=a, reynolds=re_, order=16)
    run = run_tg3d_lbm(n, amplitude=a, reynolds=re_, eddy_times=main["rows"][-1]["eddy_time"])
    t_end = run["history"][-1]["time"]
    order = min(ser.order, n // 2 - 2)
    ue, ve, we = ser.velocity_field(t_end, n=n, order=order)
    wl = run["final"]["w"]
    u_l = run["final"]["u"]
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5))
    mid = n // 2
    for a_, data, title in (
        (ax[0], we[:, :, mid], "exact series  $w$  (T=1)"),
        (ax[1], wl[:, :, mid], "D3Q27 LBM  $w$  (T=1)"),
        (ax[2], wl[:, :, mid] - we[:, :, mid], "difference  $w_{LBM}-w_{exact}$"),
    ):
        m = np.abs(data).max() if a_ is not ax[2] else np.abs(data).max()
        im = a_.imshow(data, cmap="RdBu_r", vmin=-m if m > 0 else -1, vmax=m if m > 0 else 1,
                       origin="lower")
        a_.set_title(title, fontsize=8)
        a_.set_xlabel("x"); a_.set_ylabel("y")
        fig.colorbar(im, ax=a_, shrink=0.8, format="%.1e")
    fig.suptitle(
        r"$w$ is generated only by 3-D vortex stretching — BOTH the reference and the solver "
        r"produce it", fontsize=9
    )
    fig.tight_layout()
    p = FIGS / "field_w_T1.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def write_summary(ex, main, mach, grid, verdict) -> Path:
    f = main["final"]
    d = main["discrimination"]
    lines = [
        "# CASE-TAYLOR-GREEN-3D round 3 — generated tables",
        "",
        f"Verdict: **{'PASS' if verdict['pass'] else 'FAIL'}** "
        f"({sum(1 for c in verdict['checks'] if c['pass'])}/{verdict['n_checks']} checks)",
        "",
        "## Primary metrics (N=32, A=0.05, Re=160, T=1)",
        "",
        "| quantity | LBM | exact series | rel. error |",
        "|---|---|---|---|",
        f"| E(T)/E(0) | {f['energy_ratio_lbm']:.6f} | {f['energy_ratio_series']:.6f} | {f['energy_rel_err_pct']:+.3f} % |",
        f"| Ω(T)/Ω(0) | {f['enstrophy_ratio_lbm']:.6f} | {f['enstrophy_ratio_series']:.6f} | {f['enstrophy_rel_err_pct']:+.3f} % |",
        f"| velocity L2 | — | rms {main['rows'][-1]['u_l2_ref']:.4e} | {f['u_l2_rel_err_pct']:+.3f} % |",
        f"| vorticity L2 | — | — | {f['vorticity_l2_rel_err_pct']:+.3f} % |",
        "",
        "## Reference quality",
        "",
        f"* series order {ex['series_order']}, working grid {ex['series_nfft']}³, "
        f"dealiasing margin {ex['dealias_margin']['margin']:.1f} modes",
        f"* initial energy vs analytic A²/8: {ex['energy_initial_rel_err']:+.2e}",
        f"* series vs independent spectral DNS: max |ΔE| = {ex['max_energy_rel_vs_dns']:.2e}, "
        f"max |ΔΩ| = {ex['max_enstrophy_rel_vs_dns']:.2e}",
        f"* z-invariant limit collapses to the closed form to {ex['extruded_collapse_max_abs_diff']:.2e}",
        f"* truncation at T=1: dE/E = {ex['truncation_at_main']['energy_step_rel']:.2e}, "
        f"dΩ/Ω = {ex['truncation_at_main']['enstrophy_step_rel']:.2e}",
        "",
        "## Does the case discriminate? (nonlinear signature at T=1)",
        "",
        f"* single-mode viscous decay would give ratio {d['enstrophy_ratio_pure_viscous']:.6f}",
        f"* exact series: {d['enstrophy_ratio_series']:.6f} → "
        f"{d['nonlinear_enstrophy_signature_pct']:+.2f} % above pure decay",
        f"* LBM: {d['enstrophy_ratio_lbm']:.6f} → recovers "
        f"{100 * d['lbm_recovers_fraction_of_signature']:.1f} % of that signature",
        "",
        "## Compressibility attribution (amplitude sweep, N=32, T=1)",
        "",
        "| A | Ma | Ma² | E err % | Ω err % | u L2 % |",
        "|---|---|---|---|---|---|",
    ]
    for r in mach["rows"]:
        lines.append(
            f"| {r['amplitude']:.4f} | {r['ma']:.4f} | {r['ma2']:.5f} | "
            f"{r['energy_rel_err_pct']:+.3f} | {r['enstrophy_rel_err_pct']:+.3f} | "
            f"{r['u_l2_rel_err_pct']:+.3f} |"
        )
    fit = mach["fit_energy_err_pct_vs_ma2"]
    lines += [
        "",
        f"linear fit: {fit['slope_pct_per_ma2']:.1f} %/Ma² + {fit['intercept_pct']:.2f} % "
        f"(R² = {fit['r2']:.4f})",
        "",
        "## Grid ladder (fixed A=0.05, Re=160, T=1)",
        "",
        "| N | steps | E err % | Ω err % | u L2 % | vorticity L2 % |",
        "|---|---|---|---|---|---|",
    ]
    for r in grid["rows"]:
        lines.append(
            f"| {r['n']} | {r['steps']} | {r['energy_rel_err_pct']:+.3f} | "
            f"{r['enstrophy_rel_err_pct']:+.3f} | {r['u_l2_rel_err_pct']:+.3f} | "
            f"{r['vorticity_l2_rel_err_pct']:+.3f} |"
        )
    lines += ["", f"observed order (velocity L2): **{grid['observed_order_velocity_l2']:.2f}**", "",
              "## Predeclared checks", "", "| check | value | threshold | pass |", "|---|---|---|---|"]
    for c in verdict["checks"]:
        lines.append(f"| {c['check']} | {c['value']:.4g} | {c['threshold']} | {'✅' if c['pass'] else '❌'} |")
    path = RUNS / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    ex, main_, mach, grid, verdict = (load(k) for k in ("exactness", "main", "mach_sweep",
                                                       "grid_ladder", "verdict"))
    names = [
        fig_decay(main_),
        fig_errors(main_),
        fig_dns(ex),
        fig_truncation(ex),
        fig_mach(mach),
        fig_grid(grid),
        fig_fields(main_),
    ]
    print("figures:", names)
    print("summary:", write_summary(ex, main_, mach, grid, verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
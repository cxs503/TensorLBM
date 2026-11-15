#!/usr/bin/env python
"""Figures + generated tables for CASE-TAYLOR-GREEN-3D round 4.

Reads the JSON written by :mod:`run_round4` (never hand-typed numbers) and
emits ``figures/*.png`` plus ``summary.md`` next to them::

    python benchmarks/case_taylor_green_3d/make_round4_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ART = Path(__file__).resolve().parents[2] / "artifacts" / "case_taylor_green_3d" / "round4"
FIG = ART / "figures"


def load(name: str) -> dict:
    return json.loads((ART / f"{name}.json").read_text())


def fig_decay(curves: dict) -> str:
    rows = curves["rows"]
    T = [r["eddy_time"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(T, [r["energy_ratio_series"] for r in rows], "k-", label="exact time-Taylor series")
    ax[0].plot(T, [r["energy_ratio_lbm"] for r in rows], "o", ms=4, label="D3Q27 BGK (n=32)")
    ax[0].plot(T, [np.exp(-2.0 * 0.05 * 32 / 160.0 * 3.0 * (2 * np.pi / 32) ** 2 * r["time"]) for r in rows],
               "b--", lw=1, label="single-mode viscous decay")
    ax[0].set_xlabel("eddy time $T=Ak\\,t$")
    ax[0].set_ylabel("$E(t)/E(0)$")
    ax[0].set_title("kinetic energy decay")
    ax[0].legend(fontsize=8)
    ax[1].plot(T, [r["enstrophy_ratio_series"] for r in rows], "k-", label="exact series")
    ax[1].plot(T, [r["enstrophy_ratio_lbm"] for r in rows], "o", ms=4, label="D3Q27 BGK (n=32)")
    ax[1].plot(T, [np.exp(-2.0 * 0.05 * 32 / 160.0 * 3.0 * (2 * np.pi / 32) ** 2 * r["time"]) for r in rows],
               "b--", lw=1, label="single-mode viscous decay")
    ax[1].set_xlabel("eddy time $T=Ak\\,t$")
    ax[1].set_ylabel("$\\Omega(t)/\\Omega(0)$")
    ax[1].set_title("enstrophy decay (3-D stretching signal)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "decay_vs_series.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def fig_aliasing(rep: dict) -> str:
    forced = rep["field_forced_orders"]
    orders = [r["order"] for r in forced]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(orders, [r["u_l2_rel_err_pct"] for r in forced], "o-", label="velocity L2")
    ax.plot(orders, [r["vorticity_l2_rel_err_pct"] for r in forced], "s-", label="vorticity L2")
    ax.axvline(3, color="g", ls=":", lw=1)
    ax.text(3.05, max(r["vorticity_l2_rel_err_pct"] for r in forced) * 0.95,
            "Nyquist-safe\nreference order", fontsize=8, color="g")
    ax.plot([4], [rep["round3_delivered_field_numbers"]["vorticity_l2_rel_err_pct"]], "r*", ms=14,
            label="round-3 delivered value (order 4, aliased)")
    ax.set_xlabel("forced reference truncation order (n = 32, Nyquist index 16)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("field metric vs reference truncation: the alias is not a better reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "field_reference_aliasing.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def fig_grid(grid: dict) -> str:
    rows = grid["rows"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(n, [r["band_u_l2_rel_pct"] for r in rows], "o-", label="band-projected velocity L2")
    ax[0].loglog(n, [r["band_vorticity_l2_rel_pct"] for r in rows], "s-", label="band-projected vorticity L2")
    ax[0].set_xlabel("grid n")
    ax[0].set_ylabel("relative error [%]")
    ax[0].set_title(f"M8 ladder (fixed A, Re, $T_{{end}}$)\nobserved order {grid['observed_order_band_u_l2']:.2f}")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].plot(n, [r["energy_rel_err_pct"] for r in rows], "o-", label="energy offset")
    ax[1].plot(n, [r["enstrophy_rel_err_pct"] for r in rows], "s-", label="enstrophy offset")
    ax[1].set_xlabel("grid n")
    ax[1].set_ylabel("modal relative error [%]")
    ax[1].set_title("modal offset vs grid (grid independence test)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "grid_ladder.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def fig_mach(mach: dict) -> str:
    rows = mach["rows"]
    ma2 = np.array([r["ma2"] for r in rows])
    e = np.array([r["energy_rel_err_pct"] for r in rows])
    o = np.array([r["enstrophy_rel_err_pct"] for r in rows])
    fit = mach["fit_energy_err_pct_vs_ma2"]
    xx = np.linspace(0, ma2.max() * 1.05, 50)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ma2, e, "o", label="energy offset")
    ax.plot(ma2, o, "s", label="enstrophy offset")
    ax.plot(xx, fit["slope_pct_per_ma2"] * xx + fit["intercept_pct"], "k--", lw=1,
            label=f"fit: {fit['slope_pct_per_ma2']:.2f} %/Ma$^2$, intercept {fit['intercept_pct']:.2f} %")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("$Ma^2$")
    ax.set_ylabel("relative error [%]")
    ax.set_title("M9 amplitude ladder at n=32, Re=160")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "mach_ladder.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def fig_band(band: dict) -> str:
    rows = band["rows"]
    k = [r["kmax"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(k, [r["u_l2_rel_pct"] for r in rows], "o-", label="band-projected velocity L2")
    ax[0].plot(k, [r["vorticity_l2_rel_pct"] for r in rows], "s-", label="band-projected vorticity L2")
    ax[0].axvline(band["kmax_canonical"], color="g", ls=":", lw=1)
    ax[0].text(band["kmax_canonical"] + 0.05, ax[0].get_ylim()[1] * 0.9,
               "reference exact support", fontsize=8, color="g")
    ax[0].set_xlabel("band cut-off $k_{max}$ ($|k|_\\infty$)")
    ax[0].set_ylabel("relative error [%]")
    ax[0].set_title("band-projected field metrics")
    ax[0].legend(fontsize=8)
    ax[1].plot(k, [100 * r["lbm_out_of_band_energy_fraction"] for r in rows], "d-", color="C3")
    ax[1].set_xlabel("band cut-off $k_{max}$")
    ax[1].set_ylabel("LBM energy outside band [%]")
    ax[1].set_title("unscorable content outside the exact reference band")
    fig.tight_layout()
    path = FIG / "band_metrics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name



def fig_nueff(nueff: dict) -> str:
    rows = nueff["rows"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(n, [r["energy"]["bias_pct_of_nu"] for r in rows], "o-", label="energy-implied $\\nu_{eff}/\\nu-1$")
    ax.plot(n, [r["enstrophy"]["bias_pct_of_nu"] for r in rows], "s-",
            label="enstrophy-implied $\\nu_{eff}/\\nu-1$")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("grid n")
    ax.set_ylabel("fitted viscosity bias [% of $\\nu$]")
    ax.set_title("modal offset expressed in viscosity units")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG / "nu_effective.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name

def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rep, band, grid, mach, curves = (load(n) for n in ("reproduce", "band", "grid_ladder", "mach", "curves"))
    nueff = load("nu_effective")
    verdict = load("verdict")
    names = [
        fig_decay(curves),
        fig_aliasing(rep),
        fig_grid(grid),
        fig_mach(mach),
        fig_band(band),
        fig_nueff(nueff),
    ]
    lines = ["# CASE-TAYLOR-GREEN-3D round 4 — generated tables", ""]
    lines += ["## reproduce (n=32, A=0.05, Re=160, T_end=1)", "",
              "| quantity | value |", "|---|---|",
              f"| E(T=1)/E(0) LBM | {rep['modal']['energy_ratio_lbm']:.6f} |",
              f"| E(T=1)/E(0) exact | {rep['modal']['energy_ratio_series']:.6f} |",
              f"| energy rel err | {rep['modal']['energy_rel_err_pct']:+.3f} % |",
              f"| enstrophy rel err | {rep['modal']['enstrophy_rel_err_pct']:+.3f} % |",
              f"| Nyquist-safe reference order / peak | {rep['field_nyquist_safe']['used_order']:.0f} / "
              f"{rep['field_nyquist_safe']['spectrum_peak']:.0f} (Nyquist {rep['field_nyquist_safe']['nyquist_index']:.0f}) |",
              f"| u L2 (Nyquist-safe) | {rep['field_nyquist_safe']['u_l2_rel_err_pct']:+.3f} % |",
              f"| vorticity L2 (Nyquist-safe) | {rep['field_nyquist_safe']['vorticity_l2_rel_err_pct']:+.3f} % |",
              "", "| forced order | spectrum peak | aliased | u L2 % | vorticity L2 % |", "|---|---|---|---|---|"]
    for r in rep["field_forced_orders"]:
        lines.append(f"| {r['order']} | {r['spectrum_peak']:.0f} | {r['aliased']} | "
                     f"{r['u_l2_rel_err_pct']:+.3f} | {r['vorticity_l2_rel_err_pct']:+.3f} |")
    lines += ["", "## band metrics", "", "| k_max | reference order | u L2 % | vorticity L2 % | out-of-band LBM energy % |",
              "|---|---|---|---|---|"]
    for r in band["rows"]:
        lines.append(f"| {r['kmax']} | {r['reference_field_order']:.0f} | {r['u_l2_rel_pct']:+.3f} | "
                     f"{r['vorticity_l2_rel_pct']:+.3f} | {100 * r['lbm_out_of_band_energy_fraction']:.3f} |")
    lines += ["", "## grid ladder (M8)", "",
              "| n | nu | tau | steps | band u L2 % | band vort L2 % | energy % | enstrophy % |",
              "|---|---|---|---|---|---|---|---|"]
    for r in grid["rows"]:
        lines.append(f"| {r['n']} | {r['nu']:.5f} | {r['tau']:.4f} | {r['steps']} | {r['band_u_l2_rel_pct']:+.3f} | "
                     f"{r['band_vorticity_l2_rel_pct']:+.3f} | {r['energy_rel_err_pct']:+.3f} | "
                     f"{r['enstrophy_rel_err_pct']:+.3f} |")
    lines += ["", f"observed order (band u L2): **{grid['observed_order_band_u_l2']:.2f}**; "
                  f"per-pair {['%.2f' % o for o in grid['observed_order_pairs']]}; "
                  f"modal energy offset spread {grid['energy_offset_spread_pct']:.3f} %"]
    lines += ["", "## Mach ladder (M9)", "", "| A | Ma | Ma^2 | nu | tau | steps | energy % | enstrophy % |",
              "|---|---|---|---|---|---|---|---|"]
    for r in mach["rows"]:
        lines.append(f"| {r['amplitude']} | {r['ma']:.4f} | {r['ma2']:.5f} | {r['nu']:.5f} | {r['tau']:.4f} | "
                     f"{r['steps']} | {r['energy_rel_err_pct']:+.3f} | {r['enstrophy_rel_err_pct']:+.3f} |")
    fe = mach["fit_energy_err_pct_vs_ma2"]
    lines += ["", f"fit (energy): slope {fe['slope_pct_per_ma2']:+.2f} %/Ma^2, intercept "
                  f"{fe['intercept_pct']:+.2f} %, R^2 {fe['r2']:.4f}"]
    lines += ["", "## effective viscosity (mechanism of the modal offset)", "",
              "| n | $\\tau$ | energy: $\\lambda-1$ [%] | energy err before/after [%] | enstrophy: $\\lambda-1$ [%] | enstrophy err before/after [%] |",
              "|---|---|---|---|---|---|"]
    for r in nueff["rows"]:
        lines.append(f"| {r['n']} | {r['tau']:.4f} | {r['energy']['bias_pct_of_nu']:+.3f} | "
                     f"{r['energy']['rel_err_pct_before']:+.3f} -> {r['energy']['rel_err_pct_at_nu_eff']:+.3f} | "
                     f"{r['enstrophy']['bias_pct_of_nu']:+.3f} | "
                     f"{r['enstrophy']['rel_err_pct_before']:+.3f} -> {r['enstrophy']['rel_err_pct_at_nu_eff']:+.3f} |")
    lines += ["", "## verdict", "", "| check | value | threshold | pass |", "|---|---|---|---|"]
    for c in verdict["checks"]:
        lines.append(f"| {c['check']} | {c['value']:.4g} | {c['threshold']} | {c['pass']} |")
    lines += ["", f"overall: **{'PASS' if verdict['pass'] else 'FAIL'}** ({verdict['n_checks']} checks)", "",
              "figures: " + ", ".join(names)]
    (ART / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
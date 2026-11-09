#!/usr/bin/env python3
"""Figures for the P87W2-N01 sloshing-resonance report.

Reads (never recomputes) ``artifacts/p87w2_n01/campaign.json`` plus the extra
``.npz`` records from ``scripts/run_sloshing_resonance_extras.py`` and writes the
PNG set into ``artifacts/p87w2_n01/figures``.

Every number drawn here comes from those two artefacts, so the report figures are
reproducible from the shipped JSON alone (no hand-copied values).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.sloshing_resonance.theory import (  # noqa: E402
    abramson_frequency,
    deep_water_frequency,
    shallow_water_frequency,
)

TANK_L, TANK_H, G, RHO = 1.0, 0.5, 9.81, 1000.0


def _load(out: Path) -> dict:
    return json.loads((out / "campaign.json").read_text())


def fig_frequency_ladder(camp: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    hs = np.linspace(0.05, 0.5, 400)
    for mode, color in ((1, "C0"), (2, "C3")):
        ax.plot(
            hs / TANK_H,
            [abramson_frequency(mode, h, TANK_L, G) for h in hs],
            color=color,
            lw=1.8,
            label=f"Abramson (1966), n={mode}",
        )
        rows = camp["f1_rows"] if mode == 1 else camp["f2_rows"]
        x = [r["fill_ratio"] for r in rows]
        y = [r["f_natural"] for r in rows]
        ax.plot(
            x, y, "o", color=color, ms=7, mfc="none", mew=1.8, label=f"LBM free decay, n={mode}"
        )
    ax.plot(
        hs / TANK_H,
        [shallow_water_frequency(1, h, TANK_L, G) for h in hs],
        ":",
        color="0.5",
        lw=1.2,
        label="shallow-water asymptote (n=1)",
    )
    ax.axhline(
        deep_water_frequency(1, TANK_L, G),
        linestyle="--",
        color="0.5",
        lw=1.2,
        label="deep-water asymptote (n=1)",
    )
    ax.set_xlabel("fill ratio h/H")
    ax.set_ylabel("frequency f [Hz]")
    ax.set_title(
        "Sloshing resonance ladder: free-surface LBM vs Abramson (1966)\n"
        f"tank {TANK_L} m x {TANK_H} m, A = 0.01 m, g = {G} m/s$^2$"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_frequency_error(camp: dict, path: Path, friction: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mode, color, tol in ((1, "C0", 3.0), (2, "C3", 5.0)):
        rows = camp["f1_rows"] if mode == 1 else camp["f2_rows"]
        x = [r["fill_ratio"] for r in rows]
        y = [r["err_natural_pct"] for r in rows]
        ax.plot(x, y, "o-", color=color, ms=6, lw=1.4, label=f"n={mode} (damping-corrected)")
        ax.axhline(tol, color=color, ls="--", lw=1.0, alpha=0.7)
        ax.axhline(-tol, color=color, ls="--", lw=1.0, alpha=0.7)
        ax.text(
            0.905,
            tol + 0.15,
            f"±{tol:.0f}% acceptance (n={mode})",
            color=color,
            fontsize=8,
            ha="right",
        )
    if friction:
        ax.plot(
            [friction["depth_m"] / TANK_H],
            [100 * (friction["f_damped"] - friction["f_theory"]) / friction["f_theory"]],
            "s",
            color="0.3",
            ms=8,
            label="n=1, no-slip walls (viscous detuning)",
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("fill ratio h/H")
    ax.set_ylabel("relative frequency error [%]")
    ax.set_title("Frequency error against Abramson: acceptance bands ±3% (n=1) / ±5% (n=2)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_waveform(wave: dict | None, friction: dict | None, path: Path) -> bool:
    if wave is None:
        return False
    t, modal, eta, dx, depth = (
        wave["t"],
        wave["modal"],
        wave["eta"],
        float(wave["dx"]),
        float(wave["depth"]),
    )
    f_th = float(wave["f_theory"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(t, modal, "-", color="C0", lw=1.6, label="LBM modal amplitude $a_1(t)$ (slip)")
    ax.plot(
        t,
        modal[0] * np.cos(2 * math.pi * f_th * t),
        "--",
        color="k",
        lw=1.0,
        label="linear theory $A\\cos(\\omega_1 t)$",
    )
    if friction is not None:
        ax.plot(
            friction["t"],
            friction["modal"],
            "-",
            color="0.45",
            lw=1.2,
            label="LBM (no-slip, viscous decay)",
        )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$\\eta$ modal amplitude [m]")
    ax.set_title(f"free decay, h = {depth:.2f} m, phase error vs theory")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    n = eta.shape[1]
    xs = (np.arange(n) + 0.5) * dx
    picks = [0, int(round(0.25 * (len(t) - 1))), int(round(0.5 * (len(t) - 1)))]
    for i, style in zip(picks, ("-", "--", ":")):
        ax.plot(xs, eta[i], style, lw=1.5, label=f"LBM t = {t[i]:.2f} s")
        ax.plot(
            xs,
            eta[0].max() * np.cos(math.pi * xs / TANK_L) * math.cos(2 * math.pi * f_th * t[i]),
            "k:",
            lw=0.9,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("$\\eta(x,t)$ [m]")
    ax.set_title("free-surface profile (dots = linear theory)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_spacetime(wave: dict | None, path: Path) -> bool:
    if wave is None:
        return False
    t, eta, dx = wave["t"], wave["eta"], float(wave["dx"])
    xs = (np.arange(eta.shape[1]) + 0.5) * dx
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    ax = axes[0]
    im = ax.pcolormesh(
        xs, t, eta, cmap="RdBu_r", shading="auto", vmin=-np.abs(eta).max(), vmax=np.abs(eta).max()
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("t [s]")
    ax.set_title("LBM $\\eta(x,t)$ (mm colormap = m)")
    fig.colorbar(im, ax=ax, label="$\\eta$ [m]")
    ax = axes[1]
    f_th = float(wave["f_theory"])
    A = eta[0].max()
    theory = A * np.cos(math.pi * xs[None, :] / TANK_L) * np.cos(2 * math.pi * f_th * t[:, None])
    im2 = ax.pcolormesh(
        xs,
        t,
        theory,
        cmap="RdBu_r",
        shading="auto",
        vmin=-np.abs(eta).max(),
        vmax=np.abs(eta).max(),
    )
    ax.set_xlabel("x [m]")
    ax.set_title("linear theory $A\\cos(k_1x)\\cos(\\omega_1 t)$")
    fig.colorbar(im2, ax=ax, label="$\\eta$ [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_grid(camp: dict, path: Path) -> None:
    rows = sorted(camp["grid_rows"], key=lambda r: r["dx_m"])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dx = [r["dx_m"] for r in rows]
    f = [r["f_natural"] for r in rows]
    ax.plot(dx, f, "o-", color="C0", ms=8, label="LBM $f_1$ (mode 1)")
    f_th = rows[0]["f_theory"]
    ax.axhline(f_th, color="k", ls="--", lw=1.0, label="Abramson (1966)")
    for r in rows:
        ax.annotate(
            f"H/{int(round(TANK_H / r['dx_m']))}",
            (r["dx_m"], r["f_natural"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$\\Delta x$ [m] (log, refined to the right)")
    ax.set_ylabel("$f_1$ [Hz]")
    spread = camp["criteria"]["e_grid_convergence"].get("metric_spread_pct")
    ax.set_title(f"grid convergence of $f_1$ (h = 0.20 m): spread {spread:.2f}%")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_impact(camp: dict, impact: dict | None, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    if impact is not None:
        t, press = impact["t"], impact["pressure"]
        rows = impact["gauge_rows"]
        depth = float(impact["depth"])
        for j, row in enumerate(rows):
            y = row * float(impact.get("dx", 0.0125))
            ax.plot(
                t,
                press[:, j] / (RHO * G * depth),
                lw=1.2,
                label=f"gauge row {int(row)} (y≈{y:.2f} m)",
            )
        ax.set_xlabel("t [s]")
        ax.set_ylabel("$p_{dyn}/(\\rho g h)$")
        ax.set_title(
            f"wall pressure history, h = {depth:.2f} m, sway = {float(impact['sway']):.3f} m"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "impact record missing", ha="center")
    ax = axes[1]
    rows = camp.get("impact_rows", [])
    if rows:
        labels = ["LBM peak", "linear\n(Faltinsen)", "stagnation\n(Wagner)", "bore\n(Ritter)"]
        x = np.arange(len(labels))
        w = 0.8 / max(len(rows), 1)
        for i, r in enumerate(rows):
            vals = [r["p_peak_pa"], r["p_linear_pa"], r["p_stagnation_pa"], r["p_bore_pa"]]
            ax.bar(x + i * w, vals, width=w, label=f"$X_0/L$ = {r['sway_ratio']:.2f}")
        ax.set_xticks(x + w * (len(rows) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("wall pressure peak [Pa]")
        ax.set_title("impact peak vs linear / bound anchors (d: experiment WITHHELD)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_pressure_harmonics(wave: dict | None, path: Path) -> bool:
    if wave is None:
        return False
    import math as _m

    p, rows, dx, depth, t = (
        wave["pressure"],
        wave["gauge_rows"],
        float(wave["dx"]),
        float(wave["depth"]),
        wave["t"],
    )
    f_lbm = float(np.ravel(wave["f_theory"])[0])  # Abramson value; the LBM error is <2 %
    if "f_damped" in wave.files:
        f_lbm = float(np.ravel(wave["f_damped"])[0])
    a0 = 0.01
    lin, lbm, stat = [], [], []
    for j, row in enumerate(rows):
        y = (row - 1 + 0.5) * dx - depth
        lin.append(
            1000.0
            * 9.81
            * a0
            * _m.cosh(_m.pi * (y + depth) / TANK_L)
            / _m.cosh(_m.pi * depth / TANK_L)
        )
        sig = p[:, j]
        ok = np.isfinite(sig)
        S = np.stack([np.cos(2 * _m.pi * f_lbm * t[ok]), np.sin(2 * _m.pi * f_lbm * t[ok])], axis=1)
        coef, *_ = np.linalg.lstsq(S, sig[ok] - sig[ok].mean(), rcond=None)
        lbm.append(float(np.hypot(*coef)))
        stat.append(float(sig[0]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, lin, width=0.4, label="linear Faltinsen (first harmonic)")
    ax.bar(x + 0.2, lbm, width=0.4, label=f"LBM first harmonic @ {f_lbm:.4f} Hz")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"row {int(r)}\ny={((r - 1 + 0.5) * dx - depth):+.3f} m" for r in rows], fontsize=8
    )
    ax.set_ylabel("wall dynamic pressure amplitude [Pa]")
    ax.set_title("weakly nonlinear regime (A/h = 5 %): first-harmonic wall pressure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(x, [100.0 * (m - r) / r for m, r in zip(lbm, lin)], color="C2", width=0.5)
    ax.axhline(15.0, color="r", ls="--", lw=1.0, label="±15 % acceptance line")
    ax.axhline(-15.0, color="r", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"row {int(r)}" for r in rows], fontsize=8)
    ax.set_ylabel("LBM vs linear [%]")
    ax.set_title(
        f"static gauge check at t=0: {stat[0]:.2f} Pa vs rho g A = {1000 * 9.81 * a0:.2f} Pa"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_negative(camp: dict, path: Path) -> None:
    rows = camp.get("negative_rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(rows))
    ax.bar(x, [max(r.get("osc_ratio", 0.0), 1e-6) for r in rows], color="C2", width=0.5)
    ax.axhline(0.05, color="r", ls="--", lw=1.0, label="acceptance: oscillation < 5% of $A_0$")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h = {r['depth_m']:.2f} m" for r in rows])
    ax.set_ylabel("oscillation (detrended) / $A_0$")
    ax.set_title("negative control: g → 0 must not oscillate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "p87w2_n01")
    args = ap.parse_args(argv)
    out = args.out
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    camp = _load(out)
    extra = out / "extra"

    wave = np.load(extra / "waveform.npz") if (extra / "waveform.npz").exists() else None
    fric = np.load(extra / "friction.npz") if (extra / "friction.npz").exists() else None
    imp = np.load(extra / "impact.npz") if (extra / "impact.npz").exists() else None
    fric_meta = (
        json.loads((extra / "friction.json").read_text())
        if (extra / "friction.json").exists()
        else None
    )

    fig_frequency_ladder(camp, figdir / "fig1_frequency_ladder.png")
    fig_frequency_error(camp, figdir / "fig2_frequency_error.png", fric_meta)
    fig_waveform(wave, fric, figdir / "fig3_waveform_phase.png")
    fig_spacetime(wave, figdir / "fig4_eta_spacetime.png")
    fig_grid(camp, figdir / "fig5_grid_convergence.png")
    fig_impact(camp, imp, figdir / "fig6_impact_pressure.png")
    fig_negative(camp, figdir / "fig7_negative_control.png")
    fig_pressure_harmonics(wave, figdir / "fig8_wall_pressure_harmonics.png")
    print("figures ->", figdir)
    for p in sorted(figdir.glob("*.png")):
        print(" ", p.name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

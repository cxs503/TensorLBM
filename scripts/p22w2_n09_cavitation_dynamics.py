#!/usr/bin/env python3
"""Mission P22W2-N09 -- figures of the unsteady cavitation-dynamics validation.

Every plotted number is recomputed here (or read from
``artifacts/p22w2_n09/validation.json``) -- nothing is typed in by hand.  Each
figure is registered with its sha256 in ``figures.json`` for provenance.

    PYTHONPATH=src python scripts/p22w2_n09_cavitation_dynamics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorlbm.cavitation_dynamics.kunz_fv import (  # noqa: E402
    run_dynamics,
    steady_cavity_length,
)
from tensorlbm.cavitation_dynamics.references import SHEET_REFERENCE  # noqa: E402
from tensorlbm.cavitation_dynamics.thrust import thrust_derating_vs_sigma  # noqa: E402

OUT = Path("artifacts/p22w2_n09")
FIGS: list[dict] = []


def _save(fig, name: str, caption: str) -> Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    FIGS.append({"name": name, "caption": caption, "sha256": digest, "path": str(path)})
    return path


def fig_length(data: dict, c: float) -> None:
    sigmas = np.linspace(0.62, 1.60, 17)
    model = [steady_cavity_length(float(s), n_cells=120, t_end=10.0, c_dest=c, c_prod=c)
             for s in sigmas]
    exp_sigma = SHEET_REFERENCE.sigma
    exp_l = SHEET_REFERENCE.l_cav_exp
    band_sigma = np.linspace(0.6, 1.6, 200)
    band = np.array([SHEET_REFERENCE.l_cav_curve(float(s)) for s in band_sigma])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(sigmas, model, "o-", ms=4, color="#1f77b4",
            label="Kunz-FV unsteady solve (this work)")
    ax.plot(band_sigma, band, "--", lw=1.0, color="#2ca02c",
            label="log-log curve through the measured points")
    ax.errorbar(exp_sigma, exp_l, yerr=0.10 * exp_l, fmt="s", color="#d62728", capsize=4,
                label="NACA0015 water tunnel (digitised digest)")
    ax.fill_between(band_sigma, band * 0.8, band * 1.2, color="#2ca02c", alpha=0.12,
                    label="20 % acceptance band")
    e1 = data.get("E1", {})
    ax.set_xlabel(r"cavitation number $\sigma$")
    ax.set_ylabel(r"cavity length $L_{cav}/c$")
    ax.set_title("E1: cavity length, model vs measured (max. prediction error %.1f %%)"
                 % (100.0 * e1.get("max_abs_rel_error_predictions", float("nan"))))
    ax.set_xlim(0.55, 1.65)
    ax.set_ylim(0.0, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "fig1_cavity_length_vs_sigma.png",
          "Cavity length L_cav/c against cavitation number for NACA0015 at alpha=5 deg; "
          "blue = unsteady Kunz finite-volume solve, red = digitised measured points, "
          "green band = +/-20 % acceptance band. Calibration point sigma=1.0.")


def fig_spacetime(c: float) -> None:
    res = run_dynamics(0.55, n_cells=160, t_end=18.0, c_dest=c, c_prod=c, n_snapshots=180)
    snaps = np.array(res.snapshots)          # (n_snap, n_cells)
    times = np.array(res.snap_times)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    mesh = ax.pcolormesh(res.x, times, snaps, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$t\,U/c$")
    ax.set_title(r"$\alpha_v(x,t)$ at $\sigma=0.55$ (sheet growth, break-off, cloud convection)")
    fig.colorbar(mesh, ax=ax, label=r"$\alpha_v$")

    ax2 = axes[1]
    for frac, ls in zip((0.62, 0.72, 0.82, 0.92), ("-", "--", "-.", ":")):
        i = int(frac * (snaps.shape[0] - 1))
        ax2.plot(res.x, snaps[i], ls, lw=1.2, label=f"tU/c = {times[i]:.1f}")
    ax2.set_xlabel(r"$x/c$")
    ax2.set_ylabel(r"$\alpha_v$")
    ax2.set_title("instantaneous profiles")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    _save(fig, "fig2_alpha_v_spacetime.png",
          "Space-time vapour-fraction field alpha_v(x,t) of the dynamic solve at sigma=0.55: "
          "periodic sheet growth, re-entrant-jet break-off and downstream cloud convection; "
          "right panel = instantaneous chordwise profiles.")


def fig_shedding(c: float, data: dict) -> None:
    res = run_dynamics(0.55, n_cells=140, t_end=26.0, c_dest=c, c_prod=c)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.plot(res.t, res.l_extent, color="#1f77b4", lw=1.0, label="vapour extent L_extent/c")
    ax.plot(res.t, res.l_sheet, color="#ff7f0e", lw=1.0, alpha=0.8, label="sheet L_sheet/c")
    for t_s in res.shed_times:
        ax.axvline(t_s, color="#d62728", ls=":", lw=0.7)
    ax.set_xlabel(r"$t\,U/c$")
    ax.set_ylabel(r"$L/c$")
    ax.set_title(f"break-off events (dotted): f = {res.f_shed:.3f} U/c, "
                 f"$St_L$ = {res.st_l:.3f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    y = res.vapour_mass - np.mean(res.vapour_mass)
    skip = int(0.2 * y.size)
    win = np.hanning(y[skip:].size)
    spec = np.abs(np.fft.rfft(y[skip:] * win))
    freq = np.fft.rfftfreq(y[skip:].size, d=float(res.t[1] - res.t[0]))
    ax2 = axes[1]
    ax2.semilogy(freq, spec, color="#1f77b4", lw=1.0)
    ax2.axvline(res.f_shed, color="#d62728", ls="--", lw=1.0,
                label=f"event frequency {res.f_shed:.3f}")
    ax2.axvline(res.meta.get("f_from_signal", 0.0), color="#2ca02c", ls=":", lw=1.0,
                label="spectral fundamental %.3f" % res.meta.get("f_from_signal", 0.0))
    ax2.set_xlim(0.0, 1.5)
    ax2.set_xlabel("f c/U")
    ax2.set_ylabel("|FFT(vapour mass)|")
    ax2.set_title("shedding spectrum and $St_L$ band 0.2-0.3")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    cases = data.get("E7", {}).get("cases", [])
    if cases:
        txt = "\n".join(f"sigma={x['sigma']:.2f}: shedding={x['shedding']}, "
                        f"St_L={x['st_l']:.3f}" for x in cases)
        ax2.text(0.52, 0.62, txt, transform=ax2.transAxes, fontsize=7,
                 bbox=dict(fc="white", ec="0.7", alpha=0.85))
    _save(fig, "fig3_shedding_frequency.png",
          "Cloud-shedding dynamics at sigma=0.55: vapour extent and sheet length against time "
          "with the break-off events, and the spectrum of the vapour mass with the measured "
          "frequency and the Strouhal number.")


def fig_thrust(data: dict) -> None:
    sig = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 2.2])
    area = thrust_derating_vs_sigma(sig, loss_model="area_fraction")
    plat = thrust_derating_vs_sigma(sig, loss_model="plateau")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(sig, -np.asarray(area["derating"]) * 100.0, "o-", color="#1f77b4",
            label="area-fraction convention (empirical chain)")
    ax.plot(sig, -np.asarray(plat["derating"]) * 100.0, "s--", color="#ff7f0e",
            label="plateau (section-resolved) upper bound")
    ax.axhspan(10.0, 15.0, color="#2ca02c", alpha=0.15, label="empirical 10-15 % band")
    e3 = data.get("E3", {})
    ax.set_xlabel(r"$\sigma_{0.7R}$")
    ax.set_ylabel(r"$\Delta K_T/K_{T0}$  [%]")
    ax.set_title("E3: thrust derating at $\\sigma_{0.7R}=0.6$: %.1f %%"
                 % (100.0 * abs(e3.get("derating_area_fraction", float("nan")))))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    from tensorlbm.cavitation_dynamics.thrust import propeller_design, thrust_derating

    res = thrust_derating(propeller_design(0.6), loss_model="area_fraction")
    ax2 = axes[1]
    ax2.plot(np.degrees(res["theta"]), 100.0 * np.asarray(res["derating_azimuth"]),
             color="#1f77b4", lw=1.4)
    ax2.set_xlabel("blade azimuth [deg] (90 deg = deepest)")
    ax2.set_ylabel(r"$\Delta K_T/K_{T0}$  [%]")
    ax2.set_title("blade-position resolution at $\\sigma_{0.7R}=0.6$")
    ax2.grid(alpha=0.3)
    _save(fig, "fig4_thrust_derating.png",
          "Blade-element thrust derating against the reference cavitation number in the two "
          "declared loss conventions (left) and its azimuthal, i.e. blade-position, resolution "
          "at sigma_0.7R = 0.6 (right).")


def fig_budget(c: float, data: dict) -> None:
    sigmas = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    for s in sigmas:
        res = run_dynamics(float(s), n_cells=120, t_end=10.0, c_dest=c, c_prod=c, shedding=False)
        ax.plot(res.x, res.alpha_final, lw=1.3, label=f"sigma={s:.2f}")
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"$\alpha_v$")
    ax.set_title("converged vapour profiles (Kunz-FV solve)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    e6 = data.get("E6", {})
    labels = ["evaporation", "condensation", "outflow", "limiter"]
    vals = [e6.get("mass_evap_kg_per_m2", 0.0), e6.get("mass_cond_kg_per_m2", 0.0),
            e6.get("mass_out_kg_per_m2", 0.0), e6.get("mass_clip_kg_per_m2", 0.0)]
    ax2 = axes[1]
    ax2.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"])
    ax2.set_ylabel("vapour mass per unit depth [kg/m$^2$]")
    ax2.set_title("E6: steady-state phase-change budget, residual %.1e"
                  % e6.get("budget_residual_rel", float("nan")))
    ax2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.4g}", ha="center", va="bottom", fontsize=8)
    _save(fig, "fig5_profiles_and_budget.png",
          "Left: converged vapour-fraction profiles at five cavitation numbers. Right: "
          "steady-state evaporation/condensation budget of the transport solve (the residual "
          "is the conservation check E6).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefact", default=str(OUT / "validation.json"))
    ap.add_argument("--skip", default="", help="comma separated figure numbers to skip")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.artefact).read_text()) if Path(args.artefact).exists() else {}
    c = float(data.get("calibration", {}).get("c", 800.0))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if "1" not in skip:
        fig_length(data, c)
    if "2" not in skip:
        fig_spacetime(c)
    if "3" not in skip:
        fig_shedding(c, data)
    if "4" not in skip:
        fig_thrust(data)
    if "5" not in skip:
        fig_budget(c, data)

    (OUT / "figures.json").write_text(json.dumps({"kunz_constant": c, "figures": FIGS}, indent=2))
    print(f"{len(FIGS)} figures -> {OUT}")


if __name__ == "__main__":
    main()
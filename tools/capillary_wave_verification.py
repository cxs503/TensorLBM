"""Reproduce the capillary-wave verification bundle (artifacts/capillary_wave_20261109).

Everything in ``results.json`` and in the figures is produced here; no number in the
report or the module docstring is hand-copied from a previous run.

Campaign (all parameters in lattice units):

* **static Laplace law** -- cylindrical drop, ``Delta p = sigma / R``; recovers the
  model's *actual* surface tension from a configuration that is in static
  equilibrium, i.e. independent of any wave dynamics.
* **dispersion scan** -- standing capillary waves at ``k = 2 pi m / nx`` for several
  modes and interface widths; ``omega`` and the amplitude decay rate ``gamma`` are
  fitted from the interface height time series and compared with

      omega_th = sqrt(sigma k^3 / (rho_l + rho_g)),   gamma_th = 2 nu k^2 .

* **controls** -- amplitude halved (linearity), viscosity halved (damping scaling),
  and the AC mobility varied with the hydrodynamic viscosity held fixed (does the
  phase-field relaxation leak into the wave frequency?).

Usage
-----
    python3 tools/capillary_wave_verification.py [--out DIR] [--jobs N] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorlbm.capillary_wave import (
    CapillaryWaveConfig,
    damping_theory,
    dispersion_theory,
    run_capillary_wave,
    run_flat_interface,
    run_laplace_drop,
)

R2_GATE = 0.99  # fit-quality gate: below this the record is not a single decaying mode


# ---------------------------------------------------------------------------
# Static Laplace law (cylindrical drop, uniform in z)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Capillary-wave cases
# ---------------------------------------------------------------------------


def _wave_cases(quick: bool) -> list[dict[str, Any]]:
    base = dict(
        ny=32,
        nx=64,
        nz=2,
        mode=2,
        amplitude=0.5,
        sigma=0.05,
        nu=0.05,
        width=4.0,
        n_steps=2500,
        sample_every=10,
    )
    cases: list[dict[str, Any]] = [
        {"tag": "reference_m2_W4", **{**base, "n_steps": 3000}},
        {"tag": "mode3", **{**base, "mode": 3}},
        {"tag": "width2", **{**base, "width": 2.0}},
        {"tag": "width3", **{**base, "width": 3.0}},
        {"tag": "width6", **{**base, "width": 6.0}},
        {"tag": "amplitude_halved", **{**base, "amplitude": 0.25}},
        {"tag": "viscosity_halved", **{**base, "nu": 0.03}},
        {"tag": "mobility_low", **{**base, "mobility": 0.02}},
        {"tag": "mobility_high", **{**base, "mobility": 0.20}},
    ]
    if quick:
        cases = [cases[0], cases[5]]
        for c in cases:
            c["n_steps"] = 1200
    return cases


def _run_wave(task: dict[str, Any]) -> dict[str, Any]:
    tag = task.pop("tag")
    mobility = task.pop("mobility", None)
    cfg = CapillaryWaveConfig(**task)
    torch.set_num_threads(1)
    if mobility is None:
        res = run_capillary_wave(cfg)
    else:
        # same run, but with the AC mobility decoupled from the hydrodynamic viscosity
        res = run_capillary_wave_with_mobility(cfg, mobility)
    res["tag"] = tag
    return res


def run_capillary_wave_with_mobility(
    config: CapillaryWaveConfig, mobility: float
) -> dict[str, Any]:
    """``run_capillary_wave`` with the AC mobility set independently of ``nu``."""
    import tensorlbm.capillary_wave as cw
    from tensorlbm.cac_lbm import CACSim

    config.validate()
    dev = torch.device("cpu")
    phi0 = cw.perturbed_layer_phi(config)
    sim = CACSim(
        (config.nz, config.ny, config.nx),
        rho_l=config.rho_l,
        rho_g=config.rho_g,
        mu_l=config.nu,
        mu_g=config.nu,
        sigma=config.sigma,
        width=config.width,
        mobility=mobility,
        axes="ppp",
        device=dev,
        dtype=torch.float64 if config.dtype == "float64" else torch.float32,
    )
    sim.initialize(phi0)
    k = config.k
    omega_th = dispersion_theory(config.sigma, k, config.rho_l, config.rho_g)
    steps, amps, phis = [], [], []
    blow_up = None
    for step in range(1, config.n_steps + 1):
        sim.step()
        if step % config.sample_every == 0 or step == 1:
            phi = sim.f.sum(dim=0)
            a = cw.modal_amplitude(phi, k)
            steps.append(step)
            amps.append(a)
            phis.append([float(phi.min().item()), float(phi.max().item())])
            if not math.isfinite(a) or abs(a) > 5.0 * config.amplitude:
                blow_up = step
                break
    t = np.asarray(steps, dtype=float)
    a = np.asarray(amps, dtype=float)
    fit = None
    if blow_up is None and a.size >= 4:
        start = t >= max(5.0 * config.sample_every, 0.1 * t[-1])
        try:
            fit = cw.damped_sinusoid_fit(
                t[start],
                a[start],
                omega_bounds=(0.25 * omega_th, 2.5 * omega_th),
                gamma_bounds=(0.0, max(6.0 * config.nu * k * k, 0.01)),
            )
        except ValueError:
            fit = None
    res: dict[str, Any] = {
        "config": {**asdict(config), "mobility": mobility},
        "k": k,
        "steepness": config.steepness,
        "k_times_width": k * config.width,
        "omega_theory": omega_th,
        "gamma_theory": damping_theory(config.nu, k),
        "blow_up_step": blow_up,
        "phi_range": [min(p[0] for p in phis), max(p[1] for p in phis)],
        "steps": steps,
        "amplitudes": amps,
        "fit": fit,
    }
    if fit is not None:
        res["omega_meas"] = fit["omega"]
        res["gamma_meas"] = fit["gamma"]
        res["rel_err_omega"] = fit["omega"] / omega_th - 1.0
        res["rel_err_gamma"] = fit["gamma"] / res["gamma_theory"] - 1.0
        res["sigma_eff"] = fit["omega"] ** 2 * (config.rho_l + config.rho_g) / k**3
        res["sigma_ratio"] = res["sigma_eff"] / config.sigma
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot_series(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(res["steps"], dtype=float)
    a = np.asarray(res["amplitudes"], dtype=float)
    fit = res["fit"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(t, a, "o", ms=2.5, color="tab:blue", label="LBM interface amplitude $a(t)$")
    if fit is not None:
        tt = np.linspace(t.min(), t.max(), 800)
        env = np.exp(-fit["gamma"] * tt)
        pred = env * fit["amplitude"] * np.cos(fit["omega"] * tt + fit["phase"])
        ax.plot(
            tt,
            pred,
            "-",
            lw=1.2,
            color="tab:red",
            label=(
                f"damped-sinusoid fit: $\\omega$={fit['omega']:.4f}, "
                f"$\\gamma$={fit['gamma']:.4f}, $R^2$={fit['r2']:.4f}"
            ),
        )
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("step $t$")
    ax.set_ylabel("modal amplitude $a$ (lattice units)")
    ax.set_title(f"Standing capillary wave, k={res['k']:.4f}, W={res['config']['width']}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_dispersion(disp: list[dict[str, Any]], sigma: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [
        r
        for r in disp
        if r.get("fit")
        and (r["fit"]["r2"] or 0.0) >= R2_GATE
        and "mobility" not in r["config"]
        and abs(r["config"]["nu"] - 0.05) < 1e-12
        and abs(r["config"]["width"] - 4.0) < 1e-12
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    kk = np.linspace(0.05, 0.35, 200)
    ax.plot(
        kk,
        np.sqrt(sigma * kk**3 / 2.0),
        "-",
        color="k",
        lw=1.2,
        label=r"theory $\omega=\sqrt{\sigma k^3/(\rho_l+\rho_g)}$, $\sigma$ input",
    )
    if len(good) >= 2:
        ratios = [r["sigma_ratio"] for r in good]
        fac = float(np.mean(ratios))
        ax.plot(
            kk,
            np.sqrt(fac * sigma * kk**3 / 2.0),
            "--",
            color="tab:red",
            lw=1.2,
            label=rf"theory with $\sigma_{{\rm eff}}={fac:.2f}\sigma$ (measured)",
        )
    for r in good:
        ax.plot(r["k"], r["omega_meas"], "o", ms=7, color="tab:blue")
    for r in disp:
        if r.get("fit") and r not in good:
            ax.plot(r["k"], r["omega_meas"], "x", ms=6, color="tab:gray")
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel(r"frequency $\omega$")
    ax.set_title("Capillary-wave dispersion, CAC phase-field LBM (filled = W=4, $\\nu$=0.05)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_controls(disp: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = next(r for r in disp if r["tag"] == "reference_m2_W4")
    groups = [
        ("reference", [ref]),
        ("A halved", [r for r in disp if r["tag"] == "amplitude_halved"]),
        ("W=2", [r for r in disp if r["tag"] == "width2"]),
        ("W=3", [r for r in disp if r["tag"] == "width3"]),
        ("W=6", [r for r in disp if r["tag"] == "width6"]),
        ("nu=0.03", [r for r in disp if r["tag"] == "viscosity_halved"]),
        ("M=0.02", [r for r in disp if r["tag"] == "mobility_low"]),
        ("M=0.20", [r for r in disp if r["tag"] == "mobility_high"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs, ys, labels = [], [], []
    for i, (name, rows) in enumerate(groups):
        if not rows or not rows[0].get("rel_err_omega"):
            continue
        xs.append(i)
        ys.append(100.0 * rows[0]["rel_err_omega"])
        labels.append(name)
    ax.bar(xs, ys, color=["tab:red" if abs(y) > 15 else "tab:green" for y in ys], alpha=0.85)
    ax.axhline(0.0, color="k", lw=0.7)
    ax.axhline(-15.0, color="tab:red", ls=":", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$\omega_{\rm meas}/\omega_{\rm th}-1$  (%)")
    ax.set_title("Frequency error vs the sharp-interface theory (dotted: 15% acceptance bar)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_flat_interface(flat: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ["initial profile", "relaxed profile"]
    ratios = [flat["initial"]["sigma_ratio"], flat["relaxed"]["sigma_ratio"]]
    ax.bar(bars, ratios, color=["tab:gray", "tab:blue"], alpha=0.85)
    ax.axhline(1.0, color="k", lw=1.0, ls="-", label=r"$\sigma$ input")
    ax.axhline(1.15, color="tab:red", ls=":", lw=1.0)
    ax.axhline(0.85, color="tab:red", ls=":", lw=1.0, label="15% bar")
    for i, r in enumerate(ratios):
        ax.annotate(
            f"{r:.3f}", (i, r), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9
        )
    ax.set_ylabel(r"$\sigma_{\rm profile}/\sigma_{\rm input}$")
    ax.set_title("Static surface tension from the interface energy integral")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_laplace(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    inv_r = np.array([1.0 / r["radius_eff"] for r in rows])
    dp = np.array([r["delta_p"] for r in rows])
    ax.plot(inv_r, dp, "o", ms=8, color="tab:blue", label=r"LBM: $\Delta p$ vs $1/R$")
    sigma_in = rows[0]["sigma_input"]
    xx = np.linspace(0.0, inv_r.max() * 1.15, 50)
    ax.plot(xx, sigma_in * xx, "-", color="k", lw=1.2, label=rf"$\sigma$ input = {sigma_in}")
    fit = float(np.sum(inv_r * dp) / np.sum(inv_r * inv_r))
    ax.plot(xx, fit * xx, "--", color="tab:red", lw=1.2, label=rf"fit $\sigma={fit:.4f}$")
    for r in rows:
        ax.annotate(
            f"R={r['radius_input']:.0f}",
            (1.0 / r["radius_eff"], r["delta_p"]),
            textcoords="offset points",
            xytext=(6, -10),
            fontsize=8,
        )
    ax.set_xlabel(r"curvature $1/R$")
    ax.set_ylabel(r"pressure jump $\Delta p$")
    ax.set_title("Static Laplace law of the CAC model (cylindrical drop)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/capillary_wave_20261109")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.time()

    # -- static surface tension --------------------------------------------
    # (a) interface-energy integral of a *planar* interface: no slow viscous mode
    #     is involved, so 4000 steps is fully converged;
    flat = run_flat_interface(sigma=0.05, width=4.0, nu=0.05, n_steps=4000)
    (out / "flat_interface.json").write_text(json.dumps(flat, indent=1), encoding="utf-8")

    # (b) Laplace law of a cylindrical drop.  Flagged relaxation-limited: the
    #     pressure jump approaches equilibrium on the viscous time R^2/nu
    #     (>= 1280 steps at R=8, nu=0.05), and the runs below are shorter than
    #     that, so these are *lower bounds* on the static surface tension.
    radii = [8.0, 12.0] if args.quick else [8.0, 12.0, 16.0]
    laplace = []
    for R in radii:
        row = run_laplace_drop(R)
        row["viscous_time_R2_over_nu"] = R * R / row["nu"]
        row["relaxation_limited"] = row["viscous_time_R2_over_nu"] > 0.75 * 3000
        laplace.append(row)
    (out / "laplace.json").write_text(json.dumps(laplace, indent=1), encoding="utf-8")

    # -- capillary waves ----------------------------------------------------
    cases = _wave_cases(args.quick)
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            disp = pool.map(_run_wave, cases)
    else:
        disp = [_run_wave(c) for c in cases]

    # -- checks / verdict ---------------------------------------------------
    ref = next(r for r in disp if r["tag"] == "reference_m2_W4")
    good = [r for r in disp if r.get("fit") and r["fit"]["r2"] >= R2_GATE]
    sigma_fit = float(
        np.sum([r["sigma_measured"] / r["radius_eff"] for r in laplace])
        / np.sum([1.0 / r["radius_eff"] ** 2 for r in laplace])
    )
    checks = {
        "static_profile_sigma_ratio_relaxed": flat["relaxed"]["sigma_ratio"],
        "static_profile_sigma_ratio_initial": flat["initial"]["sigma_ratio"],
        "static_profile_width_measured": flat["relaxed"]["width_measured"],
        "static_profile_width_analytic": flat["relaxed"]["width_analytic"],
        "static_profile_passed_15pct": abs(flat["relaxed"]["sigma_ratio"] - 1.0) <= 0.15,
        "laplace_sigma_ratios": [r["sigma_ratio"] for r in laplace],
        "laplace_relaxation_limited": [r["relaxation_limited"] for r in laplace],
        "laplace_sigma_fit": sigma_fit,
        "reference_omega_rel_err": ref.get("rel_err_omega"),
        "dynamic_passed_15pct": abs(ref.get("rel_err_omega", 1.0)) <= 0.15,
        "reference_sigma_ratio": ref.get("sigma_ratio"),
        "reference_fit_r2": ref["fit"]["r2"] if ref.get("fit") else None,
        "runs_total": len(disp),
        "runs_passing_fit_gate": len(good),
        "runs_excluded": [r["tag"] for r in disp if r not in good],
        "all_stable": all(r["blow_up_step"] is None for r in disp),
        "tolerance": 0.15,
    }

    payload = {
        "meta": {
            "torch_version": torch.__version__,
            "threads": torch.get_num_threads(),
            "wall_time_s": round(time.time() - started, 1),
            "note": "all numbers produced programmatically by this script",
            "model": "tensorlbm.cac_lbm (CAC phase-field, D3Q19, periodic)",
        },
        "static_flat_interface": flat,
        "static_laplace": laplace,
        "dispersion": [{k: v for k, v in r.items()} for r in disp],
        "checks": checks,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    _plot_series(ref, figs / "amplitude_series.png")
    _plot_dispersion(disp, ref["config"]["sigma"], figs / "dispersion.png")
    _plot_controls(disp, figs / "controls.png")
    _plot_laplace(laplace, figs / "laplace_static.png")
    _plot_flat_interface(flat, figs / "flat_interface_sigma.png")

    print(json.dumps(checks, indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    raise SystemExit(main())

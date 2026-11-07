"""Figures for the wave-tank benchmark (matplotlib, Agg backend).

Reads ``artifacts/wave_tank/results.json`` plus the per-case ``.npz`` records and
writes PNGs into ``artifacts/wave_tank/figures/``.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .campaign import wave_of_config  # noqa: E402

__all__ = ["make_figures"]

COLORS = {
    0.02: "#1f77b4",
    0.08: "#d62728",
}


def _load(out_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(out_dir, "results.json")) as fh:
        res = json.load(fh)
    npz = {}
    for name in res["cases"]:
        p = os.path.join(out_dir, f"{name}.npz")
        if os.path.exists(p):
            npz[name] = np.load(p, allow_pickle=True)
    return res, npz


def make_figures(out_dir: str = os.path.join("artifacts", "wave_tank")) -> list[str]:
    """Generate all benchmark figures; returns the written paths."""
    res, npz = _load(out_dir)
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    written: list[str] = []

    def save(fig, name: str) -> None:
        p = os.path.join(figdir, name)
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    cases = res["cases"]

    # ---- 1. free-surface field snapshot + analytic overlay -----------------
    for name in ("kh1.5_eps0.08", "kh3.0_eps0.08"):
        if name not in npz:
            continue
        d = npz[name]
        cfg = json.loads(str(d["config"]))
        w = wave_of_config(cfg)
        phi = np.asarray(d["phi"], dtype=float)
        eta_meas = np.asarray(d["eta_final"], dtype=float)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1.0]})
        ax = axes[0]
        ny, nx = phi.shape
        ax.pcolormesh(np.arange(nx), np.arange(ny), phi, cmap="Blues", vmin=0.0, vmax=1.0,
                      shading="auto")
        ax.axhline(cfg["h"], color="0.4", lw=0.8, ls="--", label="still water level")
        ax.set_ylabel("y [cells]")
        ax.set_ylim(0, min(ny, cfg["h"] + 6))
        ax.set_title(f"{name}: phase-field free surface (phi), t = end of run")
        ax.legend(loc="upper right", fontsize=8)
        ax2 = axes[1]
        x = np.arange(nx)
        ax2.plot(x, eta_meas, "k-", lw=0.9, label="gauge (column integral)")
        n_in = int(round(cfg["lam_in"] * w.wavelength))
        n_out = int(round(cfg["lam_out"] * w.wavelength))
        band = np.zeros(nx, dtype=bool)
        band[n_in:nx - n_out] = True
        ax2.plot(x[band], eta_meas[band], color="#d62728", lw=1.4,
                 label="propagation stretch (compared in C5)")
        ax2.axvspan(0, n_in, color="0.85", alpha=0.6, label="wavemaker zone")
        if cfg["absorb"]:
            ax2.axvspan(nx - n_out, nx, color="0.65", alpha=0.6, label="absorber zone")
        ax2.set_xlabel("x [cells]")
        ax2.set_ylabel("eta [cells]")
        ax2.legend(loc="upper right", fontsize=7, ncol=2)
        save(fig, f"fig_field_{name}.png")

    # ---- 2. gauge time series vs analytic ---------------------------------
    if "kh1.5_eps0.08" in npz:
        d = npz["kh1.5_eps0.08"]
        cfg = json.loads(str(d["config"]))
        w = wave_of_config(cfg)
        t = np.asarray(d["t"], dtype=float)
        eta = np.asarray(d["probe_eta"], dtype=float)
        warm = float(d["warm_steps"])
        cols = np.asarray(d["probe_cols"], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4))
        tt = np.linspace(warm, t[-1], 2000)
        for j, x in enumerate(cols):
            ax.plot(t, eta[:, j], lw=1.0, label=f"gauge x = {int(x)} cells (run)")
            ax.plot(tt, w.a * np.cos(w.k * x - w.omega * tt), "--", lw=0.9,
                    label=f"analytic Stokes-2 (x = {int(x)})")
        ax.axvline(warm, color="0.5", ls=":", label="measurement window start")
        ax.set_xlabel("t [steps]")
        ax.set_ylabel("eta [cells]")
        ax.set_title("kh=1.5, eps=0.08: gauge records vs analytic incident wave")
        ax.legend(fontsize=7, ncol=2)
        save(fig, "fig_probes.png")

    # ---- 3. dispersion ----------------------------------------------------
    line = sorted([(r["theory"]["kh"], r["metrics"]["c_measured"], r["theory"]["celerity"], n)
                   for n, r in cases.items()
                   if abs(r["config"]["eps"] - 0.08) < 1e-9 and "_" not in n.split("eps")[-1]])
    if line:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        kh = np.array([l[0] for l in line])
        c_err = np.array([abs(l[1] - l[2]) / l[2] * 100 for l in line])
        ax.plot(kh, c_err, "o-", label="measured vs c = sqrt((g/k)tanh kh)")
        ax.axhline(1.0, color="r", ls="--", label="C1 tolerance 1 %")
        for k_, e_, n_ in zip(kh, c_err, [l[3] for l in line]):
            ax.annotate(n_, (k_, e_), textcoords="offset points", xytext=(4, 6), fontsize=7)
        ax.set_xlabel("kh")
        ax.set_ylabel("phase-speed error [%]")
        ax.set_title("C1: dispersion relation")
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)
        save(fig, "fig_dispersion.png")

    # ---- 4. reflection ----------------------------------------------------
    base = {n: r for n, r in cases.items() if "_noreflect" not in n and "_dx2" not in n}
    if base:
        fig, ax = plt.subplots(figsize=(7.4, 4))
        names = sorted(base)
        kr = [base[n]["metrics"]["kr"] * 100 for n in names]
        neg = [r["metrics"]["kr"] * 100 for n, r in cases.items() if "_noreflect" in n]
        labels = [f"{base[n]['config']['kh']:.1f}/{base[n]['config']['eps']:.2f}" for n in names]
        ax.bar(range(len(names)), kr, color="#4c78a8", label="absorbing outlet")
        if neg:
            ax.bar([len(names)], neg, color="#e45756", label="negative control (no absorber)")
            labels.append("ctrl")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.axhline(5.0, color="r", ls="--", label="C3 tolerance 5 %")
        ax.axhline(30.0, color="k", ls=":", label="negative control > 30 %")
        ax.set_ylabel("reflection coefficient KR [%]")
        ax.set_title("C3: two-point (Goda & Suzuki) reflection coefficient")
        ax.legend(fontsize=8)
        save(fig, "fig_reflection.png")

    # ---- 5. harmonics -----------------------------------------------------
    if base:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        names = sorted(base)
        idx = np.arange(len(names))
        for ax, key, tol, title in (
            (axes[0], "a1_error_rel", 3.0, "C2a: first harmonic (|da1|/a1)"),
            (axes[1], "a2_error_rel", 10.0, "C2b: second harmonic (|da2|/a2)"),
        ):
            vals = [base[n]["metrics"][key] * 100 for n in names]
            ax.bar(idx, vals, color="#72b7b2")
            ax.axhline(tol, color="r", ls="--", label=f"tolerance {tol:.0f} %")
            ax.set_xticks(idx)
            ax.set_xticklabels([f"{base[n]['config']['kh']:.1f}/{base[n]['config']['eps']:.2f}"
                                for n in names], rotation=30, fontsize=8)
            ax.set_ylabel("error [%]")
            ax.set_title(title)
            ax.legend(fontsize=8)
        save(fig, "fig_harmonics.png")

    # ---- 6. mass drift ----------------------------------------------------
    if npz:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in sorted(npz):
            d = npz[name]
            cfg = json.loads(str(d["config"]))
            w = wave_of_config(cfg)
            t = np.asarray(d["t"], dtype=float)
            m = np.asarray(d["mass"], dtype=float)
            m0 = float(d["mass0"])
            ax.plot(t / w.period, (m - m0) / m0 * 100, lw=1.1, label=name)
        ax.set_xlabel("t [wave periods]")
        ax.set_ylabel("liquid mass drift [%]")
        ax.set_title("C4: phase-field mass conservation")
        ax.legend(fontsize=7, ncol=2)
        save(fig, "fig_mass.png")

    return written
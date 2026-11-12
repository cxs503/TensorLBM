#!/usr/bin/env python
"""Figures for the Womersley pulsatile case (CASE-LBM-WOMERSLEY-PULSATILE).

Reads the benchmark artifacts under artifacts/womersley_pulsatile_alpha5/ and
writes PNGs there.  All numbers come from the artifact JSON (no hand-typed
values).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "womersley_pulsatile_alpha5"


def load(name: str) -> dict:
    return json.loads((ART / f"{name}.json").read_text())


def fig_profiles():
    """Channel phase-averaged profiles at 4 phase instants, num vs analytic."""
    d = load("channel_H59_a5_tau0.8")
    prof = d["profile"]
    phases = np.array(prof["phase_centers"])
    u_num = np.array(prof["profile_phase_averaged"])  # (H, n_bins)
    z_ana = np.array(prof["profile_ana_phasor_re"]) + 1j * np.array(prof["profile_ana_phasor_im"])
    y = np.array(prof["y"])
    yp = y / d["H_eff"] - 0.5
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for frac, color in zip((0.0, 0.25, 0.5, 0.75), ("C0", "C1", "C2", "C3")):
        k = int(frac * len(phases))
        ua = np.real(z_ana * np.exp(1j * phases[k]))
        ax[0].plot(np.array(u_num)[:, k], yp, color=color, lw=1.2,
                   label=f"phase {math.degrees(phases[k]) % 360:.0f} deg")
        ax[0].plot(ua, yp, "--", color=color, lw=1.0)
    ax[0].set_xlabel("u_x (lattice units)")
    ax[0].set_ylabel("(y - H/2)/H")
    ax[0].set_title("channel H=59, alpha=5: phase-averaged profiles\nsolid=LBM, dashed=analytic")
    ax[0].legend(fontsize=7)
    amp = np.array(prof["amp_ratio"])
    ax[1].plot(yp, amp, "o-", ms=3, label="LBM |U|/|U_ana|")
    ax[1].axhline(1.0, color="k", ls=":", lw=0.8)
    ax2 = ax[1].twinx()
    ax2.plot(yp, np.array(prof["phase_lag_num_deg"]), "s-", ms=3, color="C3",
             label="LBM lag (deg)")
    ax2.plot(yp, np.array(prof["phase_lag_ana_deg"]), "--", color="C3", lw=1.0)
    ax[1].set_xlabel("(y - H/2)/H")
    ax[1].set_ylabel("amplitude ratio")
    ax2.set_ylabel("phase lag (deg)")
    ax[1].set_title("complex amplitude ratio and phase lag per row")
    fig.tight_layout()
    fig.savefig(ART / "fig1_channel_profiles.png", dpi=150)
    plt.close(fig)


def fig_waveforms():
    """Centreline phase-averaged waveforms, both geometries and both grids."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    cases = [
        ("channel_H59_a5_tau0.8", "channel H=59"),
        ("channel_H119_a5_tau0.8", "channel H=119"),
        ("pipe_R20.5_a5_tau0.8", "pipe R=20.5"),
    ]
    for ax, (name, title) in zip(axes, cases):
        d = load(name)
        ph = np.degrees(np.array(d["profile"]["phase_centers"]))
        ua = np.array(d["profile"]["centerline_ana_waveform"])
        un = np.array(d["profile"]["centerline_phase_averaged"])
        ax.plot(ph, ua, "k--", lw=1.2, label="analytic")
        ax.plot(ph, un, "o-", ms=2.5, lw=1.0, label="LBM phase-averaged")
        c = d["centerline"]
        ax.set_title(
            f"{title}\namp err {c['amplitude_rel_pct']:.3f}%, phase err {c['phase_rel_pct']:.3f}%",
            fontsize=8.5,
        )
        ax.set_xlabel("phase (deg)")
        ax.set_ylabel("u_x / (|U(0)|)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ART / "fig2_centerline_waveforms.png", dpi=150)
    plt.close(fig)


def fig_pipe_radial():
    d = load("pipe_R20.5_a5_tau0.8")
    prof = d["profile"]
    r = np.array(prof["r"]) / d["R"]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].plot(r, np.array(prof["amp_ratio_masked"]), "o-", ms=3, label="LBM |U|")
    zr = np.array(prof["radial_ana_phasor_re"]) + 1j * np.array(prof["radial_ana_phasor_im"])
    ax[0].plot(r, np.abs(zr) / abs(zr[0]), "k--", label="analytic |U|")
    ax[0].set_xlabel("r/R")
    ax[0].set_ylabel("|U(r)| / |U(0)|")
    ax[0].set_title("pipe R=20.5, alpha=5: radial amplitude")
    ax[0].legend(fontsize=8)
    ax2 = ax[1]
    ax2.plot(r, np.array(prof["phase_lag_num_deg_masked"]), "s-", ms=3, color="C3",
             label="LBM lag")
    ax2.plot(r, np.array(prof["phase_lag_ana_deg_masked"]), "k--", label="analytic lag")
    ax2.set_xlabel("r/R")
    ax2.set_ylabel("phase lag (deg)")
    ax2.set_title("radial phase lag")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "fig3_pipe_radial.png", dpi=150)
    plt.close(fig)


def fig_convergence():
    res = json.loads((ART / "result.json").read_text())
    cases = res["cases"]
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for geom, marker in (("channel", "o"), ("pipe", "s")):
        rows = [c for c in cases if c["geometry"] == geom and abs(c["tau"] - 0.8) < 1e-9]
        rows = sorted(rows, key=lambda r: r["resolution"])
        if len(rows) < 2:
            continue
        x = np.array([r["resolution"] for r in rows])
        ax[0].semilogy(x, [r["metrics"]["amplitude_rel_pct"] for r in rows], marker + "-",
                       label=f"{geom} amplitude")
        ax[0].semilogy(x, [r["metrics"]["phase_rel_pct"] for r in rows], marker + "--",
                       label=f"{geom} phase")
        ax[1].semilogy(x, [r["metrics"]["profile_l2_pct"] for r in rows], marker + "-",
                       label=f"{geom} profile L2")
    ax[0].axhline(5.0, color="r", ls=":", label="5% gate")
    ax[0].set_xlabel("resolution (H or R, lattice units)")
    ax[0].set_ylabel("error (%)")
    ax[0].set_title("centreline amplitude / phase error vs resolution")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("resolution (H or R, lattice units)")
    ax[1].set_ylabel("profile L2 (%)")
    ax[1].set_title("phase-averaged profile L2 vs resolution")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ART / "fig4_convergence.png", dpi=150)
    plt.close(fig)


def main() -> None:
    fig_profiles()
    fig_waveforms()
    fig_pipe_radial()
    fig_convergence()
    print("figures written to", ART)


if __name__ == "__main__":
    main()

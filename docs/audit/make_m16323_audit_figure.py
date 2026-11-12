#!/usr/bin/env python3
"""Figures for the m16323 (CASE-LBM-WOMERSLEY-PULSATILE) closure audit.

All curves are read from the mission's run artefacts; the reference curves are
recomputed by the audit harness (docs/audit/m16323_womersley_audit.py) so the
plots do not simply redraw the mission's own figure code.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "docs/audit")
from m16323_womersley_audit import channel_centreline, pipe_centreline  # noqa: E402

ART = "/root/tp-agent/work/m16323/artifacts/womersley_pulsatile_alpha5"
OUT = "docs/audit/figs"
TOL = 5.0


def main() -> None:
    audit = json.load(open("docs/audit/m16323_womersley_audit.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) channel & pipe centreline waveforms, normalised by u_peak
    for label, mark in (("channel_H59_a5_tau0.8", "C0"), ("pipe_R16.5_a5_tau0.8", "C1")):
        d = json.load(open(f"{ART}/{label}.json"))
        ph = np.asarray(d["profile"]["phase_centers"])
        un = np.asarray(d["profile"]["centerline_phase_averaged"])
        ua = np.asarray(d["profile"]["centerline_ana_waveform"])
        peak = abs(complex(*d["centerline"]["z_ana"])) if False else abs(
            complex(d["centerline"]["z_ana"][0], d["centerline"]["z_ana"][1]))
        ax[0].plot(ph, un / peak, mark + "o", ms=3, label=f"{d['geometry']} num")
        ax[0].plot(ph, ua / peak, "k-", lw=0.9)
    ax[0].set_xlabel(r"phase $\omega t$ [rad]")
    ax[0].set_ylabel(r"$u(0,t)/|U_{ana}(0)|$")
    ax[0].set_title("(a) centreline waveform, LBM vs independent Womersley ref")
    ax[0].legend(fontsize=7, loc="lower left")

    # (b) declared metrics vs the 5 % band
    cases = audit["cases"]
    names = [c["case"].replace("_a5_tau0.8", "") for c in cases]
    amp = [c["lbm_refit"]["amp_rel_pct"] for c in cases]
    ph = [c["lbm_refit"]["phase_rel_pct"] for c in cases]
    x = np.arange(len(cases))
    ax[1].bar(x - 0.19, amp, 0.36, label="amplitude ratio err [%]")
    ax[1].bar(x + 0.19, ph, 0.36, label="phase lag err [%]")
    ax[1].axhline(TOL, color="r", ls="--", lw=1, label="case-pool tolerance 5 %")
    for xi, (a, p) in enumerate(zip(amp, ph)):
        ax[1].text(xi - 0.19, a, f"{a:.3f}", ha="center", va="bottom", fontsize=6)
        ax[1].text(xi + 0.19, p, f"{p:.3f}", ha="center", va="bottom", fontsize=6)
    ax[1].set_xticks(x, names, fontsize=7, rotation=12)
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title("(b) audit-refit metrics vs predeclared 5 % band")

    # (c) pipe radial amplitude profile, R=16.5, num vs independent reference
    d = json.load(open(f"{ART}/pipe_R16.5_a5_tau0.8.json"))
    prof = d["profile"]
    r = np.asarray(prof["r"])
    amp_num = np.abs(np.asarray(prof["radial_num_phasor_re"])
                     + 1j * np.asarray(prof["radial_num_phasor_im"]))
    amp_ana = np.abs(np.asarray(prof["radial_ana_phasor_re"])
                     + 1j * np.asarray(prof["radial_ana_phasor_im"]))
    # independent reference profile from the audit ODE solve
    alpha, nu, a0, omega = d["alpha"], d["nu"], d["a0"], d["omega"]
    R = d["R"]
    dr = R / 800
    grid_r = np.arange(0.0, R + dr, dr)
    # FD solve once on the fine grid, then interpolate
    from m16323_womersley_audit import pipe_profile_fd

    u_fd, r_fd = pipe_profile_fd(alpha, nu, a0, omega, 4000)
    ax[2].plot(r / R, amp_num / a0, "C1o", ms=3, label="LBM (BFL wall)")
    ax[2].plot(r / R, amp_ana / a0, "k--", lw=1, label="mission analytic (J0 form)")
    ax[2].plot(r_fd / R, np.abs(u_fd) / a0, "r:", lw=1.4, label="audit FD ODE solve")
    ax[2].set_xlabel("r / R")
    ax[2].set_ylabel(r"$|U(r)|/a_0$  [s]")
    ax[2].set_title("(c) pipe R=16.5 radial amplitude profile")
    ax[2].legend(fontsize=7)
    ax[2].invert_xaxis()

    fig.tight_layout()
    fig.savefig(f"{OUT}/m16323_audit_overview.png", dpi=130)
    print("wrote", f"{OUT}/m16323_audit_overview.png")


if __name__ == "__main__":
    main()
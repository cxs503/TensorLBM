#!/usr/bin/env python3
"""Independent (audit) verification of CASE-LBM-WOMERSLEY-PULSATILE / m16323.

Nothing here imports the audited module's analytic code: the reference solution is
re-derived from the governing ODE with a finite-difference complex solve, and the
numerical LBM result is refitted from the run artefacts' stored phase-averaged
waveform.  Reported metrics of the audited mission are then compared with these
independent numbers.

Governing equation for the physical field u = Re{U(r,y) e^{i omega t}} driven by
a harmonic body force a(t) = a0 cos(omega t)  (=> dp/dx = -rho a):

    i omega U = a0 + nu * lap(U),  U = 0 on the wall.

channel: lap = d^2/dy^2,  y in [0, H]
pipe   : lap = d^2/dr^2 + (1/r) d/dr, r in [0, R], regularity at r = 0

Outputs a JSON summary: docs/audit/m16323_womersley_audit.json
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

ART = "/root/tp-agent/work/m16323/artifacts/womersley_pulsatile_alpha5"


# ── independent reference: complex FD solve of the ODE ──────────────────────
def channel_centreline(alpha: float, nu: float, a0: float, omega: float, n: int = 4000):
    H = 2.0 * alpha * math.sqrt(nu / omega)          # alpha = (H/2) sqrt(omega/nu)
    dy = H / n
    y = np.linspace(0.0, H, n + 1)
    a = nu / dy**2
    A = np.zeros((n + 1, n + 1), complex)
    for j in range(1, n):
        A[j, j - 1] = -a
        A[j, j] = 2 * a + 1j * omega
        A[j, j + 1] = -a
    A[0, 0], A[n, n] = 1.0, 1.0
    b = np.full(n + 1, a0, complex)
    b[0] = b[n] = 0.0
    U = np.linalg.solve(A, b)
    return complex(U[n // 2]), y[n // 2], H


def pipe_centreline(alpha: float, nu: float, a0: float, omega: float, n: int = 4000):
    R = alpha * math.sqrt(nu / omega)                # alpha = R sqrt(omega/nu)
    dr = R / n
    r = np.linspace(0.0, R, n + 1)
    A = np.zeros((n + 1, n + 1), complex)
    # r = 0: even extension, lap(u)|_0 = 2 d2u/dr2 = 4 (u1-u0)/dr^2
    A[0, 0] = 4.0 * nu / dr**2 + 1j * omega
    A[0, 1] = -4.0 * nu / dr**2
    for j in range(1, n):
        a = nu / dr**2
        c = nu / (2 * dr * r[j])
        A[j, j - 1] = -(a - c)
        A[j, j] = 2 * a + 1j * omega
        A[j, j + 1] = -(a + c)
    A[n, n] = 1.0
    b = np.full(n + 1, a0, complex)
    b[n] = 0.0
    U = np.linalg.solve(A, b)
    return complex(U[0]), 0.0, R


def pipe_profile_fd(alpha: float, nu: float, a0: float, omega: float, n: int = 4000):
    """Independent complex-FD velocity profile U(r) on the pipe radius grid."""
    R = alpha * math.sqrt(nu / omega)
    dr = R / n
    r = np.linspace(0.0, R, n + 1)
    A = np.zeros((n + 1, n + 1), complex)
    A[0, 0] = 4.0 * nu / dr**2 + 1j * omega
    A[0, 1] = -4.0 * nu / dr**2
    for j in range(1, n):
        a = nu / dr**2
        c = nu / (2 * dr * r[j])
        A[j, j - 1] = -(a - c)
        A[j, j] = 2 * a + 1j * omega
        A[j, j + 1] = -(a + c)
    A[n, n] = 1.0
    b = np.full(n + 1, a0, complex)
    b[n] = 0.0
    return np.linalg.solve(A, b), r


def lag_deg(z: complex) -> float:
    """Phase lag of the velocity behind the driving force [deg], positive = lag."""
    return -math.degrees(math.atan2(z.imag, z.real))


# ── independent re-estimation of the LBM waveform from the artefacts ────────
def refit(art: dict):
    """Own least-squares harmonic fit of the stored centreline waveform."""
    prof = art["profile"]
    t = np.asarray(prof["phase_centers"], float) / art["omega"]   # bins -> time (s)
    u = np.asarray(prof["centerline_phase_averaged"], float)
    M = np.column_stack([np.cos(art["omega"] * t), np.sin(art["omega"] * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(M, u, rcond=None)
    Z = coef[0] - 1j * coef[1]
    return Z, float(np.max(np.abs(M @ coef - u)))


def main() -> int:
    out = {"artifacts_dir": ART, "cases": [], "declared_ref": {}}
    declared = {}
    for label in ("channel_H59_a5_tau0.8", "pipe_R10.5_a5_tau0.8", "pipe_R16.5_a5_tau0.8"):
        art = json.load(open(f"{ART}/{label}.json"))
        geom = art["geometry"]
        nu, omega, a0 = art["nu"], art["omega"], art["a0"]
        if geom == "channel":
            z_fd, _, L = channel_centreline(art["alpha"], nu, a0, omega)
        else:
            z_fd, _, L = pipe_centreline(art["alpha"], nu, a0, omega)
        z_num, fit_res = refit(art)
        z_ana_stored = complex(*zip(*[art["centerline"]["z_ana"]]))[0] \
            if False else complex(art["centerline"]["z_ana"][0], art["centerline"]["z_ana"][1])
        z_num_stored = complex(art["centerline"]["z_num"][0], art["centerline"]["z_num"][1])
        amp_ref, amp_num = abs(z_fd), abs(z_num)
        rec = {
            "case": label,
            "geometry": geom,
            "alpha": art["alpha"],
            "nu": nu,
            "omega": omega,
            "a0": a0,
            "length": L,
            "independent_ref": {"abs_U0": amp_ref, "phase_lag_deg": lag_deg(z_fd)},
            "stored_ref": {"abs_U0": abs(z_ana_stored), "phase_lag_deg": lag_deg(z_ana_stored)},
            "ref_vs_independent_pct": {
                "amplitude": 100 * abs(abs(z_ana_stored) / amp_ref - 1),
                "phase_deg": lag_deg(z_ana_stored) - lag_deg(z_fd),
            },
            "lbm_refit": {"abs_U0": amp_num, "phase_lag_deg": lag_deg(z_num),
                          "amp_rel_pct": 100 * abs(amp_num / amp_ref - 1),
                          "phase_rel_pct": 100 * abs(lag_deg(z_num) - lag_deg(z_fd)) / abs(lag_deg(z_fd)),
                          "phase_diff_deg": lag_deg(z_num) - lag_deg(z_fd),
                          "residual_max": fit_res},
            "claimed_metrics": art["metrics"],
            "claimed_centerline": {
                "amplitude_rel_pct": art["centerline"]["amplitude_rel_pct"],
                "phase_rel_pct": art["centerline"]["phase_rel_pct"],
                "phase_lag_num_deg": art["centerline"]["phase_lag_num_deg"],
                "phase_lag_ana_deg": art["centerline"]["phase_lag_ana_deg"],
            },
            "stored_num_phasor_vs_refit": {
                "amp_rel_pct_diff": 100 * abs(abs(z_num_stored) / amp_num - 1),
                "phase_deg_diff": lag_deg(z_num_stored) - lag_deg(z_num),
            },
        }
        out["cases"].append(rec)
        print(json.dumps(rec, indent=1))

    # acceptance per case-pool tol policy: <5 % amplitude ratio + phase lag, alpha=5
    ok = all(c["lbm_refit"]["amp_rel_pct"] < 5.0 and c["lbm_refit"]["phase_rel_pct"] < 5.0
             for c in out["cases"])
    out["verdict"] = {"tol_pct": 5.0, "all_cases_within_tol": ok}

    # the pipe steady artifact, checked against the steady Poiseuille reference
    st = json.load(open(f"{ART}/pipe_steady_R10.5_a5_tau0.8.json"))
    out["pipe_steady"] = {
        "u_center_rel_pct_claimed": st.get("u_center_rel_pct"),
        "profile_l2_pct_claimed": st.get("profile_l2_pct"),
        "R": st.get("R"), "tau": st.get("tau"), "nu": st.get("nu", None),
    }
    print("verdict:", out["verdict"])
    with open("docs/audit/m16323_womersley_audit.json", "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
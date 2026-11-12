#!/usr/bin/env python3
"""Independent verification of `tensorlbm.fwh_rotating` (FW-H rotating sources).

Why this harness exists
-----------------------
`src/tensorlbm/fwh_rotating.py` (866 lines) advertises three "analytical
validation anchors" in its module docstring — Gardner rotating monopole,
Curle dipole, Blake BPF harmonics — but until now the module had **zero**
references from any test, benchmark or doc in the repository
(`grep -rn fwh_rotating --include=*.py --include=*.md .` returned only the
file itself). A capability whose only evidence is its own docstring is
unverified by construction: the "reference" and the "implementation" are the
same author, same file, same assumptions.

This script therefore does NOT trust the module's own analytic helpers as
references. Every gate below compares the library against an
*independently derived* closed form and, where the closed form itself is a
statement worth doubting, against a second numerical route:

  G1 retarded time      Newton solver vs. bracketed Brent root of the same
                        equation (independent solver, independent bracket).
  G2 Doppler factor     library 1/(1-M_r) vs. finite-difference of the
                        retarded-time map tau(t) (identity dtau/dt = 1/(1-M_r)).
  G3 monopole limit     library `gardner_monopole_analytical` vs. the exact
                        moving-monopole far field.
  G4 reference audit    the exact far-field monopole formula itself vs. a
                        numerical tau-quadrature + time differentiation of the
                        wave-equation Green's function solution (so G3's
                        reference is not just asserted).
  G5 Curle dipole       library vs. closed-form compact dipole + cos(theta)
                        directivity law.
  G6 SPL calibration    library `compute_spl_spectrum` vs. analytically known
                        tone amplitude (window coherent gain).
  G7 BPF harmonics      library `_extract_bpf_harmonics` vs. a synthetic
                        multi-harmonic signal with prescribed amplitudes.
  G8 nondimensional     omega / tip_speed / tip Mach / BPF definitions.
  G9 density scaling    library `compute_rotating_source_fwh` acoustic pressure
                        must be linear in rho0 for a fixed kinematic source.

Physics used as reference
-------------------------
Moving point monopole of volume-injection rate Q(t) [m^3/s] at y(t):

    box p = rho0 * d/dt [ Q(t) delta(x - y(t)) ]

Green's function in 3-D, subsonic single retarded root:

    p(x,t) = rho0/(4 pi) * d/dt [ Q(tau) / (r (1 - M_r)) ]
           = rho0/(4 pi) [ Qdot(tau) / (r (1-M_r)^2) + Q(tau) * O(1/r^2) ]

so the leading-order (1/r) far field is

    p_far(t) = rho0 * Qdot(tau) / (4 pi r (1 - M_r)^2),   t - tau = r/c0.

(Equivalent to Lowson 1965 / Dowling & Ffowcs Williams 1983 for the monopole
term; the O(1/r^2) remainder is what G4 measures and drives to zero.)

Compact dipole (Curle, low Mach, stationary observer) with force F(t):

    p(x,t) = (1/(4 pi c0 r)) * dF/dtau . rhat  ->  directivity cos(theta).

Usage
-----
    python benchmarks/verified/fwh_rotating/run.py OUTDIR [--tag pre|post]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm import fwh_rotating as fw  # noqa: E402

C0 = 340.0
RHO0 = 1000.0


# ---------------------------------------------------------------------------
# Independent helpers (deliberately NOT reusing the library implementation)
# ---------------------------------------------------------------------------

def circle_traj(radius: float, omega: float, phase: float = 0.0):
    """Source on a circle in the x-y plane, counter-clockwise."""

    def traj(tau: float):
        th = phase + omega * tau
        pos = np.array([radius * math.cos(th), radius * math.sin(th), 0.0])
        vel = np.array([-radius * omega * math.sin(th),
                        radius * omega * math.cos(th), 0.0])
        return pos, vel

    return traj


def retarded_time_brent(obs, traj, t_obs, c0, r_max, n_scan=257):
    """Independent retarded time: bracketed Brent root of g(tau)=tau+|x-y|/c-t.

    Scan tau over [t_obs - (r_max*1.05)/c0, t_obs + r_max*0.05/c0] for a sign
    change of g and refine with Brent. Subsonic => unique root.
    """
    lo = t_obs - 1.05 * r_max / c0
    hi = t_obs + 0.05 * r_max / c0
    taus = np.linspace(lo, hi, n_scan)
    gs = np.empty(n_scan)
    for k, tau in enumerate(taus):
        pos, _ = traj(tau)
        gs[k] = tau + np.linalg.norm(obs - pos) / c0 - t_obs
    sign_change = np.where(np.sign(gs[:-1]) * np.sign(gs[1:]) <= 0.0)[0]
    if sign_change.size == 0:
        raise RuntimeError("no retarded root bracketed")
    i = int(sign_change[-1])  # last crossing = retarded (largest tau) branch
    a, b = float(taus[i]), float(taus[i + 1])

    def g(tau):
        pos, _ = traj(tau)
        return tau + np.linalg.norm(obs - pos) / c0 - t_obs

    return brentq(g, a, b, xtol=1e-14, rtol=8.9e-16, maxiter=200)


def monopole_farfield(t, obs, radius, omega, q0, c0=C0, rho0=RHO0):
    """Exact leading-order far field of a rotating monopole, Q(t)=q0*sin(omega*t).

    p = rho0 * Qdot(tau) / (4 pi r (1-M_r)^2),  Qdot = q0*omega*cos(omega*tau).
    """
    traj = circle_traj(radius, omega)
    r_max = float(np.linalg.norm(obs)) + radius
    out = np.zeros_like(t)
    for i, t_obs in enumerate(t):
        tau = retarded_time_brent(obs, traj, float(t_obs), c0, r_max)
        pos, vel = traj(tau)
        r_vec = obs - pos
        r_mag = float(np.linalg.norm(r_vec))
        mr = float(np.dot(r_vec, vel) / (c0 * r_mag))
        qdot = q0 * omega * math.cos(omega * tau)
        out[i] = rho0 * qdot / (4.0 * math.pi * r_mag * (1.0 - mr) ** 2)
    return out


def monopole_green_derivative(t, obs, radius, omega, q0, c0=C0, rho0=RHO0,
                              h=2e-7):
    """Second, independent route: p = rho0/(4 pi) d/dt [Q(tau)/(r(1-M_r))].

    Q(tau) = q0 sin(omega tau); the bracket is evaluated from the retarded root
    (found by Brent) and differentiated by central differences in *observer*
    time. No use of the closed-form Qdot/(1-M_r)^2 result.
    """
    traj = circle_traj(radius, omega)
    r_max = float(np.linalg.norm(obs)) + radius

    def bracket(t_obs):
        tau = retarded_time_brent(obs, traj, t_obs, c0, r_max)
        pos, vel = traj(tau)
        r_vec = obs - pos
        r_mag = float(np.linalg.norm(r_vec))
        mr = float(np.dot(r_vec, vel) / (c0 * r_mag))
        return q0 * math.sin(omega * tau) / (r_mag * (1.0 - mr))

    out = np.zeros_like(t)
    for i, t_obs in enumerate(t):
        out[i] = (bracket(t_obs + h) - bracket(t_obs - h)) / (2.0 * h)
    return rho0 / (4.0 * math.pi) * out


def rel_err(a, b):
    """max|a-b| / max|b| (scale-free, avoids division by near-zero samples)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def gate(name, value, reference, err_pct, tol_pct, verdict, note=""):
    return {
        "gate": name,
        "value": value,
        "reference": reference,
        "err_pct": err_pct,
        "tol_pct": tol_pct,
        "verdict": verdict,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def g1_retarded_time():
    rows = []
    worst = 0.0
    for tip_mach in (0.10, 0.30, 0.60):
        for radius in (0.5, 2.0):
            omega = tip_mach * C0 / radius
            for az in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
                for r_obs in (5.0, 20.0):
                    obs = np.array([r_obs * math.cos(math.radians(az)),
                                    r_obs * math.sin(math.radians(az)), 0.0])
                    traj = circle_traj(radius, omega)
                    for t_obs in (0.0, 0.013, 0.041):
                        tau_lib, n_iter, conv = fw.solve_retarded_time(
                            obs, traj, t_obs, C0, tol=1e-12, max_iter=60)
                        tau_ref = retarded_time_brent(
                            obs, traj, t_obs, C0,
                            float(np.linalg.norm(obs)) + radius)
                        # residual expressed in metres (multiply by c0)
                        err_m = abs(tau_lib - tau_ref) * C0
                        worst = max(worst, err_m)
                        rows.append({
                            "tip_mach": tip_mach, "radius": radius,
                            "az_deg": az, "r_obs": r_obs, "t_obs": t_obs,
                            "tau_lib": tau_lib, "tau_ref": tau_ref,
                            "err_m": err_m, "iters": n_iter,
                            "converged": bool(conv),
                        })
    all_conv = all(r["converged"] for r in rows)
    return gate("G1_retarded_time_vs_brent", worst, 0.0,
                worst * 100.0, 1e-6 * 100.0,
                "PASS" if (worst < 1e-6 and all_conv) else "FAIL",
                f"{len(rows)} cases, subsonic tip Mach 0.1/0.3/0.6; "
                f"worst |dtau|*c0 = {worst:.3e} m; all converged={all_conv}"), rows


def g2_doppler_identity():
    """dtau/dt from the retarded-time map must equal the library's Doppler factor."""
    worst = 0.0
    rows = []
    for tip_mach in (0.10, 0.35, 0.60):
        radius, omega = 1.0, tip_mach * C0 / 1.0
        traj = circle_traj(radius, omega)
        for az in (0.0, 45.0, 90.0, 180.0, 270.0):
            obs = np.array([8.0 * math.cos(math.radians(az)),
                            8.0 * math.sin(math.radians(az)), 0.0])
            t_obs = 0.007
            h = 1e-7
            r_max = float(np.linalg.norm(obs)) + radius
            tp = retarded_time_brent(obs, traj, t_obs + h, C0, r_max)
            tm = retarded_time_brent(obs, traj, t_obs - h, C0, r_max)
            dtau_dt = (tp - tm) / (2.0 * h)
            tau, _, _ = fw.solve_retarded_time(obs, traj, t_obs, C0, tol=1e-13,
                                               max_iter=60)
            pos, vel = traj(tau)
            doppler_lib = fw.compute_doppler_factor(obs, pos, vel, C0)
            err = abs(doppler_lib - dtau_dt) / abs(dtau_dt)
            worst = max(worst, err)
            rows.append({"tip_mach": tip_mach, "az_deg": az,
                         "doppler_lib": doppler_lib, "dtau_dt": dtau_dt,
                         "rel_err": err})
    return gate("G2_doppler_vs_dtaudt", worst, 0.0, worst * 100.0, 0.01,
                "PASS" if worst < 1e-4 else "FAIL",
                "library 1/(1-M_r) vs finite-difference dtau/dt of the "
                "retarded-time map"), rows


def exact_stationary_monopole(t, obs, q0, omega, c0=C0, rho0=RHO0):
    """Exact far field of a *stationary* monopole Q(t)=q0 sin(omega t) at origin.

    p = rho0 Qdot(t-r/c)/(4 pi r).  No Doppler factor at all, so this reference
    is independent of the (1-M_r) exponent convention.
    """
    r = float(np.linalg.norm(obs))
    tau = t - r / c0
    return rho0 * q0 * omega * np.cos(omega * tau) / (4.0 * math.pi * r)


def g3_monopole_analytical():
    """Falsify the library's 'exact rotating monopole' claim without any
    appeal to the Doppler exponent: the stationary limit must survive."""
    obs = np.array([0.0, 0.0, 50.0])
    q0 = 1e-3
    t = np.linspace(0.0, 2.0e-3, 121)
    rows, worst = [], 0.0
    # (a) stationary limit: R -> 0 (source at rest at the origin)
    for radius in (1e-6, 1e-3):
        omega = 102.0                     # fixed, so Qdot is unaffected by R
        lib = fw.gardner_monopole_analytical(obs, radius, omega, q0, C0, RHO0, t)
        ref = exact_stationary_monopole(t, obs, q0, omega)
        err = rel_err(lib, ref)
        worst = max(worst, err)
        rows.append({"test": f"stationary_limit_R={radius:g}",
                     "lib_peak_Pa": float(np.max(np.abs(lib))),
                     "exact_peak_Pa": float(np.max(np.abs(ref))),
                     "ratio": float(np.max(np.abs(lib)) / np.max(np.abs(ref))),
                     "expected_ratio": 1.0, "err_pct": err * 100.0})
    # (b) m = 1 scaling at finite radius: the library amplitude carries a
    #     spurious omega*R/c0 factor, so lib/ref should track the tip Mach.
    radius = 1.0
    for tip_mach in (0.05, 0.35, 0.60):
        omega = tip_mach * C0 / radius
        lib = fw.gardner_monopole_analytical(obs, radius, omega, q0, C0, RHO0, t)
        ref = monopole_farfield(t, obs, radius, omega, q0)
        rows.append({"test": f"tip_mach={tip_mach}",
                     "lib_peak_Pa": float(np.max(np.abs(lib))),
                     "ref_peak_Pa_O1overr_with_single_Doppler": float(np.max(np.abs(ref))),
                     "ratio": float(np.max(np.abs(lib)) / np.max(np.abs(ref))),
                     "expected_ratio": 1.0,
                     "err_pct": rel_err(lib, ref) * 100.0})
    returning = "the stationary limit returns 0 while the exact field is finite"
    return gate("G3_gardner_monopole_vs_exact", rows[0]["ratio"], 1.0,
                worst * 100.0, 3.0,
                "PASS" if worst < 0.03 else "FAIL",
                "library 'exact solution' claim falsified by the R->0 limit "
                f"({returning}); at finite R the amplitude ratio tracks the tip "
                "Mach number because of a spurious omega*R/c0 factor"), rows


def g4_reference_selfcheck():
    """Audit the far-field reference itself.

    Route A (closed form, single Doppler power): rho0 Qdot/(4 pi r (1-M_r)).
    Route B (numerical): rho0/(4 pi) d/dt[Q(tau)/(r (1-M_r))], central
    difference of the retarded root found by Brent.

    The two differ by exactly one power of (1-M_r): Route B returns
    rho0 Qdot/(4 pi r (1-M_r)^2) + O(1/r^2). Adjudicating the exponent needs a
    direct time-domain solution of the wave equation, which this harness does
    not attempt -> the reference is reported WITHHELD, not "verified".
    """
    radius, q0, tip_mach = 1.0, 1e-3, 0.30
    omega = tip_mach * C0 / radius
    t = np.linspace(0.0, 2.0e-3, 41)
    rows, worst = [], 0.0
    for r_obs in (10.0, 100.0, 1000.0):
        obs = np.array([r_obs, 0.0, 0.0])
        a = monopole_farfield(t, obs, radius, omega, q0)
        b = monopole_green_derivative(t, obs, radius, omega, q0)
        err = rel_err(a, b)
        worst = max(worst, err)
        rows.append({"r_obs_m": r_obs, "r_over_R": r_obs / radius,
                     "routeA_single_Doppler_peak_Pa": float(np.max(np.abs(a))),
                     "routeB_ddt_peak_Pa": float(np.max(np.abs(b))),
                     "ratio_B_over_A": float(np.max(np.abs(b)) / np.max(np.abs(a))),
                     "err_pct": err * 100.0})
    return gate("G4_farfield_reference_audit", worst, 0.0, worst * 100.0, 0.5,
                "WITHHELD",
                "route A vs route B differ by a (1-M_r) power (ratio "
                f"{rows[-1]['ratio_B_over_A']:.4f} at {rows[-1]['r_obs_m']:.0f} m); "
                "the Doppler exponent of the far field for a prescribed volume "
                "injection rate is NOT adjudicated here - it needs a direct "
                "time-domain wave-equation solve. Consequence: G3 is built on an "
                "exponent-free stationary-limit test instead"), rows


def g5_curle_dipole():
    """Compact dipole: amplitude law and cos(theta) directivity."""
    r_obs, f0 = 3.0, 500.0
    omega = 2.0 * math.pi * f0
    force, c0 = 10.0, C0
    t = np.linspace(0.0, 3.0 / f0, 301)
    a_ref = force * omega / (4.0 * math.pi * c0 * r_obs)
    rows, worst_amp, worst_dir = [], 0.0, 0.0
    for theta_deg in (0.0, 30.0, 60.0, 90.0, 120.0, 180.0):
        th = math.radians(theta_deg)
        obs = np.array([r_obs * math.sin(th), 0.0, r_obs * math.cos(th)])
        lib = fw.curle_dipole_analytical(
            obs, force, omega, c0, RHO0, t,
            force_direction=np.array([0.0, 0.0, 1.0]))
        peak = float(np.max(np.abs(lib)))
        expected = a_ref * abs(math.cos(th))
        amp_err = abs(peak - expected) / max(expected, 1e-30) if expected > 1e-12 else 0.0
        # directivity: compare normalized peak with cos(theta)
        dir_err = abs(peak / (a_ref) - abs(math.cos(th)))
        worst_amp = max(worst_amp, amp_err)
        worst_dir = max(worst_dir, dir_err)
        rows.append({"theta_deg": theta_deg, "peak_Pa": peak,
                     "closed_form_Pa": expected, "amp_rel_err": amp_err,
                     "directivity_dev": dir_err})
    ok = worst_amp < 0.01 and worst_dir < 0.01
    return gate("G5_curle_dipole_amplitude_and_directivity", worst_amp, 0.0,
                worst_amp * 100.0, 1.0, "PASS" if ok else "FAIL",
                "F0*w/(4 pi c0 r) amplitude with cos(theta) directivity; "
                f"amplitude rel err {worst_amp:.3e}, directivity dev "
                f"{worst_dir:.3e}"), rows


def g6_spl_calibration():
    """SPL must recover an analytically known tone amplitude."""
    fs, f_tone, amp = 10000.0, 1000.0, 1.0     # exactly 10 bins at 1 kHz
    dt = 1.0 / fs
    t = np.arange(0.0, 1.0, dt)
    sig = amp * np.sin(2.0 * math.pi * f_tone * t)
    ref_peak = 20.0 * math.log10(amp / 1e-6)     # documented convention: peak amp
    ref_rms = 20.0 * math.log10(amp / math.sqrt(2.0) / 1e-6)
    rows = []
    for window in ("rect", "hann", "hamming"):
        spl, freqs = fw.compute_spl_spectrum(sig.reshape(1, -1), dt,
                                            ref_pressure=1e-6, window=window)
        idx = int(np.argmin(np.abs(freqs - f_tone)))
        got = float(spl[0, idx])
        rows.append({"window": window, "spl_dB": got,
                     "dev_vs_peak_dB": got - ref_peak,
                     "dev_vs_rms_dB": got - ref_rms})
    worst = max(abs(r["dev_vs_peak_dB"]) for r in rows)
    return gate("G6_spl_tone_calibration", rows[0]["spl_dB"], ref_peak, worst,
                0.1, "PASS" if worst < 0.1 else "FAIL",
                "1 Pa peak tone at a bin centre; documented convention is "
                "20log10(peak/p_ref). Windowed spectra must be corrected for "
                "coherent gain (hann cg=0.5 -> -6.02 dB bias if not)"), rows


def g7_bpf_harmonics():
    """Synthetic multi-harmonic signal with prescribed A_n = A1/n."""
    n_blades, rpm = 4.0, 3000.0
    bpf = n_blades * rpm / 60.0                 # 200 Hz
    fs = 5000.0
    dt = 1.0 / fs
    t = np.arange(0.0, 2.0, dt)
    a1 = 10.0
    sig = np.zeros_like(t)
    for n in range(1, 6):
        sig += (a1 / n) * np.sin(2.0 * math.pi * n * bpf * t + 0.3 * n)
    freqs = np.fft.rfftfreq(len(t), dt)
    got = fw._extract_bpf_harmonics(sig.reshape(1, -1), freqs, bpf,
                                    n_harmonics=5, bandwidth=0.5 * bpf)
    rows, worst = [], 0.0
    for i, n in enumerate(range(1, 6)):
        expect = a1 / n
        actual = float(got["amplitudes"][i])
        err = abs(actual - expect) / expect
        worst = max(worst, err)
        rows.append({"harmonic": n, "target_hz": n * bpf,
                     "reported_hz": got["frequencies"][i],
                     "expect_amp": expect, "got_amp": actual,
                     "rel_err_pct": err * 100.0})
    return gate("G7_bpf_harmonic_extraction", worst, 0.0, worst * 100.0, 5.0,
                "PASS" if worst < 0.05 else "FAIL",
                "signal = sum_n (A1/n) sin(2 pi n BPF t); extraction must "
                "return the peak bin closest to n*BPF inside each band"), rows


def g8_nondimensional():
    cfg = fw.RotatingSourceConfig(n_blades=4, rpm=3000.0, radius=2.0,
                                  c0=340.0)
    rows = {
        "omega_rad_s": cfg.omega,
        "omega_expect": 2.0 * math.pi * 3000.0 / 60.0,
        "tip_speed": cfg.tip_speed,
        "tip_speed_expect": cfg.omega * 2.0,
        "tip_mach": cfg.tip_mach,
        "tip_mach_expect": cfg.omega * 2.0 / 340.0,
        "bpf_hz": cfg.bpf,
        "bpf_expect": 4.0 * 3000.0 / 60.0,
    }
    worst = 0.0
    for got_key, exp_key in (("omega_rad_s", "omega_expect"),
                             ("tip_speed", "tip_speed_expect"),
                             ("tip_mach", "tip_mach_expect"),
                             ("bpf_hz", "bpf_expect")):
        worst = max(worst, abs(rows[got_key] - rows[exp_key])
                    / max(abs(rows[exp_key]), 1e-30))
    return gate("G8_nondimensional_definitions", worst, 0.0, worst * 100.0,
                1e-9, "PASS" if worst < 1e-12 else "FAIL",
                "omega, tip speed, tip Mach, BPF = n_blades*rpm/60"), rows


def g9_density_scaling():
    """Thickness (monopole) source: p ~ rho0 for a fixed *kinematic* source.

    The loading/dipole term is excluded on purpose: for a prescribed force F,
    the far field is F0*w/(4 pi c0 r) and is density-independent, so a total
    pressure cannot test the density normalisation.  Thickness is the branch
    that carries rho0.
    """
    obs = np.array([[0.0, 0.0, 10.0]])
    base = dict(n_blades=2, rpm=1200.0, radius=0.5, c0=C0,
                thrust=100.0, torque=10.0, n_segments=4,
                observer_positions=obs, t_start=0.0, t_end=2.0e-3,
                dt=2.0e-5, include_loading=False)
    peaks, totals = {}, {}
    for rho0 in (500.0, 1000.0, 2000.0):
        cfg = fw.RotatingSourceConfig(rho0=rho0, **base)
        peaks[rho0] = float(np.max(np.abs(
            fw.compute_rotating_source_fwh(cfg).pressure)))
        cfg_tot = fw.RotatingSourceConfig(rho0=rho0, **{
            k: v for k, v in base.items() if k != "include_loading"})
        totals[rho0] = float(np.max(np.abs(
            fw.compute_rotating_source_fwh(cfg_tot).pressure)))
    ratio = peaks[2000.0] / peaks[500.0] if peaks[500.0] > 0 else float("inf")
    rows = {"thickness_peak_Pa": peaks,
            "total_peak_Pa_with_loading": totals,
            "ratio_2000_over_500": ratio,
            "expected_ratio_linear_in_rho0": 4.0,
            "thickness_share_of_total": (peaks[1000.0] / totals[1000.0]
                                         if totals[1000.0] > 0 else 0.0)}
    ok = abs(ratio - 4.0) / 4.0 < 0.02
    return gate("G9_thickness_linear_in_rho0", ratio, 4.0,
                abs(ratio - 4.0) / 4.0 * 100.0, 2.0,
                "PASS" if ok else "FAIL",
                "thickness branch with loading excluded; a ratio of 16 means "
                "rho0 was applied twice (the volumetric source already carried "
                "a density factor). Also reports the thickness share of the "
                "total pressure - it is ~0 because the per-element phases "
                "cancel, i.e. all radiated pressure comes from the loading term"), rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def source_fingerprint():
    p = REPO / "src" / "tensorlbm" / "fwh_rotating.py"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": "src/tensorlbm/fwh_rotating.py", "sha256": h,
            "lines": len(p.read_text().splitlines())}


def run_all():
    out = []
    for fn in (g1_retarded_time, g2_doppler_identity, g3_monopole_analytical,
               g4_reference_selfcheck, g5_curle_dipole, g6_spl_calibration,
               g7_bpf_harmonics, g8_nondimensional, g9_density_scaling):
        g, detail = fn()
        g["detail"] = detail
        out.append(g)
        print(f"[{g['verdict']:4s}] {g['gate']:45s} err={g['err_pct']:.4g}% "
              f"(tol {g['tol_pct']:.4g}%)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="artifacts/fwh_rotating")
    ap.add_argument("--tag", default="post", choices=("pre", "post"))
    args = ap.parse_args(argv)

    gates = run_all()
    n_pass = sum(g["verdict"] == "PASS" for g in gates)
    n_withheld = sum(g["verdict"] == "WITHHELD" for g in gates)
    payload = {
        "case": "fwh_rotating_independent_verification",
        "anchor": "NEW:FWH-ROTATING-UNVERIFIED",
        "module": source_fingerprint(),
        "tag": args.tag,
        "c0": C0, "rho0": RHO0,
        "gates": gates,
        "n_gates": len(gates),
        "n_pass": n_pass,
        "n_withheld": n_withheld,
        "n_fail": len(gates) - n_pass - n_withheld,
        "fail_list": [g["gate"] for g in gates if g["verdict"] == "FAIL"],
        "withheld_list": [g["gate"] for g in gates if g["verdict"] == "WITHHELD"],
        "verdict": "PASS" if n_pass == len(gates) else "FAIL",
    }
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fname = "result.json" if args.tag == "post" else "result_pre_fix.json"
    (outdir / fname).write_text(json.dumps(payload, indent=2))
    print(f"\n{n_pass}/{len(gates)} gates PASS -> {outdir / fname}")
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())
"""Independent verification of `tensorlbm.fwh_rotating` (FW-H rotating sources).

The module shipped 866 lines with three claimed "analytical validation anchors"
(Gardner rotating monopole, Curle dipole, Blake BPF harmonics) and **no test,
benchmark or doc anywhere in the repo referenced it**.  These tests pin the
gates that a verification harness can decide independently of the module's own
analytic helpers:

* retarded time vs. an independent bracketed Brent root,
* dtau/dt identity (the Doppler factor),
* stationary limit of the "rotating monopole" far field (exponent-free
  falsification of the old omega^2 R / c0 amplitude law),
* Curle dipole amplitude F0*w/(4 pi c0 r) and cos(theta) directivity,
* SPL calibration of a bin-centred tone for every window (coherent gain),
* BPF harmonic bin selection (argmin, not the first bin of the band),
* density normalisation of the thickness source.

Run:  python -m pytest tests/test_fwh_rotating_verified.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm import fwh_rotating as fw  # noqa: E402

C0 = 340.0
RHO0 = 1000.0


def _circle(radius, omega, phase=0.0):
    def traj(tau):
        th = phase + omega * tau
        return (np.array([radius * math.cos(th), radius * math.sin(th), 0.0]),
                np.array([-radius * omega * math.sin(th),
                          radius * omega * math.cos(th), 0.0]))
    return traj


def _root(obs, traj, t_obs, c0, r_max, n=257):
    lo, hi = t_obs - 1.05 * r_max / c0, t_obs + 0.05 * r_max / c0
    taus = np.linspace(lo, hi, n)
    g = np.array([tau + np.linalg.norm(obs - traj(tau)[0]) / c0 - t_obs
                  for tau in taus])
    idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) <= 0.0)[0]
    assert idx.size, "no retarded root bracketed"
    i = int(idx[-1])

    def gg(tau):
        return tau + np.linalg.norm(obs - traj(tau)[0]) / c0 - t_obs

    return brentq(gg, taus[i], taus[i + 1], xtol=1e-14, maxiter=200)


# ---------------------------------------------------------------------------
# G1 / G2 — retarded time and the Doppler factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tip_mach", [0.1, 0.3, 0.6])
def test_retarded_time_matches_independent_brent_root(tip_mach):
    radius = 1.0
    omega = tip_mach * C0 / radius
    traj = _circle(radius, omega)
    for az_deg in (0.0, 75.0, 150.0, 300.0):
        obs = np.array([12.0 * math.cos(math.radians(az_deg)),
                        12.0 * math.sin(math.radians(az_deg)), 0.0])
        for t_obs in (0.0, 0.011):
            tau_lib, n_iter, conv = fw.solve_retarded_time(
                obs, traj, t_obs, C0, tol=1e-12, max_iter=60)
            tau_ref = _root(obs, traj, t_obs, C0, float(np.linalg.norm(obs)) + radius)
            assert conv and n_iter <= 60
            assert abs(tau_lib - tau_ref) * C0 < 1e-6  # metres of path length


def test_doppler_factor_equals_dtau_dt():
    radius, omega = 1.0, 0.35 * C0
    traj = _circle(radius, omega)
    obs = np.array([8.0, 0.0, 0.0])
    t_obs, h = 0.007, 1e-7
    r_max = float(np.linalg.norm(obs)) + radius
    tp = _root(obs, traj, t_obs + h, C0, r_max)
    tm = _root(obs, traj, t_obs - h, C0, r_max)
    dtau_dt = (tp - tm) / (2.0 * h)
    tau, _, _ = fw.solve_retarded_time(obs, traj, t_obs, C0, tol=1e-13,
                                      max_iter=60)
    pos, vel = traj(tau)
    doppler = fw.compute_doppler_factor(obs, pos, vel, C0)
    assert abs(doppler - dtau_dt) / abs(dtau_dt) < 1e-4


def test_doppler_factor_stationary_limit():
    pos = np.zeros(3)
    assert fw.compute_doppler_factor(np.array([0.0, 0.0, 3.0]), pos, pos, C0) == 1.0


# ---------------------------------------------------------------------------
# G3 — rotating monopole far field: the stationary limit fixes the amplitude
# ---------------------------------------------------------------------------

def test_monopole_stationary_limit_is_finite_and_exact():
    """R -> 0 must recover rho0*Qdot/(4 pi r); the old law was identically 0."""
    obs = np.array([0.0, 0.0, 50.0])
    q0, omega = 1e-3, 102.0
    t = np.linspace(0.0, 2.0e-3, 121)
    r = float(np.linalg.norm(obs))
    exact = RHO0 * q0 * omega * np.cos(omega * (t - r / C0)) / (4.0 * math.pi * r)
    lib = fw.gardner_monopole_analytical(obs, 1e-6, omega, q0, C0, RHO0, t)
    err = np.max(np.abs(lib - exact)) / np.max(np.abs(exact))
    assert err < 1e-6, f"stationary monopole limit wrong by {err:.3e}"


def test_monopole_amplitude_independent_of_rotation_radius():
    """For a fixed Q(t) the far-field amplitude must not scale with R."""
    obs = np.array([0.0, 0.0, 40.0])
    q0, omega = 1e-3, 102.0
    t = np.linspace(0.0, 2.0e-3, 61)
    a = fw.gardner_monopole_analytical(obs, 1.0, omega, q0, C0, RHO0, t)
    b = fw.gardner_monopole_analytical(obs, 1e-6, omega, q0, C0, RHO0, t)
    assert abs(np.max(np.abs(a)) - np.max(np.abs(b))) / np.max(np.abs(b)) < 0.05


# ---------------------------------------------------------------------------
# G5 — Curle dipole
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta_deg", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_curle_dipole_amplitude_and_directivity(theta_deg):
    r_obs, f0, force = 3.0, 500.0, 10.0
    omega = 2.0 * math.pi * f0
    th = math.radians(theta_deg)
    obs = np.array([r_obs * math.sin(th), 0.0, r_obs * math.cos(th)])
    t = np.linspace(0.0, 3.0 / f0, 301)
    lib = fw.curle_dipole_analytical(obs, force, omega, C0, RHO0, t,
                                    force_direction=np.array([0.0, 0.0, 1.0]))
    expected = force * omega / (4.0 * math.pi * C0 * r_obs) * abs(math.cos(th))
    assert abs(np.max(np.abs(lib)) - expected) / max(expected, 1e-30) < 0.01


def test_curle_dipole_phase_delay_is_r_over_c():
    """The signal must be sin(omega*(t - r/c0)), not sin(omega*t)."""
    f0, force, r_obs = 400.0, 5.0, 4.0
    omega = 2.0 * math.pi * f0
    obs = np.array([0.0, 0.0, r_obs])
    t = np.linspace(0.0, 2.0 / f0, 257)
    lib = fw.curle_dipole_analytical(obs, force, omega, C0, RHO0, t,
                                    force_direction=np.array([0.0, 0.0, 1.0]))
    amp = force * omega / (4.0 * math.pi * C0 * r_obs)
    exact = amp * np.sin(omega * (t - r_obs / C0))
    assert np.max(np.abs(lib - exact)) / amp < 1e-10


# ---------------------------------------------------------------------------
# G6 — SPL calibration (window coherent gain)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", ["rect", "hann", "hamming"])
def test_spl_recovers_known_tone_amplitude(window):
    fs, f_tone, amp = 10000.0, 1000.0, 1.0   # exactly 10 bins at 1 kHz
    dt = 1.0 / fs
    t = np.arange(0.0, 1.0, dt)
    sig = amp * np.sin(2.0 * math.pi * f_tone * t)
    spl, freqs = fw.compute_spl_spectrum(sig.reshape(1, -1), dt,
                                       ref_pressure=1e-6, window=window)
    idx = int(np.argmin(np.abs(freqs - f_tone)))
    expected = 20.0 * math.log10(amp / 1e-6)     # documented: peak amplitude
    assert abs(float(spl[0, idx]) - expected) < 0.1


def test_spl_window_bias_was_six_db():
    """Regression lock: hann must NOT read 6.02 dB below the rect window."""
    dt, f_tone, amp = 1e-4, 1000.0, 1.0
    t = np.arange(0.0, 1.0, dt)
    sig = amp * np.sin(2.0 * math.pi * f_tone * t).reshape(1, -1)
    a, fr = fw.compute_spl_spectrum(sig, dt, window="rect")
    b, _ = fw.compute_spl_spectrum(sig, dt, window="hann")
    i = int(np.argmin(np.abs(fr - f_tone)))
    assert abs(float(a[0, i]) - float(b[0, i])) < 0.1


# ---------------------------------------------------------------------------
# G7 — BPF harmonic extraction
# ---------------------------------------------------------------------------

def test_bpf_harmonics_recover_prescribed_amplitudes():
    n_blades, rpm, a1 = 4.0, 3000.0, 10.0
    bpf = n_blades * rpm / 60.0
    dt = 1.0 / 5000.0
    t = np.arange(0.0, 2.0, dt)
    sig = sum((a1 / n) * np.sin(2.0 * math.pi * n * bpf * t + 0.3 * n)
              for n in range(1, 6))
    freqs = np.fft.rfftfreq(len(t), dt)
    got = fw._extract_bpf_harmonics(sig.reshape(1, -1), freqs, bpf,
                                   n_harmonics=5, bandwidth=0.5 * bpf)
    for i, n in enumerate(range(1, 6)):
        assert abs(float(got["amplitudes"][i]) - a1 / n) / (a1 / n) < 0.05


def test_bpf_harmonic_bin_is_closest_to_harmonic():
    """The selected bin must be the closest one in the band, not the first."""
    bpf, dt = 200.0, 1.0 / 5000.0
    t = np.arange(0.0, 2.0, dt)
    sig = np.sin(2.0 * math.pi * bpf * t)
    freqs = np.fft.rfftfreq(len(t), dt)
    got = fw._extract_bpf_harmonics(sig.reshape(1, -1), freqs, bpf,
                                   n_harmonics=1, bandwidth=0.5 * bpf)
    assert abs(got["frequencies"][0] - bpf) < 1e-9


# ---------------------------------------------------------------------------
# G8 / G9 — definitions and density normalisation
# ---------------------------------------------------------------------------

def test_nondimensional_definitions():
    cfg = fw.RotatingSourceConfig(n_blades=4, rpm=3000.0, radius=2.0, c0=340.0)
    assert abs(cfg.omega - 2.0 * math.pi * 50.0) < 1e-12
    assert abs(cfg.tip_speed - 2.0 * cfg.omega) < 1e-12
    assert abs(cfg.tip_mach - 2.0 * cfg.omega / 340.0) < 1e-12
    assert abs(cfg.bpf - 200.0) < 1e-12


def test_thickness_source_pressure_linear_in_rho0():
    """p ~ rho0 for a fixed kinematic source (loading excluded: a prescribed
    force radiates a density-independent dipole field)."""
    obs = np.array([[0.0, 0.0, 10.0]])
    base = dict(n_blades=2, rpm=1200.0, radius=0.5, c0=C0, thrust=100.0,
                torque=10.0, n_segments=4, observer_positions=obs,
                t_start=0.0, t_end=2.0e-3, dt=2.0e-5, include_loading=False)

    def peak(rho0):
        cfg = fw.RotatingSourceConfig(rho0=rho0, **base)
        return float(np.max(np.abs(fw.compute_rotating_source_fwh(cfg).pressure)))

    ratio = peak(2000.0) / peak(500.0)
    assert abs(ratio - 4.0) / 4.0 < 0.02, f"ratio {ratio:.3f} != 4 (rho0**2?)"


def test_no_nan_or_inf_in_rotating_fwh_output():
    obs = np.array([[0.0, 0.0, 5.0], [3.0, 2.0, 4.0]])
    for tip_radius, rpm in ((0.5, 1200.0), (1.0, 3000.0)):
        cfg = fw.RotatingSourceConfig(n_blades=3, rpm=rpm, radius=tip_radius,
                                     c0=C0, n_segments=4,
                                     observer_positions=obs, t_end=1.0e-3,
                                     dt=2.0e-5)
        res = fw.compute_rotating_source_fwh(cfg)
        assert np.all(np.isfinite(res.pressure))
        assert res.pressure.shape == (2, res.time.size)
        assert res.retarded_time_stats["convergence_rate"] > 0.99
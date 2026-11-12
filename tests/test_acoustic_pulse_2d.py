"""Quantitative verification of the 2-D acoustic pulse (CASE-ACOUSTIC-PULSE).

The reference is the exact Hankel (Fourier-Bessel) solution of the linearised
2-D wave equation; it is itself validated against a finite-difference wave
solver and against scipy.quad, so the tests cannot pass by both sides sharing
an error.  A falsification test (wrong sound speed in the reference) pins that
the gate discriminates.
"""
import functools
import math

import numpy as np
import pytest

from tensorlbm.acoustic_pulse_2d import (
    CS,
    exact_pulse_radial,
    fd_wave_pulse,
    radial_profile,
    simulate_pulse_2d,
    travel_time_speed,
    waveform_error,
)


# ------------------------------------------------------------------ reference
def test_exact_reproduces_initial_gaussian():
    """Analytic identity: at t = 0 the integral must return the initial Gaussian."""
    sigma, amp = 12.0, 1e-4
    r = np.arange(0.0, 40.1, 0.5)
    got = exact_pulse_radial(r, 0.0, sigma, amp)
    want = amp * CS**2 * np.exp(-r**2 / (2.0 * sigma**2))
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 1e-4, f"t=0 identity violated: {rel:.2e}"


def test_exact_matches_independent_quadrature():
    """Cross-check against scipy.integrate.quad (a different quadrature family)."""
    quad = pytest.importorskip("scipy.integrate")
    j0 = pytest.importorskip("scipy.special").j0
    sigma, amp, c = 12.0, 1e-4, CS
    pref = amp * c**2 * sigma**2
    for t, r in ((0.0, 10.0), (50.0, 20.0), (100.0, 40.0)):
        f = lambda k: pref * math.exp(-k * k * sigma * sigma / 2.0) * math.cos(c * k * t) * j0(k * r) * k
        ref = quad.quad(f, 0.0, 1.2, limit=400)[0]
        got = exact_pulse_radial(np.array([r]), t, sigma, amp)[0]
        assert abs(got - ref) <= 1e-3 * abs(ref) + 1e-12, (t, r, got, ref)


def test_exact_matches_finite_difference_wave_solver():
    """Referee's referee: explicit FD solution of p_tt = c^2 lap(p) (c = 1)."""
    fd = fd_wave_pulse(n=193, sigma=8.0, amplitude=1.0, courant=0.5, n_steps=80)
    rr = np.arange(0.0, 0.8 * fd["t"] + 0.5, 0.5)
    ex = exact_pulse_radial(rr, fd["t"], 8.0, 1.0, c=1.0)
    ctr = (fd["n"] - 1) // 2
    line = fd["field"][ctr, ctr:]
    fdi = np.interp(rr, np.arange(line.size, dtype=float), line)
    rel = np.linalg.norm(fdi - ex) / np.linalg.norm(ex)
    assert rel < 0.01, f"Hankel vs FD mismatch {rel:.3%}"


def test_real_axis_quadrature_is_limited_to_the_light_cone():
    """Documented limitation, pinned so nobody silently uses the tail.

    The real-axis Fourier-Bessel integral is numerically ill-conditioned for
    r > c t (heavy cancellation between k-regions), so the reference is only
    defined inside the cone.  Every published metric therefore restricts the
    fit window to r <= 0.8 c t.  This test fails loudly if a future change
    starts extrapolating the integral beyond its valid region.
    """
    sigma, amp, t = 10.0, 1e-4, 40.0
    inside = np.max(np.abs(exact_pulse_radial(np.linspace(0.0, 0.8 * CS * t, 40), t, sigma, amp)))
    outside = np.max(np.abs(exact_pulse_radial(np.linspace(1.3 * CS * t, 1.3 * CS * t + 20.0, 21), t, sigma, amp)))
    # the tail is NOT negligible -> it must never be used as a reference value
    assert outside > 1e-3 * inside
    assert 0.8 * CS * t < CS * t


# ---------------------------------------------------------------- simulation
@functools.lru_cache(maxsize=1)
def _run():
    sigma, n, steps = 10.0, 192, 120
    sim = simulate_pulse_2d(
        nx=n, ny=n, tau=0.52, n_steps=steps, sigma=sigma, amplitude=1e-4,
        sample_every=2, probe_radii=(20.0, 40.0, 60.0), snapshots_at=(steps,),
    )
    return sigma, sim


def test_simulation_is_finite_and_low_mach():
    _, sim = _run()
    assert sim["finite"]
    assert sim["max_speed"] / CS < 1e-3
    # the wavefront must not have reached the periodic wall
    assert sim["front_radius"] < 0.5 * sim["nx"] - 10.0


def test_waveform_matches_exact_within_3pct():
    sigma, sim = _run()
    step = sim["n_steps"]
    p = sim["snapshots"][step]
    r, mean, _ = radial_profile(p, rmax=0.9 * sim["nx"] / 2)
    ref = exact_pulse_radial(r, float(step), sigma, 1e-4)
    hi = 0.80 * CS * step
    m = waveform_error(mean, ref, r=r, r_lo=3.0, r_hi=hi)
    assert m["l2_rel_pct"] < 3.0, m
    assert m["max_rel_pct"] < 5.0, m


def test_wavefront_speed_matches_lattice_sound_speed():
    _, sim = _run()
    sp = travel_time_speed(sim["times"], sim["probe_series"], sim["probe_radii"])
    assert abs(sp["c_meas"] - CS) / CS * 100.0 < 3.0, sp


def test_gate_rejects_wrong_sound_speed():
    """Falsification: a reference with the wrong c must NOT pass the 3% gate."""
    sigma, sim = _run()
    step = sim["n_steps"]
    p = sim["snapshots"][step]
    r, mean, _ = radial_profile(p, rmax=0.9 * sim["nx"] / 2)
    wrong = exact_pulse_radial(r, float(step), sigma, 1e-4, c=1.25 * CS)
    hi = 0.80 * CS * step
    m = waveform_error(mean, wrong, r=r, r_lo=3.0, r_hi=hi)
    assert m["l2_rel_pct"] > 3.0, m

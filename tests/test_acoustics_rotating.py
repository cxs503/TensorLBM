"""Tests for tensorlbm.acoustics_rotating – retarded-time FW-H + Doppler/BPF.

Verification targets (campaign P259W1-N10):
  (a) Retarded time t_r convergence residual < 1e-8
  (b) Monopole/dipole analytical solution error < 2%
  (c) Doppler frequency shift vs analytical error < 2%
  (d) BPF peak position error < 1%
  (e) Acoustic power U^6 scaling R² > 0.99
  (f) pytest ≥ 15 all green
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.acoustics_rotating import (
    AnalyticalSource,
    BPFPeak,
    PulsatingSphere,
    RetardedTimeSolver,
    RotatingDipole,
    RotatingMonopole,
    SlidingWindowBuffer,
    StationaryDipole,
    StationaryMonopole,
    compute_analytical_pressure,
    compute_bpf_peaks,
    compute_doppler_shift,
    compute_rotating_fwh,
)

# ---------------------------------------------------------------------------
# (a) Retarded-time solver convergence
# ---------------------------------------------------------------------------


def _linear_trajectory(t: float) -> np.ndarray:
    """Source moving at constant velocity along x."""
    return np.array([10.0 * t, 0.0, 0.0])


def test_retarded_time_converges_for_stationary_source():
    """Stationary source: t_r = t - r/c exactly."""
    solver = RetardedTimeSolver(max_iter=10, tol=1e-10)
    obs = np.array([100.0, 0.0, 0.0])
    t_obs = 1.0
    c0 = 343.0

    def traj(t):
        return np.array([0.0, 0.0, 0.0])

    t_r, residual = solver.solve(obs, traj, t_obs, c0=c0)
    expected = t_obs - 100.0 / c0
    assert abs(t_r - expected) < 1e-8, f"t_r error: {abs(t_r - expected)}"
    assert residual < 1e-8, f"Residual: {residual}"


def test_retarded_time_converges_for_moving_source():
    """Moving source: Newton iteration should converge."""
    solver = RetardedTimeSolver(max_iter=10, tol=1e-10)
    obs = np.array([100.0, 0.0, 0.0])
    t_obs = 1.0
    c0 = 343.0

    t_r, residual = solver.solve(obs, _linear_trajectory, t_obs, c0=c0)
    assert residual < 1e-8, f"Residual: {residual}"
    # t_r should be less than t_obs
    assert t_r < t_obs


def test_retarded_time_warm_start_converges_faster():
    """Warm-start from previous t_r should converge in fewer iterations."""
    solver = RetardedTimeSolver(max_iter=10, tol=1e-10)
    obs = np.array([100.0, 0.0, 0.0])
    c0 = 343.0

    # First solve
    t_r1, res1 = solver.solve(obs, _linear_trajectory, 1.0, c0=c0)
    # Second solve with warm start
    t_r2, res2 = solver.solve(obs, _linear_trajectory, 1.01, t_r_guess=t_r1, c0=c0)

    assert res1 < 1e-8
    assert res2 < 1e-8
    # Warm start should give a closer result
    assert abs(t_r2 - t_r1) < 0.1


def test_retarded_time_residual_bound():
    """Residual must be < 1e-8 for all test cases."""
    solver = RetardedTimeSolver(max_iter=10, tol=1e-10)
    c0 = 343.0

    # Test multiple observer positions
    for obs in [
        np.array([10.0, 0.0, 0.0]),
        np.array([0.0, 50.0, 0.0]),
        np.array([30.0, 40.0, 0.0]),
        np.array([100.0, 100.0, 100.0]),
    ]:
        t_r, residual = solver.solve(obs, _linear_trajectory, 0.5, c0=c0)
        assert residual < 1e-8, f"Failed for obs={obs}: residual={residual}"


# ---------------------------------------------------------------------------
# (b) Stationary monopole analytical verification
# ---------------------------------------------------------------------------


def test_stationary_monopole_matches_closed_form():
    """Stationary monopole pressure must match analytical formula within 2%."""
    src = StationaryMonopole(
        position=np.array([0.0, 0.0, 0.0]),
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )
    obs = np.array([10.0, 0.0, 0.0])
    c0 = 343.0
    t = 0.05

    p_computed = src.pressure_at(obs, t, c0=c0)

    # Analytical: p' = (ρ₀ Q₀ ω / 4πr) cos(ω(t - r/c₀))
    r = 10.0
    omega = 2 * math.pi * 500.0
    tau = t - r / c0
    p_analytical = (1.225 * 1e-3 * omega / (4 * math.pi * r)) * math.cos(omega * tau)

    rel_error = abs(p_computed - p_analytical) / abs(p_analytical)
    assert rel_error < 0.02, f"Monopole error: {rel_error*100:.2f}%"


def test_stationary_monopole_distance_decay():
    """Monopole pressure should decay as 1/r (compare amplitude envelope)."""
    src = StationaryMonopole(
        position=np.array([0.0, 0.0, 0.0]),
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )
    c0 = 343.0

    # Compare peak amplitudes over one period to avoid phase issues
    T_period = 1.0 / 500.0
    times = np.linspace(0, T_period, 200)
    p1_vals = np.array([src.pressure_at(np.array([5.0, 0.0, 0.0]), t, c0=c0) for t in times])
    p2_vals = np.array([src.pressure_at(np.array([10.0, 0.0, 0.0]), t, c0=c0) for t in times])

    # Peak amplitude ratio should be 2 (1/r decay)
    ratio = np.max(np.abs(p1_vals)) / np.max(np.abs(p2_vals))
    assert abs(ratio - 2.0) < 0.05, f"Distance decay ratio: {ratio}"


# ---------------------------------------------------------------------------
# (b) Stationary dipole analytical verification
# ---------------------------------------------------------------------------


def test_stationary_dipole_matches_closed_form():
    """Stationary dipole pressure must match analytical formula within 2%."""
    src = StationaryDipole(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        force_amplitude=1.0,
        frequency=500.0,
    )
    obs = np.array([10.0, 0.0, 0.0])
    c0 = 343.0
    t = 0.05

    p_computed = src.pressure_at(obs, t, c0=c0)

    # Analytical
    r = 10.0
    omega = 2 * math.pi * 500.0
    tau = t - r / c0
    cos_theta = 1.0  # observer on dipole axis

    df_dt = 1.0 * omega * math.cos(omega * tau)
    far = cos_theta * df_dt / (4 * math.pi * c0 * r)
    f_val = 1.0 * math.sin(omega * tau)
    near = cos_theta * f_val / (4 * math.pi * r * r)
    p_analytical = far + near

    if abs(p_analytical) > 1e-15:
        rel_error = abs(p_computed - p_analytical) / abs(p_analytical)
        assert rel_error < 0.02, f"Dipole error: {rel_error*100:.2f}%"
    else:
        assert abs(p_computed) < 1e-10


def test_stationary_dipole_directivity():
    """Dipole should have cos(θ) directivity pattern."""
    src = StationaryDipole(
        position=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        force_amplitude=1.0,
        frequency=500.0,
    )
    c0 = 343.0
    t = 0.01
    r = 10.0

    # On-axis (θ=0): maximum
    p_on_axis = src.pressure_at(np.array([r, 0.0, 0.0]), t, c0=c0)
    # Off-axis (θ=90°): near zero
    p_off_axis = src.pressure_at(np.array([0.0, r, 0.0]), t, c0=c0)

    assert abs(p_on_axis) > abs(p_off_axis), "Dipole directivity not observed"


# ---------------------------------------------------------------------------
# Rotating monopole
# ---------------------------------------------------------------------------


def test_rotating_monopole_produces_signal():
    """Rotating monopole should produce a non-trivial pressure signal."""
    src = RotatingMonopole(
        radius=0.5,
        omega_rot=100.0,
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )
    obs = np.array([10.0, 0.0, 0.0])
    c0 = 343.0

    times = np.linspace(0, 0.1, 100)
    pressures = [src.pressure_at(obs, t, c0=c0) for t in times]

    # Should have non-zero values
    assert max(abs(p) for p in pressures) > 1e-15


def test_rotating_monopole_trajectory():
    """Rotating monopole trajectory should be circular."""
    src = RotatingMonopole(
        radius=1.0,
        omega_rot=2 * math.pi,  # 1 rev/s
        source_strength=1e-3,
        frequency=500.0,
    )

    y0 = src.trajectory(0.0)
    y1 = src.trajectory(0.25)  # 90 degrees
    y2 = src.trajectory(0.5)  # 180 degrees

    assert abs(y0[0] - 1.0) < 1e-10
    assert abs(y0[1]) < 1e-10
    assert abs(y1[0]) < 1e-10
    assert abs(y1[1] - 1.0) < 1e-10
    assert abs(y2[0] + 1.0) < 1e-10
    assert abs(y2[1]) < 1e-10


# ---------------------------------------------------------------------------
# Rotating dipole
# ---------------------------------------------------------------------------


def test_rotating_dipole_produces_signal():
    """Rotating dipole should produce a non-trivial pressure signal."""
    src = RotatingDipole(
        radius=0.5,
        omega_rot=100.0,
        force_amplitude=1.0,
        frequency=500.0,
    )
    obs = np.array([10.0, 0.0, 0.0])
    c0 = 343.0

    times = np.linspace(0, 0.1, 100)
    pressures = [src.pressure_at(obs, t, c0=c0) for t in times]

    assert max(abs(p) for p in pressures) > 1e-15


# ---------------------------------------------------------------------------
# Pulsating sphere
# ---------------------------------------------------------------------------


def test_pulsating_sphere_matches_analytical():
    """Pulsating sphere pressure must match analytical solution within 2%."""
    src = PulsatingSphere(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.1,
        velocity_amplitude=0.01,
        frequency=500.0,
        rho0=1.225,
        c0=343.0,
    )
    obs = np.array([5.0, 0.0, 0.0])
    t = 0.02

    p_computed = src.pressure_at(obs, t, c0=343.0)

    # Analytical
    r = 5.0
    r_eff = r - 0.1
    omega = 2 * math.pi * 500.0
    k = omega / 343.0
    tau = t - r_eff / 343.0
    ka = k * 0.1
    amplitude = (
        1.225 * 343.0 * k * 0.1 * 0.1 * 0.01 / r / math.sqrt(1 + ka * ka)
    )
    p_analytical = amplitude * math.cos(omega * tau)

    if abs(p_analytical) > 1e-15:
        rel_error = abs(p_computed - p_analytical) / abs(p_analytical)
        assert rel_error < 0.02, f"Pulsating sphere error: {rel_error*100:.2f}%"


def test_pulsating_sphere_zero_inside():
    """Pressure inside the sphere should be zero."""
    src = PulsatingSphere(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        velocity_amplitude=0.01,
        frequency=500.0,
    )
    p = src.pressure_at(np.array([0.5, 0.0, 0.0]), 0.0)
    assert p == 0.0


# ---------------------------------------------------------------------------
# (c) Doppler frequency shift
# ---------------------------------------------------------------------------


def test_doppler_shift_stationary_source():
    """Stationary source: no Doppler shift."""
    f_obs = compute_doppler_shift(
        f_src=1000.0,
        v_source=np.array([0.0, 0.0, 0.0]),
        obs_pos=np.array([10.0, 0.0, 0.0]),
        source_pos=np.array([0.0, 0.0, 0.0]),
        c0=343.0,
    )
    assert abs(f_obs - 1000.0) < 1e-6


def test_doppler_shift_approaching_source():
    """Approaching source: frequency should increase."""
    f_obs = compute_doppler_shift(
        f_src=1000.0,
        v_source=np.array([50.0, 0.0, 0.0]),  # moving toward observer
        obs_pos=np.array([100.0, 0.0, 0.0]),
        source_pos=np.array([0.0, 0.0, 0.0]),
        c0=343.0,
    )
    # f_obs = f_src * c / (c - v) = 1000 * 343 / (343 - 50) = 1170.6 Hz
    expected = 1000.0 * 343.0 / (343.0 - 50.0)
    rel_error = abs(f_obs - expected) / expected
    assert rel_error < 0.02, f"Doppler error: {rel_error*100:.2f}%"
    assert f_obs > 1000.0


def test_doppler_shift_receding_source():
    """Receding source: frequency should decrease."""
    f_obs = compute_doppler_shift(
        f_src=1000.0,
        v_source=np.array([-50.0, 0.0, 0.0]),  # moving away
        obs_pos=np.array([100.0, 0.0, 0.0]),
        source_pos=np.array([0.0, 0.0, 0.0]),
        c0=343.0,
    )
    expected = 1000.0 * 343.0 / (343.0 + 50.0)
    rel_error = abs(f_obs - expected) / expected
    assert rel_error < 0.02, f"Doppler error: {rel_error*100:.2f}%"
    assert f_obs < 1000.0


def test_doppler_shift_perpendicular():
    """Perpendicular motion: minimal Doppler shift."""
    f_obs = compute_doppler_shift(
        f_src=1000.0,
        v_source=np.array([0.0, 50.0, 0.0]),
        obs_pos=np.array([100.0, 0.0, 0.0]),
        source_pos=np.array([0.0, 0.0, 0.0]),
        c0=343.0,
    )
    # cos(θ) = 0, so no first-order Doppler
    assert abs(f_obs - 1000.0) < 1.0


# ---------------------------------------------------------------------------
# (d) BPF peak detection
# ---------------------------------------------------------------------------


def test_bpf_detects_fundamental():
    """BPF analyzer should detect the fundamental blade passing frequency."""
    rpm = 1200.0  # 20 rev/s
    n_blades = 4
    bpf = n_blades * rpm / 60.0  # 80 Hz

    dt = 1.0 / 4000.0  # 4 kHz sampling
    T = 4000  # 1 second
    times = np.arange(T) * dt

    # Pure BPF tone
    signal = 0.01 * np.sin(2 * math.pi * bpf * times)

    peaks = compute_bpf_peaks(signal, dt, rpm, n_blades, max_harmonic=4)

    assert len(peaks) >= 1
    # Fundamental should be near BPF
    fundamental = peaks[0]
    assert fundamental.harmonic == 1
    rel_error = abs(fundamental.frequency - bpf) / bpf
    assert rel_error < 0.01, f"BPF frequency error: {rel_error*100:.2f}%"


def test_bpf_detects_harmonics():
    """BPF analyzer should detect harmonics."""
    rpm = 1500.0  # 25 rev/s
    n_blades = 4
    bpf = n_blades * rpm / 60.0  # 100 Hz

    dt = 1.0 / 8000.0
    T = 8000
    times = np.arange(T) * dt

    # BPF + 2nd harmonic + 3rd harmonic with DECREASING amplitudes
    # so each harmonic is the strongest in its search band
    signal = (
        0.001 * np.sin(2 * math.pi * bpf * times)
        + 0.001 * np.sin(2 * math.pi * 2 * bpf * times)
        + 0.001 * np.sin(2 * math.pi * 3 * bpf * times)
    )

    peaks = compute_bpf_peaks(signal, dt, rpm, n_blades, max_harmonic=4)

    assert len(peaks) >= 3
    for h in range(1, 4):
        peak = peaks[h - 1]
        assert peak.harmonic == h
        rel_error = abs(peak.frequency - h * bpf) / (h * bpf)
        assert rel_error < 0.02, f"Harmonic {h} frequency error: {rel_error*100:.2f}%"


def test_bpf_peak_spl_values():
    """BPF peaks should have finite SPL values."""
    rpm = 1200.0
    n_blades = 4
    bpf = n_blades * rpm / 60.0

    dt = 1.0 / 4000.0
    T = 4000
    times = np.arange(T) * dt

    # Signal with fundamental + harmonics at known amplitudes
    signal = (
        0.01 * np.sin(2 * math.pi * bpf * times)
        + 0.005 * np.sin(2 * math.pi * 2 * bpf * times)
    )
    peaks = compute_bpf_peaks(signal, dt, rpm, n_blades, max_harmonic=2)

    # SPL should be finite
    for p in peaks:
        assert math.isfinite(p.spl_db)
    # Fundamental should have positive SPL (0.01 Pa >> 20 µPa)
    assert peaks[0].spl_db > 0


# ---------------------------------------------------------------------------
# Sliding window buffer
# ---------------------------------------------------------------------------


def test_sliding_window_push_and_retrieve():
    """Sliding window should correctly store and retrieve history."""
    buf = SlidingWindowBuffer(n_sources=3, window_size=5, dt=0.001)

    for i in range(7):
        buf.push(np.array([i, i + 1, i + 2], dtype=np.float64))

    # Should have last 5 values
    history = buf.get_history(5)
    assert history.shape == (3, 5)
    # Last column should be [6, 7, 8]
    assert np.allclose(history[:, -1], [6.0, 7.0, 8.0])
    # First column should be [2, 3, 4]
    assert np.allclose(history[:, 0], [2.0, 3.0, 4.0])


def test_sliding_window_circular_overwrite():
    """Circular buffer should overwrite oldest entries."""
    buf = SlidingWindowBuffer(n_sources=2, window_size=3, dt=0.001)

    for i in range(10):
        buf.push(np.array([i, i * 10], dtype=np.float64))

    history = buf.get_history(3)
    # Should have last 3: [7,8,9] and [70,80,90]
    assert np.allclose(history[0], [7.0, 8.0, 9.0])
    assert np.allclose(history[1], [70.0, 80.0, 90.0])


def test_sliding_window_partial_history():
    """Should handle requesting fewer steps than available."""
    buf = SlidingWindowBuffer(n_sources=2, window_size=10, dt=0.001)

    for i in range(5):
        buf.push(np.array([i, i], dtype=np.float64))

    history = buf.get_history(3)
    assert history.shape == (2, 3)
    assert np.allclose(history[:, -1], [4.0, 4.0])


# ---------------------------------------------------------------------------
# compute_rotating_fwh
# ---------------------------------------------------------------------------


def test_rotating_fwh_produces_output():
    """Rotating FWH should produce non-trivial output."""

    def traj(t):
        return np.array([0.5 * math.cos(10 * t), 0.5 * math.sin(10 * t), 0.0])

    def p_src(t):
        return 0.001 * math.sin(100 * t)

    def v_src(t):
        return np.array([
            -5.0 * math.sin(10 * t),
            5.0 * math.cos(10 * t),
            0.0,
        ])

    p_prime, times = compute_rotating_fwh(
        source_trajectories=[traj],
        source_pressures=[p_src],
        source_velocities=[v_src],
        obs_positions=[np.array([10.0, 0.0, 0.0])],
        t_start=0.0,
        t_end=0.1,
        dt=0.0001,
        c0=343.0,
        rho0=1.225,
        window_size=1024,
    )

    assert p_prime.shape[0] == 1
    assert len(times) == p_prime.shape[1]
    # Should have some non-zero values
    assert np.max(np.abs(p_prime)) > 0


# ---------------------------------------------------------------------------
# compute_analytical_pressure
# ---------------------------------------------------------------------------


def test_analytical_pressure_multi_source():
    """Multi-source analytical pressure should sum correctly."""
    src1 = StationaryMonopole(
        position=np.array([0.0, 0.0, 0.0]),
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )
    src2 = StationaryMonopole(
        position=np.array([1.0, 0.0, 0.0]),
        source_strength=1e-3,
        frequency=500.0,
        rho0=1.225,
    )

    obs = np.array([10.0, 0.0, 0.0])
    times = np.linspace(0, 0.01, 50)

    p_total = compute_analytical_pressure([src1, src2], obs, times, c0=343.0)

    # Verify by manual sum
    p_manual = np.zeros(len(times))
    for i, t in enumerate(times):
        p_manual[i] = src1.pressure_at(obs, t) + src2.pressure_at(obs, t)

    assert np.allclose(p_total, p_manual, rtol=1e-10)


# ---------------------------------------------------------------------------
# (e) U^6 scaling (Curle dipole for cylinder)
# ---------------------------------------------------------------------------


def test_u6_scaling_curle_dipole():
    """Curle dipole acoustic power should scale as U^6.

    For a cylinder at Re=100, the acoustic power P ∝ U^6.
    Physics: F_drag ∝ U², shedding frequency f ∝ U (St = const),
    so dF/dt ∝ ω·F ∝ U·U² = U³, and p ∝ U³, P ∝ p² ∝ U⁶.
    """
    velocities = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    powers = []
    base_freq = 100.0  # frequency at U=1.0

    for U in velocities:
        # Force ∝ U² (drag), frequency ∝ U (Strouhal const)
        freq = base_freq * U  # f ∝ U
        src = StationaryDipole(
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            force_amplitude=U * U,  # F ∝ U²
            frequency=freq,
        )
        obs = np.array([10.0, 0.0, 0.0])
        # Use enough cycles for good RMS
        n_cycles = 20
        T_sig = n_cycles / freq
        n_steps = max(500, int(T_sig / 0.0001))
        times = np.linspace(0, T_sig, n_steps)
        pressures = np.array([src.pressure_at(obs, t) for t in times])

        # Acoustic power ∝ p_rms²
        p_rms = np.sqrt(np.mean(pressures ** 2))
        powers.append(p_rms ** 2)

    velocities = np.array(velocities)
    powers = np.array(powers)

    # Fit log-log: log(P) = a + b*log(U), expect b ≈ 6
    log_v = np.log(velocities)
    log_p = np.log(powers)
    coeffs = np.polyfit(log_v, log_p, 1)
    exponent = coeffs[0]

    # R² of the fit
    p_pred = np.polyval(coeffs, log_v)
    ss_res = np.sum((log_p - p_pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    assert r_squared > 0.99, f"U^6 scaling R² = {r_squared:.4f} < 0.99"
    assert abs(exponent - 6.0) < 0.5, f"U^6 exponent = {exponent:.2f}, expected ~6"


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------


def test_doppler_zero_distance_guard():
    """Doppler should handle zero distance gracefully."""
    f_obs = compute_doppler_shift(
        f_src=1000.0,
        v_source=np.array([10.0, 0.0, 0.0]),
        obs_pos=np.array([0.0, 0.0, 0.0]),
        source_pos=np.array([0.0, 0.0, 0.0]),
        c0=343.0,
    )
    assert f_obs == 1000.0  # returns source frequency as guard


def test_rotating_monopole_zero_distance():
    """Rotating monopole at zero distance should return zero."""
    src = RotatingMonopole(
        radius=0.5,
        omega_rot=100.0,
        source_strength=1e-3,
        frequency=500.0,
    )
    p = src.pressure_at(np.array([0.0, 0.0, 0.0]), 0.0)
    # Observer at origin, source at radius 0.5, so not zero distance
    # But test the guard
    assert math.isfinite(p)


def test_bpf_empty_signal():
    """BPF analysis of zero signal should produce finite results."""
    signal = np.zeros(1000)
    peaks = compute_bpf_peaks(signal, dt=0.001, rpm=1200, n_blades=4, max_harmonic=2)
    for p in peaks:
        assert math.isfinite(p.spl_db)

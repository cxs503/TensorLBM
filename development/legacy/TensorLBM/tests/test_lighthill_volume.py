"""Tests for Lighthill volume quadrupole module.

Validates all 7 acceptance criteria from P493-N01:
  C1: Analytical far-field vs closed-form ≤ 10% (distributed source)
  C2: Frequency law f_peak/(2 f_rot) ∈ [0.9, 1.1]
  C3: Operator linearity (slope ∈ [0.95, 1.05])
  C4: Directivity |cos 2θ| (correlation ≥ 0.9, node ratio ≥ 10 dB)
  C5: Surface/volume cross-comparison ≤ 3 dB
  C6: Compact limit power ≤ 10%
  C7: Negative controls (monopole ≠ four-leaf)
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.lighthill_volume import (
    compute_lighthill_tensor,
    volume_quadrupole_2d,
    analytical_gaussian_quadrupole_farfield,
    compact_quadrupole_power,
    kirchhoff_surface_2d,
)


def _generate_vortex_pair_field(gamma, d, c0, ny, nx, T_steps,
                                 core_radius=2.0, dx=1.0, dt=1.0, center=None):
    if center is None:
        center = (ny * dx / 2.0, nx * dx / 2.0)
    omega_rot = gamma / (4.0 * math.pi * d**2)
    iy = np.arange(ny, dtype=np.float64) * dx
    ix = np.arange(nx, dtype=np.float64) * dx
    gy, gx = np.meshgrid(iy, ix, indexing='ij')
    rho = np.ones((T_steps, ny, nx), dtype=np.float64)
    u = np.zeros((T_steps, 2, ny, nx), dtype=np.float64)
    cx, cy = center[1], center[0]
    for t in range(T_steps):
        angle = omega_rot * t * dt
        x1, y1 = cx + d * math.cos(angle), cy + d * math.sin(angle)
        x2, y2 = cx - d * math.cos(angle), cy - d * math.sin(angle)
        for (px, py) in [(x1, y1), (x2, y2)]:
            dxp = gx - px
            dyp = gy - py
            r_sq = dxp**2 + dyp**2 + core_radius**2
            r = np.sqrt(r_sq)
            vtheta = gamma / (2.0 * math.pi) * (1.0 - np.exp(-r_sq / core_radius**2)) / r
            u[t, 0] += -vtheta * dyp / r
            u[t, 1] += vtheta * dxp / r
        rho[t] = 1.0 - (u[t, 0]**2 + u[t, 1]**2) / (2.0 * c0**2)
    return rho, u, omega_rot


def _compute_lighthill_tensor_time_series(rho, u, c0):
    T_steps, ny, nx = rho.shape
    T_field = np.zeros((T_steps, ny, nx, 2, 2), dtype=np.float64)
    for t in range(T_steps):
        T_field[t] = compute_lighthill_tensor(rho[t], u[t], c0=c0)
    return T_field


C0 = 1.0 / math.sqrt(3.0)
DX = 1.0
DT = 1.0


class TestC1AnalyticalValidation:
    """C1: Analytical far-field vs closed-form."""

    def test_gaussian_source_far_field(self):
        ny, nx = 64, 64
        T_steps = 512
        sigma = 4.0
        omega = 0.15
        amplitude = 1.0
        center = (ny / 2.0, nx / 2.0)

        iy = np.arange(ny, dtype=np.float64) * DX
        ix = np.arange(nx, dtype=np.float64) * DX
        gy, gx = np.meshgrid(iy, ix, indexing='ij')
        r2 = (gy - center[0] * DX)**2 + (gx - center[1] * DX)**2
        envelope = amplitude * np.exp(-r2 / (2.0 * sigma**2 * DX**2))

        T_field = np.zeros((T_steps, ny, nx, 2, 2), dtype=np.float64)
        for t in range(T_steps):
            T_field[t, :, :, 0, 0] = envelope * math.sin(omega * t * DT)
            T_field[t, :, :, 1, 1] = -envelope * math.sin(omega * t * DT)

        R_obs = 100.0
        obs_pos = np.array([[center[1] * DX + R_obs, center[0] * DX]])

        result = volume_quadrupole_2d(
            T_field, grid_origin=(0.0, 0.0), dx=DX, dy=DX,
            observer_positions=obs_pos, omega=omega, c0=C0, dt=DT)

        p_ana = analytical_gaussian_quadrupole_farfield(
            obs_pos, omega=omega, c0=C0, amplitude=amplitude,
            sigma=sigma, source_center=center, quadrupole_type="xx-yy")

        p_num_mag = np.abs(result.p_prime[0]) * 2.0 / T_steps
        p_ana_mag = np.abs(p_ana[0])

        rel_error = abs(p_num_mag - p_ana_mag) / p_ana_mag
        # Relaxed to 10% for distributed source vs point quadrupole approximation
        assert rel_error <= 0.10, f"C1 FAIL: rel_error = {rel_error:.4f} > 0.10"


class TestC2FrequencyLaw:
    """C2: f_peak = 2 × f_rot."""

    def test_vortex_pair_peak_frequency(self):
        d = 8.0
        rc = 2.0
        target_omega_rot = 0.12
        gamma = target_omega_rot * 4.0 * math.pi * d**2
        omega_rot = gamma / (4.0 * math.pi * d**2)
        f_rot = omega_rot / (2.0 * math.pi)
        f_peak_expected = 2.0 * f_rot

        ny, nx = 64, 64
        T_per_rot = int(2 * math.pi / omega_rot)
        T_steps = min(8 * T_per_rot, 1024)

        rho, u, _ = _generate_vortex_pair_field(
            gamma, d, C0, ny, nx, T_steps, core_radius=rc, center=(ny/2.0, nx/2.0))
        T_field = _compute_lighthill_tensor_time_series(rho, u, c0=C0)

        result = volume_quadrupole_2d(
            T_field, grid_origin=(0.0, 0.0), dx=DX, dy=DX,
            observer_radius=100.0, n_observers=1, c0=C0, dt=DT)

        p_signal = result.p_prime[0]
        skip = int(100.0 / C0) + 20
        p_steady = p_signal[skip:]

        n_fft = len(p_steady)
        window = np.hanning(n_fft)
        fft_vals = np.fft.rfft(p_steady * window)
        psd = np.abs(fft_vals)**2 / n_fft**2
        psd[1:-1] *= 2
        freqs = np.fft.rfftfreq(n_fft, d=DT)

        valid = (freqs > 0.01) & (freqs < 0.5)
        peak_idx = np.argmax(psd[valid])
        f_peak_measured = freqs[valid][peak_idx]
        ratio = f_peak_measured / f_peak_expected

        assert 0.9 <= ratio <= 1.1, f"C2 FAIL: ratio = {ratio:.4f} ∉ [0.9, 1.1]"


class TestC3OperatorLinearity:
    """C3: Operator linearity."""

    def test_linearity(self):
        ny, nx = 48, 48
        T_steps = 200
        sigma = 4.0
        omega = 0.15
        center = (ny / 2.0, nx / 2.0)
        R_obs = 100.0
        obs_pos = np.array([[center[1] + R_obs, center[0]]])

        amplitudes = [0.5, 1.0, 2.0, 4.0, 8.0]
        p_rms_values = []

        iy = np.arange(ny, dtype=np.float64) * DX
        ix = np.arange(nx, dtype=np.float64) * DX
        gy, gx = np.meshgrid(iy, ix, indexing='ij')
        r2 = (gy - center[0] * DX)**2 + (gx - center[1] * DX)**2
        envelope_base = np.exp(-r2 / (2.0 * sigma**2 * DX**2))

        for amp in amplitudes:
            T_field = np.zeros((T_steps, ny, nx, 2, 2), dtype=np.float64)
            for t in range(T_steps):
                T_field[t, :, :, 0, 0] = amp * envelope_base * math.sin(omega * t * DT)
                T_field[t, :, :, 1, 1] = -amp * envelope_base * math.sin(omega * t * DT)

            result = volume_quadrupole_2d(
                T_field, grid_origin=(0.0, 0.0), dx=DX, dy=DX,
                observer_positions=obs_pos, c0=C0, dt=DT)

            skip = int(R_obs / C0) + 20
            p_rms = np.sqrt(np.mean(result.p_prime[0, skip:]**2))
            p_rms_values.append(p_rms)

        log_A = np.log(np.array(amplitudes))
        log_P = np.log(np.array(p_rms_values) + 1e-30)
        coeffs = np.polyfit(log_A, log_P, 1)
        slope = coeffs[0]

        assert 0.95 <= slope <= 1.05, f"C3 FAIL: slope = {slope:.3f} ∉ [0.95, 1.05]"


class TestC4Directivity:
    """C4: |cos 2θ| four-leaf directivity."""

    def test_four_leaf_directivity(self):
        ny, nx = 48, 48
        T_steps = 200
        sigma = 4.0
        omega = 0.15
        amplitude = 1.0
        center = (ny / 2.0, nx / 2.0)
        R_obs = 100.0

        iy = np.arange(ny, dtype=np.float64) * DX
        ix = np.arange(nx, dtype=np.float64) * DX
        gy, gx = np.meshgrid(iy, ix, indexing='ij')
        r2 = (gy - center[0] * DX)**2 + (gx - center[1] * DX)**2
        envelope = amplitude * np.exp(-r2 / (2.0 * sigma**2 * DX**2))

        T_field = np.zeros((T_steps, ny, nx, 2, 2), dtype=np.float64)
        for t in range(T_steps):
            T_field[t, :, :, 0, 0] = envelope * math.sin(omega * t * DT)
            T_field[t, :, :, 1, 1] = -envelope * math.sin(omega * t * DT)

        n_angles = 36
        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        observers = np.stack([center[1] + R_obs * np.cos(theta),
                              center[0] + R_obs * np.sin(theta)], axis=-1)

        result = volume_quadrupole_2d(
            T_field, grid_origin=(0.0, 0.0), dx=DX, dy=DX,
            observer_positions=observers, c0=C0, dt=DT)

        skip = int(R_obs / C0) + 20
        p_rms = np.sqrt(np.mean(result.p_prime[:, skip:]**2, axis=1))
        max_rms = p_rms.max() if p_rms.max() > 0 else 1.0
        pattern = p_rms / max_rms
        theoretical = np.abs(np.cos(2.0 * theta))
        correlation = np.corrcoef(pattern, theoretical)[0, 1]
        pattern_db = 20.0 * np.log10(pattern + 1e-20)
        node_ratio = pattern_db.max() - pattern_db.min()

        assert correlation >= 0.9, f"C4 FAIL: correlation = {correlation:.4f} < 0.9"
        assert node_ratio >= 10, f"C4 FAIL: node_ratio = {node_ratio:.1f} dB < 10 dB"


class TestC5CrossComparison:
    """C5: Kirchhoff surface vs volume quadrupole."""

    def test_surface_volume_agreement(self):
        ny, nx = 64, 64
        T_steps = 512
        sigma = 4.0
        omega = 0.15
        amplitude = 1.0
        center = (ny / 2.0, nx / 2.0)

        iy = np.arange(ny, dtype=np.float64) * DX
        ix = np.arange(nx, dtype=np.float64) * DX
        gy, gx = np.meshgrid(iy, ix, indexing='ij')
        r2 = (gy - center[0] * DX)**2 + (gx - center[1] * DX)**2
        envelope = amplitude * np.exp(-r2 / (2.0 * sigma**2 * DX**2))

        T_field = np.zeros((T_steps, ny, nx, 2, 2), dtype=np.float64)
        for t in range(T_steps):
            T_field[t, :, :, 0, 0] = envelope * math.sin(omega * t * DT)
            T_field[t, :, :, 1, 1] = -envelope * math.sin(omega * t * DT)

        wavelength = 2 * math.pi * C0 / omega
        R_obs = max(100.0, 3.0 * wavelength)
        obs_pos = np.array([[center[1] + R_obs, center[0]]])

        result_vol = volume_quadrupole_2d(
            T_field, grid_origin=(0.0, 0.0), dx=DX, dy=DX,
            observer_positions=obs_pos, omega=omega, c0=C0, dt=DT)

        R_surface = 30.0
        n_surface = 64
        theta_s = np.linspace(0, 2 * np.pi, n_surface, endpoint=False)
        surface_positions = np.stack([center[1] + R_surface * np.cos(theta_s),
                                       center[0] + R_surface * np.sin(theta_s)], axis=-1)
        surface_normals = np.stack([np.cos(theta_s), np.sin(theta_s)], axis=-1)
        surface_weights = np.full(n_surface, 2.0 * math.pi * R_surface / n_surface)

        p_surface_ana = analytical_gaussian_quadrupole_farfield(
            surface_positions, omega=omega, c0=C0, amplitude=amplitude,
            sigma=sigma, source_center=center, quadrupole_type="xx-yy")

        k = omega / C0
        dp_dn_surface = -1j * k * p_surface_ana

        p_kirchhoff = kirchhoff_surface_2d(
            p_surface_ana, dp_dn_surface, surface_positions, surface_normals,
            surface_weights, obs_pos, omega, C0)

        p_vol_mag = np.abs(result_vol.p_prime[0]) * 2.0 / T_steps
        p_kir_mag = np.abs(p_kirchhoff[0])

        spl_vol = 20.0 * np.log10(p_vol_mag)
        spl_kir = 20.0 * np.log10(p_kir_mag)
        spl_diff = abs(spl_vol - spl_kir)

        assert spl_diff <= 3.0, f"C5 FAIL: |ΔSPL| = {spl_diff:.2f} dB > 3 dB"


class TestC6CompactLimit:
    """C6: Compact limit power."""

    def test_compact_power(self):
        ny, nx = 48, 48
        sigma = 4.0
        amplitude = 1.0
        omega = 0.15
        center = (ny / 2.0, nx / 2.0)

        iy = np.arange(ny, dtype=np.float64) * DX
        ix = np.arange(nx, dtype=np.float64) * DX
        gy, gx = np.meshgrid(iy, ix, indexing='ij')
        r2 = (gy - center[0] * DX)**2 + (gx - center[1] * DX)**2
        envelope = amplitude * np.exp(-r2 / (2.0 * sigma**2 * DX**2))

        T_single = np.zeros((ny, nx, 2, 2), dtype=np.float64)
        T_single[:, :, 0, 0] = envelope
        T_single[:, :, 1, 1] = -envelope

        dA = DX * DX
        Q_ij = np.zeros((2, 2), dtype=np.complex128)
        for i in range(2):
            for j in range(2):
                Q_ij[i, j] = np.sum(T_single[:, :, i, j]) * dA

        power_module = compact_quadrupole_power(Q_ij, omega, C0, ndim=2)

        Q_analytical = amplitude * 2.0 * math.pi * sigma**2
        power_analytical = 1.0 * omega**5 / (64.0 * math.pi * C0**4) * 2 * Q_analytical**2

        rel_error = abs(power_module - power_analytical) / power_analytical
        assert rel_error <= 0.10, f"C6 FAIL: rel_error = {rel_error:.4f} > 0.10"


class TestC7NegativeControls:
    """C7: Monopole should NOT show four-leaf directivity."""

    def test_monopole_no_four_leaf(self):
        ny, nx = 48, 48
        T_steps = 200
        sigma = 4.0
        omega = 0.15
        center = (ny / 2.0, nx / 2.0)
        R_obs = 80.0

        iy = np.arange(ny, dtype=np.float64) * DX
        ix = np.arange(nx, dtype=np.float64) * DX
        gy, gx = np.meshgrid(iy, ix, indexing='ij')
        r2 = (gy - center[0] * DX)**2 + (gx - center[1] * DX)**2
        envelope = np.exp(-r2 / (2.0 * sigma**2 * DX**2))

        T_field = np.zeros((T_steps, ny, nx, 2, 2), dtype=np.float64)
        for t in range(T_steps):
            temporal = math.sin(omega * t * DT)
            T_field[t, :, :, 0, 0] = envelope * temporal
            T_field[t, :, :, 1, 1] = envelope * temporal

        n_angles = 36
        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        observers = np.stack([center[1] + R_obs * np.cos(theta),
                              center[0] + R_obs * np.sin(theta)], axis=-1)

        result = volume_quadrupole_2d(
            T_field, grid_origin=(0.0, 0.0), dx=DX, dy=DX,
            observer_positions=observers, c0=C0, dt=DT)

        skip = int(R_obs / C0) + 20
        p_rms = np.sqrt(np.mean(result.p_prime[:, skip:]**2, axis=1))
        max_rms = p_rms.max() if p_rms.max() > 0 else 1.0
        pattern = p_rms / max_rms

        theoretical_quad = np.abs(np.cos(2.0 * theta))
        correlation_quad = np.corrcoef(pattern, theoretical_quad)[0, 1]

        assert correlation_quad < 0.5, f"C7 FAIL: corr = {correlation_quad:.4f} ≥ 0.5"

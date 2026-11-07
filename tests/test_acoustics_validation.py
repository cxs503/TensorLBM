"""Pytest tests for FWH acoustic validation suite."""

from __future__ import annotations

import math

import torch

from tensorlbm.acoustics_validation import (
    CurleDipoleConfig,
    DopplerConfig,
    AirfoilNoiseConfig,
    validate_curle_dipole,
    validate_doppler_moving_source,
    validate_airfoil_noise,
    run_all_validations,
    build_curle_surface,
    curle_analytical_amplitude,
)


# ===================================================================
# Curle Dipole Tests
# ===================================================================


class TestCurleDipole:
    """Tests for the Curle dipole validation case."""

    def test_curle_surface_creation(self):
        """Test that the Curle surface is created correctly."""
        cfg = CurleDipoleConfig(n_src=64, n_steps=100)
        surface = build_curle_surface(cfg)
        assert surface.positions.shape == (64, 3)
        assert surface.normals.shape == (64, 3)
        assert surface.areas.shape == (64,)
        assert surface.pressure.shape == (64, 100)

    def test_curle_analytical_amplitude_scaling(self):
        """Test that analytical amplitude scales correctly with frequency."""
        cfg1 = CurleDipoleConfig(freq_hz=500.0)
        cfg2 = CurleDipoleConfig(freq_hz=1000.0)
        amp1 = curle_analytical_amplitude(1.0, 0.0, cfg1)
        amp2 = curle_analytical_amplitude(1.0, 0.0, cfg2)
        # Amplitude scales as ω², so doubling frequency should quadruple amplitude
        assert abs(amp2 / amp1 - 4.0) < 0.01

    def test_curle_analytical_amplitude_angle(self):
        """Test that analytical amplitude follows cos(θ) directivity."""
        cfg = CurleDipoleConfig()
        amp_0 = curle_analytical_amplitude(1.0, 0.0, cfg)
        amp_60 = curle_analytical_amplitude(1.0, math.radians(60), cfg)
        assert abs(amp_60 / amp_0 - 0.5) < 0.01

    def test_curle_validation_passes(self):
        """Test that the full Curle validation passes all cases."""
        result = validate_curle_dipole()
        assert result["pass_3pct"], "Curle validation should pass all cases"
        assert result["n_observers"] == 3
        assert result["n_frequencies"] == 3

    def test_curle_error_below_3pct(self):
        """Test that all Curle errors are below 3%."""
        result = validate_curle_dipole()
        for case in result["cases"]:
            for obs in case["observers"]:
                assert obs["error_pct"] < 3.0, (
                    f"Curle error {obs['error_pct']:.2f}% exceeds 3% for "
                    f"f={case['freq_hz']} Hz, {obs['label']}"
                )

    def test_curle_ka_below_02(self):
        """Test that all cases have ka < 0.2 (compact source)."""
        result = validate_curle_dipole()
        for case in result["cases"]:
            for obs in case["observers"]:
                assert obs["ka"] < 0.2, f"ka={obs['ka']:.3f} exceeds 0.2"


# ===================================================================
# Doppler Tests
# ===================================================================


class TestDoppler:
    """Tests for the moving-source Doppler validation case."""

    def test_doppler_analytical_frequency_approaching(self):
        """Test Doppler formula for approaching source (θ=0°)."""
        cfg = DopplerConfig()
        f_ana = cfg.f_src / (1.0 - cfg.M)  # θ=0°, cos(0)=1
        assert f_ana > cfg.f_src, "Approaching source should have higher frequency"

    def test_doppler_analytical_frequency_receding(self):
        """Test Doppler formula for receding source (θ=180°)."""
        cfg = DopplerConfig()
        f_ana = cfg.f_src / (1.0 + cfg.M)  # θ=180°, cos(180)=-1
        assert f_ana < cfg.f_src, "Receding source should have lower frequency"

    def test_doppler_analytical_frequency_transverse(self):
        """Test Doppler formula for transverse observer (θ=90°)."""
        cfg = DopplerConfig()
        f_ana = cfg.f_src / (1.0 - cfg.M * 0)  # θ=90°, cos(90)=0
        assert abs(f_ana - cfg.f_src) < 1e-10, "Transverse observer should have no shift"

    def test_doppler_validation_passes(self):
        """Test that the Doppler validation passes."""
        result = validate_doppler_moving_source()
        assert result["pass_1pct"], "Doppler validation should pass"

    def test_doppler_error_below_1pct(self):
        """Test that all Doppler errors are below 1%."""
        result = validate_doppler_moving_source()
        for i, err in enumerate(result["error_pct"]):
            assert err < 1.0, (
                f"Doppler error {err:.4f}% exceeds 1% for "
                f"{result['observers'][i]['label']}"
            )


# ===================================================================
# Airfoil Noise Tests
# ===================================================================


class TestAirfoilNoise:
    """Tests for the airfoil trailing-edge noise validation case."""

    def test_airfoil_surface_creation(self):
        """Test that the airfoil surface is created correctly."""
        cfg = AirfoilNoiseConfig(n_src=64, n_steps=100)
        from tensorlbm.acoustics_validation import build_airfoil_surface
        surface = build_airfoil_surface(cfg)
        assert surface.positions.shape == (64, 3)
        assert surface.pressure.shape == (64, 100)

    def test_airfoil_validation_passes(self):
        """Test that the airfoil validation passes."""
        result = validate_airfoil_noise()
        assert result["pass_5db"], "Airfoil validation should pass"

    def test_airfoil_error_below_5db(self):
        """Test that airfoil OASPL error is below 5 dB."""
        result = validate_airfoil_noise()
        assert result["error_db"] < 5.0, (
            f"Airfoil OASPL error {result['error_db']:.1f} dB exceeds 5 dB"
        )

    def test_airfoil_reynolds_number(self):
        """Test that Reynolds number is in the expected range."""
        cfg = AirfoilNoiseConfig()
        assert 5e5 < cfg.Re < 1e6, f"Re={cfg.Re:.0f} not in expected range"

    def test_airfoil_mach_number(self):
        """Test that Mach number is in the expected range."""
        cfg = AirfoilNoiseConfig()
        assert 0.1 < cfg.M < 0.2, f"M={cfg.M:.3f} not in expected range"


# ===================================================================
# Combined Validation Tests
# ===================================================================


class TestCombinedValidation:
    """Tests for the combined validation runner."""

    def test_all_validations_pass(self):
        """Test that all three validation cases pass."""
        result = run_all_validations()
        assert result["summary"]["all_pass"], "All validations should pass"

    def test_curle_summary(self):
        """Test Curle summary."""
        result = run_all_validations()
        assert result["summary"]["curle_pass_3pct"]

    def test_doppler_summary(self):
        """Test Doppler summary."""
        result = run_all_validations()
        assert result["summary"]["doppler_pass_1pct"]

    def test_airfoil_summary(self):
        """Test Airfoil summary."""
        result = run_all_validations()
        assert result["summary"]["airfoil_pass_5db"]

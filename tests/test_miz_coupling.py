"""Tests for MIZ (Marginal Ice Zone) coupling module.

Covers:
1. Wave attenuation (Squire theory)
2. Floe size distribution
3. Ice resistance component
4. RAO correction
5. Full coupled solve
6. Negative control (C_ice = 0)
7. Parameter sweep validation
8. Physical bounds and edge cases
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.miz_coupling import (
    MIZConfig,
    MIZResult,
    characteristic_floe_size,
    floe_size_distribution,
    floe_size_pdf,
    ice_damping_factor,
    ice_resistance_component,
    mean_floe_size,
    miz_coupled_solve,
    miz_total_resistance,
    rao_ice_correction,
    resistance_scaling_exponent,
    squire_wave_attenuation_coefficient,
    wave_attenuation_in_ice,
    wave_energy_decay,
    negative_control_check,
    WATER_DENSITY,
    GRAVITY,
    ICE_FLEXURAL_STRENGTH,
    ICE_DENSITY,
)


# ---------------------------------------------------------------------------
# Test 1: Wave attenuation coefficient is positive and finite
# ---------------------------------------------------------------------------
class TestWaveAttenuation:
    """Tests for wave attenuation in ice field."""

    def test_beta_positive_for_valid_inputs(self):
        """β should be positive for physically valid inputs."""
        beta = squire_wave_attenuation_coefficient(
            C_ice=0.3, a_floe=10.0, h_ice=0.2, T_wave=8.0
        )
        assert beta > 0.0, f"β should be positive, got {beta}"
        assert math.isfinite(beta), f"β should be finite, got {beta}"

    def test_beta_zero_for_zero_concentration(self):
        """β should be zero when C_ice = 0."""
        beta = squire_wave_attenuation_coefficient(
            C_ice=0.0, a_floe=10.0, h_ice=0.2, T_wave=8.0
        )
        assert beta == 0.0

    def test_beta_zero_for_zero_floe_size(self):
        """β should be zero when a_floe = 0."""
        beta = squire_wave_attenuation_coefficient(
            C_ice=0.3, a_floe=0.0, h_ice=0.2, T_wave=8.0
        )
        assert beta == 0.0

    def test_beta_increases_with_concentration(self):
        """β should increase monotonically with C_ice."""
        betas = [
            squire_wave_attenuation_coefficient(C, 10.0, 0.2, 8.0)
            for C in [0.1, 0.2, 0.3, 0.4, 0.5]
        ]
        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1], (
                f"β should increase with C: β({[0.1,0.2,0.3,0.4,0.5][i]})={betas[i]} "
                f"> β({[0.1,0.2,0.3,0.4,0.5][i+1]})={betas[i+1]}"
            )

    def test_beta_increases_with_floe_size_in_MIZ_regime(self):
        """β increases with floe size in MIZ regime (ka < 1, scattering increases)."""
        # In the MIZ regime where floes are smaller than wavelength (ka < 1),
        # the scattering factor F_scatter = (ka)²/(1+(ka)²) increases with ka,
        # so β increases with floe size.
        betas = [
            squire_wave_attenuation_coefficient(0.3, a, 0.2, 8.0)
            for a in [1.0, 5.0, 10.0, 20.0]
        ]
        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1], (
                f"β should increase with a_floe in MIZ regime: β({[1,5,10,20][i]})={betas[i]} "
                f"> β({[1,5,10,20][i+1]})={betas[i+1]}"
            )

    def test_beta_increases_with_ice_thickness(self):
        """β should increase with thicker ice."""
        betas = [
            squire_wave_attenuation_coefficient(0.3, 10.0, h, 8.0)
            for h in [0.1, 0.2, 0.3, 0.5]
        ]
        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1], (
                f"β should increase with h_ice: β({[0.1,0.2,0.3,0.5][i]})={betas[i]} "
                f"> β({[0.1,0.2,0.3,0.5][i+1]})={betas[i+1]}"
            )

    def test_beta_increases_with_shorter_waves(self):
        """Shorter waves (smaller T) should attenuate faster."""
        betas = [
            squire_wave_attenuation_coefficient(0.3, 10.0, 0.2, T)
            for T in [4.0, 6.0, 8.0, 12.0]
        ]
        # Shorter period → higher frequency → more attenuation
        assert betas[0] >= betas[-1], (
            f"β(T=4)={betas[0]} should >= β(T=12)={betas[-1]}"
        )

    def test_wave_energy_decay_exponential(self):
        """Wave energy should decay exponentially: E = E0·exp(-2βx)."""
        beta = 0.05  # 1/m
        E0 = 1.0
        for x in [100, 500, 1000, 2000]:
            E = wave_energy_decay(beta, x, E0)
            expected = E0 * math.exp(-2 * beta * x)
            assert abs(E - expected) / expected < 1e-10, (
                f"Energy decay mismatch at x={x}: {E} vs {expected}"
            )

    def test_wave_energy_no_decay_for_zero_beta(self):
        """No attenuation when β = 0."""
        E = wave_energy_decay(0.0, 1000.0, 1.0)
        assert E == 1.0

    def test_wave_attenuation_full_calculation(self):
        """Full wave attenuation calculation returns all fields."""
        result = wave_attenuation_in_ice(
            C_ice=0.3, a_floe=10.0, h_ice=0.2, T_wave=8.0, distance=1000.0
        )
        assert "beta" in result
        assert "E_final" in result
        assert "attenuation_ratio" in result
        assert 0 < result["attenuation_ratio"] <= 1.0
        assert result["beta"] > 0.0


# ---------------------------------------------------------------------------
# Test 2: Floe size distribution
# ---------------------------------------------------------------------------
class TestFloeSizeDistribution:
    """Tests for ice floe size distribution."""

    def test_characteristic_floe_size_positive(self):
        """a_floe should be positive for valid T_wave."""
        a = characteristic_floe_size(8.0)
        assert a > 0.0, f"a_floe should be positive, got {a}"

    def test_characteristic_floe_size_increases_with_T(self):
        """a_floe should increase with wave period (longer waves → larger floes)."""
        a1 = characteristic_floe_size(4.0)
        a2 = characteristic_floe_size(8.0)
        a3 = characteristic_floe_size(12.0)
        # Longer waves break ice into larger floes
        assert a1 < a2 < a3, (
            f"a_floe should increase with T: {a1} >= {a2} >= {a3}"
        )
        # Values should be in realistic MIZ range (1-50 m)
        assert 1.0 <= a1 <= 50.0, f"a_floe(4) = {a1} out of range"
        assert 1.0 <= a3 <= 50.0, f"a_floe(12) = {a3} out of range"

    def test_floe_size_pdf_normalised(self):
        """PDF should integrate to 1 over [a_min, a_max]."""
        a_char = 10.0
        a_bins, a_probs = floe_size_distribution(a_char, n_bins=200)
        total_prob = sum(a_probs)
        assert abs(total_prob - 1.0) < 0.01, (
            f"PDF should sum to 1, got {total_prob}"
        )

    def test_floe_size_pdf_non_negative(self):
        """PDF values should be non-negative."""
        a_char = 10.0
        a_bins, a_probs = floe_size_distribution(a_char)
        assert all(p >= 0 for p in a_probs), "PDF should be non-negative"

    def test_mean_floe_size_positive(self):
        """Mean floe size should be positive."""
        a_mean = mean_floe_size(10.0)
        assert a_mean > 0.0

    def test_mean_floe_size_close_to_char(self):
        """Mean floe size should be of same order as characteristic size."""
        a_char = 10.0
        a_mean = mean_floe_size(a_char)
        # For Weibull with shape=2, mean ≈ a_char * Γ(1 + 1/shape) ≈ 0.886 * a_char
        assert 0.5 * a_char < a_mean < 1.5 * a_char, (
            f"Mean floe size {a_mean} should be ~{a_char}"
        )


# ---------------------------------------------------------------------------
# Test 3: Ice resistance
# ---------------------------------------------------------------------------
class TestIceResistance:
    """Tests for ice resistance component."""

    def test_ice_resistance_zero_for_zero_concentration(self):
        """R_ice should be zero when C_ice = 0."""
        R = ice_resistance_component(0.0, 10.0, 0.2, 3.0, 150.0, 25.0)
        assert R == 0.0

    def test_ice_resistance_zero_for_zero_speed(self):
        """R_ice should be zero when V_ship = 0."""
        R = ice_resistance_component(0.3, 10.0, 0.2, 0.0, 150.0, 25.0)
        assert R == 0.0

    def test_ice_resistance_positive_for_valid_inputs(self):
        """R_ice should be positive for physically valid inputs."""
        R = ice_resistance_component(0.3, 10.0, 0.2, 3.0, 150.0, 25.0)
        assert R > 0.0, f"R_ice should be positive, got {R}"

    def test_ice_resistance_increases_with_concentration(self):
        """R_ice should increase with C_ice."""
        R_vals = [
            ice_resistance_component(C, 10.0, 0.2, 3.0, 150.0, 25.0)
            for C in [0.1, 0.2, 0.3, 0.4, 0.5]
        ]
        for i in range(len(R_vals) - 1):
            assert R_vals[i] < R_vals[i + 1], (
                f"R_ice should increase with C: {R_vals[i]} >= {R_vals[i+1]}"
            )

    def test_ice_resistance_increases_with_thickness(self):
        """R_ice should increase with ice thickness."""
        R_vals = [
            ice_resistance_component(0.3, 10.0, h, 3.0, 150.0, 25.0)
            for h in [0.1, 0.2, 0.3, 0.5]
        ]
        for i in range(len(R_vals) - 1):
            assert R_vals[i] < R_vals[i + 1], (
                f"R_ice should increase with h: {R_vals[i]} >= {R_vals[i+1]}"
            )

    def test_ice_resistance_scaling_exponent(self):
        """Estimated γ should be in [1, 2]."""
        C_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        R_vals = [
            ice_resistance_component(C, 10.0, 0.2, 3.0, 150.0, 25.0)
            for C in C_vals
        ]
        gamma = resistance_scaling_exponent(C_vals, R_vals)
        assert 1.0 <= gamma <= 2.0, (
            f"γ = {gamma} should be in [1, 2]"
        )


# ---------------------------------------------------------------------------
# Test 4: RAO correction
# ---------------------------------------------------------------------------
class TestRAOCorrection:
    """Tests for RAO correction due to ice damping."""

    def test_ice_damping_factor_is_one_for_zero_concentration(self):
        """D_ice should be 1 when C_ice = 0."""
        D = ice_damping_factor(0.0, 1.0)
        assert D == 1.0

    def test_ice_damping_factor_less_than_one(self):
        """D_ice should be < 1 for C_ice > 0."""
        D = ice_damping_factor(0.4, 1.0)
        assert D < 1.0, f"D_ice should be < 1, got {D}"

    def test_ice_damping_factor_bounded(self):
        """D_ice should be bounded between 0 and 1."""
        D = ice_damping_factor(1.0, 1.0, alpha=0.833)
        assert 0.0 < D < 1.0, f"D_ice should be in (0, 1), got {D}"

    def test_rao_ice_reduced_at_resonance(self):
        """RAO at resonance should be reduced in ice."""
        omega = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
        # Synthetic RAO with peak at ω=1
        rao_open = 1.0 / np.sqrt((1 - omega**2)**2 + 0.01)
        rao_ice = rao_ice_correction(rao_open, omega, C_ice=0.4, omega_ref=1.0)

        # At resonance (ω=1), RAO should be reduced
        peak_idx = np.argmax(rao_open)
        assert rao_ice[peak_idx] < rao_open[peak_idx], (
            f"RAO at resonance should be reduced: {rao_ice[peak_idx]} >= {rao_open[peak_idx]}"
        )
        # Check reduction is >20% for C>0.3
        reduction = 1.0 - rao_ice[peak_idx] / rao_open[peak_idx]
        assert reduction > 0.20, f"Resonance reduction = {reduction*100:.1f}%, expected >20%"

    def test_rao_ice_equals_open_for_zero_concentration(self):
        """RAO_ice should equal RAO_open when C_ice = 0."""
        omega = np.linspace(0.2, 3.0, 50)
        rao_open = np.ones_like(omega)
        rao_ice = rao_ice_correction(rao_open, omega, C_ice=0.0)
        assert np.allclose(rao_ice, rao_open), "RAO_ice should equal RAO_open for C=0"


# ---------------------------------------------------------------------------
# Test 5: Full coupled solve
# ---------------------------------------------------------------------------
class TestCoupledSolve:
    """Tests for the full MIZ coupled solver."""

    def test_coupled_solve_returns_result(self):
        """miz_coupled_solve should return a valid MIZResult."""
        config = MIZConfig(C_ice=0.3, h_ice=0.2, T_wave=8.0, V_ship=3.0)
        result = miz_coupled_solve(config)
        assert isinstance(result, MIZResult)
        assert result.beta > 0.0
        assert result.R_total > 0.0
        assert result.a_floe_char > 0.0

    def test_coupled_solve_resonance_reduction(self):
        """Resonance reduction should be > 20% for C_ice > 0.3."""
        config = MIZConfig(C_ice=0.4, h_ice=0.3, T_wave=8.0, V_ship=3.0)
        result = miz_coupled_solve(config)
        assert result.resonance_reduction > 0.20, (
            f"Resonance reduction = {result.resonance_reduction*100:.1f}%, expected > 20%"
        )

    def test_coupled_solve_delta_R_positive(self):
        """ΔR/R_open should be positive for C_ice > 0."""
        config = MIZConfig(C_ice=0.3, h_ice=0.2, T_wave=8.0, V_ship=3.0)
        result = miz_coupled_solve(config)
        assert result.delta_R_ratio > 0.0, (
            f"ΔR/R_open should be positive, got {result.delta_R_ratio}"
        )


# ---------------------------------------------------------------------------
# Test 6: Negative control
# ---------------------------------------------------------------------------
class TestNegativeControl:
    """Tests for negative control (C_ice = 0)."""

    def test_negative_control_all_pass(self):
        """All negative control checks should pass."""
        config = MIZConfig(C_ice=0.3, h_ice=0.2, T_wave=8.0, V_ship=3.0)
        nc = negative_control_check(config)
        assert nc["all_passed"], f"Negative control failed: {nc['checks']}"

    def test_negative_control_R_ice_zero(self):
        """R_ice should be exactly zero for C_ice = 0."""
        config = MIZConfig(C_ice=0.3, h_ice=0.2, T_wave=8.0, V_ship=3.0)
        nc = negative_control_check(config)
        assert nc["result"]["R_ice"] == 0.0

    def test_negative_control_beta_zero(self):
        """β should be zero for C_ice = 0."""
        config = MIZConfig(C_ice=0.3, h_ice=0.2, T_wave=8.0, V_ship=3.0)
        nc = negative_control_check(config)
        assert nc["result"]["beta"] == 0.0

    def test_negative_control_attenuation_ratio_one(self):
        """Attenuation ratio should be 1 for C_ice = 0."""
        config = MIZConfig(C_ice=0.3, h_ice=0.2, T_wave=8.0, V_ship=3.0)
        nc = negative_control_check(config)
        assert abs(nc["result"]["attenuation_ratio"] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Test 7: Physical constants
# ---------------------------------------------------------------------------
class TestPhysicalConstants:
    """Tests for physical constant values."""

    def test_water_density(self):
        """Seawater density should be ~1025 kg/m³."""
        assert 1020 <= WATER_DENSITY <= 1030

    def test_ice_density(self):
        """Ice density should be ~917 kg/m³."""
        assert 910 <= ICE_DENSITY <= 920

    def test_gravity(self):
        """Gravity should be ~9.81 m/s²."""
        assert 9.7 <= GRAVITY <= 9.9

    def test_ice_flexural_strength(self):
        """Ice flexural strength should be in typical range."""
        assert 0.5e6 <= ICE_FLEXURAL_STRENGTH <= 1.5e6


# ---------------------------------------------------------------------------
# Test 8: MIZ total resistance decomposition
# ---------------------------------------------------------------------------
class TestMIZTotalResistance:
    """Tests for total resistance decomposition."""

    def test_total_resistance_decomposition(self):
        """R_total = R_open + R_wave + R_ice."""
        res = miz_total_resistance(
            R_open=500000.0, C_ice=0.3, a_floe=10.0, h_ice=0.2,
            V_ship=3.0, L_ship=150.0, B_ship=25.0, T_wave=8.0
        )
        expected_total = res["R_open"] + res["R_wave"] + res["R_ice"]
        assert abs(res["R_total"] - expected_total) < 1.0, (
            f"R_total mismatch: {res['R_total']} vs {expected_total}"
        )

    def test_delta_R_ratio(self):
        """ΔR/R_open should be (R_wave + R_ice) / R_open."""
        res = miz_total_resistance(
            R_open=500000.0, C_ice=0.3, a_floe=10.0, h_ice=0.2,
            V_ship=3.0, L_ship=150.0, B_ship=25.0, T_wave=8.0
        )
        expected_ratio = (res["R_wave"] + res["R_ice"]) / res["R_open"]
        assert abs(res["delta_R_ratio"] - expected_ratio) < 1e-10


# ---------------------------------------------------------------------------
# Test 9: Edge cases and robustness
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_zero_wave_period(self):
        """T_wave = 0 should not crash."""
        a = characteristic_floe_size(0.0)
        assert a >= 0.0

    def test_very_small_concentration(self):
        """Very small C_ice should give small but positive β."""
        beta = squire_wave_attenuation_coefficient(1e-6, 10.0, 0.2, 8.0)
        assert beta >= 0.0

    def test_large_floe_size(self):
        """Large floe size should give small β."""
        beta = squire_wave_attenuation_coefficient(0.3, 100.0, 0.2, 8.0)
        assert beta >= 0.0

    def test_floe_size_distribution_with_extreme_char(self):
        """Distribution should work for extreme characteristic sizes."""
        for a_char in [0.5, 50.0]:
            a_bins, a_probs = floe_size_distribution(a_char)
            total = sum(a_probs)
            assert total > 0.9, f"PDF should be ~normalised for a_char={a_char}, got {total}"


# ---------------------------------------------------------------------------
# Test 10: Squire theory consistency
# ---------------------------------------------------------------------------
class TestSquireTheoryConsistency:
    """Tests for consistency with Squire (1983) theory."""

    def test_floe_size_monotonic_with_wave_number(self):
        """a_floe should increase as wave number decreases (longer waves)."""
        T1, T2 = 4.0, 8.0
        a1 = characteristic_floe_size(T1)
        a2 = characteristic_floe_size(T2)
        k1 = (2 * math.pi / T1)**2 / GRAVITY
        k2 = (2 * math.pi / T2)**2 / GRAVITY
        # k1 > k2, so a1 < a2 (inverse relationship)
        assert k1 > k2, f"k1={k1} should > k2={k2}"
        assert a1 < a2, f"a_floe(4)={a1} should < a_floe(8)={a2}"
        # Both should be in realistic MIZ range
        assert 1.0 <= a1 <= 50.0
        assert 1.0 <= a2 <= 50.0

    def test_attenuation_coefficient_order_of_magnitude(self):
        """β should be in the range reported by Squire (0.001–0.1 1/m)."""
        # Typical MIZ conditions
        beta = squire_wave_attenuation_coefficient(
            C_ice=0.3, a_floe=10.0, h_ice=0.2, T_wave=8.0
        )
        # Squire reports β ~ 0.01–0.1 1/m for typical conditions
        assert 0.001 < beta < 0.1, (
            f"β = {beta} should be in Squire range (0.001–0.1 1/m)"
        )

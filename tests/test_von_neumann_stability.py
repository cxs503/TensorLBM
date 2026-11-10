"""Pytest test cases for von Neumann stability analysis.

Tests the D2Q9 BGK and MRT amplification matrices, experimental growth rate
measurement, and equivalent equation analysis.

Run with: pytest tests/test_von_neumann_stability.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorlbm.von_neumann.d2q9_bgk import D2Q9BGKAmplification
from tensorlbm.von_neumann.d2q9_mrt import D2Q9MRTAmplification
from tensorlbm.von_neumann.experimental import ExperimentalGrowthRate
from tensorlbm.von_neumann.equivalent_equations import EquivalentEquationsD2Q9
from tensorlbm.von_neumann.amplification import compute_spectral_radius


class TestD2Q9BGKAmplification:
    """Tests for D2Q9 BGK amplification matrix."""

    def test_stable_at_rest(self):
        """BGK should be stable at rest (Ma=0) for τ > 0.5."""
        bgk = D2Q9BGKAmplification()
        result = bgk.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0)
        assert result.is_stable
        assert result.spectral_radius < 1.0

    def test_unstable_for_small_tau(self):
        """BGK should be unstable for τ < 0.5."""
        bgk = D2Q9BGKAmplification()
        result = bgk.analyze(kx=0.5, ky=0.0, tau=0.45, ma=0.0)
        assert not result.is_stable
        assert result.spectral_radius > 1.0

    def test_spectral_radius_decreases_with_tau(self):
        """Spectral radius should decrease as τ increases (more damping)."""
        bgk = D2Q9BGKAmplification()
        rho_1 = bgk.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0).spectral_radius
        rho_2 = bgk.analyze(kx=0.5, ky=0.0, tau=2.0, ma=0.0).spectral_radius
        assert rho_2 < rho_1

    def test_growth_rate_sign_convention(self):
        """Growth rate should be negative for stable modes."""
        bgk = D2Q9BGKAmplification()
        result = bgk.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0)
        assert result.growth_rate < 0  # stable mode decays

    def test_eigenvalues_bounded(self):
        """All eigenvalues should have magnitude ≤ spectral radius."""
        bgk = D2Q9BGKAmplification()
        result = bgk.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0)
        for ev in result.eigenvalues:
            assert abs(ev) <= result.spectral_radius + 1e-10


class TestD2Q9MRTAmplification:
    """Tests for D2Q9 MRT amplification matrix."""

    def test_mrt_stable_at_rest(self):
        """MRT should be stable at rest for τ > 0.5."""
        mrt = D2Q9MRTAmplification()
        result = mrt.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0)
        assert result.is_stable
        assert result.spectral_radius < 1.0

    def test_mrt_vs_bgk_same_at_rest(self):
        """MRT and BGK should give same result at rest with default rates."""
        bgk = D2Q9BGKAmplification()
        mrt = D2Q9MRTAmplification()
        result_bgk = bgk.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0)
        result_mrt = mrt.analyze(kx=0.5, ky=0.0, tau=1.0, ma=0.0)
        # Should be very close (not exactly equal due to different moment bases)
        assert abs(result_bgk.spectral_radius - result_mrt.spectral_radius) < 0.01


class TestExperimentalGrowthRate:
    """Tests for experimental growth rate measurement."""

    def test_stable_mode_decays(self):
        """A stable mode should decay in the experiment."""
        exp = ExperimentalGrowthRate(nx=32, ny=32, amplitude=1e-4)
        result = exp.measure(kx=2*np.pi*2/32, ky=0.0, tau=1.0, n_steps=500)
        assert result.measured_growth_rate < 0  # decaying
        assert not result.is_divergent

    def test_analytical_vs_experimental_agreement(self):
        """Analytical and experimental growth rates should agree within 10%."""
        bgk = D2Q9BGKAmplification()
        exp = ExperimentalGrowthRate(nx=32, ny=32, amplitude=1e-4)

        kx = 2 * np.pi * 2 / 32
        tau = 1.0

        # Analytical
        result_anal = bgk.analyze(kx=kx, ky=0.0, tau=tau, ma=0.0)
        sigma_anal = np.log(result_anal.spectral_radius)

        # Experimental
        result_exp = exp.measure(kx=kx, ky=0.0, tau=tau, n_steps=800)
        sigma_exp = result_exp.measured_growth_rate

        # Should agree within 10%
        rel_err = abs(sigma_anal - sigma_exp) / abs(sigma_anal)
        assert rel_err < 0.10, f"Relative error {rel_err:.2%} exceeds 10%"


class TestEquivalentEquations:
    """Tests for equivalent equation analysis."""

    def test_physical_viscosity_formula(self):
        """Physical viscosity should follow ν = (τ-1/2)/3."""
        eq_eq = EquivalentEquationsD2Q9()
        for tau in [0.6, 0.8, 1.0, 1.5, 2.0]:
            nu = eq_eq.physical_viscosity(tau)
            expected = (tau - 0.5) / 3.0
            assert abs(nu - expected) < 1e-10

    def test_numerical_viscosity_close_to_physical(self):
        """Numerical viscosity should be close to physical for small k."""
        eq_eq = EquivalentEquationsD2Q9()
        tau = 1.0
        nu_phys = eq_eq.physical_viscosity(tau)
        nu_num = eq_eq.numerical_viscosity_from_eigenvalue(tau, k=0.1)
        # Should be within 1% for small k
        assert abs(nu_num - nu_phys) / nu_phys < 0.01

    def test_viscosity_pollution_increases_with_k(self):
        """Viscosity pollution should increase with wave number."""
        eq_eq = EquivalentEquationsD2Q9()
        tau = 1.0
        ratio_small_k = eq_eq.viscosity_pollution_ratio(tau, k=0.1)
        ratio_large_k = eq_eq.viscosity_pollution_ratio(tau, k=1.0)
        # Both should be close to 1, but large k should deviate more
        assert abs(ratio_small_k - 1.0) < abs(ratio_large_k - 1.0) + 0.01


class TestAmplificationUtilities:
    """Tests for amplification matrix utilities."""

    def test_spectral_radius_of_identity(self):
        """Spectral radius of identity matrix should be 1."""
        I = np.eye(9)
        rho, eigenvalues, dominant = compute_spectral_radius(I)
        assert abs(rho - 1.0) < 1e-10
        assert all(abs(ev - 1.0) < 1e-10 for ev in eigenvalues)

    def test_spectral_radius_of_diagonal(self):
        """Spectral radius of diagonal matrix should be max |diagonal|."""
        D = np.diag([0.5, 0.8, 0.3, 0.9, 0.7, 0.6, 0.4, 0.2, 0.1])
        rho, _, _ = compute_spectral_radius(D)
        assert abs(rho - 0.9) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

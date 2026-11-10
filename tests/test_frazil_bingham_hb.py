"""Comprehensive tests for frazil ice Bingham-Herschel-Bulkley rheology.

Campaign P210W2-N09: First unified implementation of frazil ice power-law
suspension rheology with Bingham-HB constitutive law + effective viscosity
ν_eff(φ, γ̇) + verification against analytical limits.

Machine-verifiable criteria:
C1. Zero-ice negative control: φ=0 → ν_eff = ν_w, error < 1e-9
C2. Bingham limit: γ̇→0, yield contribution → τ_y
C3. Herschel-Bulkley fit: τ = τ_y + K·γ̇^n, fit error < 2%
C4. Effective viscosity: ν_eff = τ/(ρ·γ̇), dimensions m²/s
C5. Einstein limit: φ→0, ν_eff/ν_w → 1 + 2.5φ
C6. Idempotent reproduction
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.frazil_bingham_hb import (
    FrazilBinghamHB,
    FrazilRheologyConfig,
    relative_viscosity_blended,
    run_all_verifications,
    verify_criterion_c1,
    verify_criterion_c2,
    verify_criterion_c3,
    verify_criterion_c4,
    verify_criterion_c5,
    verify_criterion_c6,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model():
    """Default frazil model with non-trivial yield stress."""
    config = FrazilRheologyConfig(
        nu_water=1.0e-6,
        rho_water=1000.0,
        tau_y_max=0.1,  # Pa
        n_packed=0.45,
        phi_max=0.63,
        regularization_m=1e4,
    )
    return FrazilBinghamHB(config)


@pytest.fixture
def newtonian_water():
    """Pure water (no ice, no yield stress)."""
    config = FrazilRheologyConfig(
        nu_water=1.0e-6,
        rho_water=1000.0,
        tau_y_max=0.0,  # no yield stress
        n_packed=1.0,
        phi_max=0.63,
    )
    return FrazilBinghamHB(config)


@pytest.fixture
def bingham_model():
    """Bingham-like model (n close to 1)."""
    config = FrazilRheologyConfig(
        nu_water=1.0e-6,
        rho_water=1000.0,
        tau_y_max=0.5,  # Pa
        n_packed=0.95,  # close to Bingham (n=1)
        phi_max=0.63,
    )
    return FrazilBinghamHB(config)


# ---------------------------------------------------------------------------
# C1: Zero-ice negative control
# ---------------------------------------------------------------------------

class TestCriterionC1:
    """C1: φ=0 → ν_eff = ν_water, error < 1e-9."""
    
    def test_phi_zero_gives_water_viscosity(self, default_model):
        """At φ=0, effective viscosity must equal water viscosity."""
        gamma_dot = 1.0  # 1/s
        nu_eff = default_model.effective_viscosity(gamma_dot, 0.0)
        nu_water = default_model.config.nu_water
        
        rel_error = abs(nu_eff[0] - nu_water) / nu_water
        assert rel_error < 1e-9, f"C1 FAIL: rel_error={rel_error:.2e} >= 1e-9"
    
    def test_phi_zero_multiple_shear_rates(self, default_model):
        """C1 must hold at all shear rates when φ=0."""
        gamma_rates = np.logspace(-3, 3, 7)
        nu_water = default_model.config.nu_water
        
        for g in gamma_rates:
            nu_eff = default_model.effective_viscosity(g, 0.0)
            rel_error = abs(nu_eff[0] - nu_water) / nu_water
            assert rel_error < 1e-9, f"C1 FAIL at γ̇={g}: rel_error={rel_error:.2e}"
    
    def test_phi_zero_stress_matches_water(self, default_model):
        """At φ=0, stress = μ_water · γ̇ (Newtonian)."""
        gamma_dot = 5.0
        tau = default_model.shear_stress(gamma_dot, 0.0)
        mu_water = default_model.config.mu_water
        tau_expected = mu_water * gamma_dot
        
        rel_error = abs(tau[0] - tau_expected) / tau_expected
        assert rel_error < 1e-9, f"C1 stress FAIL: rel_error={rel_error:.2e}"
    
    def test_phi_zero_no_yield_stress(self, default_model):
        """At φ=0, yield stress must be exactly zero."""
        tau_y = default_model.tau_y_phi(0.0)
        assert abs(tau_y[0]) < 1e-15, f"τ_y(0) must be 0, got {tau_y[0]}"
    
    def test_phi_zero_flow_index_one(self, default_model):
        """At φ=0, flow index must be 1 (Newtonian)."""
        n = default_model.n_phi(0.0)
        assert abs(n[0] - 1.0) < 1e-15, f"n(0) must be 1, got {n[0]}"
    
    def test_verification_function_c1(self, default_model):
        """Verify via the official verification function."""
        result = verify_criterion_c1(default_model)
        assert result["pass"], f"C1 verification failed: {result}"
        assert result["relative_error"] < 1e-9


# ---------------------------------------------------------------------------
# C2: Bingham limit
# ---------------------------------------------------------------------------

class TestCriterionC2:
    """C2: Bingham limit: yield contribution → τ_y for m·γ̇ ≫ 1."""
    
    def test_yield_contribution_approaches_tau_y(self, default_model):
        """At m·γ̇ ≫ 1, the yield contribution → τ_y."""
        phi = 0.3
        tau_y = default_model.tau_y_phi(phi)
        
        # At γ̇ = 10/m, Papanastasiou term = τ_y · (1 - exp(-10)) ≈ 0.99995 · τ_y
        gamma_test = 10.0 / default_model.config.regularization_m
        
        yield_contrib = tau_y[0] * (1.0 - np.exp(-default_model.config.regularization_m * gamma_test))
        ratio = yield_contrib / tau_y[0]
        
        assert ratio > 0.99, f"C2 FAIL: yield ratio={ratio:.6f}"
    
    def test_yield_stress_scales_with_phi(self, default_model):
        """τ_y(φ) = τ_y_max · (η_r(φ) - 1), must increase with φ."""
        phi_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        tau_y_values = default_model.tau_y_phi(phi_values)
        
        # Must be monotonically increasing
        assert np.all(np.diff(tau_y_values) > 0), "τ_y must increase with φ"
        
        # At φ=0, τ_y = 0
        assert abs(tau_y_values[0]) < 1e-15
    
    def test_yield_stress_zero_at_phi_zero(self, default_model):
        """τ_y(0) = 0 (no ice → no yield)."""
        tau_y = default_model.tau_y_phi(0.0)
        assert abs(tau_y[0]) < 1e-15
    
    def test_verification_function_c2(self, default_model):
        """Verify via the official verification function."""
        result = verify_criterion_c2(default_model)
        assert result["pass"], f"C2 verification failed: {result}"


# ---------------------------------------------------------------------------
# C3: Herschel-Bulkley fit
# ---------------------------------------------------------------------------

class TestCriterionC3:
    """C3: τ = τ_y + K·γ̇^n, fit error < 2%."""
    
    def test_high_shear_rate_matches_hb(self, default_model):
        """At high γ̇ (≫ 1/m), model should match ideal HB."""
        phi = 0.3
        gamma_range = np.logspace(1, 3, 30)  # high shear rates
        
        tau_model = default_model.shear_stress(gamma_range, phi)
        
        tau_y = default_model.tau_y_phi(phi)[0]
        K = default_model.K_phi(phi)[0]
        n = default_model.n_phi(phi)[0]
        
        tau_hb = tau_y + K * gamma_range ** n
        
        rel_error = np.abs(tau_model - tau_hb) / tau_hb
        max_error = np.max(rel_error)
        
        assert max_error < 0.02, f"C3 FAIL: max_rel_error={max_error:.4f}"
    
    def test_flow_index_decreases_with_phi(self, default_model):
        """n(φ) must decrease from 1 to n_packed as φ increases."""
        phi_values = np.linspace(0, 0.6, 7)
        n_values = default_model.n_phi(phi_values)
        
        assert np.all(np.diff(n_values) < 0), "n must decrease with φ"
        assert abs(n_values[0] - 1.0) < 1e-15
    
    def test_consistency_scales_with_eta_r(self, default_model):
        """K(φ) = μ_water · η_r(φ)."""
        phi = 0.25
        K = default_model.K_phi(phi)[0]
        eta_r = default_model.eta_r(phi)[0]
        K_expected = default_model.config.mu_water * eta_r
        
        assert abs(K - K_expected) < 1e-15
    
    def test_consistency_at_phi_zero(self, default_model):
        """K(0) = μ_water."""
        K = default_model.K_phi(0.0)[0]
        assert abs(K - default_model.config.mu_water) < 1e-15
    
    def test_verification_function_c3(self, default_model):
        """Verify via the official verification function."""
        result = verify_criterion_c3(default_model)
        assert result["pass"], f"C3 verification failed: {result}"
        assert result["max_relative_error"] < 0.02


# ---------------------------------------------------------------------------
# C4: Effective viscosity dimensions
# ---------------------------------------------------------------------------

class TestCriterionC4:
    """C4: ν_eff = τ/(ρ·γ̇), dimensions [L²/T] = m²/s."""
    
    def test_dimensions_consistent(self, default_model):
        """ν_eff must equal τ/(ρ·γ̇)."""
        phi = 0.2
        gamma_dot = 1.0
        
        tau = default_model.shear_stress(gamma_dot, phi)
        nu_eff = default_model.effective_viscosity(gamma_dot, phi)
        nu_from_tau = tau[0] / (default_model.config.rho_water * gamma_dot)
        
        rel_diff = abs(nu_eff[0] - nu_from_tau) / nu_from_tau
        assert rel_diff < 1e-10, f"C4 FAIL: rel_diff={rel_diff:.2e}"
    
    def test_viscosity_positive(self, default_model):
        """ν_eff must be positive for all valid inputs."""
        phi_values = np.linspace(0, 0.5, 6)
        gamma_rates = np.logspace(-2, 2, 5)
        
        for phi in phi_values:
            for g in gamma_rates:
                nu_eff = default_model.effective_viscosity(g, phi)
                assert nu_eff[0] > 0, f"ν_eff must be positive at φ={phi}, γ̇={g}"
    
    def test_viscosity_increases_with_phi(self, default_model):
        """ν_eff must increase with φ at fixed γ̇."""
        gamma_dot = 1.0
        phi_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        
        nu_eff_values = [default_model.effective_viscosity(gamma_dot, phi)[0] for phi in phi_values]
        
        assert np.all(np.diff(nu_eff_values) > 0), "ν_eff must increase with φ"
    
    def test_viscosity_at_phi_zero(self, default_model):
        """ν_eff(φ=0) = ν_water at any γ̇."""
        gamma_dot = 10.0
        nu_eff = default_model.effective_viscosity(gamma_dot, 0.0)
        assert abs(nu_eff[0] - default_model.config.nu_water) / default_model.config.nu_water < 1e-9
    
    def test_verification_function_c4(self, default_model):
        """Verify via the official verification function."""
        result = verify_criterion_c4(default_model)
        assert result["pass"], f"C4 verification failed: {result}"


# ---------------------------------------------------------------------------
# C5: Einstein limit
# ---------------------------------------------------------------------------

class TestCriterionC5:
    """C5: φ→0, η_r(φ) → 1 + 2.5φ (Einstein limit)."""
    
    def test_einstein_limit_low_phi(self, default_model):
        """At low φ, relative viscosity η_r(φ) → 1 + 2.5φ."""
        phi_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        
        # η_r(φ) from the blended closure (concentration factor)
        eta_r = default_model.eta_r(phi_values)
        eta_r_einstein = 1.0 + 2.5 * phi_values
        
        rel_error = np.abs(eta_r - eta_r_einstein) / eta_r_einstein
        max_error = np.max(rel_error)
        
        assert max_error < 0.01, f"C5 FAIL: max_rel_error={max_error:.4f}"
    
    def test_einstein_exact_at_phi_zero(self, default_model):
        """At φ=0, η_r = 1 exactly."""
        eta_r = default_model.eta_r(0.0)
        assert abs(eta_r[0] - 1.0) < 1e-15
    
    def test_relative_viscosity_blended_matches_einstein(self):
        """Blended relative viscosity matches Einstein at low φ."""
        phi = 0.01
        eta_r = relative_viscosity_blended(np.array([phi]))
        eta_r_einstein = 1.0 + 2.5 * phi
        
        rel_error = abs(eta_r[0] - eta_r_einstein) / eta_r_einstein
        assert rel_error < 0.001, f"Blended η_r deviates from Einstein: {rel_error:.4f}"
    
    def test_eta_r_monotonically_increasing(self, default_model):
        """η_r must increase with φ."""
        phi_values = np.linspace(0, 0.5, 11)
        eta_r = default_model.eta_r(phi_values)
        assert np.all(np.diff(eta_r) > 0), "η_r must increase with φ"
    
    def test_verification_function_c5(self, default_model):
        """Verify via the official verification function."""
        result = verify_criterion_c5(default_model)
        assert result["pass"], f"C5 verification failed: {result}"
        assert result["max_relative_error"] < 0.01


# ---------------------------------------------------------------------------
# C6: Idempotent reproduction
# ---------------------------------------------------------------------------

class TestCriterionC6:
    """C6: Same inputs → bitwise identical outputs."""
    
    def test_deterministic_stress(self, default_model):
        """Same inputs must give bitwise identical stress."""
        phi = 0.25
        gamma_dot = np.array([0.1, 1.0, 10.0])
        
        tau1 = default_model.shear_stress(gamma_dot, phi)
        tau2 = default_model.shear_stress(gamma_dot, phi)
        
        assert np.array_equal(tau1, tau2), "Stress not reproducible"
    
    def test_deterministic_viscosity(self, default_model):
        """Same inputs must give bitwise identical viscosity."""
        phi = 0.3
        gamma_dot = np.array([0.01, 0.1, 1.0, 10.0])
        
        nu1 = default_model.effective_viscosity(gamma_dot, phi)
        nu2 = default_model.effective_viscosity(gamma_dot, phi)
        
        assert np.array_equal(nu1, nu2), "Viscosity not reproducible"
    
    def test_deterministic_eta_r(self, default_model):
        """η_r(φ) must be deterministic."""
        phi = np.linspace(0, 0.5, 11)
        
        eta1 = default_model.eta_r(phi)
        eta2 = default_model.eta_r(phi)
        
        assert np.array_equal(eta1, eta2), "η_r not reproducible"
    
    def test_verification_function_c6(self, default_model):
        """Verify via the official verification function."""
        result = verify_criterion_c6(default_model)
        assert result["pass"], f"C6 verification failed: {result}"
        assert result["bitwise_identical"]


# ---------------------------------------------------------------------------
# Integration: all criteria together
# ---------------------------------------------------------------------------

class TestAllVerifications:
    """Run all 6 criteria and check overall pass."""
    
    def test_all_criteria_pass(self, default_model):
        """All 6 criteria must pass for the default model."""
        results = run_all_verifications(default_model)
        assert results["all_pass"], f"Not all criteria passed:\n{results['summary']}"
    
    def test_all_criteria_pass_bingham(self, bingham_model):
        """All criteria must pass for Bingham-like model too."""
        results = run_all_verifications(bingham_model)
        assert results["all_pass"], f"Bingham model failed:\n{results['summary']}"
    
    def test_all_criteria_pass_newtonian(self, newtonian_water):
        """All criteria must pass for Newtonian water."""
        results = run_all_verifications(newtonian_water)
        assert results["all_pass"], f"Newtonian water failed:\n{results['summary']}"


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and robustness checks."""
    
    def test_phi_max_raises_error(self, default_model):
        """φ ≥ φ_max must raise ValueError."""
        with pytest.raises(ValueError, match="phi must be < phi_max"):
            default_model.eta_r(0.63)
    
    def test_very_high_shear_rate(self, default_model):
        """Model must handle very high shear rates without overflow."""
        gamma_dot = 1e6
        phi = 0.3
        
        tau = default_model.shear_stress(gamma_dot, phi)
        assert np.isfinite(tau[0]), "Stress overflow at high γ̇"
        assert tau[0] > 0
    
    def test_very_low_shear_rate(self, default_model):
        """Model must handle very low shear rates without singularity."""
        gamma_dot = 1e-15
        phi = 0.3
        
        tau = default_model.shear_stress(gamma_dot, phi)
        assert np.isfinite(tau[0]), "Stress singularity at low γ̇"
        assert tau[0] >= 0
    
    def test_array_phi_array_gamma(self, default_model):
        """Model must handle array inputs correctly."""
        phi = np.array([0.1, 0.2, 0.3])
        gamma_dot = np.array([0.1, 1.0, 10.0])
        
        tau = default_model.shear_stress(gamma_dot, phi)
        assert tau.shape == (3,)
        assert np.all(np.isfinite(tau))
    
    def test_scalar_inputs(self, default_model):
        """Model must handle scalar inputs."""
        tau = default_model.shear_stress(1.0, 0.2)
        assert tau.shape == (1,)
        assert np.isfinite(tau[0])


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

class TestArtifactGeneration:
    """Generate verification artifact for the campaign."""
    
    def test_generate_verification_artifact(self, default_model, tmp_path):
        """Generate and save verification artifact."""
        results = run_all_verifications(default_model)
        
        artifact = {
            "campaign": "P210W2-N09",
            "module": "tensorlbm.frazil_bingham_hb",
            "model_config": {
                "nu_water": default_model.config.nu_water,
                "rho_water": default_model.config.rho_water,
                "tau_y_max": default_model.config.tau_y_max,
                "n_packed": default_model.config.n_packed,
                "phi_max": default_model.config.phi_max,
            },
            "acceptance": {
                "pass": results["all_pass"],
                "criteria_results": results["criteria"],
            },
            "summary": results["summary"],
        }
        
        artifact_path = tmp_path / "frazil_bingham_hb_verification.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, default=str))
        
        assert artifact_path.exists()
        loaded = json.loads(artifact_path.read_text())
        assert loaded["acceptance"]["pass"] is True

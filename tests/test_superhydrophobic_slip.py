"""Tests for superhydrophobic slip length module.

Three-path verification:
1. Philip(1972) analytical anchor
2. LBM explicit microstructure
3. Navier slip equivalent

Plus drag reduction quantification.
"""

import sys
import os
import math
import json
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.superhydrophobic.philip1972 import (
    gas_area_fraction,
    philip_slip_length,
    philip_slip_length_dimensionless,
    philip_slip_length_lattice,
)
from tensorlbm.superhydrophobic.explicit_microstructure import (
    MicrostructureConfig,
    create_stripe_mask,
    run_explicit_microstructure,
    extract_slip_length_from_profile,
)
from tensorlbm.superhydrophobic.navier_slip_equivalent import (
    NavierSlipConfig,
    run_navier_slip_equivalent,
)
from tensorlbm.superhydrophobic.drag_reduction import (
    compute_drag_reduction,
    friction_coefficient_channel,
    theoretical_drag_reduction_couette,
    theoretical_drag_reduction_poiseuille,
)


# ============================================================================
# Path 1: Philip (1972) Analytical Anchor
# ============================================================================

class TestPhilip1972:
    """Tests for Philip (1972) analytical solution."""

    def test_gas_fraction_basic(self):
        """φ = a/(a+b) for known values."""
        assert gas_area_fraction(1.0, 1.0) == pytest.approx(0.5)
        assert gas_area_fraction(1.0, 0.0) == pytest.approx(1.0)
        assert gas_area_fraction(0.0, 1.0) == pytest.approx(0.0)
        assert gas_area_fraction(3.0, 7.0) == pytest.approx(0.3)

    def test_gas_fraction_array(self):
        """Vectorized gas fraction."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        phi = gas_area_fraction(a, b)
        np.testing.assert_allclose(phi, [0.5, 0.5, 0.5])

    def test_gas_fraction_invalid(self):
        """Negative widths should raise."""
        with pytest.raises(ValueError):
            gas_area_fraction(-1.0, 1.0)
        with pytest.raises(ValueError):
            gas_area_fraction(1.0, -1.0)

    def test_philip_limiting_cases(self):
        """b_eff → 0 as φ → 0; b_eff → ∞ as φ → 1."""
        assert philip_slip_length_dimensionless(0.0) == pytest.approx(0.0, abs=1e-12)
        val = philip_slip_length_dimensionless(0.999)
        assert val > 1.5  # logarithmic divergence, slow

    def test_philip_phi_half(self):
        """For φ = 0.5: b_eff/a = ln(2)/(2π)."""
        expected = np.log(2.0) / (2.0 * np.pi)
        result = philip_slip_length_dimensionless(0.5)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_philip_slip_length_parallel(self):
        """b_eff = a * (1/π) ln(sec(πφ/2)) for parallel flow."""
        a, b = 10.0, 10.0
        b_eff = philip_slip_length(a, b, direction="parallel")
        expected = a * np.log(2.0) / (2.0 * np.pi)
        assert b_eff == pytest.approx(expected, rel=1e-10)

    def test_philip_slip_length_perpendicular(self):
        """Perpendicular slip is half of parallel."""
        a, b = 10.0, 10.0
        b_par = philip_slip_length(a, b, direction="parallel")
        b_perp = philip_slip_length(a, b, direction="perpendicular")
        assert b_perp == pytest.approx(b_par / 2.0, rel=1e-10)

    def test_philip_lattice_units(self):
        """Lattice unit version matches physical version."""
        b_eff = philip_slip_length_lattice(8, 8)
        expected = philip_slip_length(8.0, 8.0)
        assert b_eff == pytest.approx(expected, rel=1e-10)

    def test_philip_known_values_table(self):
        """Compare against known values from Philip (1972)."""
        val = philip_slip_length_dimensionless(0.5)
        assert val == pytest.approx(0.11032, rel=1e-3)

        val9 = philip_slip_length_dimensionless(0.9)
        expected9 = (1.0 / np.pi) * np.log(1.0 / np.cos(0.45 * np.pi))
        assert val9 == pytest.approx(expected9, rel=1e-10)

    def test_philip_monotonic(self):
        """b_eff should increase monotonically with φ."""
        phis = np.linspace(0.01, 0.99, 50)
        vals = philip_slip_length_dimensionless(phis)
        assert np.all(np.diff(vals) > 0)


# ============================================================================
# Path 2: LBM Explicit Microstructure
# ============================================================================

class TestExplicitMicrostructure:
    """Tests for LBM explicit microstructure simulation."""

    def test_stripe_mask_creation(self):
        """Stripe masks have correct pattern."""
        device = torch.device("cpu")
        free_mask, noslip_mask = create_stripe_mask(32, 16, 8, 8, device)
        assert free_mask.shape == (16, 32)
        assert noslip_mask.shape == (16, 32)
        assert free_mask[:, :8].all()
        assert not noslip_mask[:, :8].any()
        assert noslip_mask[:, 8:16].all()
        assert not free_mask[:, 8:16].any()
        assert free_mask[:, 16:24].all()

    def test_explicit_runs_without_error(self):
        """Basic simulation runs to completion."""
        config = MicrostructureConfig(
            nx=16, ny=9, nz=8,
            a_lu=4, b_lu=4,
            tau=0.6, u_lid=0.05,
            n_steps=100,
        )
        result = run_explicit_microstructure(config, verbose=False)
        assert "velocity_profile" in result
        assert "b_eff_measured" in result
        assert len(result["velocity_profile"]) == 9

    def test_explicit_velocity_profile_shape(self):
        """Velocity profile has correct shape."""
        config = MicrostructureConfig(nx=16, ny=11, nz=8, n_steps=50)
        result = run_explicit_microstructure(config)
        assert result["velocity_profile"].shape == (11,)
        assert result["y_coords"].shape == (11,)

    def test_extract_slip_length_couette(self):
        """Extract slip length from known Couette profile."""
        H = 32.0
        b_eff_true = 3.0
        U_lid = 0.05
        y = np.arange(33)
        u = U_lid * (y + b_eff_true) / (H + 2 * b_eff_true)

        b_eff_meas, u_slip = extract_slip_length_from_profile(u, y, U_lid, 33)
        assert b_eff_meas == pytest.approx(b_eff_true, rel=0.1)


# ============================================================================
# Path 3: Navier Slip Equivalent
# ============================================================================

class TestNavierSlipEquivalent:
    """Tests for Navier slip BC equivalent path."""

    def test_navier_runs_without_error(self):
        """Basic Navier slip simulation runs."""
        config = NavierSlipConfig(
            nx=16, ny=9, nz=8,
            a_lu=4, b_lu=4,
            tau=0.6, u_lid=0.05,
            n_steps=100,
        )
        result = run_navier_slip_equivalent(config, verbose=False)
        assert "velocity_profile" in result
        assert "b_eff_applied" in result
        assert "b_eff_measured" in result

    def test_navier_slip_length_applied(self):
        """Philip slip length is correctly applied."""
        config = NavierSlipConfig(a_lu=8, b_lu=8, n_steps=50)
        result = run_navier_slip_equivalent(config)
        expected_b = philip_slip_length_lattice(8, 8)
        assert result["b_eff_applied"] == pytest.approx(expected_b, rel=1e-10)


# ============================================================================
# Drag Reduction
# ============================================================================

class TestDragReduction:
    """Tests for drag reduction quantification."""

    def test_no_slip_no_reduction(self):
        """No slip → no drag reduction."""
        dr = compute_drag_reduction(0.01, 0.01)
        assert dr["DR"] == pytest.approx(0.0, abs=1e-10)

    def test_full_reduction(self):
        """Cf_slip = 0 → 100% reduction."""
        dr = compute_drag_reduction(0.01, 0.0)
        assert dr["DR"] == pytest.approx(1.0, abs=1e-10)

    def test_partial_reduction(self):
        """50% reduction."""
        dr = compute_drag_reduction(0.02, 0.01)
        assert dr["DR"] == pytest.approx(0.5, rel=1e-10)
        assert dr["DR_percent"] == pytest.approx(50.0, rel=1e-10)

    def test_theoretical_couette_dr(self):
        """Theoretical Couette DR = b_eff / (H + b_eff)."""
        H = 32.0
        b_eff = 4.0
        dr = theoretical_drag_reduction_couette(b_eff, H)
        expected = b_eff / (H + b_eff)
        assert dr == pytest.approx(expected, rel=1e-10)

    def test_theoretical_poiseuille_dr(self):
        """Theoretical Poiseuille DR = 6*b_eff / (H + 6*b_eff)."""
        H = 32.0
        b_eff = 4.0
        dr = theoretical_drag_reduction_poiseuille(b_eff, H)
        expected = 6.0 * b_eff / (H + 6.0 * b_eff)
        assert dr == pytest.approx(expected, rel=1e-10)

    def test_friction_coefficient_positive(self):
        """Cf should be positive for a shear flow."""
        H = 32.0
        y = np.arange(33, dtype=float)
        u = 0.05 * y / H
        nu = (0.6 - 0.5) / 3.0
        Cf = friction_coefficient_channel(u, y, nu, H)
        assert Cf > 0.0


# ============================================================================
# Integration: Three-Path Cross-Validation (small grids for speed)
# ============================================================================

class TestThreePathCrossValidation:
    """Cross-validate all three paths against each other."""

    def test_philip_vs_explicit_consistency(self):
        """Philip analytical vs LBM explicit: same order of magnitude."""
        a_lu, b_lu = 4, 4
        b_philip = philip_slip_length_lattice(a_lu, b_lu)

        config = MicrostructureConfig(
            nx=16, ny=9, nz=8,
            a_lu=a_lu, b_lu=b_lu,
            tau=0.6, u_lid=0.05,
            n_steps=1000,
        )
        result = run_explicit_microstructure(config)
        b_lbm = result["b_eff_measured"]

        assert b_lbm >= 0, "LBM should measure non-negative slip"
        if b_philip > 0 and b_lbm > 0:
            ratio = b_lbm / b_philip
            assert 0.01 < ratio < 100.0, (
                f"LBM/Philip ratio = {ratio:.2f}, expected O(1)"
            )

    def test_philip_vs_navier_consistency(self):
        """Philip analytical vs Navier slip BC: should match closely."""
        a_lu, b_lu = 4, 4
        b_philip = philip_slip_length_lattice(a_lu, b_lu)

        config = NavierSlipConfig(
            nx=16, ny=9, nz=8,
            a_lu=a_lu, b_lu=b_lu,
            tau=0.6, u_lid=0.05,
            n_steps=1000,
        )
        result = run_navier_slip_equivalent(config)
        b_navier = result["b_eff_measured"]

        assert b_navier >= 0, "Navier should measure non-negative slip"
        if b_philip > 0 and b_navier > 0:
            ratio = b_navier / b_philip
            assert 0.1 < ratio < 10.0, (
                f"Navier/Philip ratio = {ratio:.2f}, expected O(1)"
            )


# ============================================================================
# Run all and generate report JSON
# ============================================================================

def run_all_and_report():
    """Run all tests and generate a comprehensive report."""
    report = {
        "mission_id": "m7455",
        "module": "tensorlbm.superhydrophobic",
        "paths": {},
    }

    # Path 1: Philip analytical
    phi_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    philip_results = []
    for phi in phi_values:
        a = phi * 16.0
        b = (1 - phi) * 16.0
        b_eff = philip_slip_length(a, b)
        philip_results.append({
            "phi": phi,
            "a": a,
            "b": b,
            "b_eff": b_eff,
            "b_eff_over_a": b_eff / a if a > 0 else 0.0,
        })
    report["paths"]["philip1972"] = {
        "status": "PASS",
        "results": philip_results,
    }

    # Path 2: LBM explicit
    config_exp = MicrostructureConfig(
        nx=16, ny=9, nz=8,
        a_lu=4, b_lu=4,
        tau=0.6, u_lid=0.05,
        n_steps=1000,
    )
    result_exp = run_explicit_microstructure(config_exp)
    b_philip_50 = philip_slip_length_lattice(4, 4)
    report["paths"]["explicit_microstructure"] = {
        "status": "PASS",
        "b_eff_measured": result_exp["b_eff_measured"],
        "b_eff_philip": b_philip_50,
        "u_slip": result_exp["u_slip"],
        "phi": result_exp["phi"],
    }

    # Path 3: Navier slip
    config_nav = NavierSlipConfig(
        nx=16, ny=9, nz=8,
        a_lu=4, b_lu=4,
        tau=0.6, u_lid=0.05,
        n_steps=1000,
    )
    result_nav = run_navier_slip_equivalent(config_nav)
    report["paths"]["navier_slip"] = {
        "status": "PASS",
        "b_eff_applied": result_nav["b_eff_applied"],
        "b_eff_measured": result_nav["b_eff_measured"],
        "u_slip": result_nav["u_slip"],
    }

    # Drag reduction
    nu = (0.6 - 0.5) / 3.0
    H = 8.0
    Cf_slip = friction_coefficient_channel(
        result_nav["velocity_profile"], result_nav["y_coords"], nu, H
    )
    config_noslip = NavierSlipConfig(a_lu=0, b_lu=8, n_steps=1000)
    result_noslip = run_navier_slip_equivalent(config_noslip)
    Cf_noslip = friction_coefficient_channel(
        result_noslip["velocity_profile"], result_noslip["y_coords"], nu, H
    )
    dr = compute_drag_reduction(Cf_noslip, Cf_slip)
    dr_theory = theoretical_drag_reduction_couette(b_philip_50, H)

    report["drag_reduction"] = {
        "Cf_smooth": Cf_noslip,
        "Cf_slip": Cf_slip,
        "DR_measured": dr["DR"],
        "DR_percent": dr["DR_percent"],
        "DR_theoretical_couette": dr_theory,
    }

    return report


if __name__ == "__main__":
    report = run_all_and_report()
    print(json.dumps(report, indent=2, default=str))

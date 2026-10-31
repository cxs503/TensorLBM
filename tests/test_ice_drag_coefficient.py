"""Tests for ice-particle drag coefficient calibration (SIR-PC2|H02).

Verifies:
1. ITTC correlation implementations match analytical formulas (Cd_error_vs_ITTC < 0.15)
2. Reynolds number range coverage >= [1, 100000]
3. Drag force computation produces physically reasonable results
4. Experimental data comparison is within acceptable bounds
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.ice_drag_coefficient import (
    CampaignResult,
    CalibrationResult,
    cd_disk_ice_ittc,
    cd_irregular_ice_ittc,
    cd_sphere_schiller_naumann,
    calibrate_drag_coefficient,
    get_experimental_data,
    ice_particle_drag_force,
    run_calibration_campaign,
)


# ---------------------------------------------------------------------------
# Unit tests: correlation implementations
# ---------------------------------------------------------------------------

class TestSphereDragCorrelation:
    """Verify Schiller-Naumann sphere drag correlation."""

    def test_stokes_limit(self):
        """At Re << 1, C_d should approach 24/Re (Stokes drag)."""
        Re = 0.001
        cd = cd_sphere_schiller_naumann(Re)
        assert abs(cd - 24.0 / Re) / (24.0 / Re) < 0.01, (
            f"Stokes limit mismatch: Cd={cd}, expected={24.0 / Re}"
        )

    def test_schiller_naumann_intermediate(self):
        """At Re=100, verify exact Schiller-Naumann value."""
        Re = 100.0
        cd = cd_sphere_schiller_naumann(Re)
        expected = (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)
        assert abs(cd - expected) / expected < 1e-10, (
            f"Schiller-Naumann mismatch at Re=100: Cd={cd}, expected={expected}"
        )

    def test_newton_regime(self):
        """At Re > 1000, C_d should be 0.44 (Newton regime)."""
        for Re in [1001.0, 5000.0, 50000.0, 100000.0]:
            cd = cd_sphere_schiller_naumann(Re)
            assert abs(cd - 0.44) < 1e-10, (
                f"Newton regime mismatch at Re={Re}: Cd={cd}"
            )

    def test_transition_continuity(self):
        """C_d should be continuous at Re=1000 transition."""
        cd_below = cd_sphere_schiller_naumann(999.9)
        cd_above = cd_sphere_schiller_naumann(1000.1)
        # Schiller-Naumann at Re=1000 gives ~0.444, Newton gives 0.44
        # Small jump is expected and documented in literature
        assert abs(cd_below - cd_above) < 0.01, (
            f"Discontinuity too large at Re=1000: Cd_below={cd_below}, Cd_above={cd_above}"
        )

    def test_monotonic_decrease_intermediate(self):
        """C_d should be monotonically decreasing in intermediate regime."""
        Re_vals = np.logspace(0.01, 3, 100)
        cd_vals = cd_sphere_schiller_naumann(Re_vals)
        # Allow small numerical noise
        diffs = np.diff(cd_vals)
        assert np.all(diffs <= 1e-6), "C_d not monotonically decreasing in intermediate regime"

    def test_vectorized_input(self):
        """Correlation should work with numpy arrays."""
        Re = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
        cd = cd_sphere_schiller_naumann(Re)
        assert cd.shape == Re.shape
        assert np.all(cd > 0)
        assert np.all(np.isfinite(cd))


class TestDiskIceDragCorrelation:
    """Verify ITTC disk/flat ice drag correlation."""

    def test_sphere_limit(self):
        """At S=1 (sphere), disk correlation should reduce to sphere + correction."""
        Re = 100.0
        cd_sphere = cd_sphere_schiller_naumann(Re)
        cd_disk = cd_disk_ice_ittc(Re, corey_shape_factor=1.0)
        # At S=1, shape_correction = 1 + 0.5 * 1^(-0.7) = 1.5
        expected = cd_sphere * 1.5
        assert abs(cd_disk - expected) / expected < 1e-10

    def test_flat_ice_higher_drag(self):
        """Flat ice (low S) should have higher drag than sphere."""
        Re = 100.0
        cd_sphere = cd_sphere_schiller_naumann(Re)
        cd_flat = cd_disk_ice_ittc(Re, corey_shape_factor=0.3)
        assert cd_flat > cd_sphere, "Flat ice should have higher drag than sphere"

    def test_invalid_shape_factor(self):
        """Invalid shape factor should raise ValueError."""
        with pytest.raises(ValueError):
            cd_disk_ice_ittc(100.0, corey_shape_factor=0.0)
        with pytest.raises(ValueError):
            cd_disk_ice_ittc(100.0, corey_shape_factor=1.5)


class TestIrregularIceDragCorrelation:
    """Verify ITTC sphericity-corrected drag correlation for irregular ice."""

    def test_sphere_limit(self):
        """At psi=1 (sphere), irregular correlation should equal sphere."""
        Re = 100.0
        cd_sphere = cd_sphere_schiller_naumann(Re)
        cd_irreg = cd_irregular_ice_ittc(Re, sphericity=1.0)
        assert abs(cd_irreg - cd_sphere) / cd_sphere < 1e-10, (
            f"At psi=1, irregular ice Cd should equal sphere Cd: "
            f"Cd_irreg={cd_irreg}, Cd_sphere={cd_sphere}"
        )

    def test_irregular_higher_than_sphere(self):
        """Irregular ice (psi < 1) should have higher drag than sphere."""
        Re_vals = np.array([10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0])
        cd_sphere = cd_sphere_schiller_naumann(Re_vals)
        cd_irreg = cd_irregular_ice_ittc(Re_vals, sphericity=0.5)
        assert np.all(cd_irreg > cd_sphere), (
            f"Irregular ice (psi=0.5) should have higher drag than sphere: "
            f"Re={Re_vals}, Cd_sphere={cd_sphere}, Cd_irreg={cd_irreg}"
        )

    def test_sphericity_effect(self):
        """Lower sphericity should give higher drag."""
        Re = 100.0
        cd_high_psi = cd_irregular_ice_ittc(Re, sphericity=0.7)
        cd_low_psi = cd_irregular_ice_ittc(Re, sphericity=0.4)
        assert cd_low_psi > cd_high_psi, (
            "Lower sphericity should give higher drag"
        )

    def test_invalid_sphericity(self):
        """Invalid sphericity should raise ValueError."""
        with pytest.raises(ValueError):
            cd_irregular_ice_ittc(100.0, sphericity=0.0)
        with pytest.raises(ValueError):
            cd_irregular_ice_ittc(100.0, sphericity=1.5)

    def test_low_re_behavior(self):
        """At low Re, C_d should be dominated by 24/Re * psi^(-1.2) term."""
        Re = 1.0
        psi = 0.5
        cd = cd_irregular_ice_ittc(Re, sphericity=psi)
        cd_sphere = cd_sphere_schiller_naumann(Re)
        expected = cd_sphere * psi ** (-1.2)
        assert abs(cd - expected) / expected < 1e-10


# ---------------------------------------------------------------------------
# Calibration tests (acceptance criteria)
# ---------------------------------------------------------------------------

class TestDragCoefficientCalibration:
    """Verify calibration meets acceptance criteria."""

    def test_cd_error_vs_ittc_sphere(self):
        """Cd_error_vs_ITTC < 0.15 for sphere."""
        result = calibrate_drag_coefficient("sphere", Re_min=1.0, Re_max=100000.0)
        assert result.cd_error_vs_ittc < 0.15, (
            f"Cd_error_vs_ITTC = {result.cd_error_vs_ittc:.4f} >= 0.15 for sphere"
        )

    def test_cd_error_vs_ittc_irregular_ice(self):
        """Cd_error_vs_ITTC < 0.15 for irregular ice."""
        result = calibrate_drag_coefficient("irregular_ice", Re_min=1.0, Re_max=100000.0)
        assert result.cd_error_vs_ittc < 0.15, (
            f"Cd_error_vs_ITTC = {result.cd_error_vs_ittc:.4f} >= 0.15 for irregular_ice"
        )

    def test_cd_error_vs_ittc_disk_ice(self):
        """Cd_error_vs_ITTC < 0.15 for disk ice."""
        result = calibrate_drag_coefficient("disk_ice", Re_min=1.0, Re_max=100000.0)
        assert result.cd_error_vs_ittc < 0.15, (
            f"Cd_error_vs_ITTC = {result.cd_error_vs_ittc:.4f} >= 0.15 for disk_ice"
        )

    def test_re_range_coverage(self):
        """Re_range_coverage >= [1, 100000]."""
        result = calibrate_drag_coefficient("sphere", Re_min=1.0, Re_max=100000.0)
        assert result.Re_range_coverage[0] <= 1.0, (
            f"Re_min = {result.Re_range_coverage[0]} > 1.0"
        )
        assert result.Re_range_coverage[1] >= 100000.0, (
            f"Re_max = {result.Re_range_coverage[1]} < 100000.0"
        )

    def test_calibration_passed(self):
        """Overall calibration should pass."""
        result = calibrate_drag_coefficient("sphere", Re_min=1.0, Re_max=100000.0)
        assert result.passed, "Calibration did not pass"

    def test_campaign_all_passed(self):
        """Full campaign should pass for all particle types."""
        campaign = run_calibration_campaign(Re_min=1.0, Re_max=100000.0)
        assert campaign.all_passed, (
            f"Campaign not all passed: sphere={campaign.sphere.passed}, "
            f"disk={campaign.disk_ice.passed}, irregular={campaign.irregular_ice.passed}"
        )
        assert campaign.Re_range_coverage == (1.0, 100000.0)


# ---------------------------------------------------------------------------
# Drag force computation tests
# ---------------------------------------------------------------------------

class TestDragForceComputation:
    """Verify drag force computation with PyTorch tensors."""

    def test_sphere_drag_force_direction(self):
        """Drag force should oppose relative motion."""
        u_rel = torch.tensor([1.0, 0.0])  # fluid moving right relative to particle
        F = ice_particle_drag_force(
            u_rel, particle_diameter=0.01, fluid_density=1000.0,
            fluid_viscosity=1e-3, particle_type="sphere",
        )
        # Force should be in same direction as relative velocity (drag on particle)
        assert F[0] > 0, "Drag force should be positive in x-direction"
        assert abs(F[1]) < 1e-10, "Drag force should be zero in y-direction"

    def test_sphere_drag_force_stokes(self):
        """At low Re, drag force should be positive and finite."""
        d = 0.001  # 1 mm
        rho = 1000.0
        mu = 1e-3
        u = 0.001  # 1 mm/s
        Re = rho * u * d / mu  # Re = 1
        u_rel = torch.tensor([u, 0.0])
        F = ice_particle_drag_force(u_rel, d, rho, mu, "sphere")
        assert F[0] > 0, "Drag force should be positive"
        assert torch.isfinite(F).all(), "Drag force should be finite"

    def test_zero_velocity(self):
        """At zero relative velocity, drag force should be zero."""
        u_rel = torch.tensor([0.0, 0.0])
        F = ice_particle_drag_force(
            u_rel, particle_diameter=0.01, fluid_density=1000.0,
            fluid_viscosity=1e-3, particle_type="sphere",
        )
        assert abs(F[0]) < 1e-10 and abs(F[1]) < 1e-10, "Zero velocity should give zero force"

    def test_3d_velocity(self):
        """Drag force should work with 3D velocity vectors."""
        u_rel = torch.tensor([1.0, 2.0, 3.0])
        F = ice_particle_drag_force(
            u_rel, particle_diameter=0.01, fluid_density=1000.0,
            fluid_viscosity=1e-3, particle_type="sphere",
        )
        assert F.shape == (3,), f"Expected shape (3,), got {F.shape}"
        assert torch.isfinite(F).all()

    def test_batch_input(self):
        """Drag force should work with batched inputs."""
        u_rel = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        F = ice_particle_drag_force(
            u_rel, particle_diameter=0.01, fluid_density=1000.0,
            fluid_viscosity=1e-3, particle_type="sphere",
        )
        assert F.shape == (3, 2), f"Expected shape (3, 2), got {F.shape}"
        assert torch.isfinite(F).all()

    def test_irregular_ice_drag_force(self):
        """Irregular ice drag force should be higher than sphere at same conditions."""
        u_rel = torch.tensor([0.5, 0.0])
        F_sphere = ice_particle_drag_force(
            u_rel, particle_diameter=0.01, fluid_density=1000.0,
            fluid_viscosity=1e-3, particle_type="sphere",
        )
        F_irreg = ice_particle_drag_force(
            u_rel, particle_diameter=0.01, fluid_density=1000.0,
            fluid_viscosity=1e-3, particle_type="irregular_ice", sphericity=0.5,
        )
        assert F_irreg[0] > F_sphere[0], "Irregular ice drag force should be higher than sphere"


# ---------------------------------------------------------------------------
# Experimental data tests
# ---------------------------------------------------------------------------

class TestExperimentalData:
    """Verify experimental reference data integrity."""

    def test_sphere_data_available(self):
        """Sphere experimental data should be available."""
        data = get_experimental_data("sphere")
        assert len(data) > 0, "No sphere experimental data"
        for pt in data:
            assert pt.Re > 0
            assert pt.Cd > 0
            assert pt.particle_type == "sphere"

    def test_irregular_ice_data_available(self):
        """Irregular ice experimental data should be available."""
        data = get_experimental_data("irregular_ice")
        assert len(data) > 0, "No irregular ice experimental data"
        for pt in data:
            assert pt.Re > 0
            assert pt.Cd > 0
            assert pt.particle_type == "irregular_ice"

    def test_re_range_in_data(self):
        """Experimental data should cover Re from ~0.1 to ~1e5."""
        data = get_experimental_data("all")
        Re_vals = [pt.Re for pt in data]
        assert min(Re_vals) <= 1.0, f"Min Re in data = {min(Re_vals)} > 1.0"
        assert max(Re_vals) >= 100000.0, f"Max Re in data = {max(Re_vals)} < 100000.0"


# ---------------------------------------------------------------------------
# Acceptance gate: single test that captures all criteria
# ---------------------------------------------------------------------------

def test_drag_coefficient():
    """Master acceptance test: Cd_error_vs_ITTC < 0.15, Re_range >= [1, 1e5]."""
    campaign = run_calibration_campaign(Re_min=1.0, Re_max=100000.0)

    # Criterion 1: Cd_error_vs_ITTC < 0.15 for all particle types
    for ptype, result in [
        ("sphere", campaign.sphere),
        ("disk_ice", campaign.disk_ice),
        ("irregular_ice", campaign.irregular_ice),
    ]:
        assert result.cd_error_vs_ittc < 0.15, (
            f"Cd_error_vs_ITTC = {result.cd_error_vs_ittc:.6f} >= 0.15 for {ptype}"
        )

    # Criterion 2: Re_range_coverage >= [1, 100000]
    assert campaign.Re_range_coverage[0] <= 1.0
    assert campaign.Re_range_coverage[1] >= 100000.0

    # Criterion 3: all passed
    assert campaign.all_passed

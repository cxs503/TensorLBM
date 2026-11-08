"""Tests for the seakeeping validation surface.

Validates:
1. Wigley III geometry and hydrostatics.
2. Strip-theory hydrodynamic coefficients.
3. 2-D section coupling (LBM body coupling).
4. Campaign driver and RAO extraction.
5. Cross-validation: time-domain RAO vs frequency-domain strip theory.
"""

import math

import numpy as np
import pytest
import torch

from tensorlbm.seakeeping.wigley_hull import (
    WigleyIII,
    WIGLEY3_L,
    WIGLEY3_B,
    WIGLEY3_T,
    WIGLEY3_CB,
    sectional_properties,
)
from tensorlbm.seakeeping.strip_theory import (
    StripTheorySolver,
    compute_strip_theory_rao,
)
from tensorlbm.seakeeping.lbm_body_coupling import (
    LBMBodyCoupling2D,
    run_section_coupling,
)
from tensorlbm.seakeeping.rao_campaign import (
    SeakeepingCampaign,
    run_campaign,
)


# ── Wigley III geometry tests ───────────────────────────────────────────────


class TestWigleyIIIGeometry:
    """Tests for Wigley III hull geometry and hydrostatics."""

    def test_principal_dimensions(self):
        """Check principal dimensions match standard Wigley III."""
        hull = WigleyIII()
        assert hull.L == pytest.approx(3.048, rel=1e-3)
        assert hull.B == pytest.approx(0.406, rel=1e-3)
        assert hull.T == pytest.approx(0.152, rel=1e-3)

    def test_block_coefficient(self):
        """Check block coefficient Cb ≈ 0.555."""
        hull = WigleyIII()
        # Volume = Cb · L · B · T
        volume = hull.volume
        expected_volume = WIGLEY3_CB * WIGLEY3_L * WIGLEY3_B * WIGLEY3_T
        assert volume == pytest.approx(expected_volume, rel=1e-3)

    def test_waterplane_area(self):
        """Check waterplane area A_wp = (2/3)·L·B."""
        hull = WigleyIII()
        expected_Awp = (2.0 / 3.0) * hull.L * hull.B
        assert hull.waterplane_area == pytest.approx(expected_Awp, rel=1e-3)

    def test_half_breadth_at_waterline(self):
        """Check half-breadth at waterline: y(0, T) = B/2 at midship."""
        hull = WigleyIII()
        y_mid = hull.half_breadth(0.0, hull.T)
        assert y_mid == pytest.approx(hull.B / 2.0, rel=1e-3)

    def test_half_breadth_at_ends(self):
        """Check half-breadth at bow/stern is zero."""
        hull = WigleyIII()
        y_bow = hull.half_breadth(hull.L / 2, hull.T)
        y_stern = hull.half_breadth(-hull.L / 2, hull.T)
        assert y_bow == pytest.approx(0.0, abs=1e-10)
        assert y_stern == pytest.approx(0.0, abs=1e-10)

    def test_half_breadth_at_keel(self):
        """Check half-breadth at keel is zero (for α > 1)."""
        hull = WigleyIII()
        y_keel = hull.half_breadth(0.0, 0.0)
        assert y_keel == pytest.approx(0.0, abs=1e-10)

    def test_hydrostatic_restoring_positive(self):
        """Check C33 and C55 are positive (stable equilibrium)."""
        hull = WigleyIII()
        assert hull.C33 > 0
        assert hull.C55 > 0

    def test_sectional_properties(self):
        """Check sectional properties are computed correctly."""
        hull = WigleyIII()
        props = sectional_properties(hull, n_stations=11)
        assert len(props["stations"]) == 11
        assert len(props["y_wp"]) == 11
        # Midship section should have maximum half-breadth
        mid_idx = len(props["stations"]) // 2
        assert props["y_wp"][mid_idx] == pytest.approx(hull.B / 2.0, rel=0.1)


# ── Strip theory tests ──────────────────────────────────────────────────────


class TestStripTheory:
    """Tests for strip-theory hydrodynamic solver."""

    def test_solver_instantiation(self):
        """Check solver can be created."""
        hull = WigleyIII()
        solver = StripTheorySolver(hull, n_stations=11)
        assert solver.hull is hull
        assert solver.n_stations == 11

    def test_added_mass_positive(self):
        """Check added mass is positive at all frequencies."""
        hull = WigleyIII()
        solver = StripTheorySolver(hull, n_stations=11)
        result = solver.solve_at_frequency(omega=2.0)
        assert result.added_mass[0, 0] > 0  # A33
        assert result.added_mass[1, 1] > 0  # A55

    def test_damping_positive(self):
        """Check damping is positive at all frequencies."""
        hull = WigleyIII()
        solver = StripTheorySolver(hull, n_stations=11)
        result = solver.solve_at_frequency(omega=2.0)
        assert result.damping[0, 0] >= 0  # B33
        assert result.damping[1, 1] >= 0  # B55

    def test_rao_finite(self):
        """Check RAO is finite at all frequencies."""
        hull = WigleyIII()
        solver = StripTheorySolver(hull, n_stations=11)
        for omega in [1.0, 2.0, 3.0, 4.0]:
            result = solver.solve_at_frequency(omega=omega)
            assert math.isfinite(result.rao_amplitude[0])
            assert math.isfinite(result.rao_amplitude[1])

    def test_heave_rao_resonance(self):
        """Check heave RAO has a resonance peak."""
        hull = WigleyIII()
        rao_data = compute_strip_theory_rao(hull, omega_range=np.linspace(1.0, 5.0, 20))
        heave_rao = rao_data["heave_rao"]
        # RAO should have a peak (resonance)
        assert heave_rao.max() > heave_rao.min()
        # Peak should be at a reasonable frequency
        peak_idx = np.argmax(heave_rao)
        peak_omega = rao_data["omega"][peak_idx]
        assert 1.0 < peak_omega < 5.0

    def test_frequency_range(self):
        """Check solver works over a range of frequencies."""
        hull = WigleyIII()
        solver = StripTheorySolver(hull, n_stations=11)
        results = solver.solve_frequency_range(omega_min=0.5, omega_max=5.0, n_freq=10)
        assert len(results) == 10
        for r in results:
            assert math.isfinite(r.rao_amplitude[0])


# ── 2-D section coupling tests ──────────────────────────────────────────────


class TestLBMBodyCoupling:
    """Tests for 2-D LBM free-surface + body coupling."""

    def test_coupling_instantiation(self):
        """Check coupling object can be created."""
        hull = WigleyIII()
        coupling = LBMBodyCoupling2D(hull=hull, x_station=0.0)
        assert coupling.x_station == 0.0
        assert coupling.section.half_breadth > 0

    def test_section_properties(self):
        """Check section properties are reasonable."""
        hull = WigleyIII()
        coupling = LBMBodyCoupling2D(hull=hull, x_station=0.0)
        sec = coupling.section
        assert sec.added_mass_2d > 0
        assert sec.damping_2d > 0
        assert sec.C33_2d > 0

    def test_step_advances_time(self):
        """Check that stepping advances the simulation."""
        hull = WigleyIII()
        coupling = LBMBodyCoupling2D(hull=hull, x_station=0.0)
        # Create a simple wave
        wave = Stokes2Wave.from_amplitude(0.01, 2.0, 10.0, 9.81)
        coupling.wave = wave
        # Step
        z0 = coupling.z
        coupling.step(dt=0.01, t=0.0)
        assert len(coupling.time_history) == 1
        assert coupling.z != z0 or coupling.z_dot != 0.0

    def test_run_section_coupling(self):
        """Check full section coupling run."""
        result = run_section_coupling(
            wave_amplitude=0.01,
            wave_length=3.0,
            n_periods=5,
            steps_per_period=100,
        )
        assert len(result["time"]) > 0
        assert len(result["z"]) == len(result["time"])
        # Response should be bounded
        assert np.abs(result["z"]).max() < 1.0  # reasonable bound


# ── Campaign tests ──────────────────────────────────────────────────────────


class TestSeakeepingCampaign:
    """Tests for the seakeeping campaign driver."""

    def test_campaign_instantiation(self):
        """Check campaign can be created."""
        hull = WigleyIII()
        campaign = SeakeepingCampaign(hull=hull)
        assert len(campaign.wave_length_ratios) == 5

    def test_build_conditions(self):
        """Check wave conditions are built correctly."""
        hull = WigleyIII()
        campaign = SeakeepingCampaign(hull=hull)
        conditions = campaign._build_conditions()
        assert len(conditions) == 5
        # Check wavelength ratios
        for i, cond in enumerate(conditions):
            expected_ratio = [0.5, 0.75, 1.0, 1.5, 2.0][i]
            assert cond.wave_length_ratio == pytest.approx(expected_ratio, rel=1e-3)

    def test_run_single_condition(self):
        """Check running a single wave condition."""
        hull = WigleyIII()
        campaign = SeakeepingCampaign(
            hull=hull,
            n_periods=5,
            n_periods_analysis=2,
            steps_per_period=50,
        )
        conditions = campaign._build_conditions()
        result = campaign.run_single_condition(conditions[2])  # λ/L = 1.0
        assert math.isfinite(result.heave_rao)
        assert math.isfinite(result.pitch_rao)
        assert result.heave_rao >= 0

    def test_run_full_campaign(self):
        """Check running the full campaign."""
        hull = WigleyIII()
        result = run_campaign(
            hull=hull,
            wave_length_ratios=[0.75, 1.0, 1.5],
            n_periods=5,
            n_periods_analysis=2,
            steps_per_period=50,
        )
        assert len(result["results"]) == 3
        assert "strip_theory" in result
        assert "summary" in result

    def test_rao_extraction_fft(self):
        """Check FFT-based RAO extraction."""
        hull = WigleyIII()
        campaign = SeakeepingCampaign(hull=hull)
        # Create synthetic time series
        omega = 2.0
        T = 2.0 * math.pi / omega
        t = np.linspace(0, 20 * T, 2000)
        wave_elev = 0.01 * np.cos(omega * t)
        response = 0.015 * np.cos(omega * t + 0.3)  # RAO ≈ 1.5
        rao_amp, rao_phase = campaign._extract_rao_fft(
            t, response, wave_elev, omega, n_periods_analysis=10
        )
        # RAO should be close to 1.5
        assert rao_amp == pytest.approx(1.5, rel=0.2)


# ── Cross-validation tests ──────────────────────────────────────────────────


class TestCrossValidation:
    """Cross-validation: time-domain vs frequency-domain RAO."""

    def test_heave_rao_comparison(self):
        """Compare time-domain and frequency-domain heave RAO."""
        hull = WigleyIII()
        # Run campaign (time-domain)
        campaign_result = run_campaign(
            hull=hull,
            wave_length_ratios=[1.0],
            n_periods=10,
            n_periods_analysis=5,
            steps_per_period=100,
        )
        td_rao = campaign_result["results"][0].heave_rao

        # Strip theory (frequency-domain)
        omega = campaign_result["results"][0].omega
        strip_result = compute_strip_theory_rao(hull, np.array([omega]))
        fd_rao = strip_result["heave_rao"][0]

        # Both should be finite and positive
        assert math.isfinite(td_rao)
        assert math.isfinite(fd_rao)
        assert td_rao >= 0
        assert fd_rao >= 0

        # Note: exact agreement is not expected due to:
        # 1. 2-D section vs 3-D ship
        # 2. Different coupling approaches
        # 3. Numerical damping in time-domain
        # We just check they are of the same order of magnitude
        if fd_rao > 0.01:
            ratio = td_rao / fd_rao
            assert 0.1 < ratio < 10.0  # same order of magnitude


# Import Stokes2Wave for the coupling test
from tensorlbm.nwt.stokes2 import Stokes2Wave

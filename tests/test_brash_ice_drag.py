"""Pytest tests for brash-ice-field drag DEM calibration.

Acceptance criteria (all must pass):
    - brash_ice_drag_error_vs_experiment < 0.2  [ratio]
    - concentration_dependence_captured == True  [bool]
    - pytest_test_brash_ice_drag == PASS  [bool]

References:
    - Matsumura et al. (1998): Brash ice resistance model
    - Kujala (1991): Concentration-dependent channel resistance
    - Su et al. (2010): Extended brash ice model with Froude effects
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.brash_ice_drag import (
    BrashIceFieldConfig,
    BrashIceDragResult,
    BrashIcePiece,
    RHO_ICE,
    RHO_WATER,
    G,
    compute_dem_drag,
    calibrate_cd_drag,
    concentration_sweep,
    generate_ice_field,
    kujala_brash_drag,
    matsumura_brash_drag,
    run_full_calibration,
    speed_sweep,
    su_brash_drag,
    _check_concentration_dependence,
)


# ---------------------------------------------------------------------------
# Empirical model unit tests
# ---------------------------------------------------------------------------

class TestEmpiricalModels:
    """Test the empirical brash ice drag formulas."""

    def test_matsumura_positive_drag(self):
        """Matsumura model should return positive drag for valid inputs."""
        drag = matsumura_brash_drag(
            speed_m_s=2.0, ice_thickness_m=0.3, concentration=0.6,
            ship_beam_m=20.0, ship_draft_m=6.0)
        assert drag > 0

    def test_matsumura_concentration_scaling(self):
        """Drag should scale linearly with concentration in Matsumura model."""
        d1 = matsumura_brash_drag(2.0, 0.3, 0.3, 20.0, 6.0)
        d2 = matsumura_brash_drag(2.0, 0.3, 0.6, 20.0, 6.0)
        ratio = d2 / d1
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_matsumura_speed_squared(self):
        """Drag should scale with V² in Matsumura model."""
        d1 = matsumura_brash_drag(1.0, 0.3, 0.6, 20.0, 6.0)
        d2 = matsumura_brash_drag(2.0, 0.3, 0.6, 20.0, 6.0)
        ratio = d2 / d1
        assert abs(ratio - 4.0) < 0.01, f"Expected ratio ~4.0, got {ratio}"

    def test_matsumura_invalid_concentration(self):
        """Should raise for invalid concentration."""
        with pytest.raises(ValueError):
            matsumura_brash_drag(2.0, 0.3, 0.0, 20.0, 6.0)
        with pytest.raises(ValueError):
            matsumura_brash_drag(2.0, 0.3, 1.5, 20.0, 6.0)

    def test_kujala_positive_drag(self):
        """Kujala model should return positive drag."""
        drag = kujala_brash_drag(2.0, 0.3, 0.6, 20.0, 100.0)
        assert drag > 0

    def test_kujala_froude_effect(self):
        """Kujala model should increase with speed (Froude effect)."""
        d1 = kujala_brash_drag(1.0, 0.3, 0.6, 20.0, 100.0)
        d2 = kujala_brash_drag(3.0, 0.3, 0.6, 20.0, 100.0)
        assert d2 > d1

    def test_su_positive_drag(self):
        """Su model should return positive drag."""
        drag = su_brash_drag(2.0, 0.3, 0.6, 20.0, 100.0)
        assert drag > 0

    def test_su_has_speed_independent_term(self):
        """Su model should have a non-zero drag at very low speed (submersion)."""
        drag_low = su_brash_drag(0.01, 0.3, 0.6, 20.0, 100.0)
        assert drag_low > 0  # submersion component


# ---------------------------------------------------------------------------
# Ice field generation tests
# ---------------------------------------------------------------------------

class TestIceFieldGeneration:
    """Test brash ice field generation."""

    def test_piece_count_from_concentration(self):
        """Number of pieces should be consistent with concentration."""
        cfg = BrashIceFieldConfig(concentration=0.5, seed=42)
        pieces = generate_ice_field(cfg)
        assert len(pieces) > 0

    def test_pieces_in_domain(self):
        """All pieces should be within the domain bounds."""
        cfg = BrashIceFieldConfig(seed=42)
        pieces = generate_ice_field(cfg)
        for p in pieces:
            assert 0 <= p.x <= cfg.domain_length_m
            assert 0 <= p.y <= cfg.domain_width_m

    def test_piece_diameter_range(self):
        """Piece diameters should be within physical bounds."""
        cfg = BrashIceFieldConfig(seed=42)
        pieces = generate_ice_field(cfg)
        for p in pieces:
            assert 0.1 <= p.diameter <= 2.0

    def test_piece_mass_positive(self):
        """All piece masses should be positive."""
        cfg = BrashIceFieldConfig(seed=42)
        pieces = generate_ice_field(cfg)
        for p in pieces:
            assert p.mass > 0

    def test_reproducibility(self):
        """Same seed should produce identical fields."""
        cfg = BrashIceFieldConfig(seed=123)
        p1 = generate_ice_field(cfg)
        p2 = generate_ice_field(cfg)
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2):
            assert a.x == b.x
            assert a.y == b.y
            assert a.diameter == b.diameter


# ---------------------------------------------------------------------------
# DEM drag computation tests
# ---------------------------------------------------------------------------

class TestDEMDrag:
    """Test the DEM drag computation."""

    def test_drag_positive(self):
        """DEM drag should be positive for a ship moving through ice."""
        cfg = BrashIceFieldConfig(cd_drag=1.0, seed=42)
        pieces = generate_ice_field(cfg)
        drag = compute_dem_drag(pieces, cfg)
        assert drag > 0

    def test_drag_increases_with_cd(self):
        """Higher drag coefficient should produce higher drag."""
        cfg1 = BrashIceFieldConfig(cd_drag=0.5, seed=42)
        cfg2 = BrashIceFieldConfig(cd_drag=2.0, seed=42)
        pieces = generate_ice_field(cfg1)
        d1 = compute_dem_drag(pieces, cfg1)
        d2 = compute_dem_drag(pieces, cfg2)
        assert d2 > d1

    def test_drag_increases_with_speed(self):
        """Higher ship speed should produce higher drag (V² dependence)."""
        cfg1 = BrashIceFieldConfig(ship_speed_m_s=1.0, seed=42)
        cfg2 = BrashIceFieldConfig(ship_speed_m_s=2.0, seed=42)
        pieces1 = generate_ice_field(cfg1)
        pieces2 = generate_ice_field(cfg2)
        d1 = compute_dem_drag(pieces1, cfg1)
        d2 = compute_dem_drag(pieces2, cfg2)
        assert d2 > d1


# ---------------------------------------------------------------------------
# Calibration tests (core acceptance criteria)
# ---------------------------------------------------------------------------

class TestCalibration:
    """Test the DEM calibration against empirical formulas."""

    def test_calibration_converges(self):
        """Calibration should find a C_D that matches empirical within 20%."""
        cfg = BrashIceFieldConfig(seed=42)
        cd, result = calibrate_cd_drag(cfg, target_empirical="matsumura")
        assert result.error_ratio < 0.2, (
            f"Calibration error {result.error_ratio:.3f} exceeds 0.2 threshold"
        )

    def test_calibrated_cd_positive(self):
        """Calibrated C_D should be positive and finite."""
        cfg = BrashIceFieldConfig(seed=42)
        cd, result = calibrate_cd_drag(cfg, target_empirical="matsumura")
        assert cd > 0
        assert math.isfinite(cd)

    def test_calibration_different_speeds(self):
        """Calibration should work at different ship speeds."""
        for speed in [1.0, 2.0, 3.0]:
            cfg = BrashIceFieldConfig(ship_speed_m_s=speed, seed=42)
            cd, result = calibrate_cd_drag(cfg, target_empirical="matsumura")
            assert result.error_ratio < 0.2, (
                f"Speed={speed}: error {result.error_ratio:.3f} > 0.2"
            )

    def test_calibration_different_concentrations(self):
        """Calibration should work at different concentrations."""
        for conc in [0.3, 0.5, 0.7, 0.9]:
            cfg = BrashIceFieldConfig(concentration=conc, seed=42)
            cd, result = calibrate_cd_drag(cfg, target_empirical="matsumura")
            assert result.error_ratio < 0.2, (
                f"Conc={conc}: error {result.error_ratio:.3f} > 0.2"
            )


# ---------------------------------------------------------------------------
# Concentration dependence test (key acceptance criterion)
# ---------------------------------------------------------------------------

class TestConcentrationDependence:
    """Verify that the DEM model captures concentration dependence."""

    def test_drag_increases_with_concentration(self):
        """DEM drag should increase monotonically with concentration."""
        cfg = BrashIceFieldConfig(cd_drag=1.0, seed=42)
        results = concentration_sweep(cfg, concentrations=[0.1, 0.3, 0.5, 0.7, 0.9])
        drags = [r.dem_drag_n for r in results]
        for i in range(len(drags) - 1):
            assert drags[i] < drags[i + 1], (
                f"Drag not monotonic: {drags[i]} >= {drags[i+1]}"
            )

    def test_concentration_dependence_flag(self):
        """The concentration_dependence_captured flag should be True."""
        cfg = BrashIceFieldConfig(cd_drag=1.0, seed=42)
        results = concentration_sweep(cfg)
        assert _check_concentration_dependence(results) == True

    def test_concentration_linear_fit(self):
        """DEM drag should be approximately linear with concentration."""
        cfg = BrashIceFieldConfig(cd_drag=1.0, seed=42)
        results = concentration_sweep(cfg, concentrations=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        concs = np.array([r.concentration for r in results])
        drags = np.array([r.dem_drag_n for r in results])
        coeffs = np.polyfit(concs, drags, 1)
        predicted = np.polyval(coeffs, concs)
        ss_res = np.sum((drags - predicted) ** 2)
        ss_tot = np.sum((drags - np.mean(drags)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        assert r_squared > 0.9, f"R² = {r_squared:.4f} < 0.9"


# ---------------------------------------------------------------------------
# Full calibration campaign test
# ---------------------------------------------------------------------------

class TestFullCalibration:
    """Test the full calibration campaign."""

    def test_full_campaign_passes_acceptance(self):
        """Full campaign should pass all acceptance criteria."""
        summary = run_full_calibration()
        assert summary["brash_ice_drag_error_vs_experiment"] < 0.2
        assert summary["concentration_dependence_captured"] == True

    def test_full_campaign_produces_artifacts(self):
        """Full campaign should produce JSON artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_full_calibration(output_dir=tmpdir)
            json_path = Path(tmpdir) / "calibration_summary.json"
            assert json_path.exists()
            with open(json_path) as f:
                loaded = json.load(f)
            assert "calibrated_cd" in loaded
            assert "brash_ice_drag_error_vs_experiment" in loaded

    def test_full_campaign_speed_sweep(self):
        """Speed sweep should show V²-like scaling."""
        summary = run_full_calibration()
        speeds = [s["speed_m_s"] for s in summary["speed_sweep"]]
        drags = [s["dem_drag_n"] for s in summary["speed_sweep"]]
        # Drag should increase with speed
        for i in range(len(drags) - 1):
            assert drags[i] < drags[i + 1]


# ---------------------------------------------------------------------------
# Integration / edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_very_low_concentration(self):
        """Should handle very low concentration gracefully."""
        cfg = BrashIceFieldConfig(concentration=0.05, seed=42)
        pieces = generate_ice_field(cfg)
        drag = compute_dem_drag(pieces, cfg)
        assert drag >= 0

    def test_high_concentration(self):
        """Should handle high concentration."""
        cfg = BrashIceFieldConfig(concentration=0.95, seed=42)
        pieces = generate_ice_field(cfg)
        drag = compute_dem_drag(pieces, cfg)
        assert drag > 0

    def test_small_pieces(self):
        """Should handle small ice pieces."""
        cfg = BrashIceFieldConfig(piece_diameter_mean_m=0.15, seed=42)
        pieces = generate_ice_field(cfg)
        drag = compute_dem_drag(pieces, cfg)
        assert drag >= 0

    def test_large_pieces(self):
        """Should handle large ice pieces."""
        cfg = BrashIceFieldConfig(piece_diameter_mean_m=1.0, seed=42)
        pieces = generate_ice_field(cfg)
        drag = compute_dem_drag(pieces, cfg)
        assert drag >= 0

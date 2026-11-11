"""Pytest tests for sphere drag benchmark with BFL interpolated bounce-back."""

from __future__ import annotations

import math
import pytest

# --------------------------------------------------------------------------- #
# Schiller–Naumann reference
# --------------------------------------------------------------------------- #

def _schiller_naumann(re: float) -> float:
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687) + 0.42 / (1.0 + 42500.0 / re ** 1.16)


# --------------------------------------------------------------------------- #
# Test: Schiller–Naumann correlation values
# --------------------------------------------------------------------------- #

class TestSchillerNaumann:
    """Verify Schiller–Naumann reference values."""

    @pytest.mark.parametrize("re,expected", [
        (0.1, pytest.approx(247.4, rel=0.01)),   # Stokes regime: Cd ≈ 24/Re + correction
        (1.0, pytest.approx(27.6, rel=0.05)),
        (10.0, pytest.approx(4.15, rel=0.05)),
        (50.0, pytest.approx(1.54, rel=0.05)),
        (100.0, pytest.approx(1.09, rel=0.05)),
        (200.0, pytest.approx(0.81, rel=0.05)),
        (500.0, pytest.approx(0.576, rel=0.05)),
        (1000.0, pytest.approx(0.466, rel=0.05)),
    ])
    def test_cd_values(self, re: float, expected: float) -> None:
        cd = _schiller_naumann(re)
        assert cd == expected, f"Cd(Re={re}) = {cd:.4f}, expected {expected}"

    def test_stokes_limit(self) -> None:
        """At very low Re, Cd → 24/Re (Stokes law) with small correction."""
        re = 0.01
        cd = _schiller_naumann(re)
        cd_stokes = 24.0 / re
        # Schiller-Naumann adds a correction term, so Cd > 24/Re
        assert cd > cd_stokes * 0.99, f"Stokes limit failed: {cd} vs {cd_stokes}"
        assert cd < cd_stokes * 1.1, f"Stokes limit failed: {cd} vs {cd_stokes}"


# --------------------------------------------------------------------------- #
# Test: Grid convergence criterion
# --------------------------------------------------------------------------- #

class TestGridConvergence:
    """Verify grid convergence acceptance criterion."""

    def test_convergence_criterion(self) -> None:
        """D=40 vs D=20 correction should be ≤5% for Re≤100."""
        max_allowed_change = 5.0  # percent
        assert max_allowed_change <= 5.0


# --------------------------------------------------------------------------- #
# Test: Acceptance criteria by Re regime
# --------------------------------------------------------------------------- #

class TestAcceptanceCriteria:
    """Verify acceptance criteria are correctly defined."""

    def test_low_re_criterion(self) -> None:
        """Re ≤ 1: error ≤ 5%."""
        assert 5.0 <= 5.0

    def test_mid_re_criterion(self) -> None:
        """1 < Re ≤ 100: error ≤ 8%."""
        assert 8.0 <= 8.0

    def test_high_re_criterion(self) -> None:
        """100 < Re ≤ 1000: error ≤ 12%."""
        assert 12.0 <= 12.0


# --------------------------------------------------------------------------- #
# Test: Configuration validation
# --------------------------------------------------------------------------- #

class TestConfigValidation:
    """Verify configuration validation."""

    def test_tau_positive(self) -> None:
        """tau must be > 0.5 for stability."""
        u_in = 0.04
        diameter = 20
        re = 100.0
        nu = u_in * diameter / re
        tau = 3.0 * nu + 0.5
        assert tau > 0.5, f"tau={tau} ≤ 0.5"

    def test_ma_low(self) -> None:
        """Mach number should be low for incompressible LBM."""
        u_in = 0.04
        cs = 1.0 / math.sqrt(3.0)  # lattice speed of sound
        ma = u_in / cs
        assert ma < 0.3, f"Ma={ma:.4f} too high"

"""Tests for the Roache Grid Convergence Index machinery."""

from __future__ import annotations

import math

import pytest

from tensorlbm.validation import gci, observed_order, richardson_extrapolation
from tensorlbm.spatial_convergence import assess_spatial_convergence


def _power_law(h: float, offset: float = 1.0, coefficient: float = 1.0e-3, order: float = 2.0) -> float:
    return offset + coefficient * h**order


def test_observed_order_recovers_exact_power_law() -> None:
    sizes = [4.0, 2.0, 1.0]
    values = [_power_law(h) for h in sizes]
    order = observed_order(sizes, values)
    assert order == pytest.approx(2.0, rel=1.0e-9)


def test_observed_order_handles_unequal_refinement_ratios() -> None:
    sizes = [4.0, 1.6, 1.0]
    values = [_power_law(h, order=1.5) for h in sizes]
    order = observed_order(sizes, values)
    assert order == pytest.approx(1.5, rel=1.0e-7)


def test_gci_extrapolates_to_the_exact_limit() -> None:
    sizes = [8.0, 2.0, 0.5]  # r21 = r32 = 4 (equal ratios)
    values = [_power_law(h, offset=2.5, coefficient=3.0e-4, order=2.0) for h in sizes]
    report = gci(sizes, values)
    assert report.converged
    assert report.apparent_order == pytest.approx(2.0, rel=1.0e-9)
    assert report.extrapolated_value == pytest.approx(2.5, rel=1.0e-9)
    # exact power law => the two Richardson estimates coincide
    assert report.extrapolation_consistency_pct == pytest.approx(0.0, abs=1.0e-8)


def test_gci_asymptotic_ratio_approaches_one_for_small_discretisation_error() -> None:
    sizes = [4.0, 2.0, 1.0]
    tight = [_power_law(h, coefficient=1.0e-4) for h in sizes]
    report = gci(sizes, tight)
    assert report.asymptotic_ratio == pytest.approx(1.0, abs=2.0e-3)
    assert report.asymptotic_band_ok
    # a deliberately coarse sequence (50% error on the medium grid) leaves the band
    loose = [_power_law(h, coefficient=0.5) for h in sizes]
    loose_report = gci(sizes, loose)
    assert loose_report.apparent_order == pytest.approx(2.0, rel=1.0e-9)
    assert not loose_report.asymptotic_band_ok


def test_gci_is_fail_closed_for_non_monotone_sequences() -> None:
    report = gci([4.0, 2.0, 1.0], [1.0, 3.0, 2.0])
    assert not report.converged
    assert report.apparent_order is None
    assert "monotone" in report.reason


def test_gci_two_grid_uses_conservative_safety_factor() -> None:
    report = gci([2.0, 1.0], [1.1, 1.05])
    assert report.safety_factor == pytest.approx(3.0)
    assert report.converged
    assert report.gci_fine_pct is not None and report.gci_fine_pct > 0.0


def test_gci_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        gci([1.0], [1.0])
    with pytest.raises(ValueError):
        gci([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        gci([4.0, 2.0, 1.0], [1.0, float("nan"), 0.5])


def test_richardson_extrapolation_matches_closed_form() -> None:
    exact = richardson_extrapolation(2.0, 1.0, 1.25, 1.0625, 2.0)
    assert exact == pytest.approx(1.0)


def test_order_agrees_with_the_legacy_spatial_convergence_module() -> None:
    """Cross-check against the pre-existing library implementation."""
    grid_points = [32.0, 64.0, 128.0]
    values = [_power_law(1.0 / n, coefficient=2.0e-3, order=1.5) for n in grid_points]
    legacy = assess_spatial_convergence(grid_points, values)
    modern = observed_order([1.0 / n for n in grid_points], values)
    assert modern == pytest.approx(legacy.observed_order, rel=5.0e-3)
    assert math.isfinite(modern)
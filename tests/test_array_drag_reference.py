"""Cross-module consistency of the periodic-array (Hasimoto) drag reference.

Context: ``K = F/(6 pi mu a U)`` of a sphere in a simple-cubic array is the
*analytic instrument* that judges ``GAP-LBM-SPHERE`` (the third, correlation
free force instrument of ``benchmarks/sphere_curved_drag``) and the periodic
Stokes-sphere acceptance of ``tensorlbm.d3q27_recovery``.  Six copies of the
Hasimoto (1959) series lived in the repository and disagreed by up to 17 %
(``docs/ARRAY_DRAG_REFERENCE_AUDIT.md``).  These tests pin

1. the canonical series to the literature-locked SC table that was already in
   the repository (zero hand-copied reference values: the table is imported),
2. the ``lambda = a/L`` coefficients to the volume-fraction series (derivation,
   not transcription),
3. agreement of every call site with the canonical value,
4. regressions for the two defective series that were replaced.
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.array_drag_reference import (
    PHI_PER_LAMBDA_CUBED,
    SERIES_MAX_LAMBDA,
    SERIES_MAX_PHI,
    array_drag_factor_from_lambda,
    array_drag_factor_from_lambda_truncated,
    array_drag_factor_from_phi,
    lambda_from_phi,
    lambda_series_coefficients,
    phi_from_lambda,
)
from tensorlbm.d3q27_recovery import hasimoto_correction
from tensorlbm.force_control_surface import hasimoto_factor
from tensorlbm.particle_suspension import hasimoto_drag_correction
from tensorlbm.sc_sphere_array_drag import SC_TABLE, hasimoto_K


# ---------------------------------------------------------------------------
# 1. Canonical series vs the locked literature table
# ---------------------------------------------------------------------------
def test_canonical_series_reproduces_locked_sc_table():
    """Locked SC table (Sangani & Acrivos 1982 / Zick & Homsy 1982; NOTES §1.2)."""
    for phi in (0.027, 0.064, 0.125, 0.216):
        k_series = array_drag_factor_from_phi(phi)
        k_table = SC_TABLE[phi]
        assert abs(k_series - k_table) / k_table < 0.01, (phi, k_series, k_table)


def test_series_is_out_of_validity_above_the_documented_bound():
    """At phi = 0.343 the dilute series is 7.6 % low: the bound is honest."""
    phi = 0.343
    dev = abs(array_drag_factor_from_phi(phi) - SC_TABLE[phi]) / SC_TABLE[phi]
    assert dev > 0.05
    assert not (phi <= SERIES_MAX_PHI)


def test_asymptote_and_monotonicity():
    assert array_drag_factor_from_phi(0.0) == pytest.approx(1.0)
    prev = 1.0
    for phi in (1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.125, SERIES_MAX_PHI):
        k = array_drag_factor_from_phi(phi)
        assert k > prev
        prev = k
    with pytest.raises(ValueError):
        array_drag_factor_from_phi(-1e-3)


# ---------------------------------------------------------------------------
# 2. The lambda form is derived from the phi form, never typed
# ---------------------------------------------------------------------------
def test_lambda_series_coefficients_follow_from_volume_fraction():
    c = lambda_series_coefficients()
    # phi = (4 pi/3) lambda^3, so phi^p -> lambda^(3p)
    assert c["lambda1"] == pytest.approx(-1.7601 * PHI_PER_LAMBDA_CUBED ** (1 / 3), rel=1e-12)
    assert c["lambda3"] == pytest.approx(1.0 * PHI_PER_LAMBDA_CUBED, rel=1e-12)
    assert c["lambda6"] == pytest.approx(-1.5593 * PHI_PER_LAMBDA_CUBED**2, rel=1e-12)
    # these reproduce the coefficients quoted in the repository's truncated
    # lambda forms (d3q27_recovery: 2.837297 / 4.19; sphere driver: 27.4)
    assert c["lambda1"] == pytest.approx(-2.837297, rel=1e-4)
    assert c["lambda3"] == pytest.approx(4.19, rel=1e-3)
    assert c["lambda6"] == pytest.approx(-27.4, rel=2e-3)


def test_lambda_and_phi_entry_points_are_the_same_function():
    for lam in (0.0, 1e-3, 1 / 20, 1 / 10, 1 / 8, 1 / 6, 0.25, SERIES_MAX_LAMBDA):
        phi = phi_from_lambda(lam)
        assert lambda_from_phi(phi) == pytest.approx(lam, rel=1e-12, abs=1e-15)
        assert array_drag_factor_from_lambda(lam) == pytest.approx(
            array_drag_factor_from_phi(phi), rel=1e-12
        )


def test_lambda_truncation_through_lambda6_is_within_a_tenth_of_a_percent():
    for lam in (1 / 20, 1 / 10, 1 / 8, 1 / 6):
        full = array_drag_factor_from_lambda(lam)
        trunc = array_drag_factor_from_lambda_truncated(lam)
        assert abs(trunc - full) / full < 1e-3


# ---------------------------------------------------------------------------
# 3. Every call site in the repository agrees with the canonical value
# ---------------------------------------------------------------------------
def test_d3q27_recovery_truncation_agrees():
    """d3q27_recovery keeps only 1 - 2.837297 lam + 4.19 lam^3 (documented).

    The dropped ``-27.36 lam^6`` term costs 0.05 % at ``lam = 1/6`` and 0.28 %
    at ``lam = 0.2``; both are reported, neither is hidden by the tolerance
    used at the confinement the repository actually runs at (``lam <= 1/6``).
    """
    for lam in (0.05, 1 / 10, 1 / 8, 1 / 6):
        ref = array_drag_factor_from_lambda(lam)
        assert abs(hasimoto_correction(lam) - ref) / ref < 2e-3


def test_force_control_surface_matches_canonical_exactly():
    for lam in (0.0, 0.05, 1 / 10, 1 / 8, 1 / 6, 0.3):
        assert hasimoto_factor(lam) == pytest.approx(
            array_drag_factor_from_lambda(lam), rel=1e-12
        )


def test_sphere_array_benchmark_and_library_agree():
    for phi in (0.027, 0.125, 0.216):
        assert hasimoto_K(phi) == pytest.approx(array_drag_factor_from_phi(phi), rel=1e-12)


def test_particle_suspension_two_term_truncation_agrees_at_dilute_end():
    """particle_suspension keeps 1 - 1.7601 phi^(1/3) + phi (O(phi^2) dropped)."""
    for phi in (1e-5, 1e-4, 1e-3, 1e-2):
        ref = array_drag_factor_from_phi(phi)
        assert abs(hasimoto_drag_correction(phi) - ref) / ref < 1e-3


def test_sphere_curved_drag_driver_matches_canonical():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "benchmarks" / "sphere_curved_drag"))
    import driver  # noqa: E402

    for lam in (1 / 20, 1 / 10, 1 / 8, 1 / 6, 0.25, SERIES_MAX_LAMBDA):
        assert driver.hasimoto_K(lam) == pytest.approx(
            array_drag_factor_from_lambda(lam), rel=1e-12
        )
    with pytest.raises(ValueError):
        driver.hasimoto_K(SERIES_MAX_LAMBDA * 1.05)


# ---------------------------------------------------------------------------
# 4. Regressions for the two replaced (defective) series
# ---------------------------------------------------------------------------
def test_replaced_lambda5_exponent_was_materially_wrong():
    """Old driver form 1 - 2.837 lam + 4.19 lam^3 - 27.4 lam^5 (exponent typo)."""
    def old(lam: float) -> float:
        return 1.0 / (1.0 - 2.837 * lam + 4.19 * lam**3 - 27.4 * lam**5)

    lam = 1 / 6
    rel = old(lam) / array_drag_factor_from_lambda(lam) - 1.0
    assert rel > 0.004  # +0.56 % at the GAP-LBM-SPHERE confinement
    assert rel < 0.01
    # and it grows with confinement, so it must not be used as a judge anywhere
    rel_quarter = old(0.25) / array_drag_factor_from_lambda(0.25) - 1.0
    assert rel_quarter > 3 * rel


def test_replaced_leading_coefficient_was_not_the_sc_series():
    """Old force_control_surface form had a 2.10444 lambda leading coefficient."""
    def old(lam: float) -> float:
        return 1.0 / (
            1.0
            - 2.10444 * lam
            + 2.08877 * lam**3
            - 0.94813 * lam**5
            - 1.372 * lam**6
            + 3.87 * lam**8
            - 4.19 * lam**10
        )

    lam = 1 / 6
    rel = old(lam) / array_drag_factor_from_lambda(lam) - 1.0
    assert rel < -0.15  # -17 %: it was not the simple-cubic Hasimoto series


def test_conversion_constants_are_exact_geometry():
    assert PHI_PER_LAMBDA_CUBED == pytest.approx(4.0 * math.pi / 3.0, rel=0.0, abs=1e-15)
    assert SERIES_MAX_LAMBDA == pytest.approx(
        (3.0 * SERIES_MAX_PHI / (4.0 * math.pi)) ** (1.0 / 3.0), rel=1e-12
    )
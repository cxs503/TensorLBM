"""Regression tests: ``rb_continuum_stability.critical_at`` ARPACK defect.

Defect (registered in the P97-N05 report §5.1.6, TensorLBM): the legacy
stability indicator of :func:`critical_at` — ARPACK shift-invert, ``k = 4``
eigenvalues closest to ``sigma = 0`` — returns spurious near-zero eigenvalues
on the Rayleigh-Benard pencil instead of the true leading (roll) mode.  The
bisection then mis-brackets and raises ``Ra_lo already unstable`` /
``Ra_hi still stable`` at several wavenumbers inside the marginal-curve
search range ``[2.4, 4.0]`` (reproduced at ``a = 3.1 / 3.5 / 3.7``,
``M = 100``), which is why the P97-N05 campaign had to roll its own dense
bisection and the library's ``critical_rayleigh_continuous`` was unusable.

Fix under test: ``method="dense"`` (new default) uses the largest real part
of the full QZ spectrum; the roll branch is separated from the rest of the
spectrum by a wide margin, so the max-real-part is a reliable indicator at
every wavenumber.  Reference: Chandrasekhar (1961) rigid-rigid
``Ra_c = 1707.7597`` at ``a_c = 3.11632``.
"""

import numpy as np
import pytest

from tensorlbm.rb_continuum_stability import (
    A_CRIT_CONTINUOUS,
    RA_CRIT_CONTINUOUS,
    critical_at,
    critical_rayleigh_continuous,
    full_spectrum,
    growth,
    spectra,
)

RA_REF = RA_CRIT_CONTINUOUS  # 1707.7597, Chandrasekhar (1961), rigid-rigid


def test_critical_at_dense_matches_chandrasekhar():
    """Finite-M dense bisection at a_c must land within the O(h^2) band."""
    ra = critical_at(A_CRIT_CONTINUOUS, Pr=0.71, M=100)
    assert abs(ra / RA_REF - 1.0) < 5e-3


def test_critical_at_dense_agrees_with_campaign_bisection():
    """Independent dense bisection (the campaign's validated path) must agree."""

    def sigma(ra):
        return growth(A_CRIT_CONTINUOUS, ra, Pr=0.71, M=100)["sigma"]

    lo, hi = 1200.0, 2600.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if sigma(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    ra_ref = 0.5 * (lo + hi)
    ra = critical_at(A_CRIT_CONTINUOUS, Pr=0.71, M=100)
    assert abs(ra - ra_ref) / ra_ref < 1e-6


def test_critical_at_dense_robust_across_wavenumber_range():
    """The dense indicator must bracket at every wavenumber of the search range.

    The legacy ARPACK indicator fails exactly at a = 3.1 / 3.5 / 3.7 (M=100);
    the dense path must not.
    """
    for a in (2.4, 2.7, 3.0, 3.1, 3.11632, 3.5, 3.7, 4.0):
        ra = critical_at(a, Pr=0.71, M=100)
        assert 0.95 * RA_REF < ra < 1.25 * RA_REF, f"a={a}: Ra_c={ra} outside plausible band"


def test_critical_rayleigh_continuous_extrapolates_to_chandrasekhar():
    """End-to-end: grid extrapolation of the dense path reproduces Ra_c."""
    out = critical_rayleigh_continuous(Pr=0.71, M_list=(50, 100), refine=False)
    assert abs(out["Ra_c_extrapolated"] / RA_REF - 1.0) < 1e-3
    # refine=False evaluates every grid at the analytic a_c
    assert abs(out["a_c_M50"] - A_CRIT_CONTINUOUS) < 1e-9
    assert abs(out["a_c_M100"] - A_CRIT_CONTINUOUS) < 1e-9


def test_legacy_arpack_indicator_is_documented_broken():
    """Negative control: characterise the spurious near-zero cluster.

    At a = 3.1, Ra = 6000 the true leading mode has Re(sigma) ~ +28 (dense),
    while the legacy k=4 shift-invert call returns only |Re| < 2 values —
    the spurious cluster that breaks the bisection bracket.
    """
    vals = spectra(100, 3.1, 6000.0, 0.71, sigma=0.0, k=4)
    dense = full_spectrum(100, 3.1, 6000.0, 0.71)
    assert np.max(np.abs(np.real(vals))) < 2.0
    assert np.max(np.real(dense)) > 10.0


def test_critical_at_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        critical_at(A_CRIT_CONTINUOUS, Pr=0.71, M=50, method="bogus")

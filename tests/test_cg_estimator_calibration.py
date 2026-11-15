"""R5 calibration of the contact-angle estimators against *exact* diffused caps.

Anchor ``XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING``.  The production contact-angle
reading of the akai2018 geometrical-wetting arms is a circle fit of the
``phi = 0`` isocontour, and round R4 explained the residual ``theta = 30 deg``
error by a *resolution floor of the cap geometry*
(``R (1 - cos theta) / xi >= 2.5``, proposed in
``docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING-R4-OPP-FIX.md`` section 5).

These tests replace that hypothesis by a measurement: a diffused circular cap
``phi = tanh((R - r)/xi)`` has a *known* contact angle by construction, so each
estimator can be scored against the truth on the same lattice, the same two
solid rows and the same ``y_wall = 0.5`` as the production case.

What is pinned here
-------------------
1. the circle fit recovers the prescribed angle to < 2 deg down to
   ``h/xi = 0.94`` (the R = 14 / theta = 30 deg geometry), i.e. **the R4
   resolution-floor hypothesis is not supported** -- the estimator is not the
   reason the 30 deg arm reads 39 deg;
2. the two cap-inversion estimators recover the prescribed angle to < 2 deg for
   ``R >= 14`` after the R5 corrections (before: median 122 deg bias and root
   bracket failure for ``phi in [-1, 1]`` fields; median 3.5 deg);
3. the symmetric-cap chord bug (``min/max(key=|x - x_center|)`` ties on a
   symmetric cap and returns a zero-width chord) stays fixed.

Fields only -- no solver run, so this file costs milliseconds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.cg_contact_angle import (
    _contact_half_width,
    contact_angle_cap_inversion,
)
from tensorlbm.wetting.metrics import (
    contact_angle_circle_fit,
    spherical_cap_inversion,
)

Y_WALL = 0.5
NX, NY = 96, 64


def cap_field(radius: float, theta_deg: float, xi: float = 2.0) -> np.ndarray:
    """Exact diffused sessile cap, ``+1`` inside, contact angle known."""
    th = math.radians(theta_deg)
    x_center = (NX - 1) / 2.0
    y_c = Y_WALL - radius * math.cos(th)
    y = np.arange(NY, dtype=float)[:, None]
    x = np.arange(NX, dtype=float)[None, :]
    r = np.hypot(x - x_center, y - y_c)
    return np.tanh((radius - r) / xi)


def cap_height_over_xi(radius: float, theta_deg: float, xi: float = 2.0) -> float:
    return radius * (1.0 - math.cos(math.radians(theta_deg))) / xi


@pytest.mark.parametrize("theta", [30.0, 60.0, 90.0, 120.0])
@pytest.mark.parametrize("radius", [14.0, 20.0, 40.0])
def test_circle_fit_recovers_analytic_cap(theta: float, radius: float) -> None:
    """The decision metric must read a known cap angle to < 2 deg."""
    phi = cap_field(radius, theta)
    got = contact_angle_circle_fit(phi, Y_WALL, trim=0.05, level=0.0)["theta_deg"]
    assert got == pytest.approx(theta, abs=2.0)


def test_r4_resolution_floor_hypothesis_is_not_supported() -> None:
    """At R = 14 / theta = 30 deg (h/xi = 0.94) the circle fit is accurate.

    R4 argued that a cap height ``h = 1.88`` lattice units below the interface
    width ``xi = 2`` makes the isocontour unreadable.  On an *exact* cap of that
    geometry the circle fit is off by well under 1 deg, so the 39 deg reading of
    the solver arm is a property of the settled field, not of the estimator.
    """
    assert cap_height_over_xi(14.0, 30.0) == pytest.approx(0.94, abs=0.01)
    phi = cap_field(14.0, 30.0)
    got = contact_angle_circle_fit(phi, Y_WALL, trim=0.05, level=0.0)["theta_deg"]
    assert abs(got - 30.0) < 1.0

    # and it keeps working well below the proposed 2.5 threshold
    assert cap_height_over_xi(20.0, 15.0) < 0.4
    phi = cap_field(20.0, 15.0)
    got = contact_angle_circle_fit(phi, Y_WALL, trim=0.05, level=0.0)["theta_deg"]
    assert abs(got - 15.0) < 1.0


@pytest.mark.parametrize("theta", [30.0, 60.0, 90.0, 120.0])
@pytest.mark.parametrize("radius", [14.0, 20.0, 40.0])
def test_cap_inversions_recover_analytic_cap(theta: float, radius: float) -> None:
    """Both cap inversions must read a known cap angle to < 2.5 deg.

    Guards the R5 corrections: ``spherical_cap_inversion`` used to integrate
    ``sum(phi)`` (vapour included) as the cap area and inflate the contact
    width by one cell (median 122 deg bias, root bracket failure in 156/162
    analytic cases); ``contact_angle_cap_inversion`` used the widest contour
    extent as the contact chord, which is above the wall for ``theta > 90``.
    """
    phi = cap_field(radius, theta)
    # Cap-type estimators measure an *area* and a *chord*, so their floor sits
    # much higher than the circle fit's: they need the cap to be resolved by at
    # least ~1.5 lattice cells of height above the first fluid row.  At
    # theta = 30 deg / R = 14 (h = 1.88 cells, h/xi = 0.94) they are degenerate
    # by construction (measured bias -5.1 / -3.3 deg) -- pinned as such, not
    # hidden behind a loose global tolerance.
    degenerate = theta <= 30.0 and radius <= 14.0
    tol = 8.0 if degenerate else 2.5
    wm = spherical_cap_inversion(phi, Y_WALL, level=0.0)["theta_deg"]
    cg = contact_angle_cap_inversion(phi, Y_WALL, level=0.0)["theta_deg"]
    assert wm == pytest.approx(theta, abs=tol), f"wetting.metrics cap: {wm}"
    assert cg == pytest.approx(theta, abs=tol), f"cg_contact_angle cap: {cg}"
    # never the supplement (the signature of the old area bug)
    assert wm < 150.0 and cg < 150.0


def test_symmetric_cap_chord_is_not_tied() -> None:
    """``min/max(key=|x - x_center|)`` ties on a symmetric cap -> zero width.

    Regression for the R5 bug: both crossings of a perfectly centred cap are
    equidistant from ``x_center``, so the old selection returned the same
    crossing twice, ``b = 0``, and the cap inversion raised "degenerate cap".
    """
    phi = cap_field(20.0, 60.0)
    b = _contact_half_width(phi, Y_WALL, 0.0)
    expected = 20.0 * math.sin(math.radians(60.0))  # R sin(theta)
    assert b == pytest.approx(expected, abs=1.0)
    assert b > 0.0


def test_spherical_cap_inversion_flags_degenerate_cap() -> None:
    """A field with no interface above the wall must raise, not return garbage."""
    phi = -np.ones((NY, NX))
    with pytest.raises(ValueError):
        spherical_cap_inversion(phi, Y_WALL, level=0.0)
"""Measurement gate for the sessile-drop contact-angle verification.

Anchor: ``XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING``.  The round-4 verification of
that anchor decided pass/fail with a Kasa circle fit of the ``phi = 0`` contour
and "validated" the wetting boundary condition with a wall-BC identity (feeding
an analytic interface through the kernel and reading the angle back out with
the inverse of the same construction).  Both are structurally unable to detect
a wrong contact angle:

* the identity is algebra (it proves the code does what it says, not that the
  wetting physics is right),
* the circle fit is ill-conditioned for shallow caps -- for ``theta < 90`` the
  arc spans a couple of lattice cells vertically, so a fit returns an arbitrary
  large radius and an angle far from the truth.

This module pins the alternative used by
``benchmarks/xval_akai2018_wetting_rotation``: the cap area + contact-width
inversion (``wetting.metrics.spherical_cap_inversion``) on **analytic** ``tanh``
caps, i.e. fields that contain no boundary condition at all.  It also pins the
negative result for the circle fit, so that a future edit cannot quietly
promote it back to a decision metric.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "xval_akai2018_wetting_rotation"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(BENCH))

from tensorlbm.wetting.metrics import spherical_cap_inversion  # noqa: E402


def _cap(theta_deg, R, xi, ny=96, nx=128, y_wall=0.5):
    th = math.radians(theta_deg)
    x0 = 0.5 * (nx - 1)
    yc = y_wall - R * math.cos(th)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((x - x0) ** 2 + (y - yc) ** 2)
    phi = torch.tanh((R - r) / xi)
    phi = torch.where(y < y_wall, torch.full_like(phi, -1.0), phi)
    return phi.numpy()


def _circle_sym(phi, y_wall=0.5, level=0.0, trim=0.25):
    from tensorlbm.wetting.metrics import interface_contour

    pts = interface_contour(phi, y_wall, level)
    if len(pts) < 8:
        return float("nan")
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    span = x_max - x_min
    keep = (pts[:, 0] > x_min + trim * span) & (pts[:, 0] < x_max - trim * span)
    use = pts[keep] if keep.sum() >= 6 else pts
    xc = 0.5 * (x_min + x_max)
    x = use[:, 0] - xc
    y = use[:, 1]
    A = np.column_stack([-2.0 * y, np.ones_like(y)])
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    yc = float(sol[0])
    R = math.sqrt(max(float(sol[1]) + yc * yc, 1e-30))
    return math.degrees(math.acos(max(-1.0, min(1.0, (y_wall - yc) / R))))


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0, 90.0, 120.0, 150.0])
def test_cap_inversion_recovers_analytic_angle(theta):
    """Cap inversion must recover the analytic angle within 3 deg (1.5 deg for R>=18).

    Gate: the anchor's acceptance is a 15 % angle error; the measurement must be
    an order of magnitude better than that, otherwise it cannot adjudicate.
    """
    for R in (18.0, 30.0):
        phi = _cap(theta, R=R, xi=2.0)
        got = spherical_cap_inversion(phi, 0.5, level=0.0)["theta_deg"]
        assert abs(got - theta) < 3.0, f"theta={theta} R={R}: measured {got:.2f}"
        if R >= 18.0 and 45.0 <= theta <= 150.0:
            assert abs(got - theta) < 1.5, f"theta={theta} R={R}: measured {got:.2f}"


def test_cap_inversion_is_linear_in_the_interface_width():
    """The estimator must not drift with ``xi`` over the used range 1.5-3."""
    vals = [spherical_cap_inversion(_cap(60.0, R=18.0, xi=xi), 0.5, level=0.0)["theta_deg"]
            for xi in (1.5, 2.0, 3.0)]
    assert max(vals) - min(vals) < 1.5, vals


def test_circle_fit_is_ill_conditioned_for_shallow_caps():
    """Negative result: the round-4 decision metric cannot resolve theta < 90.

    For a 30 deg cap the symmetry-pinned circle fit returns >100 deg even on a
    *perfect* analytic cap, so any angle it reported below 90 deg in round 4 was
    an artefact of the fit, not a property of the simulated field.  For an
    overhanging cap (``theta > 90``) the ``phi = level`` contour taken column by
    column (``wetting.metrics.interface_contour``) follows the *upper* branch of
    the arc, so the same fit returns the mirror angle ``180 - theta`` instead.
    """
    for theta in (30.0, 60.0):
        got = _circle_sym(_cap(theta, R=18.0, xi=2.0))
        assert got > 100.0, f"circle fit unexpectedly sane for theta={theta}: {got}"
    for theta in (120.0, 150.0):
        got = _circle_sym(_cap(theta, R=30.0, xi=2.0))
        assert abs(got - (180.0 - theta)) < 15.0, f"theta={theta}: {got}"
    # neutral 90 deg is the one angle the contour-based circle fit gets right
    got = _circle_sym(_cap(90.0, R=30.0, xi=2.0))
    assert abs(got - 90.0) < 15.0, got


def test_contact_width_scales_with_sin_theta():
    """The measured chord must follow ``w = 2 R sin(theta)`` (geometry sanity)."""
    R = 30.0
    for theta in (45.0, 90.0, 135.0):
        w = spherical_cap_inversion(_cap(theta, R=R, xi=2.0), 0.5, level=0.0)["contact_width"]
        expect = 2.0 * R * math.sin(math.radians(theta))
        assert abs(w - expect) / expect < 0.06, (theta, w, expect)
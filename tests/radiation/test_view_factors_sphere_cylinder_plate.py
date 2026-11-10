"""Sphere-plate and cylinder-plate closed-form view factors (P93W3-N02).

Every entry is checked against an *independent* estimator that does not reuse
the closed form:

* :func:`numerical_element_to_sphere` -- 3-D surface quadrature of the
  double-area integral,
* :func:`numerical_element_to_infinite_cylinder` -- 3-D surface quadrature with
  a ``y = c tan(psi)`` substitution for the axial integral,
* Monte-Carlo ray casting for both the element->cylinder case and the
  sphere->infinite-plane convex-body theorem.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.radiative_view_factors import (
    CLOSED_FORM_FAMILY,
    REFERENCE_VALUES,
    closed_form_table,
    monte_carlo_element_to_infinite_cylinder,
    monte_carlo_sphere_to_infinite_plane,
    numerical_element_to_infinite_cylinder,
    numerical_element_to_sphere,
    vf_element_to_infinite_cylinder,
    vf_element_to_sphere,
    vf_infinite_cylinder_to_parallel_plane,
    vf_sphere_to_infinite_plane,
)


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------------
# sphere <-> plate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 0.8, 0.95])
def test_element_to_sphere_matches_surface_quadrature(ratio: float) -> None:
    """F = (R/d)^2 vs the 3-D surface integral (independent quadrature)."""
    R = 1.0
    d = R / ratio
    F_cf = vf_element_to_sphere(R, d)
    F_num = numerical_element_to_sphere(R, d)
    assert rel_err(F_cf, F_num) < 1e-6
    assert rel_err(F_cf, ratio**2) < 1e-12


def test_element_to_sphere_limits() -> None:
    """Element far away -> R^2/d^2 -> 0; element on the surface -> 1."""
    assert vf_element_to_sphere(1.0, 1.0e6) < 1e-11
    assert rel_err(vf_element_to_sphere(1.0, 1.0 + 1e-9), 1.0) < 1e-8
    with pytest.raises(ValueError):
        vf_element_to_sphere(1.0, 1.0)  # element inside/on the sphere
    with pytest.raises(ValueError):
        vf_element_to_sphere(0.0, 1.0)


@pytest.mark.parametrize("height_ratio", [1.0, 1.5, 4.0])
def test_sphere_to_infinite_plane_is_one_half(height_ratio: float) -> None:
    """Convex-body theorem: 1/2 for every geometry (Monte-Carlo certified)."""
    R = 1.0
    F_cf = vf_sphere_to_infinite_plane(R, height_ratio * R)
    assert F_cf == 0.5
    mc = monte_carlo_sphere_to_infinite_plane(R, height_ratio * R, n_rays=200_000, seed=11)
    assert abs(mc["F"] - 0.5) < 4.0 * mc["stderr"] + 2e-3
    with pytest.raises(ValueError):
        vf_sphere_to_infinite_plane(1.0, 0.5)  # sphere intersects the plane


# ---------------------------------------------------------------------------
# cylinder <-> plate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.1, 0.25, 0.5, 0.8, 0.95])
def test_element_to_infinite_cylinder_closed_form(beta: float) -> None:
    """F = R/h vs two independent estimators (quadrature + ray casting)."""
    R = 1.0
    h = R / beta
    F_cf = vf_element_to_infinite_cylinder(R, h)
    assert rel_err(F_cf, beta) < 1e-12
    F_num = numerical_element_to_infinite_cylinder(R, h, n_phi=200, n_psi=200)
    assert rel_err(F_cf, F_num) < 1e-6
    mc = monte_carlo_element_to_infinite_cylinder(R, h, n_rays=300_000, seed=5)
    assert abs(mc["F"] - F_cf) < 4.0 * mc["stderr"] + 2e-3


def test_element_to_infinite_cylinder_guards() -> None:
    with pytest.raises(ValueError):
        vf_element_to_infinite_cylinder(1.0, 1.0)  # element on the surface
    with pytest.raises(ValueError):
        vf_element_to_infinite_cylinder(-1.0, 2.0)


def test_infinite_cylinder_to_parallel_plane_is_one_half() -> None:
    assert vf_infinite_cylinder_to_parallel_plane(0.5, 0.5) == 0.5
    assert vf_infinite_cylinder_to_parallel_plane(2.0, 10.0) == 0.5
    with pytest.raises(ValueError):
        vf_infinite_cylinder_to_parallel_plane(1.0, 0.9)


# ---------------------------------------------------------------------------
# registration / catalogue
# ---------------------------------------------------------------------------


def test_family_registration_and_catalogue_accuracy() -> None:
    """The new classes are in the family table and meet the 1 % gate."""
    for key in ("elem_sphere", "sphere_plane", "elem_infcyl", "cyl_plane"):
        assert key in CLOSED_FORM_FAMILY
        assert len(CLOSED_FORM_FAMILY[key]["parameters"]) >= 2
    rows = closed_form_table()
    assert len(rows) == len(REFERENCE_VALUES)
    for row in rows:
        assert row["rel_error"] <= 0.01, row


def test_new_classes_have_catalogue_rows() -> None:
    classes = {ref["class"] for ref in REFERENCE_VALUES}
    assert {"elem_sphere", "sphere_plane", "elem_infcyl", "cyl_plane"} <= classes


def test_element_to_infinite_cylinder_projection_identity() -> None:
    """The ellipse-area proof, checked numerically and independently.

    Cosine-weighted directions map uniformly onto the unit disk and the rays
    that strike the cylinder fill the ellipse ``x^2 + beta^2 y^2 <= beta^2``;
    its area ``pi*beta`` therefore *is* the view factor.  Here the ellipse
    area is re-measured by Monte Carlo (no reuse of the algebra) and compared
    with both the analytic area and the closed-form view factor.
    """
    R, h = 0.3, 1.7
    beta = R / h
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1.0, 1.0, size=(400_000, 2))
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0) & (
        pts[:, 0] ** 2 <= beta**2 * (1.0 - pts[:, 1] ** 2)
    )
    area = 4.0 * inside.mean()
    assert abs(area - math.pi * beta) < 0.02
    assert abs(area / math.pi - vf_element_to_infinite_cylinder(R, h)) < 0.01

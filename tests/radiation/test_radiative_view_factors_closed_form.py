"""Verification of the closed-form view-factor family.

Criterion (a) of P92W3-N02: every one of the six closed-form families must
agree with an independent reference to better than 1 %.

Two independent references are used:

* multi-dimensional Gauss-Legendre quadrature of the underlying double-area
  integral (``numerical_*`` helpers) where the integrand is smooth, and
* the definitional Monte-Carlo ray-tracing estimator
  (:func:`monte_carlo_view_factor_rectangles`) for the geometry whose
  double-area integrand is only slowly convergent (surfaces in contact).

Catalogue values (Howell / Siegel / Menguc, Modest) are asserted separately.
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.radiative_view_factors import (
    CLOSED_FORM_FAMILY,
    closed_form_table,
    monte_carlo_view_factor_rectangles,
    numerical_differential_element_to_disk,
    numerical_parallel_coaxial_disks,
    numerical_parallel_equal_rectangles,
    numerical_perpendicular_rectangles,
    numerical_sky_dome_to_tilted_plane,
    vf_concentric_spheres,
    vf_concentric_spheres_inner_to_outer,
    vf_differential_element_to_disk,
    vf_parallel_coaxial_disks,
    vf_parallel_equal_rectangles,
    vf_perpendicular_rectangles_common_edge,
    vf_sky_dome_to_ground,
    vf_sky_dome_to_tilted_plane,
)

TOL = 0.01  # 1 % machine criterion


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------------
# family 1: parallel equal rectangles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("X,Y", [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 3.0), (5.0, 5.0)])
def test_parallel_rectangles_vs_quadrature(X: float, Y: float) -> None:
    F_cf = vf_parallel_equal_rectangles(X, Y)
    F_q = numerical_parallel_equal_rectangles(X, Y, n=200)
    assert rel_err(F_cf, F_q) < 1e-9, (F_cf, F_q)
    assert rel_err(F_cf, F_q) < TOL
    assert 0.0 < F_cf < 1.0


def test_parallel_rectangles_catalogue_value() -> None:
    """Howell Table 5.1: X = Y = 1 -> 0.19982."""
    assert abs(vf_parallel_equal_rectangles(1.0, 1.0) - 0.19982) < 2.5e-5


def test_parallel_rectangles_symmetry() -> None:
    for X, Y in [(0.5, 3.0), (1.7, 0.9), (4.0, 2.0)]:
        assert abs(vf_parallel_equal_rectangles(X, Y) - vf_parallel_equal_rectangles(Y, X)) < 1e-15


def test_parallel_rectangles_limits() -> None:
    """F -> 0 as the separation grows (X, Y -> 0) and -> 1 as it vanishes."""
    assert vf_parallel_equal_rectangles(0.01, 0.01) < 1e-3
    assert vf_parallel_equal_rectangles(50.0, 50.0) > 0.95


# ---------------------------------------------------------------------------
# family 2: perpendicular rectangles sharing an edge
# ---------------------------------------------------------------------------


def test_perpendicular_rectangles_catalogue_value() -> None:
    """Howell Table 5.1: X = Y = 1 -> 0.20004."""
    assert abs(vf_perpendicular_rectangles_common_edge(1.0, 1.0) - 0.20004) < 5e-5


def test_perpendicular_rectangles_2d_limit() -> None:
    """Infinitely long shared edge -> 2-D crossed-plate limit (a=c)."""
    F = vf_perpendicular_rectangles_common_edge(1e-6, 1e-6)
    assert abs(F - (2.0 - math.sqrt(2.0)) / 2.0) < 1e-6


@pytest.mark.parametrize("H,W", [(1.0, 1.0), (2.0, 1.0), (1.0, 0.2)])
def test_perpendicular_rectangles_vs_monte_carlo(H: float, W: float) -> None:
    """Contact-edge geometry: certify against definitional ray tracing."""
    F_cf = vf_perpendicular_rectangles_common_edge(H, W)
    # shared edge = unit y; A1: x in [0, H]; A2: z in [0, W]
    s1 = ((0.0, 0.0, 0.0), (H, 0.0, 0.0), (0.0, 1.0, 0.0))
    s2 = ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, W))  # normal +x
    mc = monte_carlo_view_factor_rectangles(s1, s2, n_rays=2_000_000, seed=7)
    sigma = math.sqrt(mc["stderr"] ** 2 + 1e-12)
    assert abs(mc["F"] - F_cf) < 5.0 * sigma + 1e-4, (mc, F_cf)
    assert rel_err(F_cf, mc["F"]) < TOL


def test_perpendicular_rectangles_quadrature_convergence() -> None:
    """The contact-edge quadrature converges to the closed form, slowly.

    Documents the trap: naive G-L quadrature of this family is *not* a valid
    1 % reference at moderate n (0.2205 at n=40 is 10 % high).
    """
    exact = vf_perpendicular_rectangles_common_edge(1.0, 1.0)
    f20 = numerical_perpendicular_rectangles(1.0, 1.0, n=20)
    f60 = numerical_perpendicular_rectangles(1.0, 1.0, n=60)
    assert f20 > f60 > exact  # monotone from above
    assert rel_err(f60, exact) < rel_err(f20, exact)


# ---------------------------------------------------------------------------
# family 3: coaxial parallel disks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "R1,R2,S",
    [(1.0, 1.0, 1.0), (1.0, 1.0, 0.5), (0.5, 1.0, 1.0), (2.0, 1.0, 3.0)],
)
def test_coaxial_disks_vs_quadrature(R1: float, R2: float, S: float) -> None:
    F_cf = vf_parallel_coaxial_disks(R1, R2, S)
    F_q = numerical_parallel_coaxial_disks(R1, R2, S, n=40)
    assert rel_err(F_cf, F_q) < 1e-6, (F_cf, F_q)
    assert rel_err(F_cf, F_q) < TOL


def test_coaxial_disks_catalogue_value() -> None:
    """R1 = R2 = S -> (3 - sqrt 5)/2 = 0.381966."""
    assert abs(vf_parallel_coaxial_disks(1.0, 1.0, 1.0) - 0.5 * (3.0 - math.sqrt(5.0))) < 1e-15


def test_coaxial_disks_reciprocity() -> None:
    """A1 F12 = A2 F21 for the disk pair."""
    R1, R2, S = 0.7, 1.3, 2.0
    F12 = vf_parallel_coaxial_disks(R1, R2, S)
    F21 = vf_parallel_coaxial_disks(R2, R1, S)
    assert abs(R1**2 * F12 - R2**2 * F21) / (R1**2 * F12) < 1e-13


# ---------------------------------------------------------------------------
# family 4: concentric spheres
# ---------------------------------------------------------------------------


def test_concentric_spheres_exact_and_mc() -> None:
    """F(outer->inner) = (r1/r2)^2, verified by ray tracing from the outer shell."""
    import numpy as np

    r1, r2 = 1.0, 4.0
    F = vf_concentric_spheres(r1, r2)
    assert abs(F - 0.0625) < 1e-15  # (1/4)^2
    assert vf_concentric_spheres_inner_to_outer(r1, r2) == 1.0

    # rays start on the inner surface of the outer shell, cosine-weighted
    # inwards; the fraction that strikes the inner sphere is (r1/r2)^2.
    rng = np.random.default_rng(3)
    n = 1_000_000
    # uniform point on outer sphere, direction toward centre with cosine weight
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    p = r2 * u
    inward = -u  # normal of the shell pointing into the cavity
    ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(ref, inward)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(inward, e1)
    a = rng.random(n)
    b = rng.random(n)
    ct = np.sqrt(1.0 - a)
    st = np.sqrt(a)
    phi = 2.0 * np.pi * b
    d = (
        (st * np.cos(phi))[:, None] * e1
        + (st * np.sin(phi))[:, None] * e2
        + ct[:, None] * inward
    )
    # ray p + t d hits sphere of radius r1 if discriminant > 0 and t > 0
    pd = np.sum(p * d, axis=1)
    pp = np.sum(p * p, axis=1)
    disc = pd**2 - (pp - r1**2)
    t = -pd - np.sqrt(np.maximum(disc, 0.0))
    hits = (disc > 0.0) & (t > 0.0)
    F_mc = hits.mean()
    assert abs(F_mc - F) < 2e-3, (F_mc, F)
    assert rel_err(F, F_mc) < TOL


def test_concentric_spheres_fail_closed() -> None:
    with pytest.raises(ValueError):
        vf_concentric_spheres(2.0, 1.0)


# ---------------------------------------------------------------------------
# family 5: differential element -> coaxial disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("R,H", [(1.0, 1.0), (1.0, 0.25), (2.0, 5.0), (0.1, 1.0)])
def test_differential_element_to_disk_vs_quadrature(R: float, H: float) -> None:
    F_cf = vf_differential_element_to_disk(R, H)
    F_q = numerical_differential_element_to_disk(R, H, n=400)
    assert rel_err(F_cf, F_q) < 1e-9, (F_cf, F_q)
    assert rel_err(F_cf, F_q) < TOL


def test_differential_element_limits() -> None:
    assert vf_differential_element_to_disk(1.0, 1e6) < 1e-11  # far disk
    assert abs(vf_differential_element_to_disk(1e6, 1.0) - 1.0) < 1e-9  # infinite disk


# ---------------------------------------------------------------------------
# family 6: sky dome -> tilted plane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 180.0])
def test_sky_dome_vs_quadrature(beta: float) -> None:
    F_cf = vf_sky_dome_to_tilted_plane(beta)
    F_q = numerical_sky_dome_to_tilted_plane(beta)
    # the horizon (d_z = 0) makes the integrand discontinuous, so the
    # quadrature floor is ~1e-6 rather than machine precision
    assert rel_err(F_cf, F_q) < 1e-4, (F_cf, F_q)
    assert rel_err(F_cf, F_q) < TOL


def test_sky_dome_summation_rule() -> None:
    for beta in (0.0, 20.0, 45.0, 75.0, 90.0, 135.0):
        assert abs(vf_sky_dome_to_tilted_plane(beta) + vf_sky_dome_to_ground(beta) - 1.0) < 1e-15


def test_sky_dome_reference_points() -> None:
    assert abs(vf_sky_dome_to_tilted_plane(0.0) - 1.0) < 1e-15
    assert abs(vf_sky_dome_to_tilted_plane(60.0) - 0.75) < 1e-15
    assert abs(vf_sky_dome_to_tilted_plane(90.0) - 0.5) < 1e-15


# ---------------------------------------------------------------------------
# family-level audits
# ---------------------------------------------------------------------------


def test_family_has_six_classes() -> None:
    assert len(CLOSED_FORM_FAMILY) == 6
    for key, entry in CLOSED_FORM_FAMILY.items():
        assert callable(entry["func"]), key
        assert entry["description"]


def test_catalogue_table_within_tolerance() -> None:
    rows = closed_form_table()
    assert len(rows) == 7
    for row in rows:
        assert row["rel_error"] < 1e-3, row  # catalogue values are 5-digit
        assert row["rel_error"] < TOL, row


@pytest.mark.parametrize("bad", [(0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0)])
def test_fail_closed_degenerate_geometry(bad: tuple) -> None:
    with pytest.raises(ValueError):
        vf_parallel_equal_rectangles(*bad)
    with pytest.raises(ValueError):
        vf_perpendicular_rectangles_common_edge(*bad)
    with pytest.raises(ValueError):
        vf_differential_element_to_disk(*bad)
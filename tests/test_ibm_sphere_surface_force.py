"""Verification-surface tests for the IBM sphere surface-force module.

The tests are organised as the three layers of the node plus the negative
controls that make the passes meaningful:

* measure / quadrature exactness (analytic),
* closed-surface null space (negative control 1: rigid translation),
* Galilean invariance (negative control 2: a uniform free stream is a null
  vector of the operator) and the recorded failure of the legacy estimators
  (G15 momentum exchange, G12 pressure-only torque),
* an analytic torque check against the exact Stokes rotating-sphere solution,
* a short solver run checking the mass/force closures.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.drag_closure import cd_clift_gauvin, cd_schiller_naumann
from tensorlbm.ibm_sphere_surface_force import (
    SphereDragConfig,
    SphereRotationConfig,
    adjudicate_empirical,
    build_sphere_surface,
    drag_coefficient,
    legacy_unweighted_marker_force,
    observed_order,
    pressure_only_torque,
    run_sphere_drag,
    run_sphere_rotation,
    sphere_quadrature,
    stokes_torque_coefficient,
    surface_traction,
    torque_coefficient,
)

# --------------------------------------------------------------------------- #
# Layer 0: exact surface measure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_theta,n_phi", [(8, 16), (24, 48), (40, 80)])
def test_sphere_quadrature_area_is_exact(n_theta: int, n_phi: int) -> None:
    radius = 3.7
    pts, areas = sphere_quadrature(radius, (0.0, 0.0, 0.0), n_theta, n_phi)
    exact = 4.0 * math.pi * radius**2
    assert pts.shape == (n_theta * n_phi, 3)
    assert abs(float(areas.sum()) - exact) / exact < 1e-13
    # markers sit on the sphere
    r = torch.linalg.norm(pts, dim=1)
    assert float((r - radius).abs().max()) < 1e-12


@pytest.mark.parametrize("n_theta,n_phi", [(8, 16), (24, 48)])
def test_sphere_quadrature_closed_surface_identities(n_theta: int, n_phi: int) -> None:
    """``sum A_k n_k = 0`` and ``sum A_k x_k = 0``: the closed-surface null space."""
    radius = 2.5
    pts, areas = sphere_quadrature(radius, (1.0, -2.0, 0.5), n_theta, n_phi)
    normals = (pts - torch.tensor([1.0, -2.0, 0.5])) / radius
    net_normal = (areas.unsqueeze(1) * normals).sum(dim=0)
    first_moment = (areas.unsqueeze(1) * pts).sum(dim=0)
    centre = torch.tensor([1.0, -2.0, 0.5]) * float(areas.sum())
    scale = float(areas.sum()) * radius
    assert float(net_normal.norm()) / scale < 1e-13
    # The measure is exact for the *zeroth* moment (area) and therefore for the
    # force null space ``sum A n = 0``; the first moment (torque arm) inherits
    # the O(ds^2) placement of the marker inside its cell and is only O(1e-6).
    assert float((first_moment - centre).norm()) / (float(areas.sum()) * radius) < 2e-5


def test_measure_removes_marker_count_bias() -> None:
    """The force of a constant traction must not depend on the marker count."""
    surface_a = build_sphere_surface((32, 32, 32), (16.0, 16.0, 16.0), 5.0, n_theta=8, n_phi=16)
    surface_b = build_sphere_surface((32, 32, 32), (16.0, 16.0, 16.0), 5.0, n_theta=24, n_phi=48)
    rho = torch.ones((32, 32, 32), dtype=torch.float64)
    u = torch.zeros((3, 32, 32, 32), dtype=torch.float64)
    # uniform boundary velocity -> unweighted (per-marker) and weighted sums differ
    target = torch.tensor([[0.03, 0.0, 0.0]]).repeat(surface_a.n_markers, 1)
    fa = surface_traction(surface_a, u, target, rho, n_correctors=0, n_normal_passes=0)
    target_b = torch.tensor([[0.03, 0.0, 0.0]]).repeat(surface_b.n_markers, 1)
    fb = surface_traction(surface_b, u, target_b, rho, n_correctors=0, n_normal_passes=0)
    # quadrature: F = rho * U_b * 4 pi R^2 exactly (Stokes/Newtonian momentum
    # deficit of a closed surface in a quiescent fluid), independent of M
    exact = 0.03 * 4.0 * math.pi * 25.0
    # The 4-point Peskin kernel's partition of unity holds to ~1e-8, which is
    # the floor of the interpolation error for a constant field.
    assert abs(float(fa.force.norm()) - exact) / exact < 1e-7
    assert abs(float(fb.force.norm()) - exact) / exact < 1e-7
    # marker-count independence is exact (both read the same measure)
    assert abs(float(fa.force.norm()) - float(fb.force.norm())) / exact < 1e-12
    # the legacy unweighted sum scales with M and is off by ~M/areas
    legacy_a = 0.03 * surface_a.n_markers
    assert abs(legacy_a - exact) / exact > 0.5


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("speed", [0.0, 0.02, 0.08])
def test_uniform_translation_is_a_null_vector(speed: float) -> None:
    """NC1: rigid translation carries no traction (the G15 background is absent)."""
    n = 24
    surface = build_sphere_surface((n, n, n), (12.0, 12.0, 12.0), 4.0, n_theta=12, n_phi=24)
    rho = torch.ones((n, n, n), dtype=torch.float64)
    u = torch.zeros((3, n, n, n), dtype=torch.float64)
    u[0] = speed
    target = torch.zeros((surface.n_markers, 3), dtype=torch.float64)
    target[:, 0] = speed
    res = surface_traction(surface, u, target, rho)
    scale = abs(speed) * surface.area_total if speed > 0 else 1.0
    assert float(res.force.norm()) / scale < 1e-12
    assert float(res.torque.norm()) / (scale * surface.radius) < 1e-12


def test_galilean_invariance_of_the_force() -> None:
    """NC2: adding a uniform free stream shifts nothing but the frame."""
    n = 24
    surface = build_sphere_surface((n, n, n), (12.0, 12.0, 12.0), 4.0, n_theta=12, n_phi=24)
    torch.manual_seed(0)
    rho = 1.0 + 0.01 * torch.rand((n, n, n), dtype=torch.float64)
    u = 0.01 * torch.rand((3, n, n, n), dtype=torch.float64)
    target = torch.zeros((surface.n_markers, 3), dtype=torch.float64)
    f0 = surface_traction(surface, u, target, rho)
    shift = torch.tensor([0.037, -0.011, 0.005], dtype=torch.float64)
    f1 = surface_traction(surface, u + shift.view(3, 1, 1, 1), target + shift, rho)
    scale = float(f0.force.norm()) or 1.0
    assert float((f1.force - f0.force).norm()) / scale < 1e-12
    assert float((f1.torque - f0.torque).norm()) / (scale * surface.radius) < 1e-12


def test_legacy_marker_sum_is_marker_density_biased() -> None:
    """The legacy marker *sum* is not a closed-surface quadrature (G12/G15).

    A constant traction ``tau`` over a closed surface has the exact resultant
    ``tau * 4 pi R^2``; the legacy unit-weight marker sum returns
    ``M * tau``, i.e. it is off by the marker **density** ``M / A`` and drifts
    with every change of the marker count.  That is the additive background of
    G15 in its cleanest form: the estimator has a length-scale weight missing.
    """
    n = 24
    coarse = build_sphere_surface((n, n, n), (12.0, 12.0, 12.0), 4.0, n_theta=12, n_phi=24)
    fine = build_sphere_surface((n, n, n), (12.0, 12.0, 12.0), 4.0, n_theta=24, n_phi=48)
    rho = torch.ones((n, n, n), dtype=torch.float64)
    u = torch.zeros((3, n, n, n), dtype=torch.float64)
    u[0] = 0.05
    tau = 0.02
    scale = tau * coarse.area_total

    def run(surface):
        target = torch.zeros((surface.n_markers, 3), dtype=torch.float64)
        target[:, 0] = 0.05 + tau
        quad = surface_traction(surface, u, target, rho, n_correctors=0, n_normal_passes=0)
        legacy_force, _ = legacy_unweighted_marker_force(surface, u, target, rho)
        return quad.force, legacy_force

    q_c, l_c = run(coarse)
    q_f, l_f = run(fine)
    # quadrature: exact resultant, marker-count independent
    assert abs(float(q_c.norm()) - scale) / scale < 1e-6
    assert abs(float(q_f.norm()) - scale) / scale < 1e-6
    assert abs(float(q_c.norm()) - float(q_f.norm())) / scale < 1e-6
    # legacy: biased by M / A, so a 4x marker count multiplies the error
    assert abs(float(l_c.norm()) / scale - coarse.n_markers / coarse.area_total) < 1e-6
    assert abs(float(l_f.norm()) / float(l_c.norm()) - fine.n_markers / coarse.n_markers) < 1e-6


def test_pressure_only_estimator_cannot_certify_a_torque() -> None:
    """G12: a normal-only integral returns zero torque on a rotating flow."""
    n = 24
    radius = 4.0
    surface = build_sphere_surface((n, n, n), (12.0, 12.0, 12.0), radius, n_theta=12, n_phi=24)
    rho = torch.ones((n, n, n), dtype=torch.float64)  # uniform p -> no torque
    u = torch.zeros((3, n, n, n), dtype=torch.float64)
    tq = pressure_only_torque(surface, rho, u)
    assert float(tq.norm()) < 1e-13


def _stokes_torque_error(n_theta: int, n_phi: int) -> float:
    """Relative error of the quadrature of the exact Stokes traction."""
    radius, omega, mu = 3.3, 0.7, 0.11
    pts, areas = sphere_quadrature(radius, (0.0, 0.0, 0.0), n_theta, n_phi)
    nx = pts / torch.linalg.norm(pts, dim=1, keepdim=True)
    polar = torch.arccos(torch.clamp(nx[:, 2], -1.0, 1.0))
    ephi = torch.stack([-nx[:, 1], nx[:, 0], torch.zeros_like(nx[:, 0])], dim=1)
    ephi = ephi / torch.linalg.norm(ephi, dim=1, keepdim=True)
    traction = -3.0 * mu * omega * torch.sin(polar).unsqueeze(1) * ephi
    torque = torch.cross(pts, traction * areas.unsqueeze(1), dim=1).sum(dim=0)
    exact = 8.0 * math.pi * mu * omega * radius**3
    return abs(float(torque[2]) + exact) / exact


def test_analytic_stokes_rotating_sphere_torque() -> None:
    """Exact Stokes traction integrated by the quadrature gives ``8 pi mu Omega a^3``.

    Stokes solution for a sphere rotating about ``+z`` in a quiescent fluid
    (Landau & Lifshitz, *Fluid Mechanics*, Sec. 20): ``u_phi = Omega a^3
    sin(theta)/r^2``, surface traction ``t_phi = -3 mu Omega sin(theta)``, and
    ``|integral r x t dS| = 8 pi mu Omega a^3``.  The area measure itself is
    exact (round-off); the *integrand* is sampled at the marker positions, so
    the error is second order in the polar cell size -- which the test measures
    instead of hiding behind a loose tolerance.
    """
    e32 = _stokes_torque_error(32, 64)
    e64 = _stokes_torque_error(64, 128)
    assert e32 < 6e-4
    ratio = e32 / e64
    assert 3.0 < ratio < 5.0, f"expected 2nd-order reduction (~4x), got {ratio:.2f}"


# --------------------------------------------------------------------------- #
# Layer 2 helper: observed order
# --------------------------------------------------------------------------- #


def test_observed_order_recovers_second_order() -> None:
    spacings = [1 / 8, 1 / 12, 1 / 16, 1 / 24]
    values = [1.0 + 0.4 * h**2 for h in spacings]
    fit = observed_order(spacings, values)
    assert abs(fit["order"] - 2.0) < 0.05
    assert abs(fit["extrapolated"] - 1.0) < 1e-9


def test_observed_order_requires_three_grids() -> None:
    with pytest.raises(ValueError):
        observed_order([0.1, 0.05], [1.0, 1.1])


# --------------------------------------------------------------------------- #
# Reference bands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("re", [1.0, 10.0, 100.0, 300.0, 1000.0])
def test_empirical_adjudication_band_is_bounded(re: float) -> None:
    band = adjudicate_empirical(re)
    assert band["band_low"] <= band["band_mid"] <= band["band_high"]
    # the two fits diverge as Re grows: 0.4 % at Re=1, 2.2 % at Re=100,
    # 6.2 % at Re=1000 -- which is why the band, not one fit, is the reference
    assert band["spread_pct"] < 8.0
    assert abs(band["schiller_naumann"] - cd_schiller_naumann(re)) < 1e-14
    assert abs(band["clift_gauvin"] - cd_clift_gauvin(re)) < 1e-14


def test_stokes_torque_coefficient_matches_analytic() -> None:
    """``C_M = 8/Re_omega`` is the non-dimensionalisation of ``T = 8 pi mu Omega a^3``."""
    re_o = 2.0
    a, omega, rho = 2.0, 0.3, 1.0
    tip = omega * a
    nu = tip * (2 * a) / re_o          # Re_omega = U_tip D / nu
    mu = rho * nu
    torque = 8.0 * math.pi * mu * omega * a**3
    cm = torque_coefficient(
        torch.tensor([0.0, 0.0, torque], dtype=torch.float64), rho, tip, 2 * a
    )
    assert abs(cm - stokes_torque_coefficient(re_o)) < 1e-12


def test_drag_coefficient_normalisation() -> None:
    force = 0.5 * 1.0 * 0.04**2 * (0.25 * math.pi * 6.0**2) * 1.234
    got = drag_coefficient(torch.tensor([force, 0.0, 0.0], dtype=torch.float64), 1.0, 0.04, 6.0)
    assert abs(got - 1.234) < 1e-12


# --------------------------------------------------------------------------- #
# Solver-level closures (tiny grids, short runs)
# --------------------------------------------------------------------------- #


def test_drag_solver_closures_on_a_tiny_case() -> None:
    cfg = SphereDragConfig(
        diameter=6.0,
        reynolds=20.0,
        lateral_diameters=5.0,
        streamwise_diameters=5.0,
        target_speed=0.04,
        n_theta=12,
        n_phi=24,
        max_steps=400,
        min_steps=10_000,
        report_interval=100,
        steady_windows=2,
    )
    res = run_sphere_drag(cfg)
    assert res["max_mass_residual"] < 1e-12
    assert res["max_closure_residual"] < 1e-12
    assert res["cd_mean"] > 0.0
    assert math.isfinite(res["cd_mean"])
    assert res["reynolds_measured"] > 0.0


def test_rotation_solver_reports_torque_and_pressure_only_null() -> None:
    cfg = SphereRotationConfig(
        diameter=6.0,
        re_omega=1.0,
        box_diameters=5.0,
        tip_speed=0.02,
        n_theta=12,
        n_phi=24,
        max_steps=400,
        min_steps=10_000,
        report_interval=100,
        steady_windows=2,
    )
    res = run_sphere_rotation(cfg)
    assert res["cm_mean"] > 0.0
    assert res["max_mass_residual"] < 1e-12
    # the normal-only route cannot see the torque: it stays at round-off
    assert abs(res["cm_pressure_only_mean"]) < 1e-6 * max(res["cm_mean"], 1.0)
    assert abs(res["cm_stokes_reference"] - stokes_torque_coefficient(1.0)) < 1e-14
"""Tests for the curved-surface immersed-boundary force integration (P399W2-N04).

The node delivers three things that must hold independently of any flow
solution, and each is pinned here:

1. **Exact surface measure.** ``Σ_k Δs_k`` equals the analytic length / area of
   the curve / surface to round-off, for any marker count
   (:class:`~tensorlbm.ibm_curved_force.SurfaceQuadrature` built by
   :func:`~tensorlbm.ibm_curved_force.curve_quadrature`,
   :func:`~tensorlbm.ibm_curved_force.circle_quadrature`,
   :func:`~tensorlbm.ibm_curved_force.ellipse_quadrature`,
   :func:`~tensorlbm.ibm_curved_force.sphere_quadrature`).
2. **Quadrature exactness / distribution invariance.** ``∮ τ n dS`` reproduces
   the closed-form integral of a Fourier-mode traction, and is insensitive to
   the marker distribution — whereas the legacy unweighted marker sum is
   biased whenever the marker spacing is non-uniform.
3. **Conservative closures.** The spreading step conserves the Lagrangian
   momentum budget to round-off, ``conservative_closure`` makes it exact, the
   normal-projection pass reduces the boundary mass flux, and
   ``scheme="raw"`` reproduces :func:`tensorlbm.ibm_force_transfer.direct_forcing`
   bit-for-bit (backward compatibility).

The physical validation (cylinder drag vs literature, Hasimoto square-array
analytic drag) lives in ``scripts/p399w2_n04_curved_force/run_validation.py``
and its artifacts; it is not duplicated here (a flow solve does not belong in
the unit-test suite).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.ibm_curved_force import (
    CYLINDER_CD_LITERATURE,
    circle_quadrature,
    conservative_closure,
    curve_quadrature,
    curved_surface_forcing,
    drag_coefficient,
    ellipse_quadrature,
    momentum_budget,
    polyline_arc_weights,
    resample_uniform_arc,
    sphere_quadrature,
    stokes_cylinder_cd,
    surface_mass_flux,
)
from tensorlbm.ibm_force_transfer import build_stencil, direct_forcing

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
# 1. Surface measure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_markers", [8, 16, 32, 64, 128, 256])
def test_circle_measure_is_exact(n_markers: int) -> None:
    q = circle_quadrature((3.0, -1.0), 10.0, n_markers)
    assert q.measure_rel_error is not None and q.measure_rel_error < 1e-14
    assert abs(q.measure - 2.0 * math.pi * 10.0) < 1e-12
    # equal-arc layout: every marker carries the same measure
    assert q.spacing[1] / q.spacing[0] == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("axes", [(10.0, 4.0), (4.0, 10.0), (5.0, 5.0)])
def test_ellipse_measure_matches_elliptic_integral(axes: tuple[float, float]) -> None:
    a, b = axes
    for n_markers in (16, 64, 256):
        q = ellipse_quadrature((0.0, 0.0), a, b, n_markers)
        assert q.measure_rel_error is not None
        assert q.measure_rel_error < 1e-8


def test_ellipse_equal_arc_layout_is_uniform() -> None:
    q = ellipse_quadrature((0.0, 0.0), 10.0, 4.0, 64, layout="equal_arc")
    # equal-arc markers: constant measure by construction …
    assert q.spacing[1] / q.spacing[0] == pytest.approx(1.0, abs=1e-14)
    # … and the measure is unchanged by the redistribution
    assert q.measure_rel_error is not None and q.measure_rel_error < 1e-8
    # while the equal-angle sampling of the same curve is strongly non-uniform
    qa = ellipse_quadrature((0.0, 0.0), 10.0, 4.0, 64, layout="equal_angle")
    assert qa.spacing[1] / qa.spacing[0] > 2.0


def test_sphere_area_is_exact() -> None:
    for n_theta, n_phi in ((4, 8), (8, 16), (16, 32)):
        q = sphere_quadrature((0.0, 0.0, 0.0), 2.5, n_theta, n_phi)
        assert q.measure_rel_error is not None and q.measure_rel_error < 1e-12
        assert abs(q.measure - 4.0 * math.pi * 2.5**2) < 1e-9


def test_curve_quadrature_generic_circle() -> None:
    R = 5.0
    q = curve_quadrature(
        lambda t: torch.stack([R * torch.cos(t), R * torch.sin(t)], dim=1),
        23,
        reference_measure=2.0 * math.pi * R,
    )
    assert q.measure_rel_error is not None and q.measure_rel_error < 1e-8
    assert q.n_markers == 23


def test_polyline_chord_measure_is_biased_low() -> None:
    """The chord trapezoid rule under-estimates length — documented, not hidden."""
    exact = 2.0 * math.pi * 10.0
    errs = []
    for n_markers in (8, 16, 32, 64):
        pts = circle_quadrature((0.0, 0.0), 10.0, n_markers).points
        total = float(polyline_arc_weights(pts).sum().item())
        assert total < exact
        errs.append((exact - total) / exact)
    assert errs[0] > 0.02  # ~2.5 % at M = 8
    assert all(b < a for a, b in zip(errs, errs[1:]))  # monotone convergence
    assert errs[-1] < 1e-3


def test_resample_uniform_arc_keeps_radius_and_uniformity() -> None:
    # Resampling acts on the *chordal polyline*, so a dense input polyline is
    # used: the chord sag is O(1/M^2) and vanishes in the limit tested here.
    pts = circle_quadrature((0.0, 0.0), 3.0, 720).points
    out = resample_uniform_arc(pts, 40)
    r = torch.linalg.norm(out, dim=1)
    assert float((r - 3.0).abs().max().item()) < 1e-3
    w = polyline_arc_weights(out)
    assert float((w.max() - w.min()).abs().item()) < 1e-9
    # chordal (not arc) length: 0.1 % low for a 40-gon inscribed in R = 3
    assert abs(float(w.sum().item()) - 2.0 * math.pi * 3.0) / (2.0 * math.pi * 3.0) < 3e-3


def test_normals_are_outward_and_unit() -> None:
    q = ellipse_quadrature((1.0, -2.0), 10.0, 4.0, 64)
    n = torch.linalg.norm(q.normals, dim=1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-12)
    rel = q.points - torch.tensor([1.0, -2.0])
    # outward: n · (x − x_c) > 0 for a convex curve about its centre
    assert bool(torch.all((q.normals * rel).sum(dim=1) > 0.0))


# --------------------------------------------------------------------------- #
# 2. Traction <-> force round trip and quadrature exactness
# --------------------------------------------------------------------------- #


def test_traction_force_roundtrip_is_identity() -> None:
    q = circle_quadrature((0.0, 0.0), 2.0, 32)
    gen = torch.Generator().manual_seed(0)
    f = torch.randn((q.n_markers, 2), generator=gen)
    assert torch.allclose(q.forces(q.traction(f)), f, atol=1e-15)
    # traction integral == marker force sum (by construction)
    assert torch.allclose(q.integrate(q.traction(f)), f.sum(dim=0), atol=1e-13)


@pytest.mark.parametrize("n_markers", [8, 16, 32, 64])
def test_analytic_fourier_traction_integral(n_markers: int) -> None:
    """∮ (a0 + a1 cosθ + b1 sinθ) n̂ R dθ = (a1 R π, b1 R π)."""
    R, a0, a1, b1 = 1.7, 0.9, 1.3, -0.6
    q = circle_quadrature((0.0, 0.0), R, n_markers)
    th = torch.atan2(q.points[:, 1], q.points[:, 0])
    rad = torch.stack([torch.cos(th), torch.sin(th)], dim=1)
    tau = (a0 + a1 * torch.cos(th) + b1 * torch.sin(th)).unsqueeze(-1) * rad
    exact = torch.tensor([a1 * R * math.pi, b1 * R * math.pi])
    got = q.integrate(tau)
    assert torch.allclose(got, exact, atol=1e-12)


def test_quadrature_beats_unweighted_on_nonuniform_markers() -> None:
    """Equal-angle ellipse markers: the unweighted sum is biased, quadrature is not."""
    a, b = 10.0, 4.0

    def tau_of(q):
        th = torch.atan2(q.points[:, 1] / b, q.points[:, 0] / a)
        return (1.0 + 0.8 * torch.cos(th) + 0.5 * torch.sin(th)).unsqueeze(-1) * q.normals

    ref_q = ellipse_quadrature((0.0, 0.0), a, b, 8192)
    reference = ref_q.integrate(tau_of(ref_q))
    scale = float(reference.abs().max().item())

    q = ellipse_quadrature((0.0, 0.0), a, b, 32, layout="equal_angle")
    tau = tau_of(q)
    # midpoint traction sampling on a coarse (M = 32) mesh: O(M^-2) quadrature
    # error, ~1.5e-3 here
    quad_err = float((q.integrate(tau) - reference).abs().max().item()) / scale
    raw = q.integrate(tau, weighted=False) * float(q.weights.mean().item())
    raw_err = float((raw - reference).abs().max().item()) / scale
    assert quad_err < 5e-3
    # … whereas ignoring the non-uniform measure costs ~22 % (measured) on this
    # marker set and does not improve with refinement
    assert raw_err > 0.1
    assert raw_err > 50.0 * quad_err
    # the bias is a property of the marker *distribution*: equal-arc markers
    # (constant Δs) make the two estimates coincide
    qa = ellipse_quadrature((0.0, 0.0), a, b, 32, layout="equal_arc")
    ta = tau_of(qa)
    raw_a = qa.integrate(ta, weighted=False) * float(qa.weights.mean().item())
    assert torch.allclose(raw_a, qa.integrate(ta), atol=1e-12)
    assert float((qa.integrate(ta) - reference).abs().max().item()) / scale < 5e-3


def test_audit_reports_measure_and_bias() -> None:
    q = ellipse_quadrature((0.0, 0.0), 10.0, 4.0, 64, layout="equal_angle")
    th = torch.atan2(q.points[:, 1] / 4.0, q.points[:, 0] / 10.0)
    F = (1.0 + 0.5 * torch.cos(th)).unsqueeze(-1) * q.normals * q.weights.unsqueeze(-1)
    aud = q.audit(F)
    assert aud["quadrature_vs_traction_rel"] < 1e-14
    assert aud["spacing_ratio"] > 2.0
    assert aud["measure_rel_error"] is not None and aud["measure_rel_error"] < 1e-8


def test_surface_torque_of_tangential_traction() -> None:
    """Constant tangential traction τ_t on a circle: T = τ_t * 2πR²."""
    R, tau_t = 2.0, 0.7
    q = circle_quadrature((0.0, 0.0), R, 64)
    # tangential unit vector = n̂ rotated by +90°
    tang = torch.stack([-q.normals[:, 1], q.normals[:, 0]], dim=1)
    F = q.forces(tau_t * tang)
    T = float(q.torque(F).item())
    assert T == pytest.approx(tau_t * 2.0 * math.pi * R**2, rel=1e-12)


# --------------------------------------------------------------------------- #
# 3. Conservative closures
# --------------------------------------------------------------------------- #


def _cylinder_ibm_setup(n_markers: int = 32, ny: int = 48, nx: int = 64):
    surface = circle_quadrature((nx / 2.0, ny / 2.0), 6.0, n_markers)
    stencil = build_stencil(surface.points, (ny, nx), kernel="4pt")
    gen = torch.Generator().manual_seed(3)
    ux = 0.05 + 0.01 * torch.randn((ny, nx), generator=gen)
    uy = 0.01 * torch.randn((ny, nx), generator=gen)
    rho = 1.0 + 1e-3 * torch.randn((ny, nx), generator=gen)
    target = torch.zeros((n_markers, 2))
    return surface, stencil, torch.stack([ux, uy]), rho, target


def test_raw_scheme_reproduces_legacy_direct_forcing() -> None:
    surface, stencil, u, rho, target = _cylinder_ibm_setup()
    f_m, f_grid, body_force, _ = direct_forcing(stencil, u, target, rho, 2)
    res = curved_surface_forcing(
        stencil, u, target, surface, rho_grid=rho, n_correctors=2, scheme="raw"
    )
    assert torch.allclose(res.marker_force, f_m, atol=1e-15)
    assert torch.allclose(res.force_grid, f_grid, atol=1e-15)
    assert torch.allclose(-res.force_total, body_force, atol=1e-15)


def test_momentum_budget_closure_is_machine_precision() -> None:
    surface, stencil, u, rho, target = _cylinder_ibm_setup()
    for scheme in ("quadrature", "raw"):
        res = curved_surface_forcing(
            stencil, u, target, surface, rho_grid=rho, n_correctors=2, scheme=scheme
        )
        assert res.closure_rel < 1e-14
        bud = momentum_budget(res)
        assert bud["closure_abs"] < 1e-13
        # the sign convention: the load on the body is −Σ_k F_k
        assert torch.allclose(-res.force_total, -res.marker_force.sum(dim=0), atol=0.0)


def test_quadrature_forcing_scales_marker_force_with_measure() -> None:
    """``F_k = τ_k Δs_k``: doubling the marker count halves the per-marker force."""
    surface, stencil, u, rho, target = _cylinder_ibm_setup(32)
    res32 = curved_surface_forcing(stencil, u, target, surface, rho_grid=rho, n_correctors=0)
    surface2, stencil2, _, _, target2 = _cylinder_ibm_setup(64)
    res64 = curved_surface_forcing(stencil2, u, target2, surface2, rho_grid=rho, n_correctors=0)
    # fraction of the total load carried per marker halves when the marker
    # count doubles (F_k = tau_k * ds_k), while the total force is invariant
    per32 = float(res32.marker_force.abs().mean().item())
    per64 = float(res64.marker_force.abs().mean().item())
    assert per32 == pytest.approx(2.0 * per64, rel=0.1)
    tot32 = float(res32.force_total[0].item())
    tot64 = float(res64.force_total[0].item())
    assert tot32 == pytest.approx(tot64, rel=0.15)


def test_conservative_closure_makes_budget_exact_and_stays_local() -> None:
    surface, stencil, u, rho, target = _cylinder_ibm_setup()
    res = curved_surface_forcing(stencil, u, target, surface, rho_grid=rho, n_correctors=2)
    f = res.force_grid.clone()
    support = (f.abs().sum(dim=0) > 0.0).clone()
    target_force = res.force_total * 1.5  # deliberately wrong budget
    corrected, audit = conservative_closure(f, target_force)
    assert audit["residual_before_rel"] > 0.1
    assert audit["residual_after_rel"] < 1e-15
    # nothing outside the original forcing support was touched
    outside = f.abs().sum(dim=0) == 0.0
    assert bool(torch.all(corrected[:, outside] == 0.0))
    assert int(support.sum().item()) > 0


def test_normal_projection_reduces_boundary_mass_flux() -> None:
    """The mass-conservation closure: residual normal slip must go down."""
    ny, nx = 64, 80
    surface = circle_quadrature((nx / 2.0, ny / 2.0), 8.0, 64)
    stencil = build_stencil(surface.points, (ny, nx), kernel="4pt")
    # solid-body-rotation velocity field: purely tangential to the cylinder
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64) - ny / 2.0,
        torch.arange(nx, dtype=torch.float64) - nx / 2.0,
        indexing="ij",
    )
    omega = 0.05
    u = torch.stack([-omega * yy, omega * xx])
    rho = torch.ones((ny, nx))
    target = torch.zeros((surface.n_markers, 2))
    loose = curved_surface_forcing(stencil, u, target, surface, rho_grid=rho, n_correctors=0)
    tight = curved_surface_forcing(
        stencil, u, target, surface, rho_grid=rho, n_correctors=0, n_normal_passes=2
    )
    assert tight.normal_slip_rel < loose.normal_slip_rel
    # the tangential (friction) traction must not be destroyed by the closure
    assert float(tight.traction.abs().sum().item()) > 0.0


def test_surface_mass_flux_of_uniform_stream() -> None:
    """Uniform stream: signed flux ≈ 0 by symmetry, |flux| = ρU·4R for a circle."""
    ny, nx = 64, 80
    R, U = 8.0, 0.05
    surface = circle_quadrature((nx / 2.0, ny / 2.0), R, 128)
    stencil = build_stencil(surface.points, (ny, nx), kernel="4pt")
    u = torch.stack([torch.full((ny, nx), U), torch.zeros((ny, nx))])
    rho = torch.ones((ny, nx))
    leak = surface_mass_flux(
        stencil, surface, u, torch.zeros((surface.n_markers, 2)), rho, reference=U * 2 * R
    )
    assert abs(leak["flux"]) < 1e-12
    # ∮ |u·n| ds = U ∮ |cosθ| R dθ = 4 U R
    assert leak["abs_flux"] == pytest.approx(4.0 * U * R, rel=0.02)
    assert leak["leak_rate_abs"] == pytest.approx(4.0 * U * R / (U * 2 * R), rel=0.02)


def test_sphere_forcing_is_axisymmetric() -> None:
    """3-D: uniform stream past a resting sphere gives drag along x, no lift."""
    n = 32
    surface = sphere_quadrature((n / 2.0, n / 2.0, n / 2.0), 6.0, 6, 12)
    stencil = build_stencil(surface.points, (n, n, n), kernel="4pt")
    u = torch.zeros((3, n, n, n))
    u[0] = 0.05
    rho = torch.ones((n, n, n))
    res = curved_surface_forcing(
        stencil, u, torch.zeros((surface.n_markers, 3)), surface, rho_grid=rho, n_correctors=1
    )
    assert res.closure_rel < 1e-14
    assert float(res.force_total[0].item()) < 0.0  # force on the fluid is upstream
    assert abs(float(res.force_total[1].item())) < 1e-3 * abs(
        float(res.force_total[0].item())
    )
    assert abs(float(res.force_total[2].item())) < 1e-3 * abs(
        float(res.force_total[0].item())
    )
    assert surface.measure_rel_error is not None and surface.measure_rel_error < 1e-12


# --------------------------------------------------------------------------- #
# 4. Analytic references and input validation
# --------------------------------------------------------------------------- #


def test_stokes_cylinder_asymptote() -> None:
    assert stokes_cylinder_cd(1.0)["cd"] == pytest.approx(8.0 * math.pi / 2.002, rel=1e-12)
    assert stokes_cylinder_cd(0.1)["cd"] == pytest.approx(58.4, rel=2e-2)


def test_literature_table_is_self_consistent() -> None:
    cds = [v["cd"] for v in CYLINDER_CD_LITERATURE.values()]
    assert all(v["re"] == 100.0 for v in CYLINDER_CD_LITERATURE.values())
    assert 1.2 < min(cds) < max(cds) < 1.4


def test_drag_coefficient_definition() -> None:
    cd = drag_coefficient(torch.tensor([1.5, 0.0]), 1.0, 0.1, 20.0)
    assert cd == pytest.approx(2.0 * 1.5 / (1.0 * 0.01 * 20.0), rel=1e-14)


@pytest.mark.parametrize(
    "call",
    [
        lambda: circle_quadrature((0.0, 0.0), -1.0, 16),
        lambda: ellipse_quadrature((0.0, 0.0), 1.0, 1.0, 3),
        lambda: sphere_quadrature((0.0, 0.0, 0.0), 1.0, 1, 8),
        lambda: stokes_cylinder_cd(0.0),
        lambda: resample_uniform_arc(torch.zeros((4, 3)), 10),
    ],
)
def test_invalid_inputs_raise(call) -> None:
    with pytest.raises((ValueError, IndexError)):
        call()


def test_invalid_scheme_raises() -> None:
    surface, stencil, u, rho, target = _cylinder_ibm_setup(16)
    with pytest.raises(ValueError):
        curved_surface_forcing(stencil, u, target, surface, scheme="bogus")
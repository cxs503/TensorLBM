"""Unit + invariance tests for the 3D diffuse-interface MDF operator.

Covers :mod:`tensorlbm.ibm3d_mdf`:

* delta-kernel identities (parity with the historical kernels, partition of unity),
* surface-area weighting of a sphere mesh,
* Lagrangian↔Eulerian transfer: adjointness ``⟨spread g, u⟩ = ⟨g, interp u⟩``
  and exact momentum conservation ``Σ_cells spread(F) = Σ_m F_m``,
* multi-direct forcing: ``n_iter = 1`` matches an explicit single direct-forcing
  pass, the slip residual decreases with ``n_iter``, and a converged coupling
  leaves no net force when the marker velocity already equals the target,
* the physical drag coefficient formula.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import ibm3d_mdf as mdf
from tensorlbm.ibm import ibm_delta_4pt

torch.set_grad_enabled(False)


def _sphere(shape=(20, 20, 20), radius=3.0, subdivisions=2):
    center = (shape[2] / 2.0, shape[1] / 2.0, shape[0] / 2.0)
    verts, faces = mdf.sphere_surface(center, radius, n_theta=16, n_phi=10)
    return verts, faces, center


def test_peskin_4pt_matches_historical_kernel():
    r = torch.linspace(-2.5, 2.5, 501, dtype=torch.float64)
    assert torch.allclose(mdf.kernel_peskin_4pt(r), ibm_delta_4pt(r), atol=1e-15)


def test_kernel_partition_of_unity():
    """Σ_j φ(x − j) = 1 for all x (the defining property of a delta kernel)."""
    x = torch.linspace(0.0, 1.0, 31, dtype=torch.float64)
    for kernel in (mdf.kernel_peskin_4pt, mdf.kernel_cosine_4pt, mdf.kernel_peskin_3pt):
        offsets = torch.arange(-3, 5, dtype=torch.float64)
        total = kernel(x.unsqueeze(1) - offsets.unsqueeze(0)).sum(dim=1)
        tol = 1e-12 if kernel is not mdf.kernel_peskin_3pt else 1e-3
        assert torch.allclose(total, torch.ones_like(total), atol=tol)


def test_stencil_weights_sum_to_one_and_are_in_range():
    verts, faces, _ = _sphere()
    shape = (20, 20, 20)
    weights, indices = mdf.build_stencil(verts, shape, kernel="peskin4pt")
    assert weights.shape[1] == 64  # 4x4x4 tensor-product stencil
    assert torch.allclose(weights.sum(dim=1), torch.ones(verts.shape[0], dtype=weights.dtype), atol=1e-12)
    assert int(indices.min()) >= 0
    assert int(indices.max()) <= 20 * 20 * 20 - 1


def test_surface_weights_sum_to_polyhedral_area():
    radius = 4.0
    verts, faces = mdf.sphere_surface((10.0, 10.0, 10.0), radius, n_theta=32, n_phi=16)
    ds = mdf.surface_weights(verts, faces)
    assert ds.shape == (verts.shape[0],)
    assert torch.all(ds > 0)
    poly = float(mdf.surface_area(verts, faces))
    assert abs(float(ds.sum()) - poly) < 1e-10 * poly
    # an inscribed polyhedron underestimates the analytic area slightly
    analytic = 4.0 * math.pi * radius ** 2
    assert 0.9 * analytic < poly < analytic


def test_interpolate_spread_adjoint_and_momentum_conserving():
    torch.manual_seed(0)
    shape = (16, 18, 20)
    verts, faces, _ = _sphere(shape=shape)
    weights, indices = mdf.build_stencil(verts, shape)
    u = torch.randn(3, *shape, dtype=torch.float64)
    g = torch.randn(verts.shape[0], 3, dtype=torch.float64)

    interp = mdf.interpolate(u, weights, indices)          # (N, 3)
    spread = mdf.spread(g, weights, indices, shape)        # (3, nz, ny, nx)
    lhs = float((spread * u).sum())
    rhs = float((g * interp).sum())
    assert abs(lhs - rhs) < 1e-10 * max(abs(lhs), 1.0)

    # Σ_cells spread(g) = Σ_markers g_m (a partition-of-unity stencil conserves momentum)
    assert torch.allclose(spread.sum(dim=(1, 2, 3)), g.sum(dim=0), atol=1e-10)


def test_single_iteration_equals_explicit_direct_forcing():
    """``n_iter = 1`` must equal the closed-form single-pass force ``2 ds (u_t − u_m)``."""
    shape = (16, 16, 16)
    verts, _, _ = _sphere(shape=shape)
    weights, indices = mdf.build_stencil(verts, shape)
    u = torch.zeros(3, *shape, dtype=torch.float64)
    u[0] = 0.1
    ds = torch.full((verts.shape[0],), 0.25, dtype=torch.float64)
    target = torch.zeros((verts.shape[0], 3), dtype=torch.float64)

    res = mdf.multi_direct_forcing(u, weights, indices, target, ds, n_iter=1)
    slip = target - mdf.interpolate(u, weights, indices)     # u_t − u_m
    expected = mdf.spread(2.0 * ds.reshape(-1, 1) * slip, weights, indices, shape)
    assert torch.allclose(res.grid_force, expected, atol=1e-12)
    assert torch.allclose(res.marker_reaction, -2.0 * ds.reshape(-1, 1) * slip, atol=1e-12)


def test_slip_residual_decreases_with_iterations():
    shape = (16, 16, 16)
    verts, _, _ = _sphere(shape=shape)
    weights, indices = mdf.build_stencil(verts, shape)
    u = torch.zeros(3, *shape, dtype=torch.float64)
    u[0] = 0.2
    ds = torch.full((verts.shape[0],), 1.0 / verts.shape[0] * 4 * math.pi * 9.0, dtype=torch.float64)
    res = mdf.multi_direct_forcing(u, weights, indices, torch.zeros((verts.shape[0], 3), dtype=torch.float64), ds, n_iter=8)
    residuals = res.residuals
    assert residuals[0] > residuals[-1]
    # monotone (allowing round-off) and a real reduction
    assert all(b <= a + 1e-12 for a, b in zip(residuals, residuals[1:]))
    assert residuals[-1] < 0.9 * residuals[0]


def test_no_force_when_marker_velocity_matches_target():
    shape = (16, 16, 16)
    verts, _, _ = _sphere(shape=shape)
    weights, indices = mdf.build_stencil(verts, shape)
    u = torch.zeros(3, *shape, dtype=torch.float64)
    u[0] = 0.05
    u[1] = -0.02
    ds = torch.full((verts.shape[0],), 0.3, dtype=torch.float64)
    target = mdf.interpolate(u, weights, indices)
    res = mdf.multi_direct_forcing(u, weights, indices, target, ds, n_iter=5)
    assert float(res.grid_force.abs().max()) < 1e-12
    assert float(res.marker_reaction.abs().max()) < 1e-12


def test_drag_coefficient_formula():
    # F = 0.5 rho U^2 A Cd  ->  Cd = 2F/(rho U^2 A), A = pi D^2/4
    D, U, cd = 2.0, 0.1, 1.2
    area = math.pi * D * D / 4.0
    force = torch.tensor([0.5 * 1.0 * U * U * area * cd, 0.0, 0.0], dtype=torch.float64)
    got = mdf.drag_coefficient(force, 1.0, U, D)
    assert abs(got - cd) < 1e-12, f"Cd round-trip: expected {cd}, got {got}" 


def test_spread_drops_out_of_domain_markers_like_oss():
    """No wrap: a marker stencil crossing the boundary must not fold around."""
    shape = (8, 8, 8)
    coords = torch.tensor([[0.2, 4.0, 4.0]], dtype=torch.float64)
    weights, indices = mdf.build_stencil(coords, shape, kernel="peskin4pt", wrap=False)
    values = torch.ones((1, 3), dtype=torch.float64)
    field = mdf.spread(values, weights, indices, shape)
    # weight is lost (not wrapped), so the total is below the in-domain maximum
    assert float(field.sum()) < 3.0
    assert float(field.sum()) > 0.0
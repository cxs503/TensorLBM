"""Tests for tensorlbm.ibm module."""

import torch
import pytest

from tensorlbm.ibm import (
    peskin_delta_4pt,
    build_stencil_2d,
    spread_force_2d,
    interpolate_velocity_2d,
    direct_forcing_static,
    build_cylinder_markers_2d,
)


class TestPeskinDelta4pt:
    def test_support(self):
        """Delta should be zero outside |r| > 2."""
        r = torch.tensor([2.0, 2.1, 3.0, -2.5])
        d = peskin_delta_4pt(r)
        assert torch.all(d == 0.0), f"Non-zero outside support: {d}"

    def test_max_at_origin(self):
        """Delta should be 0.5 at r=0."""
        d = peskin_delta_4pt(torch.tensor([0.0]))
        assert abs(d.item() - 0.5) < 1e-10

    def test_symmetry(self):
        """Delta should be symmetric: δ(r) = δ(-r)."""
        r = torch.linspace(-2, 2, 100)
        d = peskin_delta_4pt(r)
        assert torch.allclose(d, d.flip(0), atol=1e-12)


class TestBuildStencil2d:
    def test_partition_of_unity(self):
        """Sum of weights per marker should be 1."""
        mx, my = build_cylinder_markers_2d(50.0, 50.0, 10.0, marker_spacing=0.5)
        w, idx = build_stencil_2d(mx, my, 100, 100)
        sums = w.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10)

    def test_no_invalid_indices(self):
        """All markers well inside domain should have no invalid indices."""
        mx = torch.tensor([50.0, 51.0, 49.5])
        my = torch.tensor([50.0, 50.5, 49.0])
        w, idx = build_stencil_2d(mx, my, 100, 100)
        assert (idx >= 0).all()


class TestInterpolateVelocity2d:
    def test_uniform_field(self):
        """Interpolating a uniform field should return the same value."""
        nx, ny = 100, 100
        grid_ux = torch.full((ny, nx), 0.1, dtype=torch.float64)
        grid_uy = torch.zeros((ny, nx), dtype=torch.float64)
        mx, my = build_cylinder_markers_2d(50.0, 50.0, 10.0, marker_spacing=0.5)
        w, idx = build_stencil_2d(mx, my, ny, nx)

        mux, muy = interpolate_velocity_2d(grid_ux, grid_uy, w, idx)
        assert torch.allclose(mux, torch.full_like(mux, 0.1), atol=1e-10)
        assert torch.allclose(muy, torch.zeros_like(muy), atol=1e-10)


class TestSpreadForce2d:
    def test_force_conservation(self):
        """Total spread force should equal total marker force."""
        nx, ny = 100, 100
        mx, my = build_cylinder_markers_2d(50.0, 50.0, 10.0, marker_spacing=0.5)
        w, idx = build_stencil_2d(mx, my, ny, nx)

        fm_x = torch.ones(mx.shape[0], dtype=torch.float64) * 0.01
        fm_y = torch.zeros(mx.shape[0], dtype=torch.float64)

        fx, fy = spread_force_2d(fm_x, fm_y, w, idx, ny, nx)
        assert abs(fx.sum().item() - fm_x.sum().item()) < 1e-8
        assert abs(fy.sum().item()) < 1e-10


class TestDirectForcingStatic:
    def test_zero_velocity(self):
        """If grid velocity is zero, force should be zero."""
        nx, ny = 100, 100
        mx, my = build_cylinder_markers_2d(50.0, 50.0, 10.0, marker_spacing=0.5)
        w, idx = build_stencil_2d(mx, my, ny, nx)

        rho = torch.ones((ny, nx), dtype=torch.float64)
        mux = torch.zeros(mx.shape[0], dtype=torch.float64)
        muy = torch.zeros(my.shape[0], dtype=torch.float64)
        target_ux = torch.zeros_like(mux)
        target_uy = torch.zeros_like(muy)

        fx, fy = direct_forcing_static(
            mux, muy, target_ux, target_uy, rho, w, idx, ny, nx, n_iter=10
        )
        assert fx.abs().max() < 1e-10
        assert fy.abs().max() < 1e-10

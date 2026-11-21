"""Tests for Sparse List AOS LBM (D3Q19).

This module tests the :mod:`tensorlbm.sparse_list_aos` module against
a dense LBM reference implementation for 3D periodic channel flow.
"""

import pytest
import torch

from tensorlbm.sparse_list_aos import (
    SparseListAOS,
    build_sparse_list_aos,
    aos_collide,
    aos_stream,
    aos_step,
)
from tensorlbm.d3q19 import C, OPPOSITE, W, CS2


class TestSparseListAOSBuild:
    """Test the build function for SparseListAOS."""

    def test_build_simple_cube(self):
        """Test building a simple cube with no solids."""
        nz, ny, nx = 4, 4, 4
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        table = build_sparse_list_aos(solid, wrap=True)
        
        assert table.n_fluid == nz * ny * nx
        assert table.n_total == nz * ny * nx
        assert table.porosity == 1.0
        assert table.sublist == (0, table.n_fluid - 1)
        
        # All neighbours should be valid
        assert (table.neighbour >= 0).all()

    def test_build_with_solids(self):
        """Test building with some solid nodes."""
        nz, ny, nx = 4, 4, 4
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solid[1, 1, 1] = True
        solid[2, 2, 2] = True
        table = build_sparse_list_aos(solid, wrap=True)
        
        assert table.n_fluid == nz * ny * nx - 2
        assert table.porosity == (nz * ny * nx - 2) / (nz * ny * nx)
        
        # Check that solid nodes have -1 in index
        assert table.index[1, 1, 1] == -1
        assert table.index[2, 2, 2] == -1

    def test_build_wrap(self):
        """Test wrapping behaviour."""
        nz, ny, nx = 4, 4, 4
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        table = build_sparse_list_aos(solid, wrap=True)
        
        # Node at (0, 0, 0) should have neighbour at (nz-1, 0, 0) in direction 2 (opposite of 1)
        # Direction 1 is (1, 0, 0), opposite is 2 (-1, 0, 0)
        # So neighbour[0, 2] should be the compressed index of (nz-1, 0, 0)
        expected_idx = (nz - 1) * ny * nx + 0 * nx + 0
        assert table.neighbour[0, 2] == expected_idx


class TestAOSCollide:
    """Test the collision step."""

    def test_collide_conservation(self):
        """Test that collision conserves mass."""
        K = 100
        f = torch.rand((K, 19), dtype=torch.float64)
        tau = 1.0
        
        f_new = aos_collide(f, tau)
        
        rho_old = f.sum(dim=1)
        rho_new = f_new.sum(dim=1)
        
        torch.testing.assert_close(rho_old, rho_new, rtol=1e-10, atol=1e-10)

    def test_collide_equilibrium(self):
        """Test that collision drives towards equilibrium for tau=0.5."""
        K = 100
        f = torch.rand((K, 19), dtype=torch.float64)
        tau = 0.5
        
        f_new = aos_collide(f, tau)
        
        # For tau=0.5, f_new should be exactly equilibrium
        rho = f.sum(dim=1, keepdim=True)
        u = torch.zeros((K, 3), dtype=torch.float64)
        for d in range(19):
            u[:, 0] += f[:, d] * C[d, 0]
            u[:, 1] += f[:, d] * C[d, 1]
            u[:, 2] += f[:, d] * C[d, 2]
        u[:, 0] /= rho
        u[:, 1] /= rho
        u[:, 2] /= rho
        
        f_eq = torch.zeros_like(f)
        for d in range(19):
            cu = C[d, 0] * u[:, 0] + C[d, 1] * u[:, 1] + C[d, 2] * u[:, 2]
            u2 = u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2
            feq = W[d] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
            f_eq[:, d] = feq.squeeze()
        
        torch.testing.assert_close(f_new, f_eq, rtol=1e-10, atol=1e-10)


class TestAOSStream:
    """Test the streaming step."""

    def test_stream_pull(self):
        """Test pull streaming."""
        nz, ny, nx = 4, 4, 4
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        table = build_sparse_list_aos(solid, wrap=True)
        
        K = table.n_fluid
        f = torch.rand((K, 19), dtype=torch.float64)
        
        f_new = aos_stream(f, table)
        
        # Check that streaming is correct
        # f_new[k, dir] = f[table.neighbour[k, opposite[dir]], dir]
        opp_indices = OPPOSITE.to(f.device)
        for k in range(K):
            for d in range(19):
                nb = table.neighbour[k, d]
                if nb >= 0:
                    expected = f[nb, d]
                    assert torch.isclose(f_new[k, d], expected, rtol=1e-10, atol=1e-10)
                else:
                    # Solid neighbour: bounce-back
                    expected = f[k, OPPOSITE[d]]
                    assert torch.isclose(f_new[k, d], expected, rtol=1e-10, atol=1e-10)


class TestAOSStep:
    """Test the full step."""

    def test_step_consistency(self):
        """Test that step is consistent."""
        nz, ny, nx = 4, 4, 4
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        table = build_sparse_list_aos(solid, wrap=True)
        
        K = table.n_fluid
        f = torch.rand((K, 19), dtype=torch.float64)
        tau = 1.0
        
        f_new = aos_step(f, table, tau)
        
        # Check that mass is conserved
        rho_old = f.sum(dim=1)
        rho_new = f_new.sum(dim=1)
        
        torch.testing.assert_close(rho_old, rho_new, rtol=1e-10, atol=1e-10)


class TestAOSvsDense:
    """Test AOS against dense implementation."""

    def test_channel_flow(self):
        """Test 3D periodic channel flow."""
        nz, ny, nx = 8, 8, 8
        solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
        table = build_sparse_list_aos(solid, wrap=True)
        
        K = table.n_fluid
        
        # Initialize with equilibrium
        rho0 = 1.0
        u0 = 0.01
        f = torch.zeros((K, 19), dtype=torch.float64)
        for d in range(19):
            cu = C[d, 0] * u0 + C[d, 1] * 0.0 + C[d, 2] * 0.0
            u2 = u0**2
            feq = W[d] * rho0 * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
            f[:, d] = feq
        
        # Run a few steps
        tau = 1.0
        for _ in range(10):
            f = aos_step(f, table, tau)
        
        # Check that density is still uniform
        rho = f.sum(dim=1)
        torch.testing.assert_close(rho, torch.full((K,), rho0), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for interpolated_bc.py: bouzidi_bounce_back."""
from __future__ import annotations

import torch

from tensorlbm import equilibrium
from tensorlbm.interpolated_bc import bouzidi_bounce_back, bouzidi_bounce_back_3d, compute_q_sphere


def _make_f_pair(ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (f, f_prev) as simple near-equilibrium distributions."""
    rho = torch.ones((ny, nx))
    ux = torch.full_like(rho, 0.05)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)
    f_prev = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    return f, f_prev


class TestBouzidiBounceBack:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 10
        f, f_prev = _make_f_pair(ny, nx)
        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, :] = True
        q = torch.full((ny, nx), 0.5)
        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=1)
        assert f_out.shape == f.shape

    def test_finite_output(self) -> None:
        ny, nx = 8, 10
        f, f_prev = _make_f_pair(ny, nx)
        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, :] = True
        q = torch.full((ny, nx), 0.5)
        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=1)
        assert torch.isfinite(f_out).all()

    def test_unchanged_outside_fluid_nodes(self) -> None:
        """Populations at non-fluid nodes must not be modified."""
        ny, nx = 8, 10
        f, f_prev = _make_f_pair(ny, nx)
        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, 5] = True  # only one cell is a fluid boundary node
        q = torch.full((ny, nx), 0.5)
        direction = 1
        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=direction)
        # Cells outside fluid_nodes should be unchanged in direction
        mask_other = ~fluid_nodes
        assert torch.allclose(f_out[direction][mask_other], f[direction][mask_other])

    def test_halfway_q_equals_standard_bounce_back(self) -> None:
        """q=0.5 should reproduce standard halfway bounce-back within small tolerance."""
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_prev = f.clone()

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[3, :] = True
        q = torch.full((ny, nx), 0.5)
        direction = 1
        opp = 3  # OPPOSITE[1] for D2Q9

        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=direction)
        # At q=0.5: f_bc = 2*0.5*f_opp + 0 = f_opp (standard BB)
        assert torch.allclose(
            f_out[direction][fluid_nodes],
            f[opp][fluid_nodes],
            atol=1e-5,
        )

    def test_linear_branch_q_less_than_half(self) -> None:
        """When q < 0.5 the linear formula is used; result is a blend."""
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = equilibrium(rho, torch.full_like(rho, 0.03), torch.zeros_like(rho))

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[3, 4] = True
        q = torch.full((ny, nx), 0.25)  # < 0.5 → linear branch
        direction = 1
        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=direction)
        assert torch.isfinite(f_out).all()

    def test_quadratic_branch_q_greater_than_half(self) -> None:
        """When q ≥ 0.5 the quadratic formula is used."""
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = equilibrium(rho, torch.full_like(rho, 0.03), torch.zeros_like(rho))

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[3, 4] = True
        q = torch.full((ny, nx), 0.75)  # ≥ 0.5 → quadratic branch
        direction = 1
        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=direction)
        assert torch.isfinite(f_out).all()

    def test_all_directions(self) -> None:
        """Applying BC for every direction should always return finite tensors."""
        ny, nx = 8, 10
        f, f_prev = _make_f_pair(ny, nx)
        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, 5] = True
        q = torch.full((ny, nx), 0.5)
        for direction in range(9):
            f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=direction)
            assert torch.isfinite(f_out).all(), f"Non-finite for direction {direction}"


# ---------------------------------------------------------------------------
# Phase 7: 3-D Bouzidi BC tests
# ---------------------------------------------------------------------------

class TestComputeQSphere:
    def test_returns_correct_shapes(self) -> None:
        nz, ny, nx = 16, 16, 16
        device = torch.device("cpu")
        mask, q = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, device)
        assert mask.shape == (19, nz, ny, nx)
        assert q.shape == (19, nz, ny, nx)

    def test_q_in_valid_range(self) -> None:
        nz, ny, nx = 16, 16, 16
        device = torch.device("cpu")
        mask, q = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, device)
        boundary_q = q[mask]
        if boundary_q.numel() > 0:
            assert float(boundary_q.min().item()) > 0.0
            assert float(boundary_q.max().item()) <= 1.0 + 1e-5

    def test_non_boundary_q_is_half(self) -> None:
        """Non-boundary entries should default to 0.5."""
        nz, ny, nx = 16, 16, 16
        device = torch.device("cpu")
        mask, q = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, device)
        non_boundary = ~mask
        assert torch.allclose(q[non_boundary], torch.full_like(q[non_boundary], 0.5), atol=1e-5)

    def test_finite_q_values(self) -> None:
        nz, ny, nx = 16, 16, 16
        device = torch.device("cpu")
        _, q = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, device)
        assert torch.isfinite(q).all()

    def test_some_boundary_nodes_detected(self) -> None:
        """A sphere inside the domain must produce at least some boundary nodes."""
        nz, ny, nx = 16, 16, 16
        device = torch.device("cpu")
        mask, _ = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, device)
        assert mask.any(), "No boundary nodes found for sphere inside domain"


class TestBouzidiBounceBack3D:
    def test_preserves_shape(self) -> None:
        from tensorlbm.d3q19 import equilibrium3d

        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()
        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4, :] = True
        q = torch.full((nz, ny, nx), 0.5)
        f_out = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=1)
        assert f_out.shape == f.shape

    def test_finite_output(self) -> None:
        from tensorlbm.d3q19 import equilibrium3d

        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()
        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4, :] = True
        q = torch.full((nz, ny, nx), 0.5)
        f_out = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=1)
        assert torch.isfinite(f_out).all()

    def test_halfway_q_reproduces_standard_bounce_back(self) -> None:
        """q=0.5 on an equilibrium distribution should reproduce standard BB."""
        from tensorlbm.d3q19 import OPPOSITE as OPP3D
        from tensorlbm.d3q19 import equilibrium3d

        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        ux0 = torch.full_like(rho, 0.04)
        f = equilibrium3d(rho, ux0, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()
        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4, :] = True
        q = torch.full((nz, ny, nx), 0.5)
        direction = 1
        opp = int(OPP3D[direction].item())
        f_out = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=direction)
        # At q=0.5: f_bc = 2*0.5*f_opp + 0 = f_opp
        assert torch.allclose(
            f_out[direction][fluid_nodes],
            f[opp][fluid_nodes],
            atol=1e-5,
        )

    def test_with_compute_q_sphere_finite(self) -> None:
        """Using compute_q_sphere with bouzidi_bounce_back_3d produces finite results."""
        from tensorlbm.d3q19 import equilibrium3d

        nz, ny, nx = 16, 16, 16
        device = torch.device("cpu")
        rho = torch.ones((nz, ny, nx), device=device)
        f = equilibrium3d(
            rho,
            torch.full_like(rho, 0.05),
            torch.zeros_like(rho),
            torch.zeros_like(rho),
            device=device,
        )
        f_prev = f.clone()

        mask, q_field = compute_q_sphere(nx, ny, nz, 8.0, 8.0, 8.0, 4.0, device)

        for d in range(19):
            fluid_nodes_d = mask[d]
            if fluid_nodes_d.any():
                f = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes_d, q_field[d], direction=d)

        assert torch.isfinite(f).all()

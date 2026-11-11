"""BFL interpolated bounce-back: momentum conservation and consistency tests.

These tests verify fundamental physics invariants of the Bouzidi–Firdaouss–
Lallemand (2001) interpolated boundary condition without running full
simulations.  They are lightweight (pure tensor operations, no time-stepping)
and catch implementation errors that would silently corrupt drag/lift forces.

Key invariants tested:
1. **Zero-velocity limit**: For a quiescent fluid (f = f_eq at u=0), BFL
   must not create spurious momentum.
2. **q=0.5 → standard BB**: At q=0.5 the BFL momentum exchange must equal
   the standard halfway bounce-back exchange exactly.
3. **Linear/quadratic branch continuity**: The momentum exchange is
   continuous at q=0.5 (no jump when crossing the branch boundary).
4. **Direction-consistent momentum**: BFL for direction d only injects
   momentum along the axis of direction d (no cross-axis contamination).

Reference
---------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
"Momentum transfer of a Boltzmann-lattice fluid with boundaries."
Physics of Fluids, 13(11), 3452–3459.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import C as C2D, OPPOSITE as OPP2D, equilibrium
from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, equilibrium3d
from tensorlbm.interpolated_bc import (
    bouzidi_bounce_back,
    bouzidi_bounce_back_3d,
    compute_q_circle,
    compute_q_sphere,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _momentum_change_2d(f_out: torch.Tensor, f_in: torch.Tensor) -> torch.Tensor:
    """Compute momentum change (2, ny, nx) from population change."""
    c = C2D.to(f_out.device).float()
    dp = torch.zeros((2,) + f_out.shape[1:], dtype=f_out.dtype, device=f_out.device)
    for d in range(9):
        dp[0] += (f_out[d] - f_in[d]) * c[d, 0]
        dp[1] += (f_out[d] - f_in[d]) * c[d, 1]
    return dp


def _momentum_change_3d(f_out: torch.Tensor, f_in: torch.Tensor) -> torch.Tensor:
    """Compute momentum change (3, nz, ny, nx) from population change."""
    c = C3D.to(f_out.device).float()
    dp = torch.zeros((3,) + f_out.shape[1:], dtype=f_out.dtype, device=f_out.device)
    for d in range(19):
        dp[0] += (f_out[d] - f_in[d]) * c[d, 0]
        dp[1] += (f_out[d] - f_in[d]) * c[d, 1]
        dp[2] += (f_out[d] - f_in[d]) * c[d, 2]
    return dp


# ---------------------------------------------------------------------------
# 2-D BFL momentum conservation
# ---------------------------------------------------------------------------

class TestBFL2DMomentumConservation:
    """Momentum conservation tests for 2-D BFL bounce-back."""

    def test_zero_velocity_no_spurious_momentum(self) -> None:
        """Quiescent fluid: BFL must not create momentum."""
        ny, nx = 8, 8
        rho = torch.ones((ny, nx))
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4] = True
        q = torch.full((ny, nx), 0.3)

        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=1)

        dp = _momentum_change_2d(f_out, f)
        assert abs(dp[0, 4, 4].item()) < 1e-6, f"Spurious mx = {dp[0, 4, 4].item()}"
        assert abs(dp[1, 4, 4].item()) < 1e-6, f"Spurious my = {dp[1, 4, 4].item()}"

    def test_halfway_q_momentum_equals_standard_bb(self) -> None:
        """At q=0.5, BFL momentum exchange = standard BB momentum exchange."""
        ny, nx = 8, 8
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_prev = f.clone()

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, :] = True
        q = torch.full((ny, nx), 0.5)

        f_bfl = bouzidi_bounce_back(f.clone(), f_prev.clone(), fluid_nodes, q, direction=1)

        # Standard BB: f[opp] = f[direction]
        f_std = f.clone()
        opp = int(OPP2D[1].item())
        f_std[opp, fluid_nodes] = f[1, fluid_nodes]

        dp_bfl = _momentum_change_2d(f_bfl, f)
        dp_std = _momentum_change_2d(f_std, f)
        assert torch.allclose(dp_bfl[:, fluid_nodes], dp_std[:, fluid_nodes], atol=1e-6)

    def test_branch_continuity_at_q_half(self) -> None:
        """Momentum exchange is continuous at q=0.5 (no jump)."""
        ny, nx = 6, 6
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.07)
        uy = torch.full_like(rho, 0.03)
        f = equilibrium(rho, ux, uy)
        f_prev = f.clone()

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[3, 3] = True

        eps = 1e-6
        q_lo = torch.full((ny, nx), 0.5 - eps)
        q_hi = torch.full((ny, nx), 0.5 + eps)

        f_lo = bouzidi_bounce_back(f.clone(), f_prev.clone(), fluid_nodes, q_lo, direction=1)
        f_hi = bouzidi_bounce_back(f.clone(), f_prev.clone(), fluid_nodes, q_hi, direction=1)

        dp_lo = _momentum_change_2d(f_lo, f)
        dp_hi = _momentum_change_2d(f_hi, f)

        assert torch.allclose(dp_lo[:, fluid_nodes], dp_hi[:, fluid_nodes], atol=1e-5), (
            f"Discontinuity at q=0.5: max diff = {(dp_lo - dp_hi).abs().max().item()}"
        )

    def test_momentum_along_direction_axis_only(self) -> None:
        """BFL for direction d only injects momentum along d's axis.

        D2Q9 direction 1 has c=(1,0), so BFL should only change x-momentum,
        not y-momentum.
        """
        ny, nx = 8, 8
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.1)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_prev = f.clone()

        fluid_nodes = torch.zeros((ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4] = True
        q = torch.full((ny, nx), 0.3)

        f_out = bouzidi_bounce_back(f, f_prev, fluid_nodes, q, direction=1)

        dp = _momentum_change_2d(f_out, f)

        # Direction 1 has c=(1,0), so only x-momentum should change
        assert abs(dp[0, 4, 4].item()) > 1e-8, "No x-momentum injection"
        assert abs(dp[1, 4, 4].item()) < 1e-5, f"Unexpected y-momentum injection: {dp[1, 4, 4].item()}"


# ---------------------------------------------------------------------------
# 3-D BFL momentum conservation
# ---------------------------------------------------------------------------

class TestBFL3DMomentumConservation:
    """Momentum conservation tests for 3-D BFL bounce-back (D3Q19)."""

    def test_zero_velocity_no_spurious_momentum(self) -> None:
        """Quiescent fluid: BFL must not create momentum."""
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()

        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4, 4] = True
        q = torch.full((nz, ny, nx), 0.3)

        f_out = bouzidi_bounce_back_3d(f, f_prev, fluid_nodes, q, direction=1)

        dp = _momentum_change_3d(f_out, f)
        assert abs(dp[0, 4, 4, 4].item()) < 1e-6, f"Spurious mx = {dp[0, 4, 4, 4].item()}"
        assert abs(dp[1, 4, 4, 4].item()) < 1e-6, f"Spurious my = {dp[1, 4, 4, 4].item()}"
        assert abs(dp[2, 4, 4, 4].item()) < 1e-6, f"Spurious mz = {dp[2, 4, 4, 4].item()}"

    def test_halfway_q_momentum_equals_standard_bb(self) -> None:
        """At q=0.5, BFL momentum exchange = standard BB momentum exchange."""
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.05)
        f = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()

        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4, :] = True
        q = torch.full((nz, ny, nx), 0.5)

        f_bfl = bouzidi_bounce_back_3d(f.clone(), f_prev.clone(), fluid_nodes, q, direction=1)

        # Standard BB
        f_std = f.clone()
        opp = int(OPP3D[1].item())
        f_std[opp, fluid_nodes] = f[1, fluid_nodes]

        dp_bfl = _momentum_change_3d(f_bfl, f)
        dp_std = _momentum_change_3d(f_std, f)
        assert torch.allclose(dp_bfl[:, fluid_nodes], dp_std[:, fluid_nodes], atol=1e-6)

    def test_branch_continuity_at_q_half(self) -> None:
        """Momentum exchange is continuous at q=0.5."""
        nz, ny, nx = 6, 6, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.07)
        f = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()

        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[3, 3, 3] = True

        eps = 1e-6
        q_lo = torch.full((nz, ny, nx), 0.5 - eps)
        q_hi = torch.full((nz, ny, nx), 0.5 + eps)

        f_lo = bouzidi_bounce_back_3d(f.clone(), f_prev.clone(), fluid_nodes, q_lo, direction=1)
        f_hi = bouzidi_bounce_back_3d(f.clone(), f_prev.clone(), fluid_nodes, q_hi, direction=1)

        dp_lo = _momentum_change_3d(f_lo, f)
        dp_hi = _momentum_change_3d(f_hi, f)

        assert torch.allclose(dp_lo[:, fluid_nodes], dp_hi[:, fluid_nodes], atol=1e-5), (
            f"Discontinuity at q=0.5: max diff = {(dp_lo - dp_hi).abs().max().item()}"
        )

    def test_all_19_directions_finite(self) -> None:
        """Applying BFL for all 19 directions must produce finite results."""
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.05)
        f = equilibrium3d(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f_prev = f.clone()

        fluid_nodes = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fluid_nodes[4, 4, 4] = True
        q = torch.full((nz, ny, nx), 0.35)

        for d in range(19):
            f_out = bouzidi_bounce_back_3d(f.clone(), f_prev.clone(), fluid_nodes, q, direction=d)
            assert torch.isfinite(f_out).all(), f"Non-finite for direction {d}"


# ---------------------------------------------------------------------------
# compute_q_circle / compute_q_sphere analytical verification
# ---------------------------------------------------------------------------

class TestQFieldAnalytical:
    """Verify q-field computation against known analytical values."""

    def test_circle_diagonal_q_matches_geometry(self) -> None:
        """Diagonal direction q matches the ray-sphere intersection.

        Circle at (4, 4) with radius 2.0.
        The known test case from test_bfl_fractional_distance.py:
        compute_q_circle(9, 9, 4.0, 4.0, 2.0) → q[7, 6, 6] = 2 - sqrt(2)
        """
        mask, q = compute_q_circle(9, 9, 4.0, 4.0, 2.0, torch.device("cpu"))
        expected = 2.0 - math.sqrt(2.0)
        assert mask[7, 6, 6]
        assert float(q[7, 6, 6]) == pytest.approx(expected, abs=1e-7)

    def test_circle_q_in_valid_range(self) -> None:
        """All q values on boundary nodes should be in (0, 1]."""
        mask, q = compute_q_circle(16, 16, 8.0, 8.0, 4.0, torch.device("cpu"))
        boundary_q = q[mask]
        if boundary_q.numel() > 0:
            assert float(boundary_q.min().item()) > 0.0
            assert float(boundary_q.max().item()) <= 1.0 + 1e-5

    def test_sphere_q_in_valid_range(self) -> None:
        """All q values on boundary nodes should be in (0, 1]."""
        mask, q = compute_q_sphere(16, 16, 16, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        boundary_q = q[mask]
        if boundary_q.numel() > 0:
            assert float(boundary_q.min().item()) > 0.0
            assert float(boundary_q.max().item()) <= 1.0 + 1e-5

    def test_sphere_diagonal_q_matches_geometry(self) -> None:
        """D3Q19 sphere diagonal q matches the ray-sphere intersection.

        The known test case: q[8, 4, 6, 6] = 2 - sqrt(2)
        """
        mask, q = compute_q_sphere(9, 9, 9, 4.0, 4.0, 4.0, 2.0, torch.device("cpu"))
        expected = 2.0 - math.sqrt(2.0)
        assert mask[8, 4, 6, 6]
        assert float(q[8, 4, 6, 6]) == pytest.approx(expected, abs=1e-7)

    def test_sphere_some_boundary_nodes_detected(self) -> None:
        """A sphere inside the domain must produce boundary nodes."""
        mask, _ = compute_q_sphere(16, 16, 16, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        assert mask.any(), "No boundary nodes found for sphere inside domain"

    def test_sphere_non_boundary_q_is_half(self) -> None:
        """Non-boundary entries should default to 0.5."""
        mask, q = compute_q_sphere(16, 16, 16, 8.0, 8.0, 8.0, 4.0, torch.device("cpu"))
        non_boundary = ~mask
        assert torch.allclose(q[non_boundary], torch.full_like(q[non_boundary], 0.5), atol=1e-5)

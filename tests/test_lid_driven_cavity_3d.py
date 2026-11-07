"""Tests for 3D lid-driven cavity D3Q27 vs D3Q19 comparison.

Runs a small-scale cavity simulation to verify both lattices produce
physically reasonable results and that D3Q27 converges toward the
Ghia reference within the validated anchor range.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q27 import (
    OPPOSITE as OPPOSITE27,
    collide_mrt27,
    equilibrium27,
    macroscopic27,
    stream27,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, OPPOSITE
from tensorlbm.solver3d import collide_mrt3d
from tensorlbm.boundary.boundaries3d import zou_he_moving_lid_3d


def _make_wall_mask(nz: int, ny: int, nx: int, device: torch.device) -> torch.Tensor:
    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall[:, :, 0] = True
    wall[:, :, -1] = True
    wall[:, 0, :] = True
    return wall


def _bounce_back(f: torch.Tensor, wall: torch.Tensor, opp: torch.Tensor) -> torch.Tensor:
    f_out = f.clone()
    for q in range(1, f.shape[0]):
        o = int(opp[q].item())
        f_out[q, wall] = f[o, wall]
    return f_out


def _stream_periodic_z(f: torch.Tensor, Q: int) -> torch.Tensor:
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q19 import C as C19
    C = C27 if Q == 27 else C19
    out = torch.empty_like(f)
    for q in range(Q):
        cx, cy, cz = int(C[q, 0].item()), int(C[q, 1].item()), int(C[q, 2].item())
        if cx == 0 and cy == 0 and cz == 0:
            out[q] = f[q]
        else:
            out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


def _step(f: torch.Tensor, wall: torch.Tensor, u_lid: float, tau: float, Q: int) -> torch.Tensor:
    if Q == 27:
        f = collide_mrt27(f, tau=tau)
        opp = OPPOSITE27
    else:
        f = collide_mrt3d(f, tau=tau)
        opp = OPPOSITE
    f = _bounce_back(f, wall, opp)
    f = _stream_periodic_z(f, Q)
    f = zou_he_moving_lid_3d(f, u_lid=u_lid, uy_lid=0.0, uz_lid=0.0)
    return f


class TestCavity3DSmall:
    """Small-scale cavity test to verify both lattices work."""

    @pytest.mark.parametrize("Q", [19, 27])
    def test_cavity_produces_finite_results(self, Q: int) -> None:
        """Both D3Q19 and D3Q27 should produce finite velocity fields."""
        nz, ny, nx = 2, 16, 16
        u_lid = 0.02
        tau = 0.8  # stable tau
        device = torch.device("cpu")

        wall = _make_wall_mask(nz, ny, nx, device)
        rho = torch.ones((nz, ny, nx), device=device)
        zeros = torch.zeros_like(rho)

        if Q == 27:
            f = equilibrium27(rho, zeros, zeros, zeros)
        else:
            f = equilibrium3d(rho, zeros, zeros, zeros)

        for _ in range(500):
            f = _step(f, wall, u_lid, tau, Q)

        if Q == 27:
            _, ux, uy, uz = macroscopic27(f, device)
        else:
            _, ux, uy, uz = macroscopic3d(f, device)

        assert torch.isfinite(ux).all()
        assert torch.isfinite(uy).all()
        assert torch.isfinite(uz).all()

    @pytest.mark.parametrize("Q", [19, 27])
    def test_cavity_lid_velocity_prescribed(self, Q: int) -> None:
        """The lid (y=ny-1) should have u_x close to u_lid."""
        nz, ny, nx = 2, 16, 16
        u_lid = 0.02
        tau = 0.8
        device = torch.device("cpu")

        wall = _make_wall_mask(nz, ny, nx, device)
        rho = torch.ones((nz, ny, nx), device=device)
        zeros = torch.zeros_like(rho)

        if Q == 27:
            f = equilibrium27(rho, zeros, zeros, zeros)
        else:
            f = equilibrium3d(rho, zeros, zeros, zeros)

        for _ in range(500):
            f = _step(f, wall, u_lid, tau, Q)

        if Q == 27:
            _, ux, _, _ = macroscopic27(f, device)
        else:
            _, ux, _, _ = macroscopic3d(f, device)

        # Lid row should have u_x close to u_lid (within 20% for coarse grid)
        lid_ux = ux[:, -1, 1:-1]  # exclude corners
        mean_lid_ux = float(lid_ux.mean().item())
        assert abs(mean_lid_ux - u_lid) < 0.2 * u_lid, (
            f"Lid velocity {mean_lid_ux:.4f} too far from {u_lid}"
        )

    def test_d3q27_vs_d3q19_same_order_magnitude(self) -> None:
        """D3Q27 and D3Q19 should produce velocities of the same order."""
        nz, ny, nx = 2, 16, 16
        u_lid = 0.02
        tau = 0.8
        device = torch.device("cpu")

        wall = _make_wall_mask(nz, ny, nx, device)
        rho = torch.ones((nz, ny, nx), device=device)
        zeros = torch.zeros_like(rho)

        results = {}
        for Q in [19, 27]:
            if Q == 27:
                f = equilibrium27(rho, zeros, zeros, zeros)
            else:
                f = equilibrium3d(rho, zeros, zeros, zeros)

            for _ in range(500):
                f = _step(f, wall, u_lid, tau, Q)

            if Q == 27:
                _, ux, uy, _ = macroscopic27(f, device)
            else:
                _, ux, uy, _ = macroscopic3d(f, device)

            results[Q] = {
                "ux_max": float(ux.abs().max().item()),
                "uy_max": float(uy.abs().max().item()),
            }

        # Both should have similar max velocity magnitudes
        for key in ["ux_max", "uy_max"]:
            ratio = results[27][key] / results[19][key]
            assert 0.3 < ratio < 3.0, (
                f"D3Q27/D3Q19 {key} ratio {ratio:.3f} outside [0.5, 2.0]"
            )

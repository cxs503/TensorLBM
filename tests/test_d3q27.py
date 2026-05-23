"""Tests for D3Q27 lattice, collision operators, and boundary conditions."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import (
    C27,
    OPPOSITE27,
    W27,
    apply_zou_he_channel_boundaries_27,
    bounce_back_cells_27,
    collide_bgk27,
    collide_mrt27,
    collide_smagorinsky_bgk27,
    collide_smagorinsky_mrt27,
    compute_obstacle_forces_27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
    make_channel_wall_mask_27,
    stream27,
    zou_he_inlet_velocity_27,
)
from tensorlbm.d3q27_sphere_flow import SphereFlowD3Q27Config

# ---------------------------------------------------------------------------
# D3Q27 lattice constants
# ---------------------------------------------------------------------------

class TestD3Q27Lattice:
    def test_weights_sum_to_one(self) -> None:
        assert abs(float(W27.sum().item()) - 1.0) < 1e-6

    def test_velocities_shape(self) -> None:
        assert C27.shape == (27, 3)

    def test_opposite_is_involution(self) -> None:
        for i in range(27):
            j = int(OPPOSITE27[i].item())
            assert int(OPPOSITE27[j].item()) == i

    def test_velocity_opposite_negation(self) -> None:
        for i in range(27):
            j = int(OPPOSITE27[i].item())
        assert torch.equal(C27[i], -C27[j]), (
                f"Direction {i}: C[{i}]={C27[i]} != -C[{j}]={C27[j]}"
            )


# ---------------------------------------------------------------------------
# D3Q27 equilibrium
# ---------------------------------------------------------------------------

class TestEquilibrium27:
    def test_roundtrip_zero_velocity(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, ux, ux)
        rho_out, ux_out, uy_out, uz_out = macroscopic27(f)
        assert torch.allclose(rho_out, rho, atol=1e-6)
        assert torch.allclose(ux_out, ux, atol=1e-6)

    def test_roundtrip_nonzero_velocity(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.full_like(rho, -0.02)
        uz = torch.full_like(rho, 0.01)
        f = equilibrium27(rho, ux, uy, uz)
        rho_out, ux_out, uy_out, uz_out = macroscopic27(f)
        assert torch.allclose(rho_out, rho, atol=1e-5)
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)
        assert torch.allclose(uz_out, uz, atol=1e-5)

    def test_output_shape(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        assert f.shape == (27, nz, ny, nx)


# ---------------------------------------------------------------------------
# D3Q27 streaming
# ---------------------------------------------------------------------------

class TestStream27:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        assert stream27(f).shape == f.shape

    def test_conserves_mass_periodic(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.rand((nz, ny, nx)) + 0.5
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        mass_before = float(f.sum().item())
        mass_after = float(stream27(f).sum().item())
        assert abs(mass_before - mass_after) < 1e-5 * mass_before

    def test_cache_reuse(self) -> None:
        """Second call must reuse cached indices (no error)."""
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f1 = stream27(f)
        f2 = stream27(f)
        assert torch.equal(f1, f2)


# ---------------------------------------------------------------------------
# D3Q27 collision operators
# ---------------------------------------------------------------------------

class TestCollide27BGK:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        assert collide_bgk27(f, tau=0.7).shape == f.shape

    def test_conserves_mass_and_momentum(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_bgk27(f, tau=0.7)
        rho_new, ux_new, uy_new, uz_new = macroscopic27(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-5)
        assert torch.allclose(ux_new, ux, atol=1e-5)
        assert torch.allclose(uy_new, uy, atol=1e-5)
        assert torch.allclose(uz_new, uz, atol=1e-5)

    def test_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.full_like(rho, 0.02)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        assert torch.allclose(collide_bgk27(feq, tau=0.7), feq, atol=1e-5)


class TestCollideMRT27:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_mrt27(f, tau=0.7)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_conserves_mass_and_momentum(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_mrt27(f, tau=0.7)
        rho_new, ux_new, uy_new, uz_new = macroscopic27(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-4)
        assert torch.allclose(ux_new, ux, atol=1e-4)
        assert torch.allclose(uy_new, uy, atol=1e-4)
        assert torch.allclose(uz_new, uz, atol=1e-4)

    def test_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        f_out = collide_mrt27(feq, tau=0.7)
        assert torch.allclose(f_out, feq, atol=1e-4)


class TestSmagorinsky27:
    def _feq(self, nz: int, ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.04
        uz = torch.rand_like(rho) * 0.04
        return rho, ux, uy, uz

    def test_bgk27_shape(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho, ux, uy, uz = self._feq(nz, ny, nx)
        f = equilibrium27(rho, ux, uy, uz)
        f_out = collide_smagorinsky_bgk27(f, tau=0.7)
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_mrt27_conserves_mass(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho, ux, uy, uz = self._feq(nz, ny, nx)
        f = equilibrium27(rho, ux, uy, uz)
        f_new = collide_smagorinsky_mrt27(f, tau=0.7)
        rho_new, ux_new, uy_new, uz_new = macroscopic27(f_new)
        assert torch.allclose(rho_new, rho, atol=1e-4)
        assert torch.allclose(ux_new, ux, atol=1e-4)

    def test_bgk27_at_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        feq = equilibrium27(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = collide_smagorinsky_bgk27(feq, tau=0.7)
        assert torch.allclose(f_out, feq, atol=1e-4)


# ---------------------------------------------------------------------------
# D3Q27 boundaries
# ---------------------------------------------------------------------------

class TestD3Q27Boundaries:
    def test_bounce_back_preserves_shape(self) -> None:
        nz, ny, nx = 6, 8, 10
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[0, :, :] = True
        f_out = bounce_back_cells_27(f, mask)
        assert f_out.shape == f.shape

    def test_wall_mask_has_walls(self) -> None:
        nz, ny, nx = 6, 8, 10
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        wall = make_channel_wall_mask_27(nz, ny, nx, obstacle, device=torch.device("cpu"))
        assert wall[:, 0, :].all()
        assert wall[:, -1, :].all()
        assert wall[0, :, :].all()
        assert wall[-1, :, :].all()
        assert not wall[1:-1, 1:-1, :].any()

    def test_zou_he_inlet_27_prescribes_velocity(self) -> None:
        nz, ny, nx = 4, 5, 8
        rho0 = torch.ones((nz, ny, nx))
        zeros = torch.zeros_like(rho0)
        f = equilibrium27(rho0, zeros, zeros, zeros)
        f = collide_bgk27(f, tau=0.7)
        f = stream27(f)
        u_in = 0.06
        f_out = zou_he_inlet_velocity_27(f, u_in)
        _, ux_out, uy_out, uz_out = macroscopic27(f_out)
        assert torch.allclose(ux_out[:, :, 0], torch.full((nz, ny), u_in), atol=2e-4)

    def test_apply_zou_he_channel_27_finite(self) -> None:
        nz, ny, nx = 4, 6, 10
        device = torch.device("cpu")
        obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool)
        wall = make_channel_wall_mask_27(nz, ny, nx, obstacle, device=device)
        rho = torch.ones((nz, ny, nx))
        ux0 = torch.full_like(rho, 0.05)
        zeros = torch.zeros_like(rho)
        f = equilibrium27(rho, ux0, zeros, zeros)
        f_out = apply_zou_he_channel_boundaries_27(
            f, u_in=0.05, wall_mask=wall, obstacle_mask=obstacle
        )
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_compute_obstacle_forces_27_empty_is_zero(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fx, fy, fz = compute_obstacle_forces_27(f, mask)
        assert float(fx) == pytest.approx(0.0)
        assert float(fy) == pytest.approx(0.0)
        assert float(fz) == pytest.approx(0.0)

    def test_compute_obstacle_forces_27_finite(self) -> None:
        nz, ny, nx = 4, 6, 10
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.05)
        f = equilibrium27(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
        f = stream27(f)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[2, 3, 5] = True
        fx, fy, fz = compute_obstacle_forces_27(f, mask)
        assert math.isfinite(float(fx))
        assert math.isfinite(float(fy))
        assert math.isfinite(float(fz))


# ---------------------------------------------------------------------------
# correct_mass27
# ---------------------------------------------------------------------------

class TestCorrectMass27:
    def test_restores_target_mass(self) -> None:
        nz, ny, nx = 4, 5, 6
        rho = torch.rand((nz, ny, nx)) + 0.5
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        target = 100.0
        f_corrected = correct_mass27(f, target)
        assert abs(float(f_corrected.sum().item()) - target) < 1e-4


# ---------------------------------------------------------------------------
# SphereFlowD3Q27Config
# ---------------------------------------------------------------------------

class TestSphereFlowD3Q27Config:
    def test_valid_config_does_not_raise(self) -> None:
        cfg = SphereFlowD3Q27Config(nx=32, ny=16, nz=16, re=20.0, n_steps=5)
        cfg.validate()

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"nx": 4}, "at least"),
            ({"ny": 2}, "at least"),
            ({"u_in": -0.01}, "u_in"),
        ],
    )
    def test_validate_raises(self, overrides: dict, match: str) -> None:
        base = {"nx": 32, "ny": 16, "nz": 16, "re": 20.0, "n_steps": 5}
        base.update(overrides)
        cfg = SphereFlowD3Q27Config(**base)
        with pytest.raises(ValueError, match=match):
            cfg.validate()

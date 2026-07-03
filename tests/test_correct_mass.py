"""Tests for correct_mass and correct_mass3d functions."""
import pytest
import torch

from tensorlbm.solver import correct_mass
from tensorlbm.solver3d import correct_mass3d
from tensorlbm.d3q27 import correct_mass27, equilibrium27
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm import equilibrium as equilibrium2d


# ---------------------------------------------------------------------------
# correct_mass (D2Q9)
# ---------------------------------------------------------------------------

class TestCorrectMass2D:
    """Tests for correct_mass in 2D (D2Q9)."""

    def test_restores_target_mass(self) -> None:
        """correct_mass should rescale f so that sum equals target_mass."""
        rho = torch.ones((10, 12), dtype=torch.float32) * 1.5
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        f = equilibrium2d(rho, ux, uy)

        target_mass = 100.0
        f_corrected = correct_mass(f, target_mass)

        actual_mass = float(f_corrected.sum().item())
        assert abs(actual_mass - target_mass) < 1e-4

    def test_preserves_shape(self) -> None:
        """correct_mass should return tensor of the same shape."""
        rho = torch.ones((8, 10), dtype=torch.float32)
        f = equilibrium2d(rho, torch.zeros_like(rho), torch.zeros_like(rho))

        f_corrected = correct_mass(f, target_mass=50.0)
        assert f_corrected.shape == f.shape

    def test_near_zero_current_unchanged(self) -> None:
        """When current mass is near zero, function should return f unchanged."""
        f = torch.zeros((9, 5, 5), dtype=torch.float32)
        f[4, 2, 2] = 1e-40

        f_corrected = correct_mass(f, target_mass=1.0)
        assert torch.allclose(f_corrected, f)

    def test_preserves_velocity_field(self) -> None:
        """After correction, macroscopic velocity should be unchanged."""
        from tensorlbm import macroscopic
        rho = torch.ones((10, 12), dtype=torch.float32) * 2.0
        ux = torch.ones_like(rho) * 0.1
        uy = torch.ones_like(rho) * 0.05
        f = equilibrium2d(rho, ux, uy)

        rho_before, ux_before, uy_before = macroscopic(f)
        f_corrected = correct_mass(f, target_mass=200.0)
        rho_after, ux_after, uy_after = macroscopic(f_corrected)

        assert torch.allclose(ux_before, ux_after, atol=1e-5)
        assert torch.allclose(uy_before, uy_after, atol=1e-5)

    def test_larger_than_target_scales_down(self) -> None:
        """If current mass > target, should scale down."""
        rho = torch.ones((5, 5), dtype=torch.float32) * 4.0
        f = equilibrium2d(rho, torch.zeros_like(rho), torch.zeros_like(rho))

        f_corrected = correct_mass(f, target_mass=50.0)
        actual_mass = float(f_corrected.sum().item())

        assert actual_mass < float(f.sum().item())
        assert abs(actual_mass - 50.0) < 1e-4

    def test_smaller_than_target_scales_up(self) -> None:
        """If current mass < target, should scale up."""
        rho = torch.ones((5, 5), dtype=torch.float32) * 0.5
        f = equilibrium2d(rho, torch.zeros_like(rho), torch.zeros_like(rho))

        f_corrected = correct_mass(f, target_mass=100.0)
        actual_mass = float(f_corrected.sum().item())

        assert actual_mass > float(f.sum().item())
        assert abs(actual_mass - 100.0) < 1e-4


# ---------------------------------------------------------------------------
# correct_mass3d (D3Q19)
# ---------------------------------------------------------------------------

class TestCorrectMass3D:
    """Tests for correct_mass3d in 3D (D3Q19)."""

    def test_restores_target_mass(self) -> None:
        """correct_mass3d should rescale f so that sum equals target_mass."""
        rho = torch.ones((8, 10, 12), dtype=torch.float32) * 1.2
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)

        target_mass = 500.0
        f_corrected = correct_mass3d(f, target_mass)

        actual_mass = float(f_corrected.sum().item())
        assert abs(actual_mass - target_mass) < 1e-4

    def test_preserves_shape(self) -> None:
        """correct_mass3d should return tensor of the same shape."""
        rho = torch.ones((5, 6, 7), dtype=torch.float32)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)

        f_corrected = correct_mass3d(f, target_mass=100.0)
        assert f_corrected.shape == f.shape

    def test_near_zero_current_unchanged(self) -> None:
        """When current mass is near zero, function should return f unchanged."""
        f = torch.zeros((19, 5, 5, 5), dtype=torch.float32)
        f[9, 2, 2, 2] = 1e-40

        f_corrected = correct_mass3d(f, target_mass=1.0)
        assert torch.allclose(f_corrected, f)

    def test_larger_than_target_scales_down(self) -> None:
        """If current mass > target, should scale down."""
        rho = torch.ones((4, 5, 6), dtype=torch.float32) * 5.0
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)

        f_corrected = correct_mass3d(f, target_mass=50.0)
        actual_mass = float(f_corrected.sum().item())

        assert actual_mass < float(f.sum().item())
        assert abs(actual_mass - 50.0) < 1e-4


# ---------------------------------------------------------------------------
# correct_mass27 (D3Q27)
# ---------------------------------------------------------------------------

class TestCorrectMass27:
    """Tests for correct_mass27 in 3D (D3Q27)."""

    def test_restores_target_mass(self) -> None:
        """correct_mass27 should rescale f so that sum equals target_mass."""
        rho = torch.ones((6, 8, 10), dtype=torch.float32) * 1.5
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)

        target_mass = 300.0
        f_corrected = correct_mass27(f, target_mass)

        actual_mass = float(f_corrected.sum().item())
        assert abs(actual_mass - target_mass) < 1e-4

    def test_preserves_shape(self) -> None:
        """correct_mass27 should return tensor of the same shape."""
        rho = torch.ones((4, 5, 6), dtype=torch.float32)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)

        f_corrected = correct_mass27(f, target_mass=50.0)
        assert f_corrected.shape == f.shape

    def test_near_zero_current_unchanged(self) -> None:
        """When current mass is near zero, function should return f unchanged."""
        f = torch.zeros((27, 4, 4, 4), dtype=torch.float32)
        f[13, 2, 2, 2] = 1e-40

        f_corrected = correct_mass27(f, target_mass=1.0)
        assert torch.allclose(f_corrected, f)

    def test_larger_than_target_scales_down(self) -> None:
        """If current mass > target, should scale down."""
        rho = torch.ones((3, 4, 5), dtype=torch.float32) * 6.0
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)

        f_corrected = correct_mass27(f, target_mass=40.0)
        actual_mass = float(f_corrected.sum().item())

        assert actual_mass < float(f.sum().item())
        assert abs(actual_mass - 40.0) < 1e-4

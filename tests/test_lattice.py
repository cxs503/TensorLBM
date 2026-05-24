"""Tests for tensorlbm.lattice – D2Q9 equilibrium and streaming utilities."""

from __future__ import annotations

import pytest
import torch

from tensorlbm.lattice import equilibrium, stream


class TestEquilibrium:
    """Tests for the lattice.equilibrium() function."""

    def test_output_shape(self) -> None:
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        u = torch.zeros((ny, nx, 2))
        f = equilibrium(rho, u)
        assert f.shape == (ny, nx, 9)

    def test_dtype_matches_input(self) -> None:
        rho = torch.ones((4, 5), dtype=torch.float64)
        u = torch.zeros((4, 5, 2), dtype=torch.float64)
        f = equilibrium(rho, u)
        assert f.dtype == torch.float64

    def test_weights_sum_to_one_per_cell(self) -> None:
        """f summed over the 9 directions at zero velocity must equal rho."""
        rho = torch.rand((6, 8)) + 0.5
        u = torch.zeros((6, 8, 2))
        f = equilibrium(rho, u)
        assert torch.allclose(f.sum(dim=-1), rho, atol=1e-6)

    def test_macroscopic_density_roundtrip(self) -> None:
        """Sum of f over directions = rho (at any velocity)."""
        rho = torch.rand((6, 8)) + 0.5
        u = torch.rand((6, 8, 2)) * 0.05
        f = equilibrium(rho, u)
        rho_out = f.sum(dim=-1)
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_finite_values(self) -> None:
        rho = torch.ones((6, 8))
        u = torch.full((6, 8, 2), 0.04)
        f = equilibrium(rho, u)
        assert torch.isfinite(f).all()

    def test_raises_on_device_mismatch(self) -> None:
        """rho and u on different devices must raise ValueError."""
        rho = torch.ones((4, 5))  # CPU
        u = torch.zeros((4, 5, 2))  # also CPU but mutate device attribute indirectly
        # We can only test this reliably if CUDA is available; skip otherwise.
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        u_cuda = u.cuda()
        with pytest.raises(ValueError, match="device"):
            equilibrium(rho, u_cuda)

    def test_zero_velocity_gives_symmetric_distribution(self) -> None:
        """At zero velocity, f should be spatially uniform and direction-symmetric."""
        ny, nx = 4, 4
        rho = torch.ones((ny, nx))
        u = torch.zeros((ny, nx, 2))
        f = equilibrium(rho, u)
        # Directions 1 and 3 (±x) should have equal weight at zero velocity
        assert torch.allclose(f[..., 1], f[..., 3], atol=1e-6)
        assert torch.allclose(f[..., 2], f[..., 4], atol=1e-6)

    def test_batched_shape(self) -> None:
        """equilibrium should work with arbitrary leading batch dimensions."""
        rho = torch.ones((3, 6, 8))
        u = torch.zeros((3, 6, 8, 2))
        f = equilibrium(rho, u)
        assert f.shape == (3, 6, 8, 9)


class TestStream:
    """Tests for the lattice.stream() function."""

    def test_output_shape_preserved(self) -> None:
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        u = torch.zeros((ny, nx, 2))
        f = equilibrium(rho, u)
        f_out = stream(f)
        assert f_out.shape == f.shape

    def test_conserves_total_mass_periodic(self) -> None:
        """Streaming with periodic boundaries must conserve total mass."""
        ny, nx = 8, 10
        rho = torch.rand((ny, nx)) + 0.5
        u = torch.rand((ny, nx, 2)) * 0.04
        f = equilibrium(rho, u)
        mass_before = float(f.sum().item())
        f_out = stream(f)
        mass_after = float(f_out.sum().item())
        assert abs(mass_before - mass_after) < 1e-5 * abs(mass_before)

    def test_finite_output(self) -> None:
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        u = torch.zeros((ny, nx, 2))
        f = equilibrium(rho, u)
        assert torch.isfinite(stream(f)).all()

    def test_uniform_field_unchanged(self) -> None:
        """Streaming a spatially uniform field must return the same field (periodic BC)."""
        ny, nx = 6, 8
        rho = torch.ones((ny, nx))
        u = torch.zeros((ny, nx, 2))
        f = equilibrium(rho, u)
        f_out = stream(f)
        assert torch.allclose(f_out, f, atol=1e-6)

    def test_double_stream_returns_original_for_uniform(self) -> None:
        """Streaming twice on a uniform field must restore the original (period = lattice size)."""
        ny, nx = 4, 4
        rho = torch.ones((ny, nx))
        u = torch.zeros((ny, nx, 2))
        f = equilibrium(rho, u)
        f2 = stream(stream(f))
        assert torch.allclose(f2, f, atol=1e-6)

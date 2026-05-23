import pytest
import torch

from tensorlbm import equilibrium3d, macroscopic3d


def test_equilibrium3d_roundtrip_zero_velocity() -> None:
    rho = torch.ones((4, 6, 8), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)

    f = equilibrium3d(rho, ux, uy, uz)
    rho_out, ux_out, uy_out, uz_out = macroscopic3d(f)

    assert torch.allclose(rho_out, rho, atol=1e-6)
    assert torch.allclose(ux_out, ux, atol=1e-6)
    assert torch.allclose(uy_out, uy, atol=1e-6)
    assert torch.allclose(uz_out, uz, atol=1e-6)


def test_equilibrium3d_shape() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.ones((nz, ny, nx))
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)

    f = equilibrium3d(rho, ux, uy, uz)
    assert f.shape == (19, nz, ny, nx)


def test_equilibrium3d_weights_sum_to_one() -> None:
    from tensorlbm import W3D

    assert abs(float(W3D.sum().item()) - 1.0) < 1e-6


def test_equilibrium3d_nonzero_velocity_roundtrip() -> None:
    rho = torch.ones((4, 6, 8), dtype=torch.float32)
    ux = torch.full_like(rho, 0.05)
    uy = torch.full_like(rho, 0.02)
    uz = torch.full_like(rho, -0.01)

    f = equilibrium3d(rho, ux, uy, uz)
    rho_out, ux_out, uy_out, uz_out = macroscopic3d(f)

    assert torch.allclose(rho_out, rho, atol=1e-5)
    assert torch.allclose(ux_out, ux, atol=1e-5)
    assert torch.allclose(uy_out, uy, atol=1e-5)
    assert torch.allclose(uz_out, uz, atol=1e-5)


def test_equilibrium3d_shape_mismatch_raises() -> None:
    rho = torch.ones((4, 6, 8), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros((4, 6, 7), dtype=torch.float32)
    uz = torch.zeros_like(rho)
    with pytest.raises(ValueError, match="shapes must match"):
        equilibrium3d(rho, ux, uy, uz)

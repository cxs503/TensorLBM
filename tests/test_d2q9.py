import pytest
import torch

from tensorlbm import equilibrium, macroscopic
from tensorlbm.d2q9 import W_EXACT64, W


def test_equilibrium_roundtrip_zero_velocity() -> None:
    rho = torch.ones((6, 8), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)

    f = equilibrium(rho, ux, uy)
    rho_out, ux_out, uy_out = macroscopic(f)

    assert torch.allclose(rho_out, rho, atol=1e-6)
    assert torch.allclose(ux_out, ux, atol=1e-6)
    assert torch.allclose(uy_out, uy, atol=1e-6)


def test_equilibrium_roundtrip_nonzero_velocity() -> None:
    rho = torch.ones((6, 8), dtype=torch.float32)
    ux = torch.full_like(rho, 0.05)
    uy = torch.full_like(rho, -0.02)

    f = equilibrium(rho, ux, uy)
    rho_out, ux_out, uy_out = macroscopic(f)

    assert torch.allclose(rho_out, rho, atol=1e-5)
    assert torch.allclose(ux_out, ux, atol=1e-5)
    assert torch.allclose(uy_out, uy, atol=1e-5)


def test_equilibrium_weights_sum_to_one() -> None:
    assert abs(float(W.sum().item()) - 1.0) < 1e-6


def test_float64_equilibrium_uses_unrounded_weights() -> None:
    rho = torch.ones((4, 5), dtype=torch.float64)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)

    torch.testing.assert_close(f[:, 0, 0], W_EXACT64, rtol=0.0, atol=0.0)
    assert torch.equal(W_EXACT64.float(), W)
    assert not torch.equal(W.double(), W_EXACT64)


def test_equilibrium_shape_mismatch_raises() -> None:
    rho = torch.ones((6, 8), dtype=torch.float32)
    ux = torch.zeros((6, 7), dtype=torch.float32)
    uy = torch.zeros_like(rho)
    with pytest.raises(ValueError, match="shapes must match"):
        equilibrium(rho, ux, uy)

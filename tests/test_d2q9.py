import torch

from tensorlbm import equilibrium, macroscopic


def test_equilibrium_roundtrip_zero_velocity() -> None:
    rho = torch.ones((6, 8), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)

    f = equilibrium(rho, ux, uy)
    rho_out, ux_out, uy_out = macroscopic(f)

    assert torch.allclose(rho_out, rho, atol=1e-6)
    assert torch.allclose(ux_out, ux, atol=1e-6)
    assert torch.allclose(uy_out, uy, atol=1e-6)

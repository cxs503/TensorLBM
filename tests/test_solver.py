import torch

from tensorlbm import (
    apply_simple_channel_boundaries,
    collide_bgk,
    collide_mrt,
    cylinder_mask,
    equilibrium,
    make_channel_wall_mask,
    stream,
)


def test_stream_and_collide_preserve_shape() -> None:
    rho = torch.ones((10, 12), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    f2 = collide_bgk(f, tau=0.6)
    f3 = stream(f2)

    assert f2.shape == f.shape
    assert f3.shape == f.shape


def test_channel_boundaries_returns_valid_tensor() -> None:
    ny, nx = 12, 20
    device = torch.device("cpu")
    obstacle = cylinder_mask(nx, ny, cx=5.0, cy=6.0, radius=2.0, device=device)
    wall_mask = make_channel_wall_mask(ny, nx, obstacle, device=device)

    rho = torch.ones((ny, nx), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    out = apply_simple_channel_boundaries(f, u_in=0.05, wall_mask=wall_mask, obstacle_mask=obstacle)
    assert out.shape == f.shape
    assert torch.isfinite(out).all()


def test_collide_mrt_preserves_shape() -> None:
    rho = torch.ones((10, 12), dtype=torch.float32)
    ux = torch.full_like(rho, 0.05)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    f_out = collide_mrt(f, tau=0.6)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()


def test_collide_mrt_conserves_mass_and_momentum() -> None:
    rho = torch.ones((8, 10), dtype=torch.float32)
    ux = torch.full_like(rho, 0.05)
    uy = torch.full_like(rho, 0.02)
    f = equilibrium(rho, ux, uy)

    f_out = collide_mrt(f, tau=0.7)

    assert torch.allclose(f_out.sum(dim=0), f.sum(dim=0), atol=1e-5)

    from tensorlbm import C
    cx = C[:, 0].view(9, 1, 1).float()
    cy = C[:, 1].view(9, 1, 1).float()
    assert torch.allclose((f_out * cx).sum(dim=0), (f * cx).sum(dim=0), atol=1e-5)
    assert torch.allclose((f_out * cy).sum(dim=0), (f * cy).sum(dim=0), atol=1e-5)


def test_collide_mrt_with_uniform_s_matches_bgk() -> None:
    """MRT with all relaxation rates = 1/tau must recover BGK exactly."""
    tau = 0.7
    rho = torch.ones((8, 10), dtype=torch.float32)
    ux = torch.full_like(rho, 0.04)
    uy = torch.full_like(rho, 0.01)
    f = equilibrium(rho, ux, uy)

    f_bgk = collide_bgk(f, tau=tau)

    s_uniform = torch.full((9,), 1.0 / tau)
    f_mrt = collide_mrt(f, tau=tau, s=s_uniform)

    assert torch.allclose(f_bgk, f_mrt, atol=1e-6)


def test_collide_mrt_custom_s() -> None:
    """Passing a custom s tensor should not raise and should return valid output."""
    rho = torch.ones((6, 8), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    s = torch.tensor([0.0, 1.6, 1.6, 0.0, 1.2, 0.0, 1.2, 1.5, 1.5])
    f_out = collide_mrt(f, tau=0.6, s=s)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()

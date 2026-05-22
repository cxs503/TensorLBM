import torch

from tensorlbm import (
    apply_simple_channel_boundaries,
    collide_bgk,
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

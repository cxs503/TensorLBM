import torch

from tensorlbm import (
    apply_simple_channel_boundaries_3d,
    collide_bgk3d,
    equilibrium3d,
    make_channel_wall_mask_3d,
    sphere_mask,
    stream3d,
)


def test_stream3d_and_collide3d_preserve_shape() -> None:
    nz, ny, nx = 6, 8, 10
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)

    f2 = collide_bgk3d(f, tau=0.6)
    f3 = stream3d(f2)

    assert f2.shape == f.shape
    assert f3.shape == f.shape


def test_channel_boundaries_3d_returns_valid_tensor() -> None:
    nz, ny, nx = 8, 10, 20
    device = torch.device("cpu")
    obstacle = sphere_mask(nx, ny, nz, cx=5.0, cy=5.0, cz=4.0, radius=2.0, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, obstacle, device=device)

    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)

    out = apply_simple_channel_boundaries_3d(f, u_in=0.05, wall_mask=wall_mask, obstacle_mask=obstacle)
    assert out.shape == f.shape
    assert torch.isfinite(out).all()


def test_sphere_mask_shape_and_dtype() -> None:
    device = torch.device("cpu")
    mask = sphere_mask(nx=20, ny=12, nz=10, cx=10.0, cy=6.0, cz=5.0, radius=3.0, device=device)
    assert mask.shape == (10, 12, 20)
    assert mask.dtype == torch.bool


def test_channel_wall_mask_3d_has_walls() -> None:
    nz, ny, nx = 8, 10, 20
    device = torch.device("cpu")
    obstacle = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, obstacle, device=device)

    assert wall_mask[:, 0, :].all()
    assert wall_mask[:, -1, :].all()
    assert wall_mask[0, :, :].all()
    assert wall_mask[-1, :, :].all()
    # Interior cells should not be walls
    assert not wall_mask[1:-1, 1:-1, :].any()

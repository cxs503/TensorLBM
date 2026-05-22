import torch

from tensorlbm import (
    apply_simple_channel_boundaries_3d,
    collide_bgk3d,
    collide_mrt3d,
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


def test_collide_mrt3d_preserves_shape() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.full_like(rho, 0.05)
    uy = torch.zeros_like(rho)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)

    f_out = collide_mrt3d(f, tau=0.6)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()


def test_collide_mrt3d_conserves_mass_and_momentum() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.full_like(rho, 0.05)
    uy = torch.full_like(rho, 0.02)
    uz = torch.full_like(rho, -0.01)
    f = equilibrium3d(rho, ux, uy, uz)

    f_out = collide_mrt3d(f, tau=0.7)

    assert torch.allclose(f_out.sum(dim=0), f.sum(dim=0), atol=1e-5)

    from tensorlbm import C3D
    cx = C3D[:, 0].view(19, 1, 1, 1).float()
    cy = C3D[:, 1].view(19, 1, 1, 1).float()
    cz = C3D[:, 2].view(19, 1, 1, 1).float()
    assert torch.allclose((f_out * cx).sum(dim=0), (f * cx).sum(dim=0), atol=1e-5)
    assert torch.allclose((f_out * cy).sum(dim=0), (f * cy).sum(dim=0), atol=1e-5)
    assert torch.allclose((f_out * cz).sum(dim=0), (f * cz).sum(dim=0), atol=1e-5)


def test_collide_mrt3d_with_uniform_s_matches_bgk() -> None:
    """MRT with all relaxation rates = 1/tau must recover BGK exactly."""
    tau = 0.7
    nz, ny, nx = 4, 6, 8
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.full_like(rho, 0.04)
    uy = torch.full_like(rho, 0.01)
    uz = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, uy, uz)

    f_bgk = collide_bgk3d(f, tau=tau)

    s_uniform = torch.full((19,), 1.0 / tau)
    f_mrt = collide_mrt3d(f, tau=tau, s=s_uniform)

    assert torch.allclose(f_bgk, f_mrt, atol=1e-5)

import pytest
import torch

from tensorlbm import (
    apply_simple_channel_boundaries_3d,
    apply_zou_he_channel_boundaries_3d,
    collide_bgk3d,
    collide_mrt3d,
    equilibrium3d,
    macroscopic3d,
    make_channel_wall_mask_3d,
    sphere_mask,
    stream3d,
    zou_he_inlet_velocity_3d,
)
from tensorlbm.sphere_flow import SphereFlowConfig


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

    out = apply_simple_channel_boundaries_3d(
        f,
        u_in=0.05,
        wall_mask=wall_mask,
        obstacle_mask=obstacle,
    )
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


# ---------------------------------------------------------------------------
# Mass conservation under periodic streaming (3D)
# ---------------------------------------------------------------------------

def test_stream3d_conserves_mass_periodic() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.rand((nz, ny, nx), dtype=torch.float32) + 0.5
    f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))

    mass_before = float(f.sum().item())
    f_streamed = stream3d(f)
    mass_after = float(f_streamed.sum().item())

    assert abs(mass_before - mass_after) < 1e-5 * mass_before


# ---------------------------------------------------------------------------
# MRT collision for D3Q19
# ---------------------------------------------------------------------------

def test_collide_mrt3d_preserves_shape() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
    f_out = collide_mrt3d(f, tau=0.6)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()


def test_collide_mrt3d_conserves_mass_and_momentum() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.rand((nz, ny, nx), dtype=torch.float32) + 0.5
    ux = torch.rand_like(rho) * 0.04
    uy = torch.rand_like(rho) * 0.04
    uz = torch.rand_like(rho) * 0.04
    f = equilibrium3d(rho, ux, uy, uz)

    f_new = collide_mrt3d(f, tau=0.7)
    rho_new, ux_new, uy_new, uz_new = macroscopic3d(f_new)

    assert torch.allclose(rho_new, rho, atol=1e-4)
    assert torch.allclose(ux_new, ux, atol=1e-4)
    assert torch.allclose(uy_new, uy, atol=1e-4)
    assert torch.allclose(uz_new, uz, atol=1e-4)


def test_collide_mrt3d_at_equilibrium_is_identity() -> None:
    nz, ny, nx = 4, 6, 8
    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    ux = torch.full_like(rho, 0.03)
    uy = torch.full_like(rho, 0.01)
    uz = torch.full_like(rho, -0.01)
    feq = equilibrium3d(rho, ux, uy, uz)

    f_out = collide_mrt3d(feq, tau=0.6)
    assert torch.allclose(f_out, feq, atol=1e-4)


# ---------------------------------------------------------------------------
# Zou/He 3D inlet
# ---------------------------------------------------------------------------

def test_zou_he_inlet_3d_prescribes_velocity() -> None:
    nz, ny, nx = 6, 8, 12
    rho0 = torch.ones((nz, ny, nx))
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
    f = collide_bgk3d(f, tau=0.6)
    f = stream3d(f)

    u_in = 0.06
    f_zh = zou_he_inlet_velocity_3d(f, u_in=u_in)
    _, ux_out, uy_out, uz_out = macroscopic3d(f_zh)

    assert torch.allclose(ux_out[:, :, 0], torch.full((nz, ny), u_in), atol=2e-4)


def test_zou_he_channel_3d_returns_valid_tensor() -> None:
    nz, ny, nx = 8, 10, 20
    device = torch.device("cpu")
    obstacle = sphere_mask(nx, ny, nz, cx=5.0, cy=5.0, cz=4.0, radius=2.0, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, obstacle, device=device)

    rho = torch.ones((nz, ny, nx), dtype=torch.float32)
    f = equilibrium3d(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
    out = apply_zou_he_channel_boundaries_3d(
        f,
        u_in=0.05,
        wall_mask=wall_mask,
        obstacle_mask=obstacle,
    )
    assert out.shape == f.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SphereFlowConfig.validate() error paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"nx": 4}, "nx, ny"),
        ({"ny": 2}, "nx, ny"),
        ({"nz": 2}, "nx, ny"),
        ({"n_steps": 0}, "n_steps"),
        ({"output_interval": 0}, "output_interval"),
        ({"u_in": -0.05}, "u_in, re"),
    ],
)
def test_sphere_config_validate_raises(overrides: dict, match: str) -> None:
    base = {
        "nx": 32,
        "ny": 16,
        "nz": 16,
        "u_in": 0.05,
        "re": 50.0,
        "radius": 4.0,
        "n_steps": 10,
        "output_interval": 5,
    }
    base.update(overrides)
    cfg = SphereFlowConfig(**base)
    with pytest.raises(ValueError, match=match):
        cfg.validate()

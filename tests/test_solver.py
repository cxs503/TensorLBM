import math

import pytest
import torch

from tensorlbm import (
    apply_simple_channel_boundaries,
    apply_zou_he_channel_boundaries,
    collide_bgk,
    collide_mrt,
    compute_obstacle_forces,
    cylinder_mask,
    equilibrium,
    macroscopic,
    make_channel_wall_mask,
    stream,
    zou_he_inlet_velocity,
)
from tensorlbm.cylinder_flow import CylinderFlowConfig, compute_vorticity


# ---------------------------------------------------------------------------
# Shape / basic sanity tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mass conservation under periodic streaming
# ---------------------------------------------------------------------------

def test_stream_conserves_mass_periodic() -> None:
    """Streaming on a periodic domain must not change total mass."""
    rho = torch.rand((10, 12), dtype=torch.float32) + 0.5
    ux = torch.zeros_like(rho)
    uy = torch.zeros_like(rho)
    f = equilibrium(rho, ux, uy)

    mass_before = float(f.sum().item())
    f_streamed = stream(f)
    mass_after = float(f_streamed.sum().item())

    assert abs(mass_before - mass_after) < 1e-5 * mass_before


# ---------------------------------------------------------------------------
# BGK collision conserves mass and momentum
# ---------------------------------------------------------------------------

def test_collide_bgk_conserves_mass_and_momentum() -> None:
    rho = torch.rand((8, 10), dtype=torch.float32) + 0.5
    ux = torch.rand_like(rho) * 0.05
    uy = torch.rand_like(rho) * 0.05
    f = equilibrium(rho, ux, uy)

    f_new = collide_bgk(f, tau=0.7)
    rho_new, ux_new, uy_new = macroscopic(f_new)

    # BGK preserves local mass and momentum (they are collision invariants)
    assert torch.allclose(rho_new, rho, atol=1e-5)
    assert torch.allclose(ux_new, ux, atol=1e-5)
    assert torch.allclose(uy_new, uy, atol=1e-5)


# ---------------------------------------------------------------------------
# MRT collision
# ---------------------------------------------------------------------------

def test_collide_mrt_preserves_shape() -> None:
    rho = torch.ones((10, 12), dtype=torch.float32)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    f_out = collide_mrt(f, tau=0.6)
    assert f_out.shape == f.shape
    assert torch.isfinite(f_out).all()


def test_collide_mrt_conserves_mass_and_momentum() -> None:
    """MRT must preserve local density and momentum (collision invariants)."""
    rho = torch.rand((8, 10), dtype=torch.float32) + 0.5
    ux = torch.rand_like(rho) * 0.05
    uy = torch.rand_like(rho) * 0.05
    f = equilibrium(rho, ux, uy)

    f_new = collide_mrt(f, tau=0.7)
    rho_new, ux_new, uy_new = macroscopic(f_new)

    assert torch.allclose(rho_new, rho, atol=1e-5)
    assert torch.allclose(ux_new, ux, atol=1e-5)
    assert torch.allclose(uy_new, uy, atol=1e-5)


def test_collide_mrt_at_equilibrium_is_identity() -> None:
    """MRT applied to an equilibrium distribution must leave it unchanged."""
    rho = torch.ones((8, 10), dtype=torch.float32)
    ux = torch.full_like(rho, 0.04)
    uy = torch.full_like(rho, 0.02)
    feq = equilibrium(rho, ux, uy)

    f_out = collide_mrt(feq, tau=0.6)
    assert torch.allclose(f_out, feq, atol=1e-5)


# ---------------------------------------------------------------------------
# Vorticity
# ---------------------------------------------------------------------------

def test_compute_vorticity_rigid_rotation() -> None:
    """For solid-body rotation ux = -ω·y, uy = ω·x the vorticity is 2ω."""
    ny, nx = 20, 20
    omega = 0.1
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    ux = -omega * yy
    uy = omega * xx
    vort = compute_vorticity(ux, uy)
    # Interior cells (1:-1, 1:-1) should be close to 2*omega
    interior = vort[1:-1, 1:-1]
    assert torch.allclose(interior, torch.full_like(interior, 2.0 * omega), atol=1e-4)


# ---------------------------------------------------------------------------
# CylinderFlowConfig.validate() error paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"nx": 4}, "nx and ny"),
        ({"ny": 2}, "nx and ny"),
        ({"n_steps": 0}, "n_steps"),
        ({"output_interval": 0}, "output_interval"),
        ({"u_in": -0.1}, "u_in, re"),
        ({"re": -10.0}, "u_in, re"),
        ({"radius": 0.0}, "u_in, re"),
    ],
)
def test_cylinder_config_validate_raises(overrides: dict, match: str) -> None:
    base = dict(nx=64, ny=32, u_in=0.05, re=50.0, radius=6.0, n_steps=10, output_interval=5)
    base.update(overrides)
    cfg = CylinderFlowConfig(**base)
    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_cylinder_config_validate_tau_too_small() -> None:
    # Force tau <= 0.5 by using very small nu: u_in→0 but still >0 and re→∞
    cfg = CylinderFlowConfig(u_in=1e-9, re=1e12, radius=1.0)
    with pytest.raises(ValueError, match="tau"):
        cfg.validate()


# ---------------------------------------------------------------------------
# Zou/He inlet boundary condition
# ---------------------------------------------------------------------------

def test_zou_he_inlet_prescribes_velocity() -> None:
    """After Zou/He inlet update, macroscopic ux at x=0 must equal u_in."""
    ny, nx = 16, 20
    device = torch.device("cpu")
    rho0 = torch.ones((ny, nx))
    ux0 = torch.zeros_like(rho0)
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(rho0, ux0, uy0)
    # Perturb to move away from equilibrium
    f = collide_bgk(f, tau=0.6)
    f = stream(f)

    u_in = 0.08
    f_zh = zou_he_inlet_velocity(f, u_in=u_in)
    _, ux_out, uy_out = macroscopic(f_zh)

    assert torch.allclose(ux_out[:, 0], torch.full((ny,), u_in), atol=1e-5)
    assert torch.allclose(uy_out[:, 0], torch.zeros(ny), atol=1e-5)


def test_zou_he_channel_returns_valid_tensor() -> None:
    ny, nx = 12, 20
    device = torch.device("cpu")
    obstacle = cylinder_mask(nx, ny, cx=5.0, cy=6.0, radius=2.0, device=device)
    wall_mask = make_channel_wall_mask(ny, nx, obstacle, device=device)

    rho = torch.ones((ny, nx), dtype=torch.float32)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    out = apply_zou_he_channel_boundaries(f, u_in=0.05, wall_mask=wall_mask, obstacle_mask=obstacle)
    assert out.shape == f.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# compute_obstacle_forces
# ---------------------------------------------------------------------------

def test_compute_obstacle_forces_empty_mask_is_zero() -> None:
    """Forces on an empty obstacle must be exactly zero."""
    ny, nx = 10, 12
    rho = torch.ones((ny, nx))
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    fx, fy = compute_obstacle_forces(f, mask)
    assert float(fx.item()) == pytest.approx(0.0)
    assert float(fy.item()) == pytest.approx(0.0)


def test_compute_obstacle_forces_returns_finite() -> None:
    ny, nx = 12, 20
    device = torch.device("cpu")
    obstacle = cylinder_mask(nx, ny, cx=5.0, cy=6.0, radius=2.0, device=device)
    rho = torch.ones((ny, nx))
    f = equilibrium(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho))
    f = stream(f)
    fx, fy = compute_obstacle_forces(f, obstacle)
    assert math.isfinite(float(fx.item()))
    assert math.isfinite(float(fy.item()))

import torch

from tensorlbm import (
    C,
    OPPOSITE,
    W,
    apply_simple_channel_boundaries,
    collide_bgk,
    cylinder_mask,
    equilibrium,
    macroscopic,
    stream,
)


def test_d2q9_constants_have_expected_shapes_and_weights() -> None:
    assert C.shape == (9, 2)
    assert W.shape == (9,)
    assert OPPOSITE.shape == (9,)
    assert torch.all(OPPOSITE[OPPOSITE] == torch.arange(9))
    assert torch.isclose(W.sum(), torch.tensor(1.0, dtype=W.dtype))


def test_equilibrium_has_expected_shape_and_rest_state() -> None:
    ny, nx = 6, 8
    rho = torch.ones((ny, nx))
    ux = torch.zeros((ny, nx))
    uy = torch.zeros((ny, nx))
    feq = equilibrium(rho, ux, uy)

    assert feq.shape == (9, ny, nx)
    expected = W.view(9, 1, 1) * rho.unsqueeze(0)
    assert torch.allclose(feq, expected, atol=1e-7, rtol=0.0)


def test_macroscopic_recovers_equilibrium_inputs() -> None:
    ny, nx = 5, 7
    rho_in = torch.full((ny, nx), 1.1)
    ux_in = torch.full((ny, nx), 0.03)
    uy_in = torch.full((ny, nx), -0.02)
    f = equilibrium(rho_in, ux_in, uy_in)

    rho, ux, uy = macroscopic(f)

    assert rho.shape == (ny, nx)
    assert ux.shape == (ny, nx)
    assert uy.shape == (ny, nx)
    assert torch.allclose(rho, rho_in, atol=1e-6, rtol=0.0)
    assert torch.allclose(ux, ux_in, atol=1e-6, rtol=0.0)
    assert torch.allclose(uy, uy_in, atol=1e-6, rtol=0.0)


def test_one_step_solver_smoke_cpu() -> None:
    nx, ny = 32, 16
    device = torch.device("cpu")
    obstacle = cylinder_mask(nx, ny, cx=nx * 0.25, cy=ny * 0.5, radius=3.0, device=device)
    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[-1, :] = True
    wall_mask[obstacle] = False

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), 0.05, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    ux0[obstacle] = 0.0
    f = equilibrium(rho0, ux0, uy0, device=device)

    f = collide_bgk(f, tau=0.56)
    f = stream(f)
    f = apply_simple_channel_boundaries(f, u_in=0.05, wall_mask=wall_mask, obstacle_mask=obstacle)
    rho, ux, uy = macroscopic(f)

    assert f.shape == (9, ny, nx)
    assert rho.shape == (ny, nx)
    assert ux.shape == (ny, nx)
    assert uy.shape == (ny, nx)
    assert torch.isfinite(f).all()
    assert torch.isfinite(rho).all()
    assert rho.min().item() > 0.0

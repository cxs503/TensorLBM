import torch

from tensorlbm import C, W, collide_and_stream, equilibrium, initialize_equilibrium, macroscopic


def test_d2q9_constants_shapes_and_weight_sum():
    assert C.shape == (9, 2)
    assert W.shape == (9,)
    assert torch.isclose(W.sum(), torch.tensor(1.0))


def test_equilibrium_returns_expected_shape_and_density_sum():
    rho = torch.ones((8, 10))
    u = torch.zeros((8, 10, 2))

    feq = equilibrium(rho, u)

    assert feq.shape == (8, 10, 9)
    recovered_rho = feq.sum(dim=-1)
    assert torch.allclose(recovered_rho, rho, atol=1e-6)


def test_macroscopic_recovery_shapes_and_rest_state_velocity():
    f = initialize_equilibrium(6, 9, rho0=1.0, u0=(0.0, 0.0))

    rho, u = macroscopic(f)

    assert rho.shape == (6, 9)
    assert u.shape == (6, 9, 2)
    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-6)
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)


def test_single_step_smoke_preserves_shape_and_positive_density():
    f = initialize_equilibrium(12, 16, rho0=1.0, u0=(0.03, 0.0))
    f_next = collide_and_stream(f, omega=1.0)

    assert f_next.shape == f.shape
    rho, _ = macroscopic(f_next)
    assert torch.all(rho > 0.0)

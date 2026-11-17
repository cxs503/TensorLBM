"""Cross-lattice external-force moments and closed periodic pair-force balance."""

import pytest
import torch

from tensorlbm import d2q9, d3q19, d3q27
from tensorlbm.multiphase import collide_sc_two_component
from tensorlbm.multiphase3d import collide_sc_two_component_3d
from tensorlbm.multiphase3d_d3q27 import collide_sc_two_component_27


def lattice(q):
    if q == 9:
        return d2q9.C, d2q9.equilibrium, collide_sc_two_component, (6, 8), {"force_scheme": "guo"}
    if q == 19:
        return (
            d3q19.C,
            d3q19.equilibrium3d,
            collide_sc_two_component_3d,
            (4, 6, 8),
            {"use_guo": True},
        )
    return d3q27.C, d3q27.equilibrium27, collide_sc_two_component_27, (4, 6, 8), {"use_guo": True}


@pytest.mark.parametrize("q", [9, 19, 27])
@pytest.mark.parametrize("tau", [0.6, 0.8, 1.2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_external_force_first_and_second_moments(q, tau, dtype):
    c, equilibrium, collide, shape, forcing = lattice(q)
    dim = c.shape[1]
    rho = torch.full(shape, 0.8, dtype=dtype)
    velocity = torch.tensor([0.02, -0.01, 0.03][:dim], dtype=dtype)
    acceleration = torch.tensor([2e-3, -3e-3, 4e-3][:dim], dtype=dtype)
    f = equilibrium(rho, *[torch.full_like(rho, u) for u in velocity])
    force_args = {key: float(a) for key, a in zip(("gx", "gy", "gz"), acceleration)}
    out, _ = collide(f, f, G_12=0.0, tau1=tau, tau2=tau + 0.1, **force_args, **forcing)
    delta = (out - f).reshape(q, -1)
    c = c.to(dtype)
    atol = 8 * torch.finfo(dtype).eps
    force = rho.flatten()[None] * acceleration[:, None]
    torch.testing.assert_close(c.T @ delta, force, atol=atol, rtol=0)
    torch.testing.assert_close(delta.sum(0), torch.zeros_like(rho.flatten()), atol=atol, rtol=0)
    # Starting in raw-velocity equilibrium, BGK + Guo changes the raw second
    # moment by sym(u F) + (1 - 1/(4*tau))*F F / rho. This catches using raw
    # velocity in the source despite half-force velocity in equilibrium.
    for a in range(dim):
        for b in range(dim):
            actual = (c[:, a] * c[:, b]) @ delta
            expected = velocity[a] * force[b] + velocity[b] * force[a]
            expected += (1 - 1 / (4 * tau)) * force[a] * force[b] / rho.flatten()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("q", [9, 19, 27])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_internal_pair_forces_conserve_total_momentum_at_unequal_tau(q, dtype):
    c, equilibrium, collide, shape, forcing = lattice(q)
    rng = torch.Generator().manual_seed(7)
    rhos = [0.5 + 0.1 * torch.rand(shape, generator=rng, dtype=dtype) for _ in range(2)]
    velocities = [0.02 * torch.randn(shape, generator=rng, dtype=dtype) for _ in range(c.shape[1])]
    populations = [equilibrium(rho, *velocities) for rho in rhos]
    out = collide(*populations, G_12=-0.6, tau1=0.8, tau2=1.2, **forcing)
    delta = sum(after - before for before, after in zip(populations, out))
    total_momentum_delta = c.double().T @ delta.double().reshape(q, -1).sum(1)
    atol = 1e-5 if dtype == torch.float32 else 5e-13
    torch.testing.assert_close(
        total_momentum_delta, torch.zeros(c.shape[1], dtype=torch.float64), atol=atol, rtol=0
    )

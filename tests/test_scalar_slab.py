import math

import pytest
import torch

from tensorlbm.scalar import (
    scalar_collide_bgk, scalar_diffusivity, scalar_equilibrium, scalar_stream_slab,
)


def test_slab_links_account_for_local_exchange_and_reflect_on_insulation():
    g = torch.arange(60, dtype=torch.float64).reshape(5,3,4)/60
    wall = torch.tensor([1.,2.,3.], dtype=g.dtype)
    streamed, supplied = scalar_stream_slab(g, wall, None)
    torch.testing.assert_close(streamed[1,:,0], wall/3 - g[3,:,0])
    torch.testing.assert_close(supplied[0], wall/3 - 2*g[3,:,0])
    torch.testing.assert_close(streamed[3,:,-1], g[1,:,-1])
    torch.testing.assert_close(supplied[1], torch.zeros_like(wall))
    torch.testing.assert_close(streamed.sum()-g.sum(), supplied.sum())


@pytest.mark.parametrize("tau", [.8, 1.2])
def test_mixed_wall_mode_recovers_diffusivity(tau):
    errors = []
    for n in (12,24):
        x = (torch.arange(n, dtype=torch.float64)+.5)/n
        initial = torch.sin(.5*math.pi*x)[None,:]
        zero = torch.zeros_like(initial)
        g = scalar_equilibrium(initial, zero, zero)
        steps = round(.035*n*n/scalar_diffusivity(tau))
        for _ in range(steps):
            g, _ = scalar_stream_slab(scalar_collide_bgk(g,g.sum(dim=0),zero,zero,tau), 0., None)
        time = steps*scalar_diffusivity(tau)/n**2
        errors.append(float(abs(g.sum(dim=0)-initial*math.exp(-.25*math.pi**2*time)).max()))
    assert errors[1] < errors[0]/3.5


def test_d2q5_links_balance_every_cell_including_wall_and_periodic_seam():
    from tensorlbm.scalar import scalar_link_transfer

    g = torch.arange(60, dtype=torch.float64).cos().reshape(5,3,4)
    after, walls = scalar_stream_slab(g, 3., None)
    internal = scalar_link_transfer(g, "x")
    x = torch.cat((walls[0,:,None], internal, -walls[1,:,None]), dim=1)
    y = scalar_link_transfer(g, "y", periodic=True)
    torch.testing.assert_close(after.sum(0)-g.sum(0), -x.diff(dim=1)+torch.roll(y,1,0)-y)
    torch.testing.assert_close(scalar_link_transfer(g, "y"), y[:-1])
    torch.testing.assert_close(scalar_link_transfer(g, "x", periodic=True)[:,:-1], internal)

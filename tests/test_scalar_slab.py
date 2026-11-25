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


def test_legacy_scalar_populations_stream_in_their_declared_directions():
    from tensorlbm import scalar_transport as transport

    g = torch.zeros(5,5,5,dtype=torch.float64)
    g[:,2,2] = 1.
    streamed = transport.scalar_stream(g)
    for i,(cx,cy) in enumerate(transport.C5.tolist()):
        assert streamed[i,2+cy,2+cx] == 1
    c = torch.ones_like(g[0])
    ux, uy = .05*c, -.03*c
    eq = transport.scalar_equilibrium(c, ux, uy)
    torch.testing.assert_close((eq * transport.C5[:,0,None,None]).sum(0), ux)
    torch.testing.assert_close((eq * transport.C5[:,1,None,None]).sum(0), uy)
    torch.testing.assert_close(transport.scalar_collision(eq, c, ux, uy, .8), eq)


def test_step_mean_preserves_substep_pulse_and_rowwise_wall_levels():
    import numpy as np
    from tensorlbm.scalar import scalar_wall_step_means

    # Both short-pulse edges lie before the midpoint of the first step.
    means = scalar_wall_step_means([0., 1., 3.], [-2., .1, .3, 1., 4.],
                                   [[99., 99.], [0., 1.], [2., -1.], [0., 1.], [3., 4.], [9., 9.]])
    np.testing.assert_allclose(means, [[.4, .6], [3., 4.]])
    np.testing.assert_allclose(scalar_wall_step_means([0., .2, 1.], [], [5.]), [5., 5.])


def test_linear_wall_step_means_integrate_short_ramps_and_hold_end_values():
    import numpy as np
    from tensorlbm.scalar import scalar_wall_step_means

    means = scalar_wall_step_means([-1., 0., 1., 2.], [.1, .2, .4],
                                   [[1., 0.], [3., -2.], [2., 1.]], interpolation="linear")
    np.testing.assert_allclose(means, [[1., 0.], [2., .4], [2., 1.]])
    np.testing.assert_allclose(scalar_wall_step_means([0., 1.], [.3], [7.], interpolation="linear"), [7.])

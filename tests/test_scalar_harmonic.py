import math

import pytest
import torch

from tensorlbm.scalar import scalar_collide_bgk, scalar_stream_slab
from tensorlbm.scalar_harmonic import solve_harmonic_scalar_slab


@pytest.mark.parametrize("insulated", [False, True])
def test_complex_slab_response_satisfies_original_population_update(insulated):
    ny, nx, tau, omega = 7, 4, 1.1, .23
    west = torch.arange(ny, dtype=torch.float64).sin() + .3j
    east = None if insulated else .4-.2j
    sol = solve_harmonic_scalar_slab(nx, ny, tau, omega, west, east)
    # Independent real-time update at arbitrary phases, including quadrature.
    for angle in (.1, 1.7):
        phase = complex(math.cos(angle), math.sin(angle))
        mid = complex(math.cos(angle+.5*omega), math.sin(angle+.5*omega))
        end = complex(math.cos(angle+omega), math.sin(angle+omega))
        g = (sol.populations*phase).real
        zero = torch.zeros_like(g[0])
        post = scalar_collide_bgk(g,g.sum(0),zero,zero,tau)
        new,_ = scalar_stream_slab(post,(west*mid).real,None if insulated else (east*mid).real)
        torch.testing.assert_close(new,(sol.populations*end).real,rtol=0,atol=2e-14)
    divergence = sol.x_transfer.diff(dim=1)+sol.y_transfer-torch.roll(sol.y_transfer,1,0)
    torch.testing.assert_close(divergence,-2j*math.sin(omega/2)*sol.scalar,rtol=0,atol=2e-14)


def test_dc_solution_reproduces_linear_field_and_insulated_degeneracy():
    sol = solve_harmonic_scalar_slab(5, 1, .8, 0., 1., 0.)
    x = (torch.arange(5,dtype=torch.float64)+.5)/5
    torch.testing.assert_close(sol.scalar.real[0],1-x)
    torch.testing.assert_close(sol.x_transfer.real,torch.full((1,6),.02,dtype=x.dtype))
    with pytest.raises(ValueError,match="reference"):
        solve_harmonic_scalar_slab(5,1,.8,0.,None,None)

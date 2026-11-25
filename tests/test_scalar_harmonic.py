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


def test_anisotropic_collision_conserves_density_and_relaxes_first_moments():
    from tensorlbm.scalar import scalar_collide_diffusion

    g = torch.arange(40,dtype=torch.float64).cos().reshape(5,2,4)
    post = scalar_collide_diffusion(g,.82,1.14,.95)
    torch.testing.assert_close(post.sum(0),g.sum(0))
    for pos, neg, tau in ((1,3,.82),(2,4,1.14)):
        torch.testing.assert_close(post[pos]-post[neg],(1-1/tau)*(g[pos]-g[neg]))
    zero = torch.zeros_like(g[0])
    torch.testing.assert_close(scalar_collide_diffusion(g,.82,.82,.82),
                               scalar_collide_bgk(g,g.sum(0),zero,zero,.82))
    with pytest.raises(ValueError,match="relaxation"):
        scalar_collide_diffusion(g,.8,.5,1.)


@pytest.mark.parametrize("tau_xy", [.3, -.3])
def test_rotated_flux_relaxation(tau_xy):
    from tensorlbm.scalar import scalar_collide_diffusion

    g = torch.arange(40, dtype=torch.float64).sin().reshape(5, 2, 4)
    matrix = g.new_tensor([[1.2, tau_xy], [tau_xy, .85]])
    post = scalar_collide_diffusion(g, 1.2, .85, .94, tau_xy=tau_xy)
    moments = (g[1:3]-g[3:5]).reshape(2, -1)
    expected = moments-torch.linalg.solve(matrix, moments)
    torch.testing.assert_close((post[1:3]-post[3:5]).reshape(2, -1), expected)
    torch.testing.assert_close(post.sum(0), g.sum(0))
    with pytest.raises(ValueError, match="positive definite"):
        scalar_collide_diffusion(g, .75, .75, 1., tau_xy=.25)


@pytest.mark.parametrize("direction", [-1, 1])
def test_oblique_periodic_decay_converges(direction):
    from tensorlbm.scalar import scalar_collide_diffusion, scalar_equilibrium, scalar_stream

    errors = []
    for n in (16, 32, 64):
        x = torch.arange(n, dtype=torch.float64)*2*math.pi/n
        mode = torch.cos(x[None, :]+direction*x[:, None])
        zero = torch.zeros_like(mode)
        g = scalar_equilibrium(mode, zero, zero)
        steps = n*n//16
        for _ in range(steps):
            g = scalar_stream(scalar_collide_diffusion(g, 1.2, .85, 1., tau_xy=.21))
        decay = math.exp(-steps*(2*math.pi/n)**2*(.7+.35+2*direction*.21)/3)
        reference = decay*mode
        errors.append(float((g.sum(0)-reference).norm()/reference.norm()))
    assert max(errors[1]/errors[0], errors[2]/errors[1]) < .28
    assert errors[-1] < .001

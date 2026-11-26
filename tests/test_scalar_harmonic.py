import cmath
import math

import pytest
import torch

from tensorlbm.scalar import scalar_collide_bgk, scalar_stream_slab
from tensorlbm.scalar_harmonic import solve_harmonic_scalar_slab


@pytest.mark.parametrize("insulated", [False, True])
def test_complex_slab_response_satisfies_original_population_update(insulated):
    ny, nx, tau, omega = 7, 4, 1.1, 0.23
    west = torch.arange(ny, dtype=torch.float64).sin() + 0.3j
    east = None if insulated else 0.4 - 0.2j
    sol = solve_harmonic_scalar_slab(nx, ny, tau, omega, west, east)
    # Independent real-time update at arbitrary phases, including quadrature.
    for angle in (0.1, 1.7):
        phase = complex(math.cos(angle), math.sin(angle))
        mid = complex(math.cos(angle + 0.5 * omega), math.sin(angle + 0.5 * omega))
        end = complex(math.cos(angle + omega), math.sin(angle + omega))
        g = (sol.populations * phase).real
        zero = torch.zeros_like(g[0])
        post = scalar_collide_bgk(g, g.sum(0), zero, zero, tau)
        new, _ = scalar_stream_slab(
            post, (west * mid).real, None if insulated else (east * mid).real
        )
        torch.testing.assert_close(new, (sol.populations * end).real, rtol=0, atol=2e-14)
    divergence = sol.x_transfer.diff(dim=1) + sol.y_transfer - torch.roll(sol.y_transfer, 1, 0)
    torch.testing.assert_close(
        divergence, -2j * math.sin(omega / 2) * sol.scalar, rtol=0, atol=2e-14
    )


def test_dc_solution_reproduces_linear_field_and_insulated_degeneracy():
    sol = solve_harmonic_scalar_slab(5, 1, 0.8, 0.0, 1.0, 0.0)
    x = (torch.arange(5, dtype=torch.float64) + 0.5) / 5
    torch.testing.assert_close(sol.scalar.real[0], 1 - x)
    torch.testing.assert_close(sol.x_transfer.real, torch.full((1, 6), 0.02, dtype=x.dtype))
    with pytest.raises(ValueError, match="reference"):
        solve_harmonic_scalar_slab(5, 1, 0.8, 0.0, None, None)


def test_anisotropic_collision_conserves_density_and_relaxes_first_moments():
    from tensorlbm.scalar import scalar_collide_diffusion

    g = torch.arange(40, dtype=torch.float64).cos().reshape(5, 2, 4)
    post = scalar_collide_diffusion(g, 0.82, 1.14, 0.95)
    torch.testing.assert_close(post.sum(0), g.sum(0))
    for pos, neg, tau in ((1, 3, 0.82), (2, 4, 1.14)):
        torch.testing.assert_close(post[pos] - post[neg], (1 - 1 / tau) * (g[pos] - g[neg]))
    zero = torch.zeros_like(g[0])
    torch.testing.assert_close(
        scalar_collide_diffusion(g, 0.82, 0.82, 0.82),
        scalar_collide_bgk(g, g.sum(0), zero, zero, 0.82),
    )
    with pytest.raises(ValueError, match="relaxation"):
        scalar_collide_diffusion(g, 0.8, 0.5, 1.0)


@pytest.mark.parametrize("tau_xy", [0.3, -0.3])
def test_rotated_flux_relaxation(tau_xy):
    from tensorlbm.scalar import scalar_collide_diffusion

    g = torch.arange(40, dtype=torch.float64).sin().reshape(5, 2, 4)
    matrix = g.new_tensor([[1.2, tau_xy], [tau_xy, 0.85]])
    post = scalar_collide_diffusion(g, 1.2, 0.85, 0.94, tau_xy=tau_xy)
    moments = (g[1:3] - g[3:5]).reshape(2, -1)
    expected = moments - torch.linalg.solve(matrix, moments)
    torch.testing.assert_close((post[1:3] - post[3:5]).reshape(2, -1), expected)
    torch.testing.assert_close(post.sum(0), g.sum(0))
    with pytest.raises(ValueError, match="positive definite"):
        scalar_collide_diffusion(g, 0.75, 0.75, 1.0, tau_xy=0.25)


@pytest.mark.parametrize("direction", [-1, 1])
def test_oblique_periodic_decay_converges(direction):
    from tensorlbm.scalar import scalar_collide_diffusion, scalar_equilibrium, scalar_stream

    errors = []
    for n in (16, 32, 64):
        x = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
        mode = torch.cos(x[None, :] + direction * x[:, None])
        zero = torch.zeros_like(mode)
        g = scalar_equilibrium(mode, zero, zero)
        steps = n * n // 16
        for _ in range(steps):
            g = scalar_stream(scalar_collide_diffusion(g, 1.2, 0.85, 1.0, tau_xy=0.21))
        decay = math.exp(-steps * (2 * math.pi / n) ** 2 * (0.7 + 0.35 + 2 * direction * 0.21) / 3)
        reference = decay * mode
        errors.append(float((g.sum(0) - reference).norm() / reference.norm()))
    assert max(errors[1] / errors[0], errors[2] / errors[1]) < 0.28
    assert errors[-1] < 0.001


@pytest.mark.parametrize("omega,insulated", [(0.0, False), (0.27, False), (0.27, True), (3.1, True)])
def test_volume_heat_increment_matches_population_step_and_local_balance(omega, insulated):
    from tensorlbm.scalar import scalar_equilibrium

    source = torch.arange(15, dtype=torch.float64).reshape(3, 5).sin() + 0.37j
    west = None if insulated else torch.tensor([1.0, 2.0, -0.4], dtype=torch.float64)
    solution = solve_harmonic_scalar_slab(5, 3, 0.83, omega, west, None, source=source)
    phase = complex(math.cos(0.6), math.sin(0.6))
    mid = phase * complex(math.cos(omega / 2), math.sin(omega / 2))
    end = phase * complex(math.cos(omega), math.sin(omega))
    g = (phase * solution.populations).real
    zero = torch.zeros_like(g[0])
    post = scalar_collide_bgk(g, g.sum(0), zero, zero, 0.83)
    new, _ = scalar_stream_slab(post, None if insulated else (mid * west).real, None)
    new += scalar_equilibrium((mid * source).real, zero, zero)
    torch.testing.assert_close(new, (end * solution.populations).real, rtol=0, atol=2e-14)
    div = solution.x_transfer.diff(dim=1) + solution.y_transfer - solution.y_transfer.roll(1, 0)
    torch.testing.assert_close(2j * math.sin(omega / 2) * solution.scalar + div, source)


def test_insulated_balanced_heat_source_converges_with_prescribed_mean(record_property):
    # Unit square and unit physical diffusivity. The x cosine is Neumann
    # and the independent y cosine is periodic; both integrate to zero.
    tau = .9
    mean = 3.+.25j
    previous = None
    for n in (8, 16, 32, 64):
        dx = 1/n
        dt = (tau-.5)*dx**2/3
        x = (torch.arange(n, dtype=torch.float64)+.5)*dx
        longitudinal = torch.cos(math.pi*x)[None, :]
        transverse = .4j*torch.cos(2*math.pi*x)[:, None]
        source = dt*math.pi**2*(longitudinal+4*transverse)
        sol = solve_harmonic_scalar_slab(n, n, tau, 0., None, None, source=source,
                                         mean_temperature=mean)
        error = (sol.scalar-(mean+longitudinal+transverse)).abs().max().item()
        faces = torch.arange(n+1, dtype=torch.float64)*dx
        flux_x = math.pi*torch.sin(math.pi*faces)
        flux_y = .8j*math.pi*torch.sin(2*math.pi*faces[1:])[:, None]
        flux_error = max((sol.x_transfer*dx/dt-flux_x).abs().max().item(),
                         (sol.y_transfer*dx/dt-flux_y).abs().max().item())
        record_property(f"flux_error_{n}_W_per_m2", flux_error)
        divergence = sol.x_transfer.diff(dim=1) + sol.y_transfer - sol.y_transfer.roll(1, 0)
        balance = (divergence-source).abs().max().item()
        record_property(f"field_error_{n}_K", error)
        record_property(f"max_cell_balance_{n}_K_per_step", balance)
        assert abs(sol.scalar.mean().item()-mean) < 2e-14
        assert sol.x_transfer[:, [0, -1]].abs().max().item() == 0.
        assert balance < 5e-14
        if previous is not None:
            assert previous[0]/error > 3.8
            assert previous[1]/flux_error > 3.8
        previous = error, flux_error
    assert error < 4e-5 and flux_error < .002


def test_insulated_dc_mean_and_source_compatibility():
    sol = solve_harmonic_scalar_slab(1, 1, .85, 0., None, None, mean_temperature=2.+1j)
    assert sol.scalar.item() == pytest.approx(2.+1j)
    with pytest.raises(ValueError, match="zero net source"):
        solve_harmonic_scalar_slab(4, 3, .85, 0., None, None, source=1j, mean_temperature=0.)
    with pytest.raises(ValueError, match="only"):
        solve_harmonic_scalar_slab(4, 3, .85, .2, None, None, mean_temperature=2.)


@pytest.mark.parametrize("omega", [1e-12, 1e-18])
def test_slow_insulated_response_separates_mean_storage_and_link_transfer(omega, record_property):
    # Constant source fills the entire insulated slab. On top of it, a
    # balanced source/sink sends heat across the interior toward increasing x.
    uniform = .75-.25j
    perturbation = torch.tensor([.125, .125, 0., 0., -.125, -.125], dtype=torch.float64)
    source = uniform + perturbation
    sol = solve_harmonic_scalar_slab(6, 3, .85, omega, None, None, source=source)
    storage = 2j*math.sin(omega/2)
    mean_error = abs(storage*sol.scalar.mean().item()-uniform)/abs(uniform)
    # At these frequencies the nonuniform storage correction is O(omega).
    dc_transfer = torch.cat((perturbation.new_zeros(1), perturbation.cumsum(0)))
    flux_error = (sol.x_transfer-dc_transfer).abs().max().item()
    div = sol.x_transfer.diff(dim=1) + sol.y_transfer-sol.y_transfer.roll(1, 0)
    residual = (storage*sol.scalar+div-source).abs().max().item()
    record_property("relative_mean_storage_error", mean_error)
    record_property("max_transfer_difference_from_dc", flux_error)
    record_property("max_cell_balance_error", residual)
    assert mean_error < 1e-14
    assert flux_error < 20*omega+5e-15
    assert residual < 1e-14


@pytest.mark.parametrize("fixed", [None, "west", "east"])
def test_prescribed_wall_transfer_recovers_steady_gradient(fixed, record_property):
    nx, tau, slope = 12, .9, -.125
    mean = 2.5
    transfer = -(tau-.5)*slope/3
    west = mean-slope*nx/2 if fixed == "west" else None
    east = mean+slope*nx/2 if fixed == "east" else None
    flux = (transfer if west is None else None, -transfer if east is None else None)
    sol = solve_harmonic_scalar_slab(nx, 3, tau, 0., west, east, wall_transfer=flux,
                                     mean_temperature=mean if fixed is None else None)
    exact = mean + slope*(torch.arange(nx, dtype=torch.float64)+.5-nx/2)
    field_error = (sol.scalar-exact).abs().max().item()
    rate_error = (sol.x_transfer-transfer).abs().max().item()
    record_property("max_temperature_error_K", field_error)
    record_property("max_link_transfer_error", rate_error)
    assert field_error < 2e-13
    assert rate_error < 3e-15


def test_rowwise_wall_flux_closes_harmonic_storage_and_time_update(record_property):
    from tensorlbm.scalar import scalar_equilibrium

    nx, ny, omega, tau = 4, 3, .19, .87
    west = torch.tensor([.1j, .3, -.4], dtype=torch.complex128)
    east = -.02+.1j
    source = .03+.04j
    sol = solve_harmonic_scalar_slab(nx, ny, tau, omega, None, None, source=source,
                                     wall_transfer=(west, east))
    div = sol.x_transfer.diff(dim=1)+sol.y_transfer-sol.y_transfer.roll(1, 0)
    storage = 2j*math.sin(omega/2)*sol.scalar
    residual = (storage+div-source).abs().max().item()
    assert abs(storage.mean()-(source+(west.mean()+east)/nx)) < 1e-14
    mid = complex(math.cos(omega/2), math.sin(omega/2))
    end = mid**2
    g = sol.populations.real
    zero = torch.zeros_like(g[0])
    post = scalar_collide_bgk(g, g.sum(0), zero, zero, tau)
    streamed, _ = scalar_stream_slab(post, None, None, wall_transfer=((mid*west).real, (mid*east).real))
    streamed += scalar_equilibrium(zero+(mid*source).real, zero, zero)
    update_error = (streamed-(end*sol.populations).real).abs().max().item()
    record_property("max_cell_balance_error", residual)
    record_property("max_population_step_error", update_error)
    assert max(residual, update_error) < 1e-14


def test_flux_wall_rejects_inconsistent_data():
    with pytest.raises(ValueError, match="both"):
        solve_harmonic_scalar_slab(3, 1, .9, .2, 0., None, wall_transfer=(1., None))
    with pytest.raises(ValueError, match="zero net"):
        solve_harmonic_scalar_slab(3, 1, .9, 0., None, None, mean_temperature=0.,
                                   wall_transfer=(1., None))


def test_steady_wall_input_balances_uniform_volume_sink(record_property):
    source = -.0625
    nx = 7
    sol = solve_harmonic_scalar_slab(nx, 2, .87, 0., None, None, source=source,
                                     wall_transfer=(-nx*source, None), mean_temperature=4.)
    expected = (torch.arange(nx+1, dtype=torch.float64)-nx)*source
    error = (sol.x_transfer-expected).abs().max().item()
    record_property("max_link_transfer_error", error)
    assert error < 5e-15
    assert abs(sol.scalar.mean()-4.) < 1e-14


@pytest.mark.parametrize("kxy", [0., -.35, .35])
def test_anisotropic_traveling_wall_flux_refines_against_continuum(kxy, record_property):
    # Unit square, unit volumetric heat capacity. Both walls have traveling
    # inward fluxes with a quarter-period phase offset. The bounded solution
    # is exp(i*b*y)*X(x), with qx=-kx*X'-i*b*kxy*X.
    kx, ky = 1., .5
    omega, b = 8., 2*math.pi
    west, east = 1., .5j
    beta = b*kxy/kx
    wave = ((b*b*(ky-kxy*kxy/kx)+1j*omega)/kx)**.5
    # Two decaying axial modes. Their coefficients impose qx(0)=west
    # and qx(1)=-east with the common transverse phase factored out.
    left_rate, right_rate = -wave-1j*beta, wave-1j*beta
    scale = kx*wave*(1-cmath.exp(-2*wave))
    left = (west+east*cmath.exp(-right_rate))/scale
    right = (east+west*cmath.exp(left_rate))/scale
    refinements = []
    for n in (8, 16, 32, 64):
        dx = 1/n
        dt = (.9-.5)*dx**2/(3*kx)
        faces = torch.arange(n+1, dtype=torch.float64)*dx
        centers = (faces[1:]+faces[:-1])/2
        y_average = torch.exp(1j*b*centers)*math.sin(b*dx/2)/(b*dx/2)
        left_face = left*torch.exp(left_rate*faces)
        right_face = right*torch.exp(right_rate*(faces-1))
        x_average = (left_face.diff()/left_rate + right_face.diff()/right_rate)/dx
        gradient_average = (left_face+right_face).diff()/dx
        qx = kx*wave*(left_face-right_face)
        sol = solve_harmonic_scalar_slab(
            n, n, .9, omega*dt, None, None, tau_y=.5+3*ky*dt/dx**2, tau_xy=3*kxy*dt/dx**2,
            wall_transfer=(west*y_average*dt/dx, east*y_average*dt/dx))
        exact_y_flux = torch.exp(1j*b*faces[1:, None])*(-kxy*gradient_average-1j*b*ky*x_average)
        errors = (
            (sol.scalar-y_average[:, None]*x_average).abs().max().item(),
            (sol.x_transfer*dx/dt-y_average[:, None]*qx).abs().max().item(),
            (sol.y_transfer*dx/dt-exact_y_flux).abs().max().item(),
        )
        for name, error in zip(("temperature_K", "x_flux_W_m2", "y_flux_W_m2"), errors):
            record_property(f"max_error_{name}_{n}", error)
        refinements.append(errors)
    for previous, current in zip(refinements, refinements[1:]):
        assert all(old > 3*new for old, new in zip(previous, current))
    assert all(old > 3.7*new for old, new in zip(refinements[-2], refinements[-1]))
    assert errors[0] < .001 and max(errors[1:]) < .003

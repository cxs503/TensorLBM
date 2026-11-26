"""Adiabatic-wall conservation regression for the 3D D3Q7 thermal BC fix.

The pre-fix rule copied the whole post-stream distribution of the adjacent
interior slab onto the four adiabatic walls (y = 0/ny-1, z = 0/nz-1) of the
periodic ``stream_thermal_3d``: a zero-gradient condition referencing the
interior arrivals instead of the wall node's own outgoing distributions
that the periodic stream had wrapped around to the opposite wall.  Under
convection (and even in pure diffusion with wall-normal temperature
structure) the two differ, so the old rule acted as a spurious heat
source/sink (2026-09 evidence, 32^3 fp64 budget: ~1e-2/step in modulated
pure diffusion, -7.8e-3/step in developed Ra=1e3 convection, -0.46/step at
Ra=1e4, worsening under refinement; same disease family as the 2D D2Q5 fix,
PR #300).  The fixed rule unwraps the wrapped-around distribution and
reflects it back, so each adiabatic wall pair exchanges energy in exact
antisymmetry.

This test drives the temperature pipeline alone (no flow solver) with a
deterministic velocity field so advection -- the leak driver -- is nonzero
from the first step, and asserts (a) exact budget identity closure and
(b) net-zero adiabat exchange.  On the old rule (b) fails by ~0.5/step.
"""

import math

import torch

from tensorlbm import thermal3d
from tensorlbm.thermal3d import (
    collide_thermal_bgk_3d,
    equilibrium_thermal_3d,
    macroscopic_thermal_3d,
    stream_thermal_3d,
)

KICK = 0.05
STEPS = 20
TOL = 1e-12
TAU_T = 0.8  # thermal3d driver value
WALLS = ("x0", "xn", "y0", "yn", "z0", "zn")

# the fixed module exports the public alias; the pre-fix module only has
# the private implementation (getattr fallback keeps this test runnable
# against both generations -- it must FAIL on the old rule, not error)
apply_bc = getattr(
    thermal3d, "apply_temperature_boundaries_3d", thermal3d._apply_temperature_boundaries_3d
)


def _wall_deltas(g: torch.Tensor, g_ps: torch.Tensor) -> dict[str, float]:
    return {
        "x0": (g[:, :, :, 0] - g_ps[:, :, :, 0]).sum().item(),
        "xn": (g[:, :, :, -1] - g_ps[:, :, :, -1]).sum().item(),
        "y0": (g[:, :, 0, 1:-1] - g_ps[:, :, 0, 1:-1]).sum().item(),
        "yn": (g[:, :, -1, 1:-1] - g_ps[:, :, -1, 1:-1]).sum().item(),
        "z0": (g[:, 0, 1:-1, 1:-1] - g_ps[:, 0, 1:-1, 1:-1]).sum().item(),
        "zn": (g[:, -1, 1:-1, 1:-1] - g_ps[:, -1, 1:-1, 1:-1]).sum().item(),
    }


def _adiabat_budget(n=16):
    nz = ny = nx = n
    z = torch.arange(nz, dtype=torch.float64).view(nz, 1, 1)
    y = torch.arange(ny, dtype=torch.float64).view(1, ny, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, 1, nx)
    # deterministic cellular velocity field: advection nonzero everywhere,
    # wall slabs included (worst case for the adiabat reflection rule)
    ux = (
        (KICK * torch.sin(2 * math.pi * y / ny) * torch.sin(math.pi * (x + 1) / nx))
        .expand(nz, ny, nx)
        .contiguous()
    )
    uy = (
        (-KICK * torch.cos(2 * math.pi * y / ny) * torch.cos(math.pi * (x + 1) / nx))
        .expand(nz, ny, nx)
        .contiguous()
    )
    uz = (
        (KICK * torch.sin(2 * math.pi * z / nz) * torch.cos(math.pi * (x + 1) / nx))
        .expand(nz, ny, nx)
        .contiguous()
    )
    T0 = (x / (nx - 1)).expand(nz, ny, nx).contiguous()
    g = equilibrium_thermal_3d(T0, ux, uy, uz)
    tb, ident = 0.0, 0.0
    for _ in range(STEPS):
        g = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), ux, uy, uz, tau_T=TAU_T)
        g = stream_thermal_3d(g)
        g_ps = g
        g = apply_bc(g, 1.0, 0.0)
        d = _wall_deltas(g, g_ps)
        tot = (g.sum() - g_ps.sum()).item()
        tb += d["y0"] + d["yn"] + d["z0"] + d["zn"]
        ident = max(ident, abs(tot - sum(d[k] for k in WALLS)))
    return tb / STEPS, ident / STEPS


def test_3d_adiabat_exchange_net_zero():
    tb, _ = _adiabat_budget()
    assert abs(tb) < TOL, (
        f"net adiabatic-wall leak {tb:+.3e}/step -- the reflection rule "
        "must use the wrapped-around outgoing distribution (old rule: "
        "~5e-1/step at 16^3)"
    )


def test_3d_budget_identity_closes():
    _, ident = _adiabat_budget()
    assert ident < TOL, f"budget decomposition leak {ident:.3e}/step"


def test_3d_uniform_field_is_fixed_point():
    nz = ny = nx = 10
    T0 = torch.full((nz, ny, nx), 1.0, dtype=torch.float64)
    zeros = torch.zeros_like(T0)
    g = equilibrium_thermal_3d(T0, zeros, zeros, zeros)
    g = apply_bc(g, 1.0, 1.0)
    for _ in range(5):
        g = collide_thermal_bgk_3d(g, T0.clone(), zeros, zeros, zeros, tau_T=TAU_T)
        g = stream_thermal_3d(g)
        g = apply_bc(g, 1.0, 1.0)
    assert torch.equal(g, equilibrium_thermal_3d(T0, zeros, zeros, zeros)), (
        "uniform pure-diffusion state must stay bit-identical"
    )


def test_transient_slab_with_isothermal_and_insulated_walls(record_property):
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    # Isothermal x surfaces pass through nodes; transverse insulating walls
    # lie half a lattice link beyond the first and last nodes.
    tau, time = .8, 1/32
    alpha = (tau-.5)/4
    hot, cold = 1.2, .4
    wave = (math.pi, 2*math.pi, 4*math.pi/3)
    decay = math.exp(-alpha*sum(k*k for k in wave)*time)
    previous = None
    for n in (16, 32, 64):
        z, y, x = torch.meshgrid(
            (torch.arange(3*n//4, dtype=torch.float64)+.5)/n,
            (torch.arange(n//2, dtype=torch.float64)+.5)/n,
            torch.arange(n+1, dtype=torch.float64)/n, indexing="ij")
        transverse = torch.cos(wave[1]*y)*torch.cos(wave[2]*z)
        perturbation = .3*torch.sin(wave[0]*x)*transverse
        baseline = hot+(cold-hot)*x
        initial = baseline+perturbation
        zero = torch.zeros_like(initial)
        g = equilibrium_thermal_3d(initial, zero, zero, zero)
        for _ in range(int(time*n*n)):
            post = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), zero, zero, zero, tau)
            g = apply_bc(stream_thermal_3d(post), hot, cold)
        field = macroscopic_thermal_3d(g)
        # Second-order one-sided Fourier estimates, both positive toward +x.
        left = -alpha*n*(-3*field[..., 0]+4*field[..., 1]-field[..., 2])/2
        right = -alpha*n*(3*field[..., -1]-4*field[..., -2]+field[..., -3])/2
        varying = .3*wave[0]*decay*transverse[..., 0]
        errors = dict(field=(field-baseline-decay*perturbation).abs().max().item(),
                      left_flux=(left+alpha*(cold-hot+varying)).abs().max().item(),
                      right_flux=(right+alpha*(cold-hot-varying)).abs().max().item())
        # The last streaming step transports energy between neighboring nodes.
        # Compare its face fluxes with Fourier's law at link and time midpoints.
        dx, dt = 1/n, 1/n**2
        qx, qy, qz = thermal_link_fluxes_3d(post, spacing=dx, dt=dt)
        mid_decay = math.exp(-alpha*sum(k*k for k in wave)*(time-dt/2))
        amplitude = .3*alpha*mid_decay
        exact_x = alpha*(hot-cold) - amplitude*wave[0]*torch.cos(wave[0]*(x[..., :-1]+dx/2))*transverse[..., :-1]
        exact_y = (amplitude*wave[1]*torch.sin(wave[0]*x[:, :-1])*torch.sin(wave[1]*(y[:, :-1]+dx/2))
                   *torch.cos(wave[2]*z[:, :-1]))
        exact_z = (amplitude*wave[2]*torch.sin(wave[0]*x[:-1])*torch.cos(wave[1]*y[:-1])
                   *torch.sin(wave[2]*(z[:-1]+dx/2)))
        errors.update(x_link_flux=(qx-exact_x).abs().max().item(),
                      y_link_flux=(qy-exact_y).abs().max().item(),
                      z_link_flux=(qz-exact_z).abs().max().item())

        # The x boundary nodes are reservoirs. All other nodes, including
        # transverse wall nodes, have no source and zero exterior y/z flux.
        div_y = torch.diff(torch.nn.functional.pad(qy, (0, 0, 1, 1)), dim=1)
        div_z = torch.diff(torch.nn.functional.pad(qz, (0, 0, 0, 0, 1, 1)), dim=0)
        delta = (field-post.sum(dim=0))[..., 1:-1]
        balance = delta + (dt/dx)*(qx.diff(dim=2)+(div_y+div_z)[..., 1:-1])
        residual = balance.abs().max().item()
        record_property(f"n{n}_local_balance_error", residual)
        assert residual < 2e-14
        boundary = max((field[..., 0]-hot).abs().max().item(), (field[..., -1]-cold).abs().max().item())
        record_property(f"n{n}_wall_temperature_error", boundary)
        assert boundary < 1e-14
        for name, error in errors.items():
            record_property(f"n{n}_{name}_error", error)
            if previous is not None:
                order = math.log2(previous[name]/error)
                record_property(f"n{n}_{name}_order", order)
                if name == "field" or name.endswith("_link_flux"):
                    assert order > 1.9
                else:
                    assert order > (1.65 if n == 64 else 1.3)
        previous = errors


def test_cumulative_link_heat_and_slab_storage(record_property):
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    # Unit-length slab with constant x-wall temperatures, insulated transverse
    # walls and unit-area reporting. A decaying mode gives nonzero net cooling.
    tau, capacity, end_time = .8, 2.3, .125
    alpha = (tau-.5)/4
    hot, cold = 1.2, .4
    amplitude = .35
    previous = None
    for n in (16, 32, 64):
        dx, dt = 1/n, 1/n**2
        x = torch.arange(n+1, dtype=torch.float64)*dx
        initial = (hot+(cold-hot)*x+amplitude*torch.sin(math.pi*x)).expand(3, 4, -1)
        zero = torch.zeros_like(initial)
        g = equilibrium_thermal_3d(initial, zero, zero, zero)
        heat = torch.zeros_like(x[:-1])
        faces = x[:-1]+.5*dx
        worst_cell = worst_slab = 0.
        for step in range(1, int(end_time/dt)+1):
            post = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), zero, zero, zero, tau)
            qx, _, _ = thermal_link_fluxes_3d(post, dt=dt, spacing=dx, volumetric_heat_capacity=capacity)
            heat += dt*qx.mean(dim=(0, 1))
            g = apply_bc(stream_thermal_3d(post), hot, cold)
            delta = macroscopic_thermal_3d(g)-initial
            stored = capacity*dx*delta[..., 1:-1].mean(dim=(0, 1))
            worst_cell = max(worst_cell, (stored+heat.diff()).abs().max().item())
            worst_slab = max(worst_slab, abs((stored.sum()-heat[0]+heat[-1]).item()))
        # Integrate -C*alpha*T_x in time at each face. Integrate the change
        # in C*T separately in space between the two reservoir-adjacent links.
        decay_change = math.expm1(-alpha*math.pi**2*end_time)
        exact_heat = capacity*(alpha*(hot-cold)*end_time
                               + amplitude*torch.cos(math.pi*faces)*decay_change/math.pi)
        exact_storage = capacity*amplitude*decay_change*(
            math.cos(math.pi*faces[0])-math.cos(math.pi*faces[-1]))/math.pi
        errors = dict(link_heat=(heat-exact_heat).abs().max().item(),
                      slab_storage=abs(stored.sum().item()-exact_storage))
        record_property(f"n{n}_max_cell_balance_J_m2", worst_cell)
        record_property(f"n{n}_max_slab_balance_J_m2", worst_slab)
        assert max(worst_cell, worst_slab) < 1e-12
        for name, error in errors.items():
            record_property(f"n{n}_{name}_error_J_m2", error)
            if previous is not None:
                order = math.log2(previous[name]/error)
                record_property(f"n{n}_{name}_order", order)
                assert order > 1.8
        previous = errors


def test_reversing_section_heat_in_cooling_slab(record_property):
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    # At x=1/4 the initial sine gradient dominates the imposed wall gradient.
    # Its exponential decay leaves a positive steady flux after one reversal.
    capacity, tau, amplitude, end_time = 2.3, .8, .43, .5
    hot, cold = 1.2, .4
    alpha = (tau-.5)/4
    steady = capacity*alpha*(hot-cold)
    transient = capacity*alpha*amplitude*math.pi*math.cos(math.pi/4)
    decay = alpha*math.pi**2
    reversal = math.log(transient/steady)/decay

    def net_heat(t):
        return steady*t+transient*math.expm1(-decay*t)/decay

    exact_reverse = -net_heat(reversal)
    exact_forward = net_heat(end_time)+exact_reverse
    previous = None
    for n in (18, 34, 66):
        # n=4m+2 places the same physical section at a link midpoint on every grid.
        dx, dt = 1/n, 1/n**2
        x = torch.arange(n+1, dtype=torch.float64)*dx
        initial = (hot+(cold-hot)*x+amplitude*torch.sin(math.pi*x)).expand(3, 4, -1)
        zero = torch.zeros_like(initial)
        g = equilibrium_thermal_3d(initial, zero, zero, zero)
        rates = []
        for _ in range(round(end_time/dt)):
            post = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), zero, zero, zero, tau)
            qx, _, _ = thermal_link_fluxes_3d(post, spacing=dx, dt=dt, volumetric_heat_capacity=capacity)
            rates.append(qx[..., n//4].mean().item())
            g = apply_bc(stream_thermal_3d(post), hot, cold)
        flux = torch.tensor(rates, dtype=torch.float64)
        assert flux[0] < 0 and flux[-1] > 0
        crossings = torch.nonzero((flux[:-1] < 0) != (flux[1:] < 0)).flatten()
        assert crossings.numel() == 1
        j = crossings.item()
        crossing = dt*(j+.5-flux[j].item()/(flux[j+1]-flux[j]).item())
        forward = dt*flux.clamp_min(0).sum().item()
        reverse = dt*(-flux).clamp_min(0).sum().item()
        errors = dict(forward_J_m2=abs(forward-exact_forward), reverse_J_m2=abs(reverse-exact_reverse),
                      reversal_s=abs(crossing-reversal))
        for name, error in errors.items():
            record_property(f"n{n}_{name}_error", error)
            if previous is not None:
                order = math.log(previous[name]/error)/math.log(n/previous_n)
                record_property(f"n{n}_{name}_order", order)
                assert order > 1.8
        previous, previous_n = errors, n
    record_property("reference_forward_J_m2", exact_forward)
    record_property("reference_reverse_J_m2", exact_reverse)
    record_property("reference_reversal_s", reversal)


def test_harmonic_walls_recover_temperature_and_link_flux_phasors(record_property):
    import cmath
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    # Independent continuum solution of alpha*U''=i*omega*U on [0,1].
    tau, period, capacity = .8, .5, 2.3
    alpha = (tau-.5)/4
    omega = 2*math.pi/period
    wave = cmath.sqrt(1j*omega/alpha)
    left, right = .4+.15j, -.2+.3j
    previous = None
    for n in (12, 24, 48, 96):
        dx, dt = 1/n, 1/n**2
        x = torch.arange(n+1, dtype=torch.float64)*dx
        face = x[:-1]+dx/2
        exact_t = (left*torch.sinh(wave*(1-x))+right*torch.sinh(wave*x))/cmath.sinh(wave)
        exact_q = capacity*alpha*wave*(left*torch.cosh(wave*(1-face))-right*torch.cosh(wave*face))/cmath.sinh(wave)
        initial = (1.+exact_t.real).expand(3, 4, -1)
        zero = torch.zeros_like(initial)
        g = equilibrium_thermal_3d(initial, zero, zero, zero)
        steps = round(period/dt)
        measured_t = torch.zeros_like(exact_t)
        measured_q = torch.zeros_like(exact_q)
        for step in range(1, 2*steps+1):
            post = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), zero, zero, zero, tau)
            phase = cmath.exp(1j*omega*step*dt)
            g = apply_bc(stream_thermal_3d(post), 1.+(left*phase).real, 1.+(right*phase).real)
            if step > steps:
                # Temperatures are at step ends; link transport spans the step.
                qx, _, _ = thermal_link_fluxes_3d(post, spacing=dx, dt=dt, volumetric_heat_capacity=capacity)
                measured_q += qx.mean((0, 1))*cmath.exp(-1j*omega*(step-.5)*dt)*(2/steps)
                measured_t += (macroscopic_thermal_3d(g).mean((0, 1))-1.)*phase.conjugate()*(2/steps)
        errors = dict(temperature_K=(measured_t-exact_t).abs().max().item(),
                      flux_W_m2=(measured_q-exact_q).abs().max().item())
        for name, error in errors.items():
            record_property(f"n{n}_{name}_error", error)
            if previous is not None:
                order = math.log2(previous[name]/error)
                record_property(f"n{n}_{name}_order", order)
                assert order > {24: 1.2, 48: 1.5, 96: 1.75}[n]
        previous = errors


def test_fixed_wall_pulse_cooldown_against_sine_series(record_property):
    from tensorlbm.thermal_common import thermal_link_fluxes_3d

    # Difference of delayed unit wall steps in a unit slab, initially at zero.
    # After shutoff both walls are cold and the constant-in-time lift cancels.
    tau, shutoff, end = .8, 1/16, 1/4
    alpha = (tau-.5)/4
    wave = math.pi*torch.arange(1, 81, dtype=torch.float64)
    rate = alpha*wave.square()
    amplitude = 2*torch.exp(-rate*(end-shutoff))*(-torch.expm1(-rate*shutoff))/wave
    start, capacity = 1/8, 2.3
    start_amplitude = 2*torch.exp(-rate*(start-shutoff))*(-torch.expm1(-rate*shutoff))/wave
    integrated = start_amplitude*(-torch.expm1(-rate*(end-start)))/rate
    previous = None
    # Odd multiples of 16 put both probes exactly at kinetic link midpoints.
    probes = torch.tensor([29/32, 31/32], dtype=wave.dtype)
    reference = -capacity*alpha*(torch.cos(probes[:, None]*wave) @ (wave*integrated))
    previous_probe_errors = None
    for n in (16, 32, 48, 64, 80):
        dt = 1/n**2
        x = torch.arange(n+1, dtype=torch.float64)/n
        initial = torch.zeros((3, 4, n+1), dtype=x.dtype)
        g = equilibrium_thermal_3d(initial, initial, initial, initial)
        g = apply_bc(g, 1., 0.)
        heat = torch.zeros_like(x[:-1])
        for step in range(1, round(end/dt)+1):
            post = collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), initial, initial, initial, tau)
            if step > round(start/dt):
                qx, _, _ = thermal_link_fluxes_3d(post, dt=dt, spacing=1/n, volumetric_heat_capacity=capacity)
                heat += dt*qx.mean((0, 1))
            g = apply_bc(stream_thermal_3d(post), float(step*dt < shutoff), 0.)
            if step == round(start/dt):
                start_field = macroscopic_thermal_3d(g).mean((0, 1))
        measured = macroscopic_thermal_3d(g).mean((0, 1))
        exact = torch.sin(x[:, None]*wave) @ amplitude
        error = (measured-exact).abs().max().item()
        record_property(f"n{n}_temperature_error_K", error)
        faces = .5*(x[:-1]+x[1:])
        exact_heat = -capacity*alpha*(torch.cos(faces[:, None]*wave) @ (wave*integrated))
        link_error = (heat-exact_heat).abs().max().item()
        stored = capacity*(measured-start_field)[1:-1].sum()/n
        balance = abs((stored-heat[0]+heat[-1]).item())
        record_property(f"n{n}_link_heat_error_J_m2", link_error)
        record_property(f"n{n}_storage_balance_J_m2", balance)
        record_property(f"n{n}_east_link_J_m2", heat[-1].item())
        record_property(f"n{n}_exact_east_link_J_m2", exact_heat[-1].item())
        assert heat[0] < 0 and balance < 1e-12
        if n == 64:
            assert heat[-1] > 0
        if n in (16, 32, 64):
            if previous is not None:
                assert previous[0]/error > 3.5 and previous[1]/link_error > 2.5
            previous = error, link_error
        if n in (16, 48, 80):
            indices = (probes*n).long()
            assert torch.equal(faces[indices], probes)
            relative = ((heat[indices]-reference)/reference).abs()
            for numerator, computed, target, rel in zip((29, 31), heat[indices], reference, relative):
                record_property(f"x{numerator}_over_32_reference_J_m2", target.item())
                record_property(f"n{n}_x{numerator}_over_32_heat_J_m2", computed.item())
                record_property(f"n{n}_x{numerator}_over_32_relative_error", rel.item())
            if previous_probe_errors is not None:
                assert torch.all(relative < previous_probe_errors)
                assert torch.all(heat[indices] > 0)
            previous_probe_errors = relative

"""Tests for the 3D thermal lattice Boltzmann module."""

from __future__ import annotations

import torch

from tensorlbm import equilibrium3d
from tensorlbm.thermal3d import (
    ThermalCavity3DConfig,
    apply_buoyancy_force_3d,
    collide_thermal_bgk_3d,
    equilibrium_thermal_3d,
    macroscopic_thermal_3d,
    run_thermal_cavity_3d,
    stream_thermal_3d,
)


def test_thermal3d_equilibrium_shape() -> None:
    nz, ny, nx = 4, 5, 6
    T = torch.ones((nz, ny, nx))
    zeros = torch.zeros_like(T)
    geq = equilibrium_thermal_3d(T, zeros, zeros, zeros)
    assert geq.shape == (7, nz, ny, nx)


def test_thermal3d_macroscopic() -> None:
    nz, ny, nx = 4, 5, 6
    T = torch.rand((nz, ny, nx)) + 0.5
    zeros = torch.zeros_like(T)
    geq = equilibrium_thermal_3d(T, zeros, zeros, zeros)
    assert torch.allclose(macroscopic_thermal_3d(geq), T, atol=1e-5)


def test_thermal3d_bgk_conserves_energy() -> None:
    nz, ny, nx = 4, 5, 6
    T = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand((nz, ny, nx)) * 0.01
    uy = torch.rand((nz, ny, nx)) * 0.01
    uz = torch.rand((nz, ny, nx)) * 0.01
    g = equilibrium_thermal_3d(T, ux, uy, uz) + 1e-3 * torch.rand((7, nz, ny, nx))
    T0 = g.sum(dim=0)
    gout = collide_thermal_bgk_3d(g, T0, ux, uy, uz, tau_T=0.8)
    assert torch.allclose(gout.sum(dim=0), T0, atol=1e-5)


def test_thermal3d_stream() -> None:
    nz, ny, nx = 4, 5, 6
    T = torch.rand((nz, ny, nx)) + 0.5
    zeros = torch.zeros_like(T)
    g = equilibrium_thermal_3d(T, zeros, zeros, zeros)
    assert torch.allclose(stream_thermal_3d(g).sum(), g.sum(), atol=1e-5)


def test_buoyancy_3d_shape() -> None:
    nz, ny, nx = 4, 5, 6
    rho = torch.ones((nz, ny, nx))
    zeros = torch.zeros_like(rho)
    f = equilibrium3d(rho, zeros, zeros, zeros)
    fout = apply_buoyancy_force_3d(f, rho, T_ref=1.0, beta=0.01)
    assert fout.shape == (19, nz, ny, nx)


def test_cavity_3d_smoke() -> None:
    result = run_thermal_cavity_3d(ThermalCavity3DConfig(nx=8, ny=8, nz=8, n_steps=5))
    assert "nusselt" in result


def test_d3q7_oblique_advection_diffusion_refinement(record_property):
    import numpy as np

    # Fixed physical box (1, 3/4, 1/2) and time, with dt=dx².
    tau, time = .8, 1/32
    velocity = np.array([.4, -.25, .3])
    wave = 2*np.pi*np.array([1., -4/3, 2.])
    alpha = (tau-.5)/4
    exact = np.exp(-time*(alpha*(wave @ wave)+1j*(velocity @ wave)))
    directions = np.vstack(([0., 0., 0.], np.eye(3), -np.eye(3)))
    weights = np.array([.25]+[.125]*6)
    previous = None
    for n in (16, 32, 64):
        z, y, x = torch.meshgrid(
            *(torch.arange(m, dtype=torch.float64) for m in (n//2, 3*n//4, n)), indexing="ij")
        phase = (wave[0]*x+wave[1]*y+wave[2]*z)/n
        initial = 1.25+torch.cos(phase)
        flow = [torch.full_like(initial, v/n) for v in velocity]
        g = equilibrium_thermal_3d(initial, *flow)
        steps = int(time*n*n)
        for _ in range(steps):
            g = stream_thermal_3d(collide_thermal_bgk_3d(g, macroscopic_thermal_3d(g), *flow, tau))
        field = macroscopic_thermal_3d(g)
        harmonic = torch.exp(1j*phase)
        observed = (2*(field-1.25)*harmonic.conj()).mean().item()

        # Independently propagate all seven Fourier populations from equilibrium.
        eq = weights*(1+4*(directions @ velocity)/n)
        collision = (1-1/tau)*np.eye(7)+eq[:, None]/tau
        propagator = np.exp(-1j*(directions @ wave)/n)[:, None]*collision
        discrete = (np.linalg.matrix_power(propagator, steps) @ eq).sum()
        discrete_error = abs(observed-discrete)
        mean_error = abs(field.mean().item()-1.25)
        record_property(f"n{n}_discrete_error", discrete_error)
        record_property(f"n{n}_mean_error", mean_error)
        assert max(discrete_error, mean_error) < 1e-12
        errors = dict(decay_rate=abs(-np.log(abs(observed))/time-alpha*(wave @ wave)),
                      phase_rad=abs(np.angle(observed/exact)),
                      field=(field-1.25-(exact*harmonic).real).abs().max().item())
        for name, error in errors.items():
            record_property(f"n{n}_{name}_error", float(error))
            if previous is not None:
                order = np.log2(previous[name]/error)
                record_property(f"n{n}_{name}_order", float(order))
                assert order > 1.8
        previous = errors

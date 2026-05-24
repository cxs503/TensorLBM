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

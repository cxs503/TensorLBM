"""Tests for tensorlbm.simulation – LBMSimulation class."""

from __future__ import annotations

import pytest
import torch

from tensorlbm import LBMSimulation


class TestLBMSimulationInit:
    def test_default_construction(self) -> None:
        sim = LBMSimulation()
        assert sim.nx == 64
        assert sim.ny == 32
        assert sim.tau == 0.6
        assert sim.device == torch.device("cpu")

    def test_custom_grid(self) -> None:
        sim = LBMSimulation(nx=32, ny=16, tau=0.8)
        assert sim.nx == 32
        assert sim.ny == 16
        assert sim.tau == 0.8

    def test_distribution_shape(self) -> None:
        """Initial f must have shape (ny, nx, 9)."""
        sim = LBMSimulation(nx=10, ny=8)
        assert sim.f.shape == (8, 10, 9)

    def test_distribution_is_finite(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        assert torch.isfinite(sim.f).all()

    def test_initial_density_is_one(self) -> None:
        """Initial density (sum over directions) should equal 1 everywhere."""
        sim = LBMSimulation(nx=10, ny=8)
        rho = sim.f.sum(dim=-1)
        assert torch.allclose(rho, torch.ones(8, 10), atol=1e-6)

    def test_invalid_nx_raises(self) -> None:
        with pytest.raises(ValueError, match="nx"):
            LBMSimulation(nx=0)

    def test_invalid_ny_raises(self) -> None:
        with pytest.raises(ValueError, match="ny"):
            LBMSimulation(ny=-1)

    def test_invalid_tau_raises(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            LBMSimulation(tau=0.0)


class TestLBMSimulationMacroscopic:
    def test_output_shapes(self) -> None:
        nx, ny = 10, 8
        sim = LBMSimulation(nx=nx, ny=ny)
        rho, u = sim.macroscopic()
        assert rho.shape == (ny, nx)
        assert u.shape == (ny, nx, 2)

    def test_initial_density_all_ones(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        rho, _ = sim.macroscopic()
        assert torch.allclose(rho, torch.ones_like(rho), atol=1e-6)

    def test_initial_velocity_all_zeros(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        _, u = sim.macroscopic()
        assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)

    def test_finite_outputs(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        rho, u = sim.macroscopic()
        assert torch.isfinite(rho).all()
        assert torch.isfinite(u).all()


class TestLBMSimulationStep:
    def test_step_preserves_shape(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        shape_before = sim.f.shape
        sim.step()
        assert sim.f.shape == shape_before

    def test_step_produces_finite_values(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        sim.step()
        assert torch.isfinite(sim.f).all()

    def test_step_conserves_total_mass(self) -> None:
        """Total mass must be conserved (periodic domain)."""
        sim = LBMSimulation(nx=12, ny=10)
        mass_before = float(sim.f.sum().item())
        sim.step()
        mass_after = float(sim.f.sum().item())
        assert abs(mass_before - mass_after) < 1e-4

    def test_multiple_steps(self) -> None:
        sim = LBMSimulation(nx=12, ny=10)
        for _ in range(10):
            sim.step()
        assert torch.isfinite(sim.f).all()


class TestLBMSimulationRun:
    def test_run_returns_rho_and_u(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        rho, u = sim.run(steps=5)
        assert rho.shape == (8, 10)
        assert u.shape == (8, 10, 2)

    def test_run_produces_finite_values(self) -> None:
        sim = LBMSimulation(nx=10, ny=8)
        rho, u = sim.run(steps=5)
        assert torch.isfinite(rho).all()
        assert torch.isfinite(u).all()

    def test_run_default_steps(self) -> None:
        """run() with default steps=10 should complete without error."""
        sim = LBMSimulation(nx=8, ny=6)
        rho, u = sim.run()
        assert rho.shape == (6, 8)

    def test_run_zero_steps_returns_initial(self) -> None:
        sim = LBMSimulation(nx=8, ny=6)
        rho_before, u_before = sim.macroscopic()
        rho_after, u_after = sim.run(steps=0)
        assert torch.allclose(rho_before, rho_after, atol=1e-6)
        assert torch.allclose(u_before, u_after, atol=1e-6)

    def test_run_conserves_mass(self) -> None:
        sim = LBMSimulation(nx=12, ny=10, tau=0.6)
        mass_initial = float(sim.f.sum().item())
        sim.run(steps=20)
        mass_final = float(sim.f.sum().item())
        assert abs(mass_initial - mass_final) < 1e-3

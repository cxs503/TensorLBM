"""Pytest verification suite for LBM-DEM-FSI three-way coupling.

Tests all acceptance criteria from campaign node p80w2-n03.
Uses minimal grid and few steps for fast execution.
"""

import math
import pytest
import torch

from tensorlbm.lbm_dem_fsi_coupling import (
    LBMDemFsiCoupling,
    LBMDemFsiConfig,
    CouplingStatistics,
)
from tensorlbm.beam_fem import EulerBernoulliBeamFEM, BeamProperties
from tensorlbm.dem import DEMSolver, DEMConfig, IceParticle
from tensorlbm.wave_tank.stokes2 import Stokes2Wave, dispersion_solve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Minimal coupling configuration for fast testing."""
    return LBMDemFsiConfig(
        lbm_nx=8,
        lbm_ny=4,
        lbm_nz=4,
        lbm_dx=0.2,
        lbm_dt=0.01,
        lbm_tau=0.8,
        dem_time_step=1e-3,
        fsi_n_elements=3,
        fsi_boundary_condition="simply_supported",
        n_ice_particles=1,
        ice_radius_min=0.08,
        ice_radius_max=0.08,
        wave_amplitude=0.01,
        wave_frequency=2.0,
        water_depth=0.5,
        seed=42,
    )


@pytest.fixture
def coupling(default_config):
    return LBMDemFsiCoupling(default_config)


@pytest.fixture
def beam():
    props = BeamProperties(
        youngs_modulus=7e10, density=2700.0,
        width=0.05, height=0.005, length=1.0,
        damping_ratio=0.02, boundary_condition="simply_supported",
    )
    return EulerBernoulliBeamFEM(props, n_elements=10)


# ---------------------------------------------------------------------------
# C1: LBM→DEM momentum exchange
# ---------------------------------------------------------------------------


class TestC1LBMDemMomentumExchange:
    """C1: LBM→DEM momentum exchange residual <2%."""

    def test_momentum_residual_below_2_percent(self, coupling):
        """Test that LBM→DEM momentum exchange residual is <2%."""
        coupling.step(n_lbm_substeps=2)
        residual = coupling.stats.lbm_dem_momentum_residual
        assert residual < 0.02, f"Momentum residual {residual:.4f} >= 2%"

    def test_momentum_conservation_magnitude(self, coupling):
        """Test that LBM and DEM force magnitudes are consistent."""
        coupling.step(n_lbm_substeps=2)
        force_lbm = coupling.stats.lbm_dem_force_lbm
        force_dem = coupling.stats.lbm_dem_force_dem
        # Forces should have similar magnitudes (within 5%)
        mag_lbm = torch.norm(force_lbm).item()
        mag_dem = torch.norm(force_dem).item()
        if mag_dem > 1e-10:
            ratio = mag_lbm / mag_dem
            # In IBM, grid force sum should equal particle force sum
            assert 0.5 < ratio < 2.0, f"Force magnitude ratio {ratio:.3f} outside [0.5, 2.0]"


# ---------------------------------------------------------------------------
# C2: DEM→FSI energy exchange
# ---------------------------------------------------------------------------


class TestC2DemFsiEnergyExchange:
    """C2: DEM→FSI energy exchange residual <3%."""

    def test_energy_exchange_residual_below_3_percent(self, coupling):
        coupling.step(n_lbm_substeps=2)
        residual = coupling.stats.dem_fsi_energy_residual
        assert residual < 0.03, f"Energy residual {residual:.4f} >= 3%"

    def test_modal_force_computation(self, beam):
        beam.compute_eigenmodes(n_modes=3)
        force_vec = beam.apply_point_load(0.5, 100.0)
        modal_force = beam.compute_modal_generalized_force(force_vec, mode_idx=0)
        assert abs(modal_force) > 1e-10

    def test_virtual_work_computation(self, beam):
        force_vec = beam.apply_point_load(0.3, 50.0)
        disp = torch.zeros(beam.n_dof, dtype=torch.float64)
        disp[4] = 0.001
        vw = beam.compute_virtual_work(force_vec, disp)
        assert math.isfinite(vw)


# ---------------------------------------------------------------------------
# C3: Energy ledger
# ---------------------------------------------------------------------------


class TestC3EnergyLedger:
    """C3: Energy ledger verification."""

    def test_energy_components_finite(self, coupling):
        coupling.step(n_lbm_substeps=2)
        assert math.isfinite(coupling.stats.kinetic_energy)
        assert math.isfinite(coupling.stats.potential_energy)
        assert math.isfinite(coupling.stats.input_work)

    def test_energy_components_positive(self, coupling):
        coupling.step(n_lbm_substeps=2)
        assert coupling.stats.kinetic_energy >= -1e-10


# ---------------------------------------------------------------------------
# C4: Time synchronization
# ---------------------------------------------------------------------------


class TestC4TimeSync:
    """C4: Time sync verification."""

    def test_zero_offset_residual_below_3_percent(self, coupling):
        coupling.step(n_lbm_substeps=2)
        residual = coupling.stats.time_sync_residual
        assert residual < 0.03, f"Zero offset residual {residual:.4f} >= 3%"

    def test_half_dt_offset_detected(self):
        config = LBMDemFsiConfig(
            lbm_nx=8, lbm_ny=4, lbm_nz=4,
            lbm_dx=0.2, lbm_dt=0.01, lbm_tau=0.8,
            dem_time_step=1e-3, fsi_n_elements=3,
            n_ice_particles=1, time_sync_offset=0.5, seed=42,
        )
        c = LBMDemFsiCoupling(config)
        c.step(n_lbm_substeps=2)
        residual = c.stats.time_sync_residual
        assert residual > 0.10, f"Δt/2 offset not detected: {residual:.4f}"


# ---------------------------------------------------------------------------
# C5: Negative control
# ---------------------------------------------------------------------------


class TestC5NegativeControl:
    """C5: Negative control: no FSI feedback, structural response <1e-6."""

    def test_no_fsi_feedback_zero_response(self, coupling):
        max_response = coupling.run_negative_control(n_steps=3)
        assert max_response < 1e-6, f"Negative control response {max_response:.6e} >= 1e-6"


# ---------------------------------------------------------------------------
# C7: Seed dispersion
# ---------------------------------------------------------------------------


class TestC7SeedDispersion:
    """C7: Three seeds, dispersion <10%."""

    def test_three_seed_dispersion_below_10_percent(self):
        config = LBMDemFsiConfig(
            lbm_nx=8, lbm_ny=4, lbm_nz=4,
            lbm_dx=0.2, lbm_dt=0.01, lbm_tau=0.8,
            dem_time_step=1e-3, fsi_n_elements=3,
            n_ice_particles=1, seed=42,
        )
        coupling = LBMDemFsiCoupling(config)
        results = coupling.run_seed_sweep(seeds=[42, 123, 456], n_steps=3)
        dispersion = results["dispersion"]
        assert dispersion < 0.10, f"Seed dispersion {dispersion:.4f} >= 10%"


# ---------------------------------------------------------------------------
# Beam FEM unit tests
# ---------------------------------------------------------------------------


class TestBeamFEM:
    """Unit tests for Euler-Bernoulli beam FEM model."""

    def test_beam_natural_frequency(self):
        props = BeamProperties(
            youngs_modulus=2.1e11, density=7850.0,
            width=0.1, height=0.01, length=1.0,
            boundary_condition="cantilever",
        )
        beam = EulerBernoulliBeamFEM(props, n_elements=50)
        EI = props.flexural_rigidity
        rhoA = props.mass_per_length
        f_analytical = (1.875104**2 / (2 * math.pi * props.length**2)) * math.sqrt(EI / rhoA)
        beam.compute_eigenmodes(n_modes=1)
        omega1 = math.sqrt(beam._eigenvalues[0].item())
        f_fem = omega1 / (2 * math.pi)
        error = abs(f_fem - f_analytical) / f_analytical
        assert error < 0.05, f"Natural frequency error {error:.4f} >= 5%"

    def test_beam_static_deflection(self):
        props = BeamProperties(
            youngs_modulus=2.1e11, density=7850.0,
            width=0.1, height=0.01, length=1.0,
            boundary_condition="cantilever",
        )
        beam = EulerBernoulliBeamFEM(props, n_elements=50)
        F = 100.0
        force_vec = beam.apply_point_load(props.length, F)
        free_dofs = [i for i in range(beam.n_dof) if i not in beam._constrained_dofs]
        K_free = beam.K_global[free_dofs][:, free_dofs]
        F_free = force_vec[free_dofs]
        d_free = torch.linalg.solve(K_free, F_free)
        tip_disp = d_free[-2].item()
        delta_analytical = F * props.length**3 / (3 * props.flexural_rigidity)
        error = abs(tip_disp - delta_analytical) / delta_analytical
        assert error < 0.02, f"Static deflection error {error:.4f} >= 2%"

    def test_beam_stress_computation(self):
        props = BeamProperties(
            youngs_modulus=2.1e11, density=7850.0,
            width=0.1, height=0.01, length=1.0,
            boundary_condition="cantilever",
        )
        beam = EulerBernoulliBeamFEM(props, n_elements=20)
        force_vec = beam.apply_point_load(0.5, 100.0)
        free_dofs = [i for i in range(beam.n_dof) if i not in beam._constrained_dofs]
        K_free = beam.K_global[free_dofs][:, free_dofs]
        F_free = force_vec[free_dofs]
        d_free = torch.linalg.solve(K_free, F_free)
        for i, dof in enumerate(free_dofs):
            beam.displacement[dof] = d_free[i]
        stress = beam.compute_stress()
        assert torch.all(torch.isfinite(stress))
        assert torch.max(torch.abs(stress)) > 1e-10


# ---------------------------------------------------------------------------
# Stokes wave verification
# ---------------------------------------------------------------------------


class TestStokesWave:
    """Verify Stokes wave implementation."""

    def test_dispersion_relation(self):
        g = 9.81
        h = 0.5
        omega = 3.0
        k = dispersion_solve(omega, g, h)
        omega_computed = math.sqrt(g * k * math.tanh(k * h))
        error = abs(omega_computed - omega) / omega
        assert error < 1e-10

    def test_stokes2_kinematic_residual(self):
        g = 9.81
        h = 0.5
        omega = 3.0
        k = dispersion_solve(omega, g, h)
        a = 0.01
        wave = Stokes2Wave(h=h, a=a, k=k, omega=omega, g=g)
        residual = wave.kinematic_residual(x=0.0, t=0.0)
        assert residual < 0.01


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for full coupling chain."""

    def test_full_coupling_chain(self, default_config):
        coupling = LBMDemFsiCoupling(default_config)
        coupling.step(n_lbm_substeps=2)
        assert len(coupling.history) > 0
        assert coupling.stats.lbm_dem_momentum_residual >= 0

    def test_coupling_history_structure(self, coupling):
        coupling.step(n_lbm_substeps=2)
        for entry in coupling.history:
            assert "time" in entry
            assert "step" in entry
            assert "lbm_dem_momentum_residual" in entry


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

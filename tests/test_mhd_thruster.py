"""Tests for the channel MHD thruster ledger (thrust mouths, energy guard, propeller)."""

from __future__ import annotations

import math

import pytest

from tensorlbm.mhd.thruster import (
    MHDThrusterConfig,
    propeller_ideal_efficiency,
    run_thruster_ledger,
)


@pytest.fixture(scope="module")
def ledger():
    """Closed-form ledger of the seawater reference design (no lattice run)."""
    return run_thruster_ledger(MHDThrusterConfig(), run_lattice=False)


class TestDesignPoint:
    def test_hartmann_number(self) -> None:
        cfg = MHDThrusterConfig()
        expected = cfg.b0_t * cfg.half_width_m * math.sqrt(cfg.sigma_s_per_m / cfg.mu)
        assert cfg.ha == pytest.approx(expected, rel=1e-12)
        assert cfg.ha == pytest.approx(12.494, rel=1e-3)

    def test_actuator_disk_contraction(self) -> None:
        cfg = MHDThrusterConfig()
        assert cfg.duct_speed_m_s == pytest.approx(
            0.5 * (cfg.ship_speed_m_s + cfg.jet_speed_m_s), rel=1e-15
        )
        assert cfg.jet_speed_m_s > cfg.ship_speed_m_s

    def test_validates_inputs(self) -> None:
        with pytest.raises(ValueError):
            MHDThrusterConfig(half_width_m=0.0)
        with pytest.raises(ValueError):
            MHDThrusterConfig(ny=127)


class TestThrustMouths:
    def test_two_mouths_close(self, ledger) -> None:
        assert ledger.thrust_closure_residual < 1e-6
        assert ledger.thrust_lorentz_n > 0.0
        assert ledger.thrust_momentum_n == pytest.approx(ledger.thrust_lorentz_n, rel=1e-6)

    def test_thrust_equals_pressure_force(self, ledger) -> None:
        # ideal actuator disk: T = dp * A
        assert ledger.thrust_lorentz_n == pytest.approx(
            ledger.delta_p_pa * ledger.config.area_m2, rel=1e-6
        )

    def test_lorentz_force_exceeds_drag(self, ledger) -> None:
        assert ledger.lorentz_force_n > ledger.wall_drag_n > 0.0
        assert ledger.lorentz_force_n - ledger.wall_drag_n == pytest.approx(
            ledger.thrust_lorentz_n, rel=1e-9
        )

    def test_thrust_density(self, ledger) -> None:
        assert ledger.thrust_density_pa == pytest.approx(
            ledger.thrust_lorentz_n / ledger.config.area_m2, rel=1e-12
        )


class TestEnergy:
    def test_energy_guard(self, ledger) -> None:
        assert ledger.energy_residual < 1e-6
        total = ledger.power_joule_w + ledger.power_viscous_w + ledger.power_hydraulic_w
        assert total == pytest.approx(ledger.power_electric_w, rel=1e-6)

    def test_joule_dissipation_dominates_seawater(self, ledger) -> None:
        # low conductivity of seawater: almost all electric power becomes heat
        assert ledger.power_joule_w > 0.9 * ledger.power_electric_w

    def test_efficiency_below_one(self, ledger) -> None:
        assert 0.0 < ledger.efficiency < 1.0
        assert ledger.efficiency == pytest.approx(
            ledger.thrust_lorentz_n * ledger.config.ship_speed_m_s / ledger.power_electric_w,
            rel=1e-12,
        )

    def test_mhd_efficiency_below_propeller(self, ledger) -> None:
        assert ledger.efficiency < ledger.efficiency_propeller_ideal
        assert ledger.efficiency < ledger.efficiency_jet_ideal


class TestPropellerComparison:
    def test_ideal_efficiency_formula(self) -> None:
        # zero thrust -> zero efficiency; large thrust -> small efficiency
        assert propeller_ideal_efficiency(0.0, 1.0, 1025.0, 2.0) == 0.0
        eta_low = propeller_ideal_efficiency(100.0, 0.24, 1025.0, 2.0)
        eta_high = propeller_ideal_efficiency(1000.0, 0.24, 1025.0, 2.0)
        assert 0.0 < eta_high < eta_low < 1.0

    def test_propeller_matches_froude_form(self, ledger) -> None:
        # for the same (T, A, U) the actuator disk gives eta = 2U/(U+u_e,prop)
        u_e = math.sqrt(
            ledger.config.ship_speed_m_s**2
            + 2.0 * ledger.thrust_lorentz_n / (ledger.config.rho_kg_m3 * ledger.config.area_m2)
        )
        assert ledger.efficiency_propeller_ideal == pytest.approx(
            2.0 * ledger.config.ship_speed_m_s / (ledger.config.ship_speed_m_s + u_e), rel=1e-12
        )


class TestLatticeVerification:
    def test_lattice_momentum_and_energy_closure(self) -> None:
        cfg = MHDThrusterConfig(ny=64, nu_lattice=0.1, steps=60000, check_every=1000)
        led = run_thruster_ledger(cfg, run_lattice=True)
        lat = led.lattice
        assert lat["converged"]
        assert lat["u_mean_relative_error"] < 1e-2
        assert lat["closure_residual_lbm"] < 2e-2
        assert lat["energy_residual_lattice"] < 5e-2
        assert lat["wall_shear_coefficient_lbm"] == pytest.approx(
            lat["wall_shear_coefficient_exact"], rel=5e-2
        )

    def test_ledger_serialises(self, ledger) -> None:
        payload = ledger.to_dict()
        assert payload["thrust"]["closure_residual"] < 1e-6
        assert payload["efficiency"]["eta_total"] < 1.0
        assert payload["design"]["area_m2"] > 0.0

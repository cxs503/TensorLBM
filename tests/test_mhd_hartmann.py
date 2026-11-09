"""Tests for the MHD Hartmann duct surface (closed form + D2Q9 Lorentz forcing)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.mhd.hartmann import (
    HartmannConfig,
    ha_from_params,
    hartmann_mean_factor,
    hartmann_mean_velocity_coefficient,
    hartmann_normalised_profile,
    hartmann_shape_factor,
    hartmann_wall_shear_coefficient,
    poiseuille_profile,
    run_hartmann_channel,
)


class TestClosedForm:
    def test_mean_factor_matches_definition(self) -> None:
        for ha in (0.3, 1.0, 5.0, 10.0):
            assert hartmann_mean_factor(ha) == pytest.approx(1.0 - math.tanh(ha) / ha, rel=1e-14)

    def test_mean_factor_small_ha_series(self) -> None:
        # m(Ha) = Ha^2/3 - 2 Ha^4/15 + ..., so m(1e-6) ~ 3.33e-13
        assert hartmann_mean_factor(1e-6) == pytest.approx(1.0 / 3.0 * 1e-12, rel=1e-12)

    def test_shape_factor_boundary_conditions(self) -> None:
        y = np.array([-1.0, 0.0, 1.0])
        assert float(hartmann_shape_factor(y, 5.0)[0]) == pytest.approx(0.0, abs=1e-15)
        assert float(hartmann_shape_factor(y, 5.0)[2]) == pytest.approx(0.0, abs=1e-15)
        assert float(hartmann_shape_factor(y, 5.0)[1]) == pytest.approx(
            1.0 - 1.0 / math.cosh(5.0), rel=1e-14
        )

    def test_small_ha_has_no_cancellation(self) -> None:
        # naive 1 - cosh(ha y)/cosh(ha) loses all digits at ha = 1e-6
        val = float(hartmann_shape_factor(np.array([0.5]), 1e-6)[0])
        assert val == pytest.approx(1e-12 * (1.0 - 0.25) / 2.0, rel=1e-9)

    def test_poiseuille_two_thirds_limit(self) -> None:
        # u_mean / u_centre -> 2/3 as Ha -> 0
        ha = 1e-6
        centre = float(hartmann_shape_factor(np.array([0.0]), ha)[0])
        ratio = hartmann_mean_factor(ha) / centre
        assert abs(ratio - 2.0 / 3.0) / (2.0 / 3.0) <= 1e-6

    def test_normalised_profile_is_poiseuille_at_zero_ha(self) -> None:
        y = np.linspace(-0.9, 0.9, 19)
        np.testing.assert_allclose(
            hartmann_normalised_profile(y, 0.0), poiseuille_profile(y, 1.0), rtol=1e-9, atol=1e-12
        )

    def test_wall_shear_coefficient_limits(self) -> None:
        assert hartmann_wall_shear_coefficient(0.0) == pytest.approx(3.0)
        assert hartmann_wall_shear_coefficient(1e-3) == pytest.approx(3.0, rel=1e-6)
        # large Ha: tau_w H/(mu u_mean) -> Ha
        assert hartmann_wall_shear_coefficient(20.0) == pytest.approx(21.0526, rel=1e-4)

    def test_large_ha_asymptotic_slope(self) -> None:
        lo, hi = 20.0, 40.0
        slope = (
            math.log(hartmann_mean_velocity_coefficient(hi))
            - math.log(hartmann_mean_velocity_coefficient(lo))
        ) / (math.log(hi) - math.log(lo))
        assert abs(slope + 2.0) / 2.0 <= 0.03

    def test_ha_from_params(self) -> None:
        # B0 = 1 T, H = 0.2 m, sigma = 4 S/m, mu = 1025 * 1e-6 Pa s -> Ha ~ 12.49
        ha = ha_from_params(b0=1.0, half_width=0.2, sigma=4.0, nu=1e-6, rho0=1025.0)
        assert ha == pytest.approx(12.494, rel=1e-3)
        assert ha_from_params(0.0, 0.2, 4.0, 1e-6) == 0.0


class TestConfig:
    def test_rejects_odd_ny(self) -> None:
        with pytest.raises(ValueError):
            HartmannConfig(ny=33)

    def test_rejects_high_mach(self) -> None:
        with pytest.raises(ValueError):
            HartmannConfig(u_mean=0.5)

    def test_sigma_realises_hartmann_number(self) -> None:
        cfg = HartmannConfig(ny=64, ha=10.0, nu=0.1)
        ha = cfg.half_width * math.sqrt(cfg.sigma / cfg.mu)
        assert ha == pytest.approx(10.0, rel=1e-12)
        assert cfg.loading > cfg.u_mean

    def test_force_coefficient_is_sigma_b0_squared(self) -> None:
        # the Lorentz force density is sigma B0^2 (K-u): Ohm's law and the cross
        # product each contribute one B0, so sigma B0 must not be used instead
        cfg = HartmannConfig(ny=64, ha=8.0, nu=0.1, b0=2.0)
        assert cfg.magnetic_force_coefficient == pytest.approx(cfg.sigma * 4.0, rel=1e-14)
        assert cfg.magnetic_force_coefficient == pytest.approx(
            cfg.mu * 64.0 / cfg.half_width**2, rel=1e-12
        )

    def test_profile_is_invariant_under_b0_rescaling(self) -> None:
        # with Ha held fixed only sigma B0^2 enters the momentum balance, so the
        # same Ha must give the same profile for B0 = 1 and B0 = 2 (sigma/4)
        common = dict(ny=32, ha=5.0, nu=0.1, u_mean=0.01, steps=6000, check_every=500)
        r1 = run_hartmann_channel(HartmannConfig(b0=1.0, **common))
        r2 = run_hartmann_channel(HartmannConfig(b0=2.0, **common))
        assert r2.config.sigma == pytest.approx(0.25 * r1.config.sigma, rel=1e-12)
        np.testing.assert_allclose(r1.u_profile, r2.u_profile, rtol=1e-9, atol=1e-12)

    def test_poiseuille_control_derives_body_force(self) -> None:
        cfg = HartmannConfig(ny=64, ha=0.0, nu=0.1, u_mean=0.01)
        expected = 3.0 * cfg.mu * cfg.u_mean / cfg.half_width**2
        assert cfg.body_force == pytest.approx(expected, rel=1e-14)
        assert cfg.sigma == 0.0


class TestSolver:
    @pytest.fixture(scope="class")
    def hartmann_run(self):
        cfg = HartmannConfig(ny=48, ha=5.0, nu=0.1, u_mean=0.01, steps=40000, check_every=500)
        return run_hartmann_channel(cfg)

    def test_converges_and_matches_profile(self, hartmann_run) -> None:
        assert hartmann_run.converged, "steady state not reached"
        assert hartmann_run.rel_linf < 5e-3
        assert hartmann_run.rel_l2 < 2e-3
        assert hartmann_run.max_mach < 0.05

    def test_flow_rate_identity(self, hartmann_run) -> None:
        rel = abs(hartmann_run.u_mean - hartmann_run.u_mean_analytic) / abs(
            hartmann_run.u_mean_analytic
        )
        assert rel < 5e-3

    def test_wall_shear_matches_closed_form(self, hartmann_run) -> None:
        assert hartmann_run.wall_shear_coefficient == pytest.approx(
            hartmann_wall_shear_coefficient(5.0), rel=1e-2
        )

    def test_momentum_closure(self, hartmann_run) -> None:
        # Lorentz volume integral == momentum exchanged with the walls
        assert hartmann_run.closure_residual < 1e-6

    def test_wall_rows_and_symmetry(self, hartmann_run) -> None:
        # node centres sit half a cell from the wall, so the wall row is small but
        # not zero; the profile must however be symmetric and much slower there
        assert hartmann_run.u_profile[0] < 0.3 * hartmann_run.u_centre
        assert hartmann_run.u_profile[0] == pytest.approx(hartmann_run.u_profile[-1], rel=1e-6)
        assert hartmann_run.u_profile[0] < hartmann_run.u_profile[len(hartmann_run.u_profile) // 2]

    def test_negative_control_returns_poiseuille(self) -> None:
        cfg = HartmannConfig(ny=32, ha=0.0, nu=0.1, u_mean=0.01, steps=80000, check_every=500)
        res = run_hartmann_channel(cfg)
        assert res.converged
        assert res.rel_linf < 5e-3
        # B0 = 0 must strictly rebuild plane Poiseuille flow: u_mean/u_centre = 2/3
        ratio = res.u_mean / res.u_centre
        assert ratio == pytest.approx(2.0 / 3.0, abs=5e-3)
        assert res.lorentz_force == 0.0

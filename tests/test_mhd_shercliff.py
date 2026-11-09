"""Tests for the insulating-wall (Shercliff) MHD channel branch.

Two layers of verification:

* closed-form algebra (mean-velocity law, damping exponent, wall shear, energy
  closure, zero net current) - pure functions, no solver;
* a small number of LBM runs that check the *implementation* of the insulating
  closure (K = <u_x>, zero mean Lorentz force, current reversal) against the
  closed form inside the 2 % gate of the campaign node.

The expensive ladder (``Ha`` in {0,1,2,5,10,20} x ``ny`` in {64,128,256}) lives in
``scripts/run_mhd_shercliff.py``; here only the cheapest configurations are run so
that the suite stays in the one-minute class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.mhd.hartmann import (
    HartmannConfig,
    hartmann_mean_factor,
    hartmann_mean_velocity_coefficient,
    poiseuille_profile,
    run_hartmann_channel,
    shercliff_grad_factor,
)
from tensorlbm.mhd.shercliff import (
    SHERCLIFF_THRESHOLDS,
    ShercliffCase,
    interpolate_profile,
    run_shercliff_channel,
    shercliff_current_density,
    shercliff_efficiency_insulating,
    shercliff_grad_for_mean,
    shercliff_mean_velocity,
    shercliff_normalised_profile,
    shercliff_plan,
    shercliff_profile,
    shercliff_wall_shear_coefficient,
    evaluate_shercliff_criteria,
)

MU = 0.1
HALF = 16.0


class TestClosedForm:
    """Algebra of the insulating-wall solution (no LBM)."""

    def test_poiseuille_limit_of_mean_velocity_law(self) -> None:
        grad = 3.0e-5
        u = shercliff_mean_velocity(grad, 0.0, HALF, MU)
        assert u == pytest.approx(grad * HALF**2 / (3.0 * MU), rel=1e-12)

    def test_grad_factor_series_matches_exact_branch(self) -> None:
        # series branch (Ha < 0.05) must join the exact branch continuously
        lo = shercliff_grad_factor(0.0499)
        hi = shercliff_grad_factor(0.0501)
        assert lo == pytest.approx(hi, rel=1e-4)
        assert shercliff_grad_factor(0.0) == pytest.approx(3.0, rel=1e-12)

    def test_grad_mean_round_trip(self) -> None:
        for ha in (0.0, 0.5, 1.0, 3.0, 10.0):
            g = shercliff_grad_for_mean(0.02, ha, HALF, MU)
            assert shercliff_mean_velocity(g, ha, HALF, MU) == pytest.approx(0.02, rel=1e-10)

    def test_large_ha_damping_is_inverse_in_ha(self) -> None:
        """Insulating walls: u_mean ~ 1/Ha (one power less braking than Ha^-2)."""
        grad = 1.0e-4
        u20 = shercliff_mean_velocity(grad, 20.0, HALF, MU)
        u40 = shercliff_mean_velocity(grad, 40.0, HALF, MU)
        ratio = u20 / u40
        # 1.949 at Ha = 20/40, approaching 2 asymptotically (1/f is not yet pure
        # 1/Ha there because m(Ha) = 1 - tanh(Ha)/Ha still has a 1/Ha tail)
        assert 1.9 < ratio < 2.0
        f20, f40 = shercliff_grad_factor(20.0), shercliff_grad_factor(40.0)
        assert ratio == pytest.approx(f40 / f20, rel=1e-10)
        # The conducting-wall (electrode) law u ~ G H^2 m/(mu Ha^2) brakes twice as
        # hard over the same Ha range: ratio (m20/20^2)/(m40/40^2) = 3.90 > ratio.
        m20, m40 = hartmann_mean_factor(20.0), hartmann_mean_factor(40.0)
        cond_ratio = (m20 / 20.0**2) / (m40 / 40.0**2)
        assert 3.8 < cond_ratio < 4.0
        assert cond_ratio > ratio

    def test_wall_shear_coefficient(self) -> None:
        assert shercliff_wall_shear_coefficient(0.0) == pytest.approx(3.0, rel=1e-12)
        for ha in (0.5, 2.0, 5.0, 20.0):
            expect = ha * math.tanh(ha) / hartmann_mean_factor(ha)
            assert shercliff_wall_shear_coefficient(ha) == pytest.approx(expect, rel=1e-12)

    def test_profile_reduces_to_poiseuille(self) -> None:
        y = np.linspace(-1, 1, 201)
        u = shercliff_profile(y, 0.0, 0.01)
        assert np.max(np.abs(u - poiseuille_profile(y, 0.01))) < 1e-4 * 0.01
        assert np.allclose(shercliff_normalised_profile(y, 0.0), 1.5 * (1 - y * y), atol=1e-6)

    def test_zero_net_current_and_reversal(self) -> None:
        y = np.linspace(-1, 1, 200001)
        for ha in (0.5, 2.0, 10.0):
            j = shercliff_current_density(y, ha, 0.01)
            total = float(np.trapezoid(j, y)) if hasattr(np, "trapezoid") else float(np.trapz(j, y))
            assert abs(total) < 1e-6
            assert j[0] < 0.0 < j[len(j) // 2]  # wall layer vs core sign reversal

    def test_energy_closure_analytic(self) -> None:
        """Hydraulic input = Joule + viscous dissipation (the closed form is exact)."""
        for ha in (0.0, 0.5, 2.0, 5.0, 20.0):
            cfg = HartmannConfig(ny=32, ha=ha, u_mean=0.01, nu=MU, wall_mode="insulating")
            half = cfg.half_width
            grad = shercliff_grad_for_mean(0.01, ha, half, MU)
            e = shercliff_efficiency_insulating(ha, 0.01, grad, cfg.sigma, cfg.b0, MU, half)
            assert e["closure_relative"] < 1e-6
            assert 0.0 <= e["joule_fraction"] < 1.0
            assert e["joule_fraction"] + e["viscous_fraction"] == pytest.approx(1.0, rel=1e-6)

    def test_hartmann_mean_coefficient_unchanged(self) -> None:
        """Guard: the new series helper must not perturb the Hartmann normalisation."""
        assert hartmann_mean_velocity_coefficient(0.0) == pytest.approx(1.0 / 3.0, rel=1e-12)
        assert hartmann_mean_velocity_coefficient(1e-5) == pytest.approx(1.0 / 3.0, rel=1e-12)


class TestConfig:
    def test_wall_mode_validation(self) -> None:
        with pytest.raises(ValueError):
            HartmannConfig(ny=32, ha=2.0, wall_mode="superconducting")

    def test_insulating_body_force_is_the_closed_form_gradient(self) -> None:
        cfg = HartmannConfig(ny=32, ha=5.0, u_mean=0.01, nu=MU, wall_mode="insulating")
        grad = shercliff_grad_for_mean(0.01, 5.0, cfg.half_width, MU)
        assert cfg.body_force == pytest.approx(grad, rel=1e-12)
        # conducting default keeps the old behaviour (no body force for Ha > 0)
        assert HartmannConfig(ny=32, ha=5.0, u_mean=0.01, nu=MU).body_force == 0.0

    def test_interpolate_profile_round_trip(self) -> None:
        coarse = [0.1 * i for i in range(32)]
        fine = interpolate_profile(coarse, 64)
        assert len(fine) == 64
        assert fine[0] == pytest.approx(coarse[0], rel=0.05)
        assert fine[-1] == pytest.approx(coarse[-1], rel=0.05)

    def test_plan_covers_every_hartmann_number(self) -> None:
        plan = shercliff_plan((0.0, 1.0, 5.0, 20.0))
        assert set(plan) == {0.0, 1.0, 5.0, 20.0}
        assert all(len(v) >= 2 for v in plan.values())
        assert plan[20.0][-1] == 256


@pytest.fixture(scope="module")
def insulating_ha2():
    """Cheap Shercliff run (Ha=2, ny=32) shared by the solver tests."""
    case = ShercliffCase(ha=2.0, ny=32, u_mean=0.01, nu=MU, steps=200_000, check_every=1000,
                         tol=1e-9)
    return case, run_shercliff_channel(case)


class TestSolver:
    def test_profile_matches_closed_form(self, insulating_ha2) -> None:
        _, run = insulating_ha2
        assert run["converged"]
        assert run["rel_l2"] <= SHERCLIFF_THRESHOLDS["c_shercliff_profile_l2"]
        assert run["rel_linf"] <= SHERCLIFF_THRESHOLDS["c_shercliff_profile_linf"]
        assert run["rel_mean"] <= SHERCLIFF_THRESHOLDS["c_shercliff_mean"]

    def test_zero_mean_lorentz_force(self, insulating_ha2) -> None:
        """Insulating walls: the magnetic field cannot add net streamwise momentum."""
        _, run = insulating_ha2
        assert run["lorentz_mean_relative"] < 1e-6
        assert run["net_current_relative"] < 1e-6

    def test_current_reverses(self, insulating_ha2) -> None:
        _, run = insulating_ha2
        assert run["current_reversal_check"]["sign_reversal"] is True

    def test_wall_shear(self, insulating_ha2) -> None:
        _, run = insulating_ha2
        assert run["rel_wall_shear"] <= 2.0e-2

    def test_energy_closure_of_the_simulation(self, insulating_ha2) -> None:
        _, run = insulating_ha2
        assert run["energy"]["closure_relative"] < 1e-3

    def test_ha_zero_is_poiseuille(self) -> None:
        """Negative control: with Ha = 0 the insulating branch collapses onto Poiseuille."""
        case = ShercliffCase(ha=0.0, ny=32, u_mean=0.01, nu=MU, steps=200_000, check_every=1000,
                             tol=1e-9)
        run = run_shercliff_channel(case)
        y = np.asarray(run["y_hat"])
        u_poiseuille = poiseuille_profile(y, run["u_mean_lbm"])
        assert np.max(np.abs(np.asarray(run["u_profile"]) - u_poiseuille)) < 2e-3 * run["u_mean_lbm"]
        assert run["lorentz_mean_relative"] < 1e-12

    def test_prescribed_gradient_prediction(self) -> None:
        """Independent check: prescribe G, predict u_mean from the law alone."""
        ha, ny = 5.0, 32
        half = 0.5 * ny
        grad = shercliff_grad_for_mean(0.01, 0.0, half, MU)
        case = ShercliffCase(ha=ha, ny=ny, u_mean=0.01, nu=MU, grad=grad, steps=200_000,
                             check_every=1000, tol=1e-9)
        run = run_shercliff_channel(case)
        predicted = shercliff_mean_velocity(grad, ha, half, MU)
        assert abs(run["u_mean_lbm"] - predicted) / predicted <= 2.0e-2
        assert run["rel_l2"] <= 2.0e-2


def test_criteria_aggregation(tmp_path) -> None:
    """The criteria aggregator must flag a deliberately bad run as FAIL."""
    good = {
        "ha": 2.0,
        "ny": 64,
        "tau_wall_coefficient_lbm": shercliff_wall_shear_coefficient(2.0) * (1 + 9.0e-6),
        "tau_wall_coefficient_analytic": shercliff_wall_shear_coefficient(2.0),
        "rel_l2": 1e-3,
        "rel_linf": 1e-3,
        "rel_mean": 1e-3,
        "rel_wall_shear": 1e-3,
        "net_current_relative": 1e-14,
        "net_current_integral_relative": 1e-3,
        "lorentz_mean_relative": 1e-14,
        "u_mean_lbm": 0.01,
        "converged": True,
    }
    _g = shercliff_wall_shear_coefficient(2.0)
    medium = dict(good, ny=32, rel_l2=4e-3, rel_linf=4e-3, u_mean_lbm=0.0100015,
                  tau_wall_coefficient_lbm=_g * (1 + 3.7e-5))
    coarse = dict(good, ny=16, rel_l2=1.6e-2, rel_linf=1.6e-2, u_mean_lbm=0.010006,
                  tau_wall_coefficient_lbm=_g * (1 + 1.5e-4))
    ok = evaluate_shercliff_criteria({2.0: [coarse, medium, good]})
    assert ok["all_passed"] is True
    bad = dict(good, rel_l2=5e-2, rel_linf=5e-2)
    failed = evaluate_shercliff_criteria({2.0: [coarse, medium, bad]})
    assert failed["all_passed"] is False
    assert failed["verdicts"]["c_shercliff_profile_l2"] is False
    assert "c_shercliff_gci" in failed["verdicts"]
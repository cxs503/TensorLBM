"""Invariant tests for ``free_surface_verification`` (audit m16285).

The pre-existing tests in ``tests/test_free_surface_verification.py`` only assert
that keys exist ("the code executed"), which is why two tautological "verification"
cases could sit in the module unnoticed.  The tests below assert *invariants*:

1. the closed forms agree with the independent Wagner implementation in
   :mod:`tensorlbm.lng_membrane_sloshing.impact` (cross-implementation);
2. the force equals the integral of the pressure it ships (analytic closure);
3. the reported ``measured_force_slope`` does **not** follow a corrupted closed
   form (falsifiability: the measurement is an independent path);
4. the Wigley "dispersion check" is declared an algebraic identity and the case
   is not certified, while the geometry block coefficient converges to 4/9.

See ``docs/AUDIT_m16285_FREE_SURFACE_VERIFICATION.md``.
"""

import math

import numpy as np
import pytest

from tensorlbm.free_surface_verification import (
    WaterEntryConfig,
    WaterEntrySlamming,
    WigleyWaveConfig,
    WigleyWaveVerification,
    summarise_verifications,
)


def _we(angle=4.0, v=0.01, n_steps=1000):
    return WaterEntrySlamming(
        WaterEntryConfig(wedge_angle=angle, V_entry=v, n_steps=n_steps)
    )


# --- 1. cross-implementation against lng_membrane_sloshing.impact -----------


@pytest.mark.parametrize("angle", [2.0, 4.0, 20.0])
def test_force_slope_matches_independent_implementation(angle):
    """dF/dt must match wagner_slam_force_per_length from the LNG module.

    Fail-before: the audited revision used rho V^3 pi^3 / (8 tan^2 beta), i.e.
    exactly half of this value.
    """
    from tensorlbm.lng_membrane_sloshing.impact import (
        WagnerImpact,
        wagner_slam_force_per_length,
    )

    we = _we(angle=angle)
    t = 50.0
    ref = wagner_slam_force_per_length(
        t, WagnerImpact(v_impact=we.config.V_entry, beta_deg=angle, rho=1.0)
    )
    assert we.wagner_force_slope() == pytest.approx(ref / t, rel=1e-12)


def test_pressure_matches_independent_implementation():
    """Pointwise pressure must match wagner_pressure_profile (same xi convention)."""
    from tensorlbm.lng_membrane_sloshing.impact import (
        WagnerImpact,
        wagner_pressure_profile,
    )

    we = _we()
    imp = WagnerImpact(v_impact=we.config.V_entry, beta_deg=we.config.wedge_angle)
    t = 10.0
    x = we._probe_x(t)
    np.testing.assert_allclose(
        we.wagner_pressure_distribution(x, t),
        wagner_pressure_profile(x, t, imp),
        rtol=1e-12,
    )


# --- 2. analytic closure: force == integral of the shipped pressure ----------


def test_pressure_integral_closes_with_force():
    """The fitted slope of the quadrature force series must equal dF/dt.

    Fail-before: the pressure lacked the cdot/V = pi/(2 tan beta) factor, so
    int p dx / (dF/dt * t) was (4 tan beta)/pi = 0.089 at beta = 4 deg.
    """
    we = _we()
    res = we.run()
    closure = res["checks"]["pressure_integral_closure"]
    assert closure["relative_error"] < 2e-3, closure
    # direct spot check at one time
    t = 500.0
    F_quad = we.wagner_force_by_quadrature(t)
    assert F_quad == pytest.approx(we.wagner_force(t), rel=2e-3)


def test_pressure_from_potential_time_derivative():
    """p = -rho dphi/dt on the plate must reproduce the closed-form pressure."""
    we = _we()
    t = 200.0
    x = we._probe_x(t)
    p_fd = we.pressure_from_potential(x, t)
    p_cf = we.wagner_pressure_distribution(x, t)
    assert np.max(np.abs(p_fd - p_cf) / p_cf) < 1e-5


# --- 3. falsifiability: the "measurement" must not be the theory ------------


def test_measurement_is_not_generated_from_the_closed_form():
    """Corrupting the closed form must NOT move the measured slope.

    This is the anti-tautology invariant: the audited revision built its
    "measured" force from ``dF_dt_theory * t`` and then fitted a line to it, so
    ``relative_error`` was 1.4e-16 and the verdict could never be FAIL.
    """
    we = _we()
    base = we.run()
    assert base["verdict"] == "PASS"

    we.wagner_force_slope = lambda: 2.0 * base["wagner_force_slope"]  # corrupt theory
    corrupted = we.run()

    # measurement did not move (it comes from quadrature of p(x,t))
    assert corrupted["measured_force_slope"] == pytest.approx(
        base["measured_force_slope"], rel=1e-12
    )
    # ... and the closure check now fails by ~100 %
    assert corrupted["checks"]["pressure_integral_closure"]["pass"] is False
    assert corrupted["verdict"] == "FAIL"


def test_no_simulation_claim():
    """The class must not claim a solver-backed validation."""
    res = _we().run()
    assert res["solver_invoked"] is False
    assert res["simulation_verified"] is False
    assert res["calibration_status"] == "WITHHELD"
    assert res["verdict_scope"] == "analytic self-consistency only"


def test_force_slope_scaling():
    """dF/dt ~ V^3 and ~ 1/tan^2(beta) (the audited revision had the right
    far-field scaling but a factor-2 coefficient error)."""
    a = _we(angle=4.0, v=0.01).wagner_force_slope()
    b = _we(angle=4.0, v=0.02).wagner_force_slope()
    assert b / a == pytest.approx(8.0, rel=1e-12)
    c1 = _we(angle=4.0).wagner_force_slope()
    c2 = _we(angle=8.0).wagner_force_slope()
    expected = (math.tan(math.radians(4.0)) / math.tan(math.radians(8.0))) ** 2
    assert c2 / c1 == pytest.approx(expected, rel=1e-12)


# --- 4. Wigley: identity is not a verification, geometry is -----------------


def test_wigley_dispersion_is_an_identity_not_a_pass():
    res = WigleyWaveVerification(WigleyWaveConfig(L=80, B=8, T=5, Fr=0.3)).run()
    # the old "dispersion_check" is exactly 0 for any Fr/L
    assert res["dispersion_identity_residual"] < 1e-12
    assert res["dispersion_check_kind"] == "algebraic_identity"
    # ... therefore it must not certify the case
    assert res["verdict"] == "WITHHELD"
    assert res["simulation_verified"] is False


@pytest.mark.parametrize("fr", [0.2, 0.45])
def test_wigley_identity_holds_for_any_froude(fr):
    res = WigleyWaveVerification(WigleyWaveConfig(L=123.0, B=9.0, T=4.0, Fr=fr)).run()
    assert res["dispersion_identity_residual"] < 1e-12


def test_wigley_geometry_block_coefficient():
    """Discretised Wigley hull must integrate to Cb = 4/9 (second order)."""
    wig = WigleyWaveVerification(WigleyWaveConfig(L=80, B=8, T=5))
    check = wig.geometry_check()
    assert check["pass"] is True
    assert check["monotone_convergence"] is True
    assert abs(check["observed_order"] - 2.0) < 0.15
    assert check["finest_rel_error"] < 1e-4


# --- 5. aggregate summary must not inflate certification -------------------


def test_summary_does_not_fold_withheld_or_analytic_cases_into_pass():
    fake = {
        "dam_break": {"verdict": "PASS"},
        "wigley_wave": {"verdict": "WITHHELD", "solver_invoked": False},
        "water_entry": {"verdict": "PASS", "solver_invoked": False},
        "negative_controls": {
            "single_phase": {"verdict": "PASS"},
            "zero_sigma": {"verdict": "PASS"},
        },
    }
    summary = summarise_verifications(fake)
    assert summary["all_pass"] is True
    assert summary["scope"].startswith("solver-backed")
    assert "wigley_wave" in summary["withheld"]
    assert set(summary["analytic_only"]) == {"wigley_wave", "water_entry"}


def test_summary_propagates_failing_negative_control():
    fake = {
        "dam_break": {"verdict": "PASS"},
        "negative_controls": {
            "single_phase": {"verdict": "PASS"},
            "zero_sigma": {"verdict": "FAIL"},
        },
    }
    summary = summarise_verifications(fake)
    assert summary["all_pass"] is False
    assert summary["negative_control_cases"]["zero_sigma"] == "FAIL"
"""Verification tests for the zig-zag (overshoot) manoeuvre in tensorlbm.maneuvering.

Two independent routes are exercised on the KVLCC2-class linear sway-yaw model:

* ``simulate_zigzag`` -- event-driven numerical integration (helm reversals and
  heading extrema located *inside* a step by bisection);
* ``analytic_zigzag`` -- the exact piecewise-modal solution (closed-form roots).

The verification claim is *internal consistency + order of accuracy* (two
independent routes on the same ODE), **not** absolute accuracy against a real
ship: the hydrodynamic derivatives are a representative linear set, so absolute
overshoot angles remain WITHHELD (see docs/maneuvering_zigzag_20261107.md).
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.maneuvering import (
    NOMINAL_TANKER_PRIME,
    ManeuveringModel,
    analytic_zigzag,
    check_imo_msc137_zigzag,
    imo_msc137_zigzag_limits,
    prime_to_dimensional,
    simulate_zigzag,
)

# --------------------------------------------------------------------------- #
# KVLCC2-class large tanker, full scale (same parameters as
# benchmarks/verified/maneuvering_steady_turn/run.py)
# --------------------------------------------------------------------------- #
L = 320.0
B = 58.0
D = 20.8
CB = 0.81
RHO = 1000.0
U = 7.97  # m/s ~ 15.5 kn
MASS = RHO * CB * L * B * D
IZ = MASS * (0.25 * L) ** 2


def build(prime=None, **overrides) -> ManeuveringModel:
    """Build the tanker model, optionally with overridden prime derivatives."""
    p = dict(NOMINAL_TANKER_PRIME) if prime is None else dict(prime)
    dim = prime_to_dimensional(p, L, D, U, RHO)
    kwargs = dict(L=L, d=D, U=U, mass=MASS, inertia_z=IZ, rho=RHO, **dim)
    kwargs.update(overrides)
    return ManeuveringModel(**kwargs)


@pytest.fixture(scope="module")
def tanker() -> ManeuveringModel:
    return build()


# --------------------------------------------------------------------------- #
# 1. the two routes must agree (the verification claim)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_deg", [10.0, 20.0])
def test_numerical_route_matches_analytic_route(tanker, delta_deg):
    sim = simulate_zigzag(tanker, delta_deg, dt=0.05, integrator="rk4")
    ana = analytic_zigzag(tanker, delta_deg)
    assert sim.terminated == "ok"
    assert len(sim.overshoots_deg) == 2 and len(ana.overshoots_deg) == 2
    for i, (s, a) in enumerate(zip(sim.overshoots_deg, ana.overshoots_deg)):
        assert s == pytest.approx(a, rel=1e-4), f"overshoot #{i + 1}: {s} vs {a}"
    for s, a in zip(sim.reversal_times, ana.reversal_times):
        assert s == pytest.approx(a, rel=1e-5)
    for s, a in zip(sim.peak_times, ana.peak_times):
        assert s == pytest.approx(a, rel=1e-5)


def test_overshoot_is_a_fraction_of_the_rudder_angle(tanker):
    """A directionally stable ship overshoots by far less than the helm order."""
    res = analytic_zigzag(tanker, 20.0)
    assert 0.0 < res.first_overshoot_deg < 20.0
    assert 0.0 < res.second_overshoot_deg < 20.0


# --------------------------------------------------------------------------- #
# 2. physics sanity: linear model scale invariance, mirror symmetry
# --------------------------------------------------------------------------- #
def test_linear_model_is_scale_invariant_in_rudder_angle(tanker):
    """In the linear model the 10/10 and 20/20 tests are the same solution.

    ``psi(t) = delta * g(t)`` with ``g`` independent of ``delta``, so scaling the
    rudder *and* the ordered heading by two leaves every switch time unchanged and
    doubles every overshoot angle.  This is a property of the linear model (a real
    ship breaks it through nonlinear hull forces -- see the benchmark's
    cross-flow case); the test pins the property so a future change to the helm
    logic cannot hide behind it.
    """
    a10 = analytic_zigzag(tanker, 10.0)
    a20 = analytic_zigzag(tanker, 20.0)
    for s10, s20 in zip(a10.reversal_times, a20.reversal_times):
        assert s20 == pytest.approx(s10, rel=1e-9)
    for p10, p20 in zip(a10.peak_times, a20.peak_times):
        assert p20 == pytest.approx(p10, rel=1e-9)
    for o10, o20 in zip(a10.overshoots_deg, a20.overshoots_deg):
        assert o20 == pytest.approx(2.0 * o10, rel=1e-9)


def test_port_and_starboard_manoeuvres_are_mirror_images(tanker):
    stbd = simulate_zigzag(tanker, 10.0, dt=0.1)
    port = simulate_zigzag(tanker, -10.0, dt=0.1)
    assert port.overshoots_deg == pytest.approx(stbd.overshoots_deg, rel=1e-9)
    assert port.reversal_times == pytest.approx(stbd.reversal_times, rel=1e-6)
    assert port.events[0]["heading_deg"] == pytest.approx(-stbd.events[0]["heading_deg"], rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. order of accuracy + negative control on the discretisation
# --------------------------------------------------------------------------- #
def test_rk4_beats_euler_and_euler_is_first_order(tanker):
    ana = analytic_zigzag(tanker, 20.0).first_overshoot_deg
    coarse = simulate_zigzag(tanker, 20.0, dt=0.5, integrator="semi_implicit_euler")
    fine = simulate_zigzag(tanker, 20.0, dt=0.125, integrator="semi_implicit_euler")
    e_coarse = abs(coarse.first_overshoot_deg - ana)
    e_fine = abs(fine.first_overshoot_deg - ana)
    # dt / 4 for a first-order scheme -> error / ~4 (accept 2.5x .. 6x)
    assert 2.5 < e_coarse / e_fine < 6.0, (e_coarse, e_fine)
    rk = simulate_zigzag(tanker, 20.0, dt=0.125, integrator="rk4")
    assert abs(rk.first_overshoot_deg - ana) < 0.2 * e_fine


def test_negative_control_coarse_time_step_is_detectable(tanker):
    """A deliberately bad discretisation must be *caught* by the 1e-4 criterion."""
    ana = analytic_zigzag(tanker, 20.0)
    bad = simulate_zigzag(tanker, 20.0, dt=20.0, integrator="semi_implicit_euler")
    assert len(bad.overshoots_deg) >= 1
    rel_err = abs(bad.first_overshoot_deg - ana.first_overshoot_deg) / ana.first_overshoot_deg
    assert rel_err > 1e-2, rel_err
    good = simulate_zigzag(tanker, 20.0, dt=0.05, integrator="rk4")
    rel_good = abs(good.first_overshoot_deg - ana.first_overshoot_deg) / ana.first_overshoot_deg
    assert rel_good < 1e-5, rel_good


def test_crossflow_nonlinearity_breaks_scale_invariance():
    """A representative cross-flow term makes 20/20 != 2 x 10/10 (as in reality)."""
    dim = prime_to_dimensional(NOMINAL_TANKER_PRIME, L, D, U, RHO)
    m = build(Y_vv=abs(dim["Y_v"]) / U, N_vv=abs(dim["N_v"]) / U)
    s10 = simulate_zigzag(m, 10.0, dt=0.1)
    s20 = simulate_zigzag(m, 20.0, dt=0.1)
    ratio = s20.first_overshoot_deg / s10.first_overshoot_deg
    assert 1.5 < ratio < 1.99, ratio
    # the analytic route is linear-only: it must refuse rather than silently
    # drop the cross-flow terms
    with pytest.raises(ValueError, match="linear"):
        analytic_zigzag(m, 10.0)


# --------------------------------------------------------------------------- #
# 4. IMO MSC.137(76) criterion implementation
# --------------------------------------------------------------------------- #
def test_imo_10_10_limit_depends_on_length_speed_ratio():
    fast = imo_msc137_zigzag_limits(10.0, 10.0, L_over_V=8.0)
    slow = imo_msc137_zigzag_limits(10.0, 10.0, L_over_V=25.0)
    assert fast["applicable"] and fast["limits_deg"]["first"] == 10.0
    assert slow["applicable"] and slow["limits_deg"]["first"] == 20.0
    assert imo_msc137_zigzag_limits(10.0, 10.0, L_over_V=None)["applicable"] is False


def test_imo_20_20_limits_and_unsupported_combination():
    lim = imo_msc137_zigzag_limits(20.0, 20.0, L_over_V=L / U)
    assert lim["limits_deg"] == {"first": 25.0, "second": 40.0}
    odd = imo_msc137_zigzag_limits(20.0, 10.0, L_over_V=L / U)
    assert odd["applicable"] is False and "does not define" in odd["reason"]


def test_imo_gate_passes_nominal_tanker(tanker):
    verdict = check_imo_msc137_zigzag(analytic_zigzag(tanker, 20.0), L, U)
    assert verdict["pass"] is True
    assert all(c["pass"] for c in verdict["checks"])
    assert len(verdict["checks"]) == 2  # first + second overshoot


def test_imo_gate_rejects_degraded_course_keeper():
    """The gate must have teeth: a weakly damped ship exceeds the 25 deg limit.

    The family scales the yaw-damping derivative ``N'_r``; it is a *sensitivity*
    scan (the low-damping members give turning radii below one ship length and are
    therefore not physical ships), used only to prove the criterion discriminates.
    """
    nominal = check_imo_msc137_zigzag(analytic_zigzag(build(), 20.0), L, U)
    assert nominal["pass"] is True

    weak = dict(NOMINAL_TANKER_PRIME)
    weak["N_r"] *= 0.3
    degraded = check_imo_msc137_zigzag(analytic_zigzag(build(prime=weak), 20.0), L, U)
    assert degraded["pass"] is False
    first = degraded["checks"][0]
    assert first["measured"] > first["limit"]
    assert first["margin_deg"] < 0.0


def test_imo_gate_is_monotone_in_yaw_damping():
    """Overshoot must grow as yaw damping falls (the gate's discriminating axis)."""
    vals = []
    for scale in (1.0, 0.7, 0.5):
        p = dict(NOMINAL_TANKER_PRIME)
        p["N_r"] *= scale
        m = build(prime=p)
        vals.append(analytic_zigzag(m, 20.0).first_overshoot_deg)
    assert vals[0] < vals[1] < vals[2], vals
    assert math.isfinite(vals[-1])
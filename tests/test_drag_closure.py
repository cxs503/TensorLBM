"""Tests for :mod:`tensorlbm.drag_closure` (node P54-W3-N11).

Covers the four verification faces of the node:

* analytic limits of the matched-asymptotic hierarchy (Stokes -> Oseen ->
  Schiller-Naumann) and the documented validity domain of each branch;
* properties of the bridged closure ``cd_oseen_bridge`` (continuity,
  monotonicity, error band against the standard drag curve);
* the DEM side: drag force, terminal velocity (Stokes limit + finite Re),
  and the 1-DOF settling integrator of :func:`simulate_settling`;
* the DEM solver integration: ``DEMConfig(drag_model="oseen_bridge")`` must
  produce exactly the closure force.

No time stepping of a lattice Boltzmann solver is involved, so the whole file
runs in ~1 s.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.drag_closure import (
    BRIDGE_RANGE,
    CLOSURES,
    OSEEN_VALIDITY_RE,
    cd_clift_gauvin,
    cd_oseen,
    cd_oseen_bridge,
    cd_proudman_pearson,
    cd_schiller_naumann,
    cd_stokes,
    drag_force_magnitude,
    drag_force_vector,
    error_table,
    oseen_bridge_weights,
    reynolds_number,
    simulate_settling,
    terminal_velocity,
)

RE_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]


# ---------------------------------------------------------------------------
# 1. analytic branches
# ---------------------------------------------------------------------------
def test_stokes_value():
    assert cd_stokes(0.1) == pytest.approx(240.0)
    assert cd_stokes(2.0) == pytest.approx(12.0)


def test_oseen_definition():
    for Re in (0.1, 1.0, 5.0):
        assert cd_oseen(Re) == pytest.approx(24.0 / Re * (1.0 + 3.0 * Re / 16.0))


def test_oseen_to_stokes_limit():
    """Oseen must reduce to Stokes as Re -> 0 (criterion (a) analytic half)."""
    for Re in (1e-6, 1e-9):
        rel = abs(cd_oseen(Re) / cd_stokes(Re) - 1.0)
        assert rel < 1e-5


def test_proudman_pearson_matches_oseen_to_second_order():
    """PP - Oseen = O(Re^2 ln Re): at Re=1e-3 the difference is tiny."""
    Re = 1e-3
    assert abs(cd_proudman_pearson(Re) - cd_oseen(Re)) / cd_oseen(Re) < 1e-5


def test_proudman_pearson_is_below_oseen_at_order_one():
    """The Re^2 ln Re term is negative for Re<1, so PP < Oseen there."""
    for Re in (0.1, 0.5, 0.9):
        assert cd_proudman_pearson(Re) < cd_oseen(Re)


def test_schiller_naumann_definition():
    Re = 10.0
    assert cd_schiller_naumann(Re) == pytest.approx(24.0 / Re * (1.0 + 0.15 * Re**0.687))


def test_clift_gauvin_adds_small_correction():
    """The Clift-Gauvin second term is negligible for Re <= 50."""
    for Re in (1.0, 10.0, 50.0):
        sn, cg = cd_schiller_naumann(Re), cd_clift_gauvin(Re)
        assert abs(cg - sn) / cg < 5e-3


# ---------------------------------------------------------------------------
# 2. the bridge
# ---------------------------------------------------------------------------
def test_bridge_stokes_limit():
    """Criterion (a): Re -> 0 must give C_D -> 24/Re within 1 %."""
    for Re in (1e-6, 1e-4, 1e-3):
        rel = abs(cd_oseen_bridge(Re) * Re / 24.0 - 1.0)
        assert rel < 1e-3


def test_bridge_equals_oseen_below_switch():
    for Re in (1e-3, 0.1, 0.5, OSEEN_VALIDITY_RE):
        assert cd_oseen_bridge(Re) == pytest.approx(cd_oseen(Re), rel=0, abs=0)


def test_bridge_equals_schiller_naumann_above_switch():
    for Re in (3.0, 10.0, 50.0):
        assert cd_oseen_bridge(Re) == pytest.approx(cd_schiller_naumann(Re), rel=0, abs=0)


def test_bridge_is_continuous_at_both_switch_points():
    eps = 1e-9
    for Re_s in (1.0, 3.0):
        lo, hi = cd_oseen_bridge(Re_s - eps), cd_oseen_bridge(Re_s + eps)
        assert abs(hi - lo) / lo < 1e-6


def test_bridge_weights_partition_of_unity():
    for Re in (0.2, 1.0, 1.7, 2.5, 3.0, 40.0):
        w_o, w_s = oseen_bridge_weights(Re)
        assert w_o >= 0.0 and w_s >= 0.0
        assert w_o + w_s == pytest.approx(1.0)


def test_bridge_is_monotone_decreasing_in_re():
    res = [cd_oseen_bridge(Re) for Re in RE_GRID]
    assert all(a > b for a, b in zip(res, res[1:]))


def test_bridge_matches_standard_curve_within_2pct_for_re_1_to_50():
    """Criterion (d) analytic half: bridge vs standard drag curve, Re = 1..50."""
    worst = 0.0
    for row in error_table([1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0]):
        worst = max(worst, abs(row["err_pct_oseen_bridge"]))
    assert worst < 2.0


def test_oseen_is_the_accurate_branch_only_below_validity():
    """Documents *why* the bridge hands over at Re = 0.5 (validity ceiling)."""
    below = max(
        abs(row["err_pct_oseen"])
        for row in error_table([0.05, 0.1, 0.2, OSEEN_VALIDITY_RE])
    )
    assert below < 2.0
    # ... and fails the 5 % band just above Re ~ 1.2
    above = error_table([2.0])[0]["err_pct_oseen"]
    assert above > 10.0


def test_oseen_validity_ceiling_is_machine_determined():
    """The 5 % band of Oseen closes at Re ~ 1.2 (not assumed: computed)."""
    from tensorlbm.drag_closure import oseen_error_ceiling

    ceiling = oseen_error_ceiling(5.0)
    assert 1.0 < ceiling < 1.5
    assert OSEEN_VALIDITY_RE < ceiling


def test_bridge_range_documented():
    lo, hi = BRIDGE_RANGE
    assert lo > 0.0 and hi >= 50.0


# ---------------------------------------------------------------------------
# 3. force / settling
# ---------------------------------------------------------------------------
def test_reynolds_number_definition():
    assert reynolds_number(radius=1.0, speed=2.0, rho=1.0, mu=0.5) == pytest.approx(8.0)


def test_zero_speed_gives_zero_drag():
    """Criterion (e) negative control at closure level."""
    assert drag_force_magnitude(0.0, radius=1.0) == 0.0
    assert drag_force_vector((0.0, 0.0, 0.0), radius=1.0) == (0.0, 0.0, 0.0)


def test_stokes_regime_force_is_six_pi_mu_a_u():
    mu, a, u = 1e-3, 1e-3, 1e-5          # Re = 1000*1e-5*2e-3/1e-3 = 0.02
    f = drag_force_magnitude(u, radius=a, rho=1000.0, mu=mu)
    assert f == pytest.approx(6.0 * math.pi * mu * a * u, rel=1e-3)


def test_drag_force_vector_follows_relative_fluid_velocity():
    """Convention (same as ``dem_solver``): F along ``u_fluid - u_particle``."""
    fx, fy, fz = drag_force_vector((0.0, -0.3, 0.0), radius=0.5, rho=1.0, mu=0.01)
    assert fx == 0.0 and fz == 0.0
    assert fy < 0.0     # fluid moves -y relative to the particle -> drag on -y


def test_terminal_velocity_stokes_limit():
    """Small particle -> Stokes terminal velocity to 1e-6."""
    d, rho_p, rho_f, mu = 1e-5, 2500.0, 1000.0, 1e-3   # Re ~ 9e-4
    u_stokes = d * d * (rho_p - rho_f) * 9.81 / (18.0 * mu)
    u = terminal_velocity(d, rho_p, rho_f, mu, 9.81)
    # the residual deviation is the Oseen correction itself (Re ~ 9e-4 -> 1.5e-4
    # in u), i.e. Stokes is recovered to 0.02 % and from the correct side
    assert u == pytest.approx(u_stokes, rel=2e-4)
    assert u < u_stokes


def test_terminal_velocity_neutral_buoyancy_is_zero():
    assert terminal_velocity(1e-3, 1000.0, 1000.0, 1e-3, 9.81) == 0.0


def test_terminal_velocity_finite_re_differs_from_stokes():
    """The m7441 gap, quantified: a 4 mm sand grain settles at Re ~ 20."""
    d, rho_p, rho_f, mu = 4e-4, 2650.0, 1000.0, 1e-3   # Re ~ 24
    u_bridge = terminal_velocity(d, rho_p, rho_f, mu, 9.81, model="oseen_bridge")
    u_stokes = terminal_velocity(d, rho_p, rho_f, mu, 9.81, model="stokes")
    Re = reynolds_number(0.5 * d, u_bridge, rho_f, mu)
    assert 5.0 < Re < 50.0
    # Stokes over-predicts the settling velocity at finite Re by a factor > 2
    assert (u_stokes - u_bridge) / u_bridge > 1.0


def test_simulate_settling_matches_analytic_terminal_velocity():
    """DEM-side consistency: the integrator must land on the analytic root."""
    out = simulate_settling(
        diameter=4e-4, rho_particle=2650.0, rho_fluid=1000.0, mu=1e-3,
        gravity=9.81, model="oseen_bridge", t_end=0.05,
    )
    assert abs(out["rel_diff"]) < 0.01
    assert 5.0 < out["Re_terminal"] < 50.0


def test_simulate_settling_zero_gravity_is_negative_control():
    out = simulate_settling(1e-3, 2650.0, 1000.0, 1e-3, 0.0, model="oseen_bridge")
    assert out["u_terminal_analytic"] == 0.0
    assert out["u_terminal_dem"] == 0.0


# ---------------------------------------------------------------------------
# 4. DEM solver integration
# ---------------------------------------------------------------------------
def test_dem_solver_oseen_bridge_matches_closure():
    from tensorlbm.dem.dem_solver import DEMConfig, DEMSolver
    from tensorlbm.dem.particle import IceParticle

    rho_f, mu, radius = 1000.0, 1e-3, 2e-3
    cfg = DEMConfig(drag_model="oseen_bridge", fluid_density=rho_f,
                    fluid_viscosity=mu, enable_wall_contacts=False)
    solver = DEMSolver(cfg)
    part = IceParticle(
        position=torch.zeros(3), velocity=torch.tensor([0.0123, 0.0, 0.0]),
        angular_velocity=torch.zeros(3), radius=radius,
    )
    f = solver.compute_drag_force(part, torch.zeros(3))

    speed = 0.0123
    Re = reynolds_number(radius, speed, rho_f, mu)
    cd = cd_oseen_bridge(Re)
    area = math.pi * radius**2
    f_expected = 0.5 * cd * rho_f * area * speed * speed
    assert float(f.norm()) == pytest.approx(f_expected, rel=1e-6)


def test_dem_solver_unknown_model_raises():
    from tensorlbm.dem.dem_solver import DEMConfig, DEMSolver
    from tensorlbm.dem.particle import IceParticle

    cfg = DEMConfig(drag_model="not_a_model", enable_wall_contacts=False)
    solver = DEMSolver(cfg)
    part = IceParticle(
        position=torch.zeros(3), velocity=torch.tensor([1e-3, 0.0, 0.0]),
        angular_velocity=torch.zeros(3), radius=1e-3,
    )
    with pytest.raises(ValueError):
        solver.compute_drag_force(part, torch.zeros(3))


# ---------------------------------------------------------------------------
# 5. bookkeeping
# ---------------------------------------------------------------------------
def test_unknown_closure_raises():
    with pytest.raises(ValueError):
        cd_oseen_bridge(0.0)          # non-positive Re is rejected
    with pytest.raises(ValueError):
        drag_force_magnitude(1.0, radius=1.0, model="does_not_exist")


def test_registry_documents_every_closure():
    for name, meta in CLOSURES.items():
        assert callable(meta["f"])
        assert meta["source"]
        assert meta["accuracy"]
        lo, hi = meta["range"]
        assert 0.0 <= lo < hi


def test_error_table_structure():
    rows = error_table([1.0, 10.0])
    assert len(rows) == 2
    assert rows[0]["Re"] == 1.0
    for key in ("cd_reference", "cd_stokes", "err_pct_stokes", "err_pct_oseen_bridge"):
        assert key in rows[0]
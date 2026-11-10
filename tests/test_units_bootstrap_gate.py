"""Acceptance tests for the lattice -> physical unit bootstrap gate.

The gate (:mod:`tensorlbm.units.gate`) is what protects reports from the
``F/2``-class defect: a wrong dimensionalisation coefficient rescales the
answer while the solver looks healthy.  These tests pin the campaign
criteria:

* round-trip closure ``<= 1e-14``
* four analytic anchors inside their pre-declared tolerances
  (Poiseuille 0.4 %, Couette 0.5 %, Stokes 2 %, Taylor-Green 2 %)
* three ``(dx, dt)`` groups agree on the dimensionless anchor number to
  ``<= 0.1 %`` with ``Ma < 0.1`` and ``tau in [0.51, 2]``
* five single-site coefficient mutations are all caught, every anchor catches
  at least one, and the pristine bridge is green for three seeds

The full gate is a session fixture so the ladder is simulated once.
"""

from __future__ import annotations

import json
import math

import pytest

from tensorlbm.units import (
    ANCHORS,
    GRID_SPREAD_TOL,
    MUTATIONS,
    ROUNDTRIP_TOL,
    TOLERANCES,
    LatticeBridge,
    bootstrap_bridge,
    build_case,
    evaluate,
    mutate,
)
from tensorlbm.units import gate as gate_mod


@pytest.fixture(scope="session")
def gate_report():
    return gate_mod.run_gate(seeds=(0, 1, 2))


# --------------------------------------------------------------------- units
@pytest.mark.unit
def test_bootstrap_bridge_derives_consistent_scales():
    bridge = bootstrap_bridge(
        length_phys=1.0e-3, velocity_phys=2.5e-3, nu_phys=1.0e-6, nx=48, u_lb=0.02
    )
    # dt is defined by u_lb = u_phys dt / dx
    assert bridge.velocity_to_lattice(2.5e-3) == pytest.approx(0.02, rel=1e-15)
    # the similarity invariant holds exactly
    nu_lb = bridge.viscosity_to_lattice(1.0e-6)
    re_lb = bridge.reynolds_lattice(0.02, 24.0, nu_lb)
    re_phys = bridge.reynolds_physical(
        bridge.velocity_to_phys(0.02), bridge.length_to_phys(24.0),
        bridge.viscosity_to_phys(nu_lb),
    )
    assert re_lb == pytest.approx(re_phys, rel=1e-15)
    assert 0.51 <= bridge.tau_from_viscosity(nu_lb) <= 2.0


@pytest.mark.unit
@pytest.mark.physics
def test_roundtrip_closure_of_every_site():
    rt = gate_mod._roundtrip()
    assert rt["max_rel_error"] <= ROUNDTRIP_TOL
    assert rt["passed"] is True
    for site in ("length", "time", "velocity", "viscosity", "density",
                 "stress", "force", "reynolds_identity"):
        assert rt["checks"][site] <= ROUNDTRIP_TOL, site


@pytest.mark.unit
def test_bootstrap_rejects_out_of_window_operating_points():
    with pytest.raises(ValueError):
        # huge lattice velocity -> Ma > 0.1
        bootstrap_bridge(length_phys=1.0, velocity_phys=1.0, nu_phys=1e-3,
                         nx=8, u_lb=0.5)
    with pytest.raises(ValueError):
        # vanishing lattice viscosity -> tau < 0.51
        bootstrap_bridge(length_phys=1.0e-3, velocity_phys=1.0, nu_phys=1e-9,
                         nx=8, u_lb=0.02)


@pytest.mark.unit
def test_mutation_sites_change_only_the_output_coefficient():
    base = bootstrap_bridge(length_phys=1.0e-3, velocity_phys=2.5e-3,
                            nu_phys=1e-6, nx=48, u_lb=0.02)
    x2 = mutate(base, "length_x2")
    assert x2.length_to_phys(1.0) == pytest.approx(2.0 * base.length_to_phys(1.0))
    # the set-up direction keeps the true scale (bug model: output site only)
    assert x2.length_to_lattice(1.0) == pytest.approx(base.length_to_lattice(1.0))
    for name in MUTATIONS:
        assert isinstance(mutate(base, name), LatticeBridge)
    with pytest.raises(KeyError):
        mutate(base, "no_such_site")


@pytest.mark.unit
@pytest.mark.physics
def test_analytic_anchors_hit_their_predeclared_tolerances(gate_report):
    for name in ANCHORS:
        rows = gate_report["anchors"][name]
        assert rows, f"no anchor cases for {name}"
        worst = max(r["worst_rel_error"] for r in rows)
        assert worst <= TOLERANCES[name], (name, worst, TOLERANCES[name])
        assert all(r["passed"] for r in rows)


@pytest.mark.integration
@pytest.mark.physics
def test_grid_invariance_over_three_dx_dt_groups(gate_report):
    for name, gi in gate_report["grid_invariance"].items():
        assert len(gi["grids"]) >= 3, (name, gi["grids"])
        assert gi["regime_ok"], (name, gi["regime"])
        assert gi["regime"]["Ma_max"] < 0.1, name
        if name == "stokes":
            # conversion-chain anchor: no time marching, so the tau window of
            # the marching anchors does not apply (documented limitation)
            assert gi["time_marching"] is False
        else:
            assert gi["time_marching"] is True
            assert 0.51 <= gi["regime"]["tau_min"]
            assert gi["regime"]["tau_max"] <= 2.0
        assert gi["spread"] <= GRID_SPREAD_TOL, (name, gi["spread"])


@pytest.mark.regression
@pytest.mark.physics
def test_mutation_battery_catches_all_five_mutations(gate_report):
    bat = gate_report["mutation_battery"]
    assert bat["caught_count"] == len(MUTATIONS), bat["missing"]
    assert bat["missing"] == []
    assert bat["anchors_without_catch"] == []
    for mut, anchors in bat["caught_by"].items():
        assert anchors, f"{mut} passed undetected"
    for name, hits in bat["anchor_catches"].items():
        assert hits, f"anchor {name} catches no mutation"


@pytest.mark.regression
@pytest.mark.unit
def test_mutation_battery_is_sensitive_at_physical_scale(gate_report):
    """A caught mutation must be off by a *physical* factor, not by roundoff."""
    matrix = gate_report["mutation_battery"]["matrix_worst_rel_error"]
    for mut, row in matrix.items():
        assert max(row.values()) > 0.1, f"{mut} only weakly detected: {row}"


@pytest.mark.regression
def test_no_false_positives_for_three_seeds(gate_report):
    fp = gate_report["false_positive"]
    assert fp["all_anchors_green"] is True
    assert len(fp["seeds"]) == 3
    for name in ANCHORS:
        rows = [r for r in gate_report["anchors"][name] if r["seed"] in fp["seeds"]]
        assert len(rows) >= len(fp["seeds"]), (name, len(rows))
        assert all(r["passed"] for r in rows), name


@pytest.mark.regression
def test_overall_gate_and_report_serialisable(gate_report, tmp_path):
    assert gate_report["overall_passed"] is True
    path = tmp_path / "gate_report.json"
    path.write_text(json.dumps(gate_report, sort_keys=True, default=float))
    reloaded = json.loads(path.read_text())
    assert reloaded["overall_passed"] is True
    assert reloaded["criteria"]["roundtrip_tolerance"] == ROUNDTRIP_TOL


@pytest.mark.unit
@pytest.mark.physics
def test_stokes_anchor_closes_analytically():
    """Conversion-chain anchor: drag must close to machine precision."""
    case = build_case("stokes", 0, 32)
    res = evaluate(case)
    assert abs(res["checks"]["drag"]) < 1e-14
    assert res["dimensionless"]["drag_over_reference"] == pytest.approx(1.0, rel=1e-14)


@pytest.mark.unit
@pytest.mark.physics
def test_taylor_green_decay_matches_two_nu_k_squared():
    case = build_case("taylor_green", 0, 32)
    res = evaluate(case)
    assert abs(res["checks"]["decay_rate"]) < TOLERANCES["taylor_green"]
    nu_implied = res["dimensionless"]["nu_implied_over_nu"]
    assert math.isfinite(nu_implied) and nu_implied == pytest.approx(1.0, rel=2e-2)
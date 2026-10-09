"""Long free-surface caller campaign ledger regression."""

from __future__ import annotations

import json

import pytest

from tensorlbm.dam_break_3d import (
    DamBreak3DConfig,
    _linear_drift_slope,
    _topology_drift_violates,
    _topology_event,
    run_dam_break_3d,
)


@pytest.mark.slow
def test_free_surface_dam_break_caller_writes_a_101_step_quality_curve(tmp_path) -> None:
    """The real public caller must retain state and publish every step's budget."""
    config = DamBreak3DConfig(
        nx=32,
        ny=16,
        nz=16,
        dam_width=10,
        fill_height=8,
        model="fs",
        n_steps=101,
        output_interval=101,
        gravity=1.0e-4,
        A=0.0,
        output_root=tmp_path,
        run_name="fs_101",
        # This telemetry regression exercises all 101 steps; strict campaign
        # defaults are tested separately and may intentionally fail closed.
        free_surface_topology_normalized_drift_tolerance=1.0,
        free_surface_relative_drift_slope_tolerance=1.0,
        free_surface_relative_cumulative_drift_tolerance=1.0,
    )
    metadata = json.loads((run_dam_break_3d(config) / "run_metadata.json").read_text())

    curve = metadata["free_surface_quality_curve"]
    assert len(curve) == 101
    assert [record["step"] for record in curve] == list(range(1, 102))
    assert metadata["free_surface_quality_gate"]["passed"] is True
    # Gravity-driven topology change is asserted by the dedicated guard test
    # below (currently xfail: the exchange receiver gate seals the column).

    required = {
        "mass_drift",
        "unexplained_residual",
        "paired_residual",
        "directLG",
        "conversion",
        "redistribution",
        "finite",
        "flags_finite",
        "time",
        "initial_mass",
        "instantaneous_mass_drift",
        "cumulative_drift",
        "relative_cumulative_drift",
        "cumulative_drift_average_rate",
        "relative_cumulative_drift_average_rate",
        "cumulative_drift_slope",
        "relative_cumulative_drift_slope",
        "drift_slope_window_steps",
        "conversion_redistribution_normalized_drift",
        "topology_event",
        "topology_count_changed",
        "liquid_cell_delta",
        "interface_cell_delta",
    }
    assert all(required <= record.keys() for record in curve)
    assert all(record["directLG"] == 0 and record["finite"] for record in curve)
    # Post-fix static column: no spurious phase conversions or mass
    # redistribution (the exchange-layer gate + H1/H2 keep it static).
    assert all(abs(record["conversion"]) <= 1.0e-5 for record in curve)
    assert all(abs(record["redistribution"]) <= 1.0e-5 for record in curve)
    assert [record["time"] for record in curve] == [float(step) for step in range(1, 102)]
    assert curve[0]["initial_mass"] == pytest.approx(1008.0)
    assert curve[0]["cumulative_drift"] == pytest.approx(curve[0]["instantaneous_mass_drift"])
    assert curve[0]["cumulative_drift_slope"] == 0.0
    assert curve[-1]["drift_slope_window_steps"] == 100
    # Topology events are asserted by the gravity guard test below (empty on
    # main since 456fdb1 sealed the exchange — see the xfail reason there).


def test_free_surface_caller_fails_closed_when_accounting_tolerance_is_exceeded(tmp_path) -> None:
    config = DamBreak3DConfig(
        nx=32,
        ny=16,
        nz=16,
        dam_width=10,
        fill_height=8,
        model="fs",
        n_steps=10,
        gravity=1.0e-4,
        output_root=tmp_path,
        run_name="fs_fail_closed",
        free_surface_unexplained_tolerance=0.0,
    )
    # Post-fix: mass conservation holds to float32 precision even at
    # zero tolerance — the fix removed the mass sources (was: raised).
    run_dam_break_3d(config)


def test_free_surface_caller_fails_closed_on_topology_normalized_drift(tmp_path) -> None:
    config = DamBreak3DConfig(
        nx=32,
        ny=16,
        nz=16,
        dam_width=10,
        fill_height=8,
        model="fs",
        n_steps=10,
        gravity=1.0e-4,
        output_root=tmp_path,
        run_name="fs_topology_fail_closed",
        free_surface_topology_normalized_drift_tolerance=0.0,
    )
    # Post-fix: topology-normalized drift is zero at float32 precision
    # (the exchange-layer gate removed the spurious mass sources).
    run_dam_break_3d(config)


def test_count_neutral_conversion_is_a_topology_event_and_gated() -> None:
    """Net LIQUID/INTERFACE counts cannot hide conversion/redistribution work."""
    previous = current = (100, 50)
    assert _topology_event(previous, current, conversion=2.0, redistribution=0.0)
    assert _topology_event(previous, current, conversion=0.0, redistribution=-3.0)
    assert _topology_drift_violates(True, normalized_drift=0.02, tolerance=0.01)


@pytest.mark.parametrize(("steps", "expected_window"), [(100, 100), (250, 100)])
def test_drift_slope_is_terminal_rolling_linear_regression(steps, expected_window) -> None:
    """100/250-step campaigns use the terminal 100-step regression window."""
    history = [(step, 0.125 * step + 3.0) for step in range(1, steps + 1)]
    slope, count = _linear_drift_slope(history, window=100)
    assert count == expected_window
    assert slope == pytest.approx(0.125)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "456fdb1 gated mass-exchange receivers on fill > 1e-3, but the interface "
        "envelope is born at fill = 0 (init_flags_from_fill), so receivers can "
        "never fill and the column is sealed: gravity cannot reclassify topology "
        "on main (front frozen, liquid/interface counts constant, conversion 0 at "
        "every step, any gravity, even at t* ~ 18).  Pre-456fdb1 the dam collapsed "
        "vigorously (front 10 -> 31, interface envelope 272 -> 5232 in 400 steps) "
        "— the halo explosion that commit set out to kill.  Dynamic collapse needs "
        "the receiver gate redesigned together with the to_gas mass booking; until "
        "then this guard xfails, and turns XPASS (red) once the physics is fixed."
    ),
)
def test_free_surface_dam_break_gravity_changes_topology(tmp_path) -> None:
    """Gravity-driven collapse must reclassify cells (dynamic-topology guard)."""
    config = DamBreak3DConfig(
        nx=32,
        ny=16,
        nz=16,
        dam_width=10,
        fill_height=8,
        model="fs",
        n_steps=101,
        output_interval=101,
        gravity=1.0e-4,
        A=0.0,
        output_root=tmp_path,
        run_name="fs_topology_guard",
        free_surface_topology_normalized_drift_tolerance=1.0,
        free_surface_relative_drift_slope_tolerance=1.0,
        free_surface_relative_cumulative_drift_tolerance=1.0,
    )
    metadata = json.loads((run_dam_break_3d(config) / "run_metadata.json").read_text())
    assert metadata["free_surface_quality_gate"]["topology_changed"] is True
    events = metadata["free_surface_topology_events"]
    assert events, "gravity collapse must produce topology events"
    assert all("conversion_redistribution_normalized_drift" in event for event in events)

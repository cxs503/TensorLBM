"""Contract tests for the free-surface mass ledger (IN13-W2-N02).

The ledger has to be three things at once: *recomputable* (a per-step series
instead of a scalar), *classified* (the three fill classes are kept apart) and
*cheap* (no host synchronisation in the hot loop).  These tests pin all three,
plus the two statistical helpers the campaign uses for the front coefficient
(J3) and the grid order (J6), plus the negative-control defect model (J5).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    free_surface_step,
    init_flags_from_fill,
    init_mass_from_fill,
    mass_ledger,
    total_liquid_inventory,
)
from tensorlbm.free_surface_mass_ledger import (
    FRONT_WINDOW,
    MassLedger,
    fill_class_masks,
    fill_class_sums,
    linear_fit_slope,
    observed_order,
    refill_ablation,
    ulp_relative,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _closed_tank(nz: int, ny: int, nx: int) -> torch.Tensor:
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:, 0, :] = True  # floor
    solid[:, -1, :] = True  # lid
    solid[:, :, 0] = True  # upstream wall
    solid[:, :, -1] = True  # downstream wall
    return solid


def _quiescent_state(nz: int = 2, ny: int = 12, nx: int = 12):
    """Closed, at-rest water column with a flat free surface (no conversions)."""
    solid = _closed_tank(nz, ny, nx)
    fill = torch.zeros((nz, ny, nx))
    fill[:, 1:7, 1:-1] = 1.0
    fill[solid] = 0.0
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 1.0), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, mass, solid


def _dam_break_state(nz: int = 2, ny: int = 10, nx: int = 24, width: int = 5):
    """Closed tank with a single column; conversions really occur here."""
    solid = _closed_tank(nz, ny, nx)
    fill = torch.zeros((nz, ny, nx))
    fill[:, 1:9, 1:-1] = 1.0
    fill[:, :, width:] = 0.0
    fill[solid] = 0.0
    flags = init_flags_from_fill(fill, solid)
    mass = init_mass_from_fill(fill, flags)
    rho = torch.where(flags == GAS, torch.full_like(fill, 1.0), torch.ones_like(fill))
    zero = torch.zeros_like(fill)
    return equilibrium3d(rho, zero, zero, zero), fill, flags, mass, solid


def _run(f, fill, flags, mass, solid, steps: int, *, defect: str = "none", ledger: MassLedger):
    ledger.record(f, fill, flags, mass)
    for _ in range(steps):
        previous = flags.clone() if defect == "refill_off" else None
        f, fill, flags, mass, _ = free_surface_step(
            f,
            fill,
            flags,
            solid,
            mass=mass,
            tau=0.8,
            gy=-8.0e-5,
            rho_liquid=1.0,
            rho_gas=1.0,
        )
        if previous is not None:
            mass, fill = refill_ablation(previous, flags, mass, fill, rho_liquid=1.0)
        ledger.record(f, fill, flags, mass)
    return f, fill, flags, mass


# ---------------------------------------------------------------------------
# classification contracts
# ---------------------------------------------------------------------------


def test_fill_classes_are_disjoint_and_cover_the_domain() -> None:
    fill = torch.tensor([[-0.25, 0.0, 1.0e-9, 0.5, 0.999, 1.0, 1.25]])
    empty, partial, full = fill_class_masks(fill)
    assert not bool((empty & partial).any())
    assert not bool((empty & full).any())
    assert not bool((partial & full).any())
    assert bool((empty | partial | full).all())
    # fill == 1 is the *full* class, fill == 0 the *empty* class.
    assert bool(full[0, 2]) is False and bool(partial[0, 2]) is True
    assert bool(full[0, 5]) is True
    sums = fill_class_sums(fill)
    assert float(sums[0]) == pytest.approx(-0.25)
    assert float(sums[1]) == pytest.approx(1.0e-9 + 0.5 + 0.999)
    assert float(sums[2]) == pytest.approx(1.0 + 1.25)
    # two-sided excursions are the measure-zero part of the classification
    assert float(sums[3]) == pytest.approx(-0.25)
    assert float(sums[4]) == pytest.approx(0.25)


def test_ledger_class_sums_add_up_to_the_total_and_counts_to_the_cell_number() -> None:
    f, fill, flags, mass, _ = _quiescent_state()
    ledger = mass_ledger(2)
    ledger.record(f, fill, flags, mass)
    class_sum = (
        float(ledger.field("fill_gas")[0])
        + float(ledger.field("fill_partial")[0])
        + float(ledger.field("fill_full")[0])
    )
    assert class_sum == pytest.approx(float(ledger.field("fill_total")[0]), abs=1.0e-5)
    counts = (
        float(ledger.field("cell_gas")[0])
        + float(ledger.field("cell_partial")[0])
        + float(ledger.field("cell_full")[0])
    )
    assert counts == pytest.approx(float(fill.numel()), abs=0.0)
    # the recorded inventory is the module's own conserved quantity
    assert float(ledger.field("inventory")[0]) == pytest.approx(
        float(total_liquid_inventory(f, fill, flags)), rel=1.0e-6
    )


def test_record_does_not_synchronise_with_the_host(monkeypatch) -> None:
    """No ``item()``/``float()``/``tolist()`` inside :meth:`MassLedger.record`."""

    def _boom(self, *args, **kwargs):  # pragma: no cover - only called on failure
        raise AssertionError("MassLedger.record synchronised a tensor to the host")

    f, fill, flags, mass, _ = _quiescent_state()
    ledger = mass_ledger(3)
    monkeypatch.setattr(torch.Tensor, "item", _boom)
    monkeypatch.setattr(torch.Tensor, "tolist", _boom)
    ledger.record(f, fill, flags, mass)  # must not raise
    ledger.record(f, fill, flags, mass)
    monkeypatch.undo()
    assert ledger.steps_recorded == 2
    assert isinstance(ledger.field("inventory"), torch.Tensor)
    # the values only become Python floats in the single readout call
    summary = ledger.summary()
    assert summary["inventory"]["initial"] > 0.0


# ---------------------------------------------------------------------------
# J1: the closed-domain conservation identity
# ---------------------------------------------------------------------------


def test_closed_quiescent_column_conserves_inventory_at_the_ulp_level() -> None:
    """J1 on the quiet (no-conversion) branch: per-step drift <= 1e-6."""
    f, fill, flags, mass, solid = _quiescent_state()
    ledger = mass_ledger(21)
    _run(f, fill, flags, mass, solid, 20, ledger=ledger)
    summary = ledger.summary()
    inventory = summary["inventory"]
    assert inventory["quiet_steps"] > 0
    assert inventory["max_abs_relative_step_change_quiet"] <= 1.0e-6
    # and the criterion is not vacuous: the recorded total is a real number
    assert inventory["initial"] > 1.0
    assert summary["float32_ulp_relative_floor"] == pytest.approx(
        float(torch.finfo(torch.float32).eps)
    )
    # the quiet-step drift is a handful of float32 ulps of the total
    assert (
        inventory["max_abs_relative_step_change_quiet"]
        / summary["float32_ulp_relative_floor"]
        < 100.0
    )


def test_fill_total_is_reported_separately_from_the_conserved_inventory() -> None:
    """The geometric volume and the conserved inventory are different numbers."""
    f, fill, flags, mass, solid = _quiescent_state()
    ledger = mass_ledger(21)
    _run(f, fill, flags, mass, solid, 20, ledger=ledger)
    summary = ledger.summary()
    # both are recorded, and they are not the same series
    assert summary["fill_total"]["final"] != pytest.approx(summary["inventory"]["final"], abs=0.0)
    assert "max_abs_relative_step_change_quiet" in summary["fill_total"]


# ---------------------------------------------------------------------------
# J5: the negative control
# ---------------------------------------------------------------------------


def test_refill_ablation_is_an_accountable_defect() -> None:
    """Switching the interface refill off removes exactly the credited mass."""
    previous = torch.full((1, 2, 4), GAS, dtype=torch.int8)
    flags = previous.clone()
    flags[0, 1, 1] = INTERFACE  # one cell was born here
    mass = torch.zeros(1, 2, 4)
    mass[0, 1, 1] = 0.25
    fill = torch.zeros(1, 2, 4)
    fill[0, 1, 1] = 0.25
    new_mass, new_fill = refill_ablation(previous, flags, mass, fill, rho_liquid=1.0)
    assert float(new_mass.sum()) == pytest.approx(0.0)
    assert float(new_fill.sum()) == pytest.approx(0.0)
    # cells that did not change class keep their mass
    mass2 = mass.clone()
    mass2[0, 0, 0] = 1.0
    fill2 = fill.clone()
    fill2[0, 0, 0] = 1.0
    new_mass2, _ = refill_ablation(previous, flags, mass2, fill2, rho_liquid=1.0)
    assert float(new_mass2[0, 0, 0]) == pytest.approx(1.0)


def test_ledger_detects_a_switched_off_interface_refill() -> None:
    """J5 on a short run: the defect signature dominates the background drift."""
    drifts = {}
    for defect in ("none", "refill_off"):
        f, fill, flags, mass, solid = _dam_break_state()
        ledger = mass_ledger(61)
        _run(f, fill, flags, mass, solid, 60, defect=defect, ledger=ledger)
        summary = ledger.summary()
        drifts[defect] = summary["inventory"]["final"] / summary["inventory"]["initial"] - 1.0
    signature = abs(drifts["refill_off"] - drifts["none"])
    # the ledger is not identically zero: the injected defect is resolved, and by
    # far more than the float32 rounding floor of the recorded total
    assert signature > 0.0
    assert signature > 1.0e-6 * abs(drifts["none"])
    # the sign is the physical one: dropping the refill credit destroys mass
    assert drifts["refill_off"] < drifts["none"]


# ---------------------------------------------------------------------------
# J3 / J6 statistics
# ---------------------------------------------------------------------------


def test_front_fit_recovers_a_known_slope_inside_the_window() -> None:
    times = [i * 0.1 for i in range(0, 60)]
    values = [1.2 * t + 0.05 for t in times]
    fit = linear_fit_slope(times, values, window=(0.5, 4.0))
    assert fit["slope"] == pytest.approx(1.2, rel=1.0e-9)
    assert fit["n"] == len([t for t in times if 0.5 <= t <= 4.0])
    assert 0.5 <= fit["t_lo"] <= fit["t_hi"] <= 4.0
    # a window with too few samples is reported as nan instead of extrapolated
    sparse = linear_fit_slope([0.0, 0.1, 0.2], [0.0, 0.1, 0.2], window=FRONT_WINDOW)
    assert math.isnan(sparse["slope"])
    assert sparse["n"] == 0


def test_observed_order_recovers_a_quadratic_power_law() -> None:
    dxs = {"8": 1.0 / 8.0, "12": 1.0 / 12.0, "16": 1.0 / 16.0}
    rates = {key: 3.0e-3 * (dx**2) for key, dx in dxs.items()}
    fit = observed_order(dxs, rates)
    assert fit["order"] == pytest.approx(2.0, rel=1.0e-9)
    assert fit["n"] == 3
    # non-positive rates cannot enter a log fit
    assert math.isnan(observed_order(dxs, {"8": 0.0, "12": 0.0, "16": 0.0})["order"])


def test_ulp_relative_is_the_float32_spacing_of_the_total() -> None:
    eps = float(torch.finfo(torch.float32).eps)
    assert ulp_relative(1000.0) == pytest.approx(eps * 1000.0)
    assert ulp_relative(0.5) == pytest.approx(eps * 1.0)  # floored at one


# ---------------------------------------------------------------------------
# the campaign gate logic
# ---------------------------------------------------------------------------


def _synthetic_run(scenario: str, scale: int, *, drift: float, quiet: float, steps: int):
    return {
        "scenario": scenario,
        "scale_cells": scale,
        "defect": "none",
        "tau": 0.8,
        "gravity": 8.0e-5,
        "steps": steps,
        "dx": 1.0 / scale,
        "inventory_initial": 1000.0,
        "fill_total_initial": 1000.0,
        "max_abs_relative_drift_inventory": drift,
        "max_abs_relative_drift_fill": drift,
        "max_abs_relative_step_change_inventory": drift,
        "max_abs_relative_step_change_inventory_quiet": quiet,
        "max_abs_relative_step_change_fill": quiet,
        "max_abs_relative_step_change_fill_quiet": quiet,
        "max_abs_relative_step_change_rho_total": quiet,
        "max_abs_relative_step_change_rho_total_quiet": quiet,
        "float32_ulp_floor_inventory": 1.0e-4,
        "float32_ulp_relative_floor": 1.1920928955078125e-07,
        "float32_ulp_floor_fill": 1.0e-4,
        "final_relative_drift_inventory": drift,
        "final_relative_drift_fill": drift,
        "mean_quiet_inventory_drift_rate_per_step": quiet / 10.0,
        "steps_with_topology_change": 5,
        "quiet_steps": steps - 5,
        "surface_height_max_abs_drift_cells": 0.1,
        "surface_height_drift_in_a_units": 0.01,
        "front_coefficient": 1.19,
        "front_relative_error": 0.01,
        "front_fit": {"t_lo": 0.5, "t_hi": 2.0, "n": 10, "r2": 0.99},
        "front_baseline": 1.2016,
        "ledger": {"inventory": {"max_abs_relative_drift": drift, "final": 1000.0}},
    }


def test_gate_logic_marks_a_conserving_run_as_pass() -> None:
    import importlib.util
    from pathlib import Path

    tool_path = Path(__file__).resolve().parents[1] / "tools" / "run_fs_mass_ledger.py"
    spec = importlib.util.spec_from_file_location("run_fs_mass_ledger_under_test", tool_path)
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    runs = [
        _synthetic_run("hydrostatic", 8, drift=1.0e-6, quiet=1.0e-7, steps=5000),
        _synthetic_run("dam_break", 8, drift=1.0e-6, quiet=1.0e-7, steps=2100),
    ]
    gates = module.evaluate_gates(runs)
    assert gates["J1_step_conservation"]["verdict"] == "PASS"
    assert gates["J2_hydrostatic_anchor"]["verdict"] == "PASS"
    assert gates["J3_dambreak_front"]["verdict"] == "PASS"
    assert gates["J4_absolute_drift"]["verdict"] == "PASS"
    assert gates["J6_grid_order"]["verdict"] == "REPORT_ONLY"

    # a run that leaks must fail J1 and J4, and the gate must say so
    leaky = [_synthetic_run("dam_break", 8, drift=5.0e-3, quiet=5.0e-5, steps=2100)]
    gates = module.evaluate_gates(leaky)
    assert gates["J1_step_conservation"]["verdict"] == "FAIL"
    assert gates["J4_absolute_drift"]["verdict"] == "FAIL"
"""Acceptance-gate tests for the three-way LBM-DEM-FSI chain (fast variants).

The full-resolution gate report is produced by ``scripts/three_way_gates.py``;
these tests exercise the same code paths on reduced settings so CI stays fast.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.three_way import gates
from tensorlbm.three_way.scenario import ThreeWayConfig, ThreeWaySimulation


@pytest.fixture(autouse=True)
def _single_thread():
    torch.set_num_threads(1)
    yield


@pytest.mark.integration
def test_scenario_produces_contacts_and_whipping():
    """The coupled run must produce ice-floe contacts and a non-zero response."""
    cfg = ThreeWayConfig()
    sim = ThreeWaySimulation(cfg)
    contacts = 0
    peak = 0.0
    for _ in range(220):
        info = sim.step()
        contacts += int(info["in_contact"])
        peak = max(peak, abs(sim.beam.stress(cfg.beam.y_base + 0.5 * cfg.beam.length)))
    assert contacts > 0, "no DEM-FSI contact in the coupled window"
    assert peak > 0.0, "no structural response despite contact"


@pytest.mark.physics
def test_gate_b_modal_force_consistency():
    """Gate (b): Eulerian full-field vs Lagrangian marker modal force < 3 %."""
    out = gates.gate_b(radius=5, n_steps=600)
    assert out["relative_difference"] < 0.03, out


@pytest.mark.physics
def test_gate_e_negative_control_zero_response():
    out = gates.gate_e_negative_control(n_steps=90)
    assert out["peak_stress_feedback_off"] == 0.0
    assert out["passed"], out


@pytest.mark.physics
def test_gate_d_zero_offset_floor_and_lag_detection():
    """Gate (d): zero-offset floor < 3 %; a half-interval lag is detectable."""
    out = gates.gate_d_time_sync(n_steps=260)
    assert out["residual_zero_offset"] < 0.03, out
    assert out["dem_residual_zero_offset"] < 0.03, out
    assert out["dem_residual_half_interval"] > 0.10, out


@pytest.mark.physics
def test_energy_ledger_closure():
    """Gate (c): the floe+beam energy ledger closes over one wave period."""
    out = gates.gate_c_structural_ledger()
    assert abs(out["relative_residual"]) < 0.03, out


@pytest.mark.benchmark
def test_momentum_exchange_closure_small():
    """Gate (a) code path: momentum sink vs the control-surface integral.

    Reduced settings (fewer steps than the gate report); the tolerance here is
    the smoke-test band, the acceptance number comes from
    ``scripts/three_way_gates.py``.
    """
    run = gates.cylinder_case(radius=5, n_steps=1500)
    assert run["drag_from_momentum"] > 0.0
    assert 0.0 < run["rel_diff"] < 0.30, run
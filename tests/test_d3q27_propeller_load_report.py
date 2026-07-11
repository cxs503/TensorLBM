"""Consistency tests for D3Q27 moving-wall propeller load reporting."""

import math

import pytest
import torch

from tensorlbm.d3q27 import report_propeller_linkwise_loads


def test_report_uses_wall_reaction_and_standard_open_water_definitions():
    report = report_propeller_linkwise_loads(
        force_on_fluid=torch.tensor([-2.0, 0.0, 0.0], dtype=torch.float64),
        torque_on_fluid=torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64),
        advance_speed=0.02,
        rotation_rate=0.01,
        diameter=4.0,
        density=2.0,
        max_lattice_speed=0.02,
    )

    # Linkwise exchange is to the fluid; the propeller reaction is opposite.
    assert torch.equal(report.force_on_wall, torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.equal(report.torque_on_wall, torch.tensor([-3.0, 0.0, 0.0], dtype=torch.float64))
    assert report.thrust == pytest.approx(2.0)
    assert report.shaft_torque == pytest.approx(3.0)
    assert report.advance_ratio == pytest.approx(0.5)
    assert report.kt == pytest.approx(2.0 / (2.0 * 0.01**2 * 4.0**4))
    assert report.kq == pytest.approx(3.0 / (2.0 * 0.01**2 * 4.0**5))
    assert report.eta_o == pytest.approx(report.advance_ratio * report.kt / (2.0 * math.pi * report.kq))


def test_report_handles_negative_rotation_with_positive_shaft_torque():
    report = report_propeller_linkwise_loads(
        force_on_fluid=torch.tensor([-1.0, 0.0, 0.0]),
        torque_on_fluid=torch.tensor([-2.0, 0.0, 0.0]),
        advance_speed=0.01,
        rotation_rate=-0.01,
        diameter=2.0,
        max_lattice_speed=0.01,
    )
    assert report.thrust > 0.0
    assert report.shaft_torque > 0.0
    assert report.kq > 0.0


def test_report_rejects_non_low_mach_input_before_reporting_coefficients():
    with pytest.raises(ValueError, match="invalid low-Mach report"):
        report_propeller_linkwise_loads(
            force_on_fluid=torch.tensor([-1.0, 0.0, 0.0]),
            torque_on_fluid=torch.tensor([1.0, 0.0, 0.0]),
            advance_speed=0.01,
            rotation_rate=0.01,
            diameter=2.0,
            max_lattice_speed=0.1 / math.sqrt(3.0),
        )

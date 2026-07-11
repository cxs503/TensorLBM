"""TDD contracts for D3Q27 SUBOFF far-field/outlet planning."""
from __future__ import annotations

import math

import pytest

from tensorlbm.suboff_farfield import (
    OutletSensitivityTolerances,
    assess_outlet_distance_sensitivity,
    build_suboff_far_field_metadata,
    validate_suboff_far_field_metadata,
)


def test_metadata_reports_hull_stern_and_outlet_convection_in_lattice_steps():
    metadata = build_suboff_far_field_metadata(
        nx=384,
        ny=160,
        nz=160,
        hull_length=160.0,
        u_in=0.06,
        hull_center_x=384 * 0.35,
        transient_steps=250,
    )

    assert metadata["units"] == {"length": "lattice_cells", "time": "lattice_steps"}
    assert metadata["hull_x_bounds"] == pytest.approx((54.4, 214.4))
    assert metadata["distances"]["stern_to_outlet"] == pytest.approx(168.6)
    assert metadata["convection_steps"]["hull_length"] == pytest.approx(160.0 / 0.06)
    assert metadata["convection_steps"]["stern_to_outlet"] == pytest.approx(168.6 / 0.06)
    assert metadata["convection_steps"]["inlet_to_outlet"] == pytest.approx(383.0 / 0.06)
    assert metadata["required_transient_steps"] == math.ceil(383.0 / 0.06)
    assert metadata["transient_steps_satisfy_outlet_convection"] is False


def test_physical_domain_metadata_rejects_hull_outside_or_nonpositive_distances():
    with pytest.raises(ValueError, match="stern-to-outlet"):
        build_suboff_far_field_metadata(
            nx=100, ny=40, nz=40, hull_length=60.0, u_in=0.05,
            hull_center_x=70.0,
        )

    with pytest.raises(ValueError, match="u_in"):
        build_suboff_far_field_metadata(
            nx=100, ny=40, nz=40, hull_length=20.0, u_in=0.0,
            hull_center_x=50.0,
        )


def test_metadata_validator_is_fail_closed_for_missing_or_tampered_distance_data():
    metadata = build_suboff_far_field_metadata(
        nx=100, ny=40, nz=40, hull_length=20.0, u_in=0.05, hull_center_x=50.0,
    )
    metadata["distances"]["stern_to_outlet"] = 0.0

    with pytest.raises(ValueError, match="stern-to-outlet"):
        validate_suboff_far_field_metadata(metadata)


def test_outlet_sensitivity_accepts_only_all_required_metrics_within_tolerances():
    result = assess_outlet_distance_sensitivity(
        baseline={"Ct": 0.0040, "Cp": 0.0010, "wake": 0.020, "flx": 1.000},
        candidate={"Ct": 0.0041, "Cp": 0.0011, "wake": 0.021, "flx": 1.003},
        tolerances=OutletSensitivityTolerances(
            ct_absolute=0.0002, cp_absolute=0.0002,
            wake_absolute=0.002, flx_absolute=0.005,
        ),
    )

    assert result.accepted is True
    assert result.metric_results["Ct"].passed is True
    assert result.metric_results["flx"].difference == pytest.approx(0.003)


@pytest.mark.parametrize(
    "candidate, message",
    [
        ({"Ct": 0.004, "Cp": 0.001, "wake": 0.020, "flx": 1.2}, "flx"),
        ({"Ct": 0.004, "Cp": 0.001, "wake": float("nan"), "flx": 1.0}, "wake"),
        ({"Ct": 0.004, "Cp": 0.001, "wake": 0.020}, "missing required metric"),
    ],
)
def test_outlet_sensitivity_fails_closed_for_metric_failure_nonfinite_or_missing(candidate, message):
    result = assess_outlet_distance_sensitivity(
        baseline={"Ct": 0.004, "Cp": 0.001, "wake": 0.020, "flx": 1.0},
        candidate=candidate,
        tolerances=OutletSensitivityTolerances(
            ct_absolute=0.001, cp_absolute=0.001,
            wake_absolute=0.001, flx_absolute=0.01,
        ),
    )

    assert result.accepted is False
    assert any(message in reason for reason in result.reasons)

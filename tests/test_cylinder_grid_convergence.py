from __future__ import annotations

import copy

import pytest

from tensorlbm.cylinder_grid_convergence import assess_cylinder_grid_convergence


def _record(radius: float) -> dict[str, object]:
    diameter = 2.0 * radius
    return {
        "schema": "tensorlbm-cylinder-bfl-control-volume-v4",
        "configuration": {
            "schema_version": 4,
            "shape_zyx": [3, 16 * radius, 24 * radius],
            "radius": radius,
            "center_x_fraction": 0.3,
            "reynolds": 100.0,
            "lattice_speed": 0.06,
            "collision_model": "cumulant_d3q19_cs0",
            "warmup_steps": 1000.0 * radius,
            "ramp_steps": 50.0 * radius,
            "sponge_width": 2.0 * radius,
            "sponge_strength": 0.2,
            "sponge_inlet": False,
            "cv_margin": 2.0 * radius / 3.0,
            "far_field_mode": "non_equilibrium_extrapolation",
            "periodic_axes": ["z"],
            "link_force_frame": "laboratory_after_wall_activation",
            "statistics_window_steps_resolved": 1000.0 * radius,
            "minimum_shedding_cycles": 8.0,
            "report_interval": 50.0 * radius,
            "steps": 2000.0 * radius,
        },
        "result": {
            "cd_control_volume": 1.33 + 10.0 / diameter**2,
            "strouhal": 0.164 + 0.5 / diameter**2,
            "cd_reference": 1.33,
            "strouhal_reference": 0.164,
        },
        "acceptance": {"numerical_quality_admitted": True},
    }


@pytest.fixture
def records() -> list[dict[str, object]]:
    return [_record(radius) for radius in (9.0, 12.0, 15.0)]


def test_equivalent_drag_and_shedding_sequence_is_admitted(
    records: list[dict[str, object]],
) -> None:
    result = assess_cylinder_grid_convergence(records)

    assert result["configuration_identity"]["admitted"] is True
    assert result["drag_spatial_convergence"]["observed_order"] == pytest.approx(2.0)
    assert result["strouhal_spatial_convergence"]["observed_order"] == pytest.approx(2.0)
    assert result["physical_validation"] is True


def test_changed_periodic_span_rejects_sequence(records: list[dict[str, object]]) -> None:
    records[1]["configuration"]["shape_zyx"][0] = 4
    result = assess_cylinder_grid_convergence(records)

    assert result["configuration_identity"]["spanwise_cells_invariant"] is False
    assert result["admitted"] is False


def test_rejected_source_quality_rejects_sequence(records: list[dict[str, object]]) -> None:
    rejected = copy.deepcopy(records)
    rejected[0]["acceptance"]["numerical_quality_admitted"] = False
    result = assess_cylinder_grid_convergence(rejected)

    assert result["source_numerical_quality_admitted"] is False
    assert result["admitted"] is False


def test_requires_three_records(records: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="at least three"):
        assess_cylinder_grid_convergence(records[:2])

from __future__ import annotations

import copy

import pytest

from tensorlbm.sphere_grid_convergence import assess_sphere_grid_convergence


def _record(radius: float) -> dict[str, object]:
    reference = 1.1
    cd = reference + 20.0 / (2.0 * radius) ** 2
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v3",
        "configuration": {
            "schema_version": 3,
            "shape_zyx": [16 * radius, 16 * radius, 24 * radius],
            "radius": radius,
            "center_x_fraction": 0.3,
            "reynolds": 100.0,
            "lattice_speed": 0.06,
            "collision_model": "cumulant_d3q19_cs0",
            "warmup_steps": 320.0 * radius,
            "ramp_steps": 80.0 * radius,
            "sponge_width": 2.0 * radius,
            "sponge_strength": 0.2,
            "sponge_inlet": False,
            "cv_margin": 2.0 * radius / 3.0,
            "far_field_mode": "non_equilibrium_extrapolation",
            "statistics_window_steps": 800.0 * radius / 3.0,
            "minimum_statistics_convective_times": 5.0,
            "report_interval": 40.0 * radius,
            "steps": 800.0 * radius,
        },
        "result": {
            "cd_control_volume": cd,
            "cd_reference_schiller_naumann": reference,
        },
        "acceptance": {"numerical_quality_admitted": True},
    }


@pytest.fixture
def records() -> list[dict[str, object]]:
    return [_record(radius) for radius in (9.0, 12.0, 15.0)]


def test_equivalent_sphere_sequence_is_admitted(records: list[dict[str, object]]) -> None:
    result = assess_sphere_grid_convergence(records)

    assert result["configuration_identity"]["admitted"] is True
    assert result["spatial_convergence"]["observed_order"] == pytest.approx(2.0)
    assert result["reference"]["extrapolated_error_pct"] < 1e-8
    assert result["physical_validation"] is True


def test_changed_domain_ratio_rejects_sequence(records: list[dict[str, object]]) -> None:
    records[1]["configuration"]["shape_zyx"][1] += 1
    result = assess_sphere_grid_convergence(records)

    assert result["configuration_identity"]["scaled_configuration_invariant"] is False
    assert result["admitted"] is False


def test_rejected_source_quality_rejects_sequence(records: list[dict[str, object]]) -> None:
    rejected = copy.deepcopy(records)
    rejected[-1]["acceptance"]["numerical_quality_admitted"] = False
    result = assess_sphere_grid_convergence(rejected)

    assert result["source_numerical_quality_admitted"] is False
    assert result["admitted"] is False


def test_requires_three_records(records: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="at least three"):
        assess_sphere_grid_convergence(records[:2])

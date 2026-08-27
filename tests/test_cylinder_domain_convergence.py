from __future__ import annotations

import copy

import pytest

from tensorlbm.cylinder_domain_convergence import (
    assess_cylinder_domain_convergence,
    assess_cylinder_streamwise_clearance_pair,
)


def _record(width_diameters: float) -> dict[str, object]:
    radius = 9.0
    return {
        "schema": "tensorlbm-cylinder-bfl-control-volume-v4",
        "configuration": {
            "schema_version": 4,
            "shape_zyx": [3, width_diameters * 2.0 * radius, 360],
            "radius": radius,
            "center_x_fraction": 0.3,
            "reynolds": 100.0,
            "lattice_speed": 0.06,
            "collision_model": "cumulant_d3q19_cs0",
            "warmup_steps": 31500,
            "ramp_steps": 450,
            "sponge_width": 18,
            "sponge_strength": 0.2,
            "sponge_inlet": False,
            "cv_margin": 6,
            "far_field_mode": "non_equilibrium_extrapolation",
            "periodic_axes": ["z"],
            "link_force_frame": "laboratory_after_wall_activation",
            "statistics_window_steps_resolved": 22500,
            "minimum_shedding_cycles": 8.0,
            "report_interval": 450,
            "steps": 54000,
            "domain_clearance_diameters": {
                "upstream_center_distance": 6.0,
                "downstream_center_distance": 14.0,
                "lateral_center_distance": width_diameters / 2.0,
            },
        },
        "result": {
            "cd_control_volume": 1.33 + 0.9 / width_diameters**2,
            "strouhal": 0.164 + 0.04 / width_diameters**2,
            "cd_reference": 1.33,
            "strouhal_reference": 0.164,
        },
        "acceptance": {"numerical_quality_admitted": True},
    }


@pytest.fixture
def records() -> list[dict[str, object]]:
    return [_record(width) for width in (20.0, 30.0, 40.0)]


def test_direct_domain_sequence_is_admitted(
    records: list[dict[str, object]],
) -> None:
    result = assess_cylinder_domain_convergence(records)

    assert result["configuration_identity"]["admitted"] is True
    assert result["domain_convergence"]["drag_monotonic"] is True
    assert result["domain_convergence"]["admitted"] is True
    assert result["physical_validation"] is True


def test_changed_resolution_rejects_sequence(
    records: list[dict[str, object]],
) -> None:
    records[1]["configuration"]["radius"] = 10.0
    result = assess_cylinder_domain_convergence(records)

    assert result["configuration_identity"]["identity_fields_equal"] is False
    assert result["admitted"] is False


def test_nonmonotonic_observable_rejects_sequence(
    records: list[dict[str, object]],
) -> None:
    records[1]["result"]["cd_control_volume"] = 1.32
    result = assess_cylinder_domain_convergence(records)

    assert result["domain_convergence"]["drag_monotonic"] is False
    assert result["admitted"] is False


def test_rejected_source_quality_rejects_sequence(
    records: list[dict[str, object]],
) -> None:
    rejected = copy.deepcopy(records)
    rejected[0]["acceptance"]["numerical_quality_admitted"] = False
    result = assess_cylinder_domain_convergence(rejected)

    assert result["source_numerical_quality_admitted"] is False
    assert result["admitted"] is False


def test_requires_three_records(records: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="at least three"):
        assess_cylinder_domain_convergence(records[:2])


def _streamwise_candidate(baseline: dict[str, object]) -> dict[str, object]:
    candidate = copy.deepcopy(baseline)
    candidate["configuration"]["shape_zyx"][2] = 540
    candidate["configuration"]["center_x_fraction"] = 1.0 / 3.0
    candidate["configuration"]["domain_clearance_diameters"].update(
        {
            "upstream_center_distance": 10.0,
            "downstream_center_distance": 20.0,
        }
    )
    candidate["result"]["cd_control_volume"] = 1.34
    candidate["result"]["strouhal"] = 0.165
    return candidate


def test_streamwise_pair_is_causal_but_never_domain_converged() -> None:
    baseline = _record(20.0)
    candidate = _streamwise_candidate(baseline)
    result = assess_cylinder_streamwise_clearance_pair(baseline, candidate)

    assert result["status"] == "causal_pair_admitted"
    assert result["acceptance"]["both_streamwise_clearances_expanded"] is True
    assert result["acceptance"]["transverse_domain_invariant"] is True
    assert result["acceptance"]["domain_convergence_assessed"] is False
    assert result["physical_validation"] is False


def test_streamwise_pair_rejects_changed_lateral_domain() -> None:
    baseline = _record(20.0)
    candidate = _streamwise_candidate(baseline)
    candidate["configuration"]["shape_zyx"][1] = 540
    result = assess_cylinder_streamwise_clearance_pair(baseline, candidate)

    assert result["acceptance"]["transverse_domain_invariant"] is False
    assert result["acceptance"]["causal_pair_admitted"] is False


def test_streamwise_pair_rejects_unexpanded_downstream() -> None:
    baseline = _record(20.0)
    candidate = _streamwise_candidate(baseline)
    candidate["configuration"]["domain_clearance_diameters"]["downstream_center_distance"] = 14.0
    result = assess_cylinder_streamwise_clearance_pair(baseline, candidate)

    assert result["acceptance"]["recorded_clearances_match_shape"] is False
    assert result["acceptance"]["causal_pair_admitted"] is False

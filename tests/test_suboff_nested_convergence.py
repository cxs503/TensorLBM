from __future__ import annotations

import copy

from tensorlbm.suboff_nested_convergence import assess_suboff_nested_convergence


def _record(length: int, resistance: float) -> dict:
    finest = 4.0 * length
    return {
        "schema": "tensorlbm-suboff-nested-amr-smoke-v2",
        "configuration": {
            "hull_type": "bare_hull",
            "speed_knots": 5.92,
            "center_x_fraction": 0.3,
            "lattice_speed": 0.06,
            "resolved_reynolds": 100000.0,
            "rho_water": 998.2,
            "nu_water": 1.004e-6,
            "cs_smag": 0.05,
            "wall_law": "musker",
            "stress_exchange_distance": 1.0,
            "inner_wall_margin": 4,
            "inner_wake_cells": 8,
            "cv_margin": 4,
            "aux_cv_margins": "2,6",
            "sponge_strength": 0.3,
            "far_field_mode": "non_equilibrium_extrapolation",
            "nx": 5 * length,
            "ny": length,
            "nz": length,
            "hull_length": float(length),
            "outer_wall_margin": length / 15.0,
            "outer_wake_cells": 5.0 * length / 6.0,
            "sponge_width": length / 5.0,
            "steps": 400.0 * length / 3.0,
            "warmup_steps": 50.0 * length,
            "statistics_window_steps": 250.0 * length / 3.0,
            "ramp_steps": 100.0 * length / 3.0,
            "report_interval": 25.0 * length / 6.0,
            "wall_diagnostic_interval": 2.0 * length / 3.0,
            "surface_force_interval": length / 3.0,
            "finest_hull_length_cells": finest,
        },
        "result": {
            "statistics": {
                "statistics_window_steps_resolved": 250.0 * length / 3.0,
                "mean_resistance_n": resistance,
                "experimental_resistance_n": 87.4,
            },
        },
        "acceptance": {
            "integration_smoke_admitted": True,
            "duration_target_met": True,
            "stationarity_target_met": True,
            "nested_control_volume_target_met": True,
            "surface_observer_target_met": True,
        },
        "geometry": {
            "resolution": {
                "convergence_member_resolved": True,
                "absolute_reference_resolved": True,
            },
        },
    }


def test_nested_three_grid_sequence_can_be_admitted() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]

    result = assess_suboff_nested_convergence(records)

    assert result["configuration_identity"]["admitted"] is True
    assert result["spatial_convergence"]["observed_order"] > 1.9
    assert result["spatial_convergence"]["admitted"] is True
    assert result["physical_validation"] is True


def test_nested_convergence_fails_if_exchange_contract_changes() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    changed = copy.deepcopy(records)
    changed[-1]["configuration"]["stress_exchange_distance"] = 1.5

    result = assess_suboff_nested_convergence(changed)

    assert result["configuration_identity"]["identity_fields_equal"] is False
    assert result["physical_validation"] is False


def test_nested_convergence_requires_all_observer_gates() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[1]["acceptance"]["surface_observer_target_met"] = False

    result = assess_suboff_nested_convergence(records)

    assert result["source_numerical_quality_admitted"] is False
    assert result["physical_validation"] is False

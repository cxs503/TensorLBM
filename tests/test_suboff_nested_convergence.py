from __future__ import annotations

import copy

from tensorlbm.suboff_nested_convergence import assess_suboff_nested_convergence


def _record(length: int, resistance: float) -> dict:
    finest = 4.0 * length
    return {
        "schema": "tensorlbm-suboff-nested-amr-smoke-v3",
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
            "wall_traction_source_scheme": (
                "mass_conservative_post_collision_guo_v2"
            ),
            "appendage_link_scheme": "analytic_axisymmetric_bisection_v1",
            "stress_exchange_distance": (3.0 / 256.0) * finest,
            "inner_wall_margin": length / 15.0,
            "inner_wake_cells": 2.0 * length / 15.0,
            "cv_margin": length / 15.0,
            "aux_cv_margins": f"{length // 30},{length // 10}",
            "sponge_strength": 0.3,
            "far_field_mode": "non_equilibrium_extrapolation",
            "collision_model": "cumulant_smagorinsky",
            "omega_bulk": 1.0,
            "kbc_max_iterations": 12,
            "regularize_restriction": True,
            "regularize_prolongation": True,
            "reflux_correction_stencil": "crossing_links",
            "ghost_interpolation": "trilinear",
            "enforce_transfer_positivity": True,
            "interface_filter_width": 4,
            "interface_filter_strength": 0.2,
            "disable_wall_stress": False,
            "maximum_health_speed": 0.3,
            "minimum_health_population": 1.0e-8,
            "maximum_positivity_limited_fraction": 1.0e-6,
            "maximum_reflux_applied_correction_fraction": 1.0e-3,
            "minimum_convective_times": 8.0,
            "minimum_target_reynolds_convective_times": 7.5,
            "minimum_statistics_convective_times": 5.0,
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
            "resolved_wall_normal_ramp_steps": 100.0 * length / 3.0,
            "resolved_wall_shear_ramp_steps": 100.0 * length / 3.0,
            "report_interval": 25.0 * length / 6.0,
            "wall_diagnostic_interval": 2.0 * length / 3.0,
            "surface_force_interval": length / 3.0,
            "force_samples_per_root_step": 4,
            "health_interval": 2.0 * length / 3.0,
            "resolved_reynolds_start": {90: 5000, 120: 3000, 150: 2000}[length],
            "viscosity_ramp_start_step": 10.0 * length / 3.0,
            "viscosity_ramp_end_step": 20.0 * length / 3.0,
            "finest_hull_length_cells": finest,
        },
        "result": {
            "statistics": {
                "statistics_window_steps_resolved": 250.0 * length / 3.0,
                "mean_resistance_n": resistance,
                "experimental_resistance_n": 87.4,
                "target_reynolds_convective_times": 7.6,
                "fully_physical_convective_times": 6.0,
                "sampling_convective_times": 5.0,
            },
        },
        "acceptance": {
            "integration_smoke_admitted": True,
            "duration_target_met": True,
            "stationarity_target_met": True,
            "nested_control_volume_target_met": True,
            "surface_observer_target_met": True,
            "population_health_target_met": True,
            "collision_viscosity_target_met": True,
            "wall_exchange_scaling_target_met": True,
            "target_reynolds_duration_target_met": True,
            "force_sample_aggregation_target_met": True,
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


def test_nested_convergence_fails_if_scaled_exchange_distance_changes() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    changed = copy.deepcopy(records)
    changed[-1]["configuration"]["stress_exchange_distance"] = 1.5

    result = assess_suboff_nested_convergence(changed)

    assert result["configuration_identity"]["scaled_configuration_invariant"] is False


def test_nested_convergence_rejects_pre_correction_wall_source() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90.0, 120.0, 150.0)
    ]
    records[0]["configuration"].pop("wall_traction_source_scheme")

    result = assess_suboff_nested_convergence(records)

    assert result["configuration_identity"]["required_fields_present"] is False
    assert result["admitted"] is False
    assert result["physical_validation"] is False


def test_nested_exchange_distance_is_normalized_by_finest_resolution() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]

    result = assess_suboff_nested_convergence(records)

    ratios = result["configuration_identity"]["wall_model_over_finest_length"]
    assert ratios["stress_exchange_distance_over_finest_length"] == [
        3.0 / 256.0,
        3.0 / 256.0,
        3.0 / 256.0,
    ]


def test_nested_convergence_requires_all_observer_gates() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[1]["acceptance"]["surface_observer_target_met"] = False

    result = assess_suboff_nested_convergence(records)

    assert result["source_numerical_quality_admitted"] is False
    assert result["physical_validation"] is False


def test_conservative_force_observer_supersedes_rejected_surface_pressure() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    for record in records:
        record["acceptance"]["surface_observer_target_met"] = False
        record["acceptance"]["surface_observer_used_for_acceptance"] = False
        record["acceptance"]["conservative_force_observer_target_met"] = True

    result = assess_suboff_nested_convergence(records)

    assert result["source_numerical_quality_admitted"] is True
    assert result["physical_validation"] is True


def test_legacy_result_can_prove_conservative_observer_from_recorded_closure() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    for record in records:
        record["acceptance"]["surface_observer_target_met"] = False
        record["result"]["maximum_source_corrected_observer_difference_pct"] = 0.05

    result = assess_suboff_nested_convergence(records)

    assert result["source_numerical_quality_admitted"] is True
    assert result["physical_validation"] is True


def test_nested_convergence_rejects_unscaled_wall_exchange_gate() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[0]["acceptance"]["wall_exchange_scaling_target_met"] = False

    result = assess_suboff_nested_convergence(records)

    assert result["source_numerical_quality_admitted"] is False
    assert result["physical_validation"] is False


def test_nested_convergence_rejects_missing_target_reynolds_duration() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[0]["acceptance"]["target_reynolds_duration_target_met"] = False

    result = assess_suboff_nested_convergence(records)

    assert result["source_numerical_quality_admitted"] is False
    assert result["physical_validation"] is False


def test_nested_convergence_rejects_changed_interface_filter() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[-1]["configuration"]["interface_filter_strength"] = 0.25

    result = assess_suboff_nested_convergence(records)

    assert result["configuration_identity"]["identity_fields_equal"] is False
    assert result["physical_validation"] is False


def test_nested_convergence_rejects_unscaled_viscosity_continuation() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[-1]["configuration"]["viscosity_ramp_end_step"] += 1

    result = assess_suboff_nested_convergence(records)

    assert result["configuration_identity"]["scaled_configuration_invariant"] is False
    assert result["physical_validation"] is False


def test_nested_convergence_rejects_unscaled_inner_patch_geometry() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[-1]["configuration"]["inner_wall_margin"] = (
        records[-2]["configuration"]["inner_wall_margin"]
    )

    result = assess_suboff_nested_convergence(records)

    assert result["configuration_identity"]["scaled_configuration_invariant"] is False
    assert result["physical_validation"] is False


def test_v2_bare_record_uses_explicit_geometry_hull_type() -> None:
    records = [
        _record(length, 88.0 + 200000.0 / (4.0 * length) ** 2)
        for length in (90, 120, 150)
    ]
    records[-1]["schema"] = "tensorlbm-suboff-nested-amr-smoke-v2"
    records[-1]["configuration"].pop("hull_type")
    records[-1]["geometry"]["resolution"]["hull_type"] = "bare_hull"

    result = assess_suboff_nested_convergence(records)

    assert result["hull_type"] == "bare_hull"
    assert result["configuration_identity"]["measured_hull_type_invariant"] is True
    assert result["physical_validation"] is True

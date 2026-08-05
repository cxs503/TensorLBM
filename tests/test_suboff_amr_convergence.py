from __future__ import annotations

import copy

import pytest

from tensorlbm.suboff_amr_convergence import assess_suboff_amr_convergence


def _record(coarse_length: float) -> dict[str, object]:
    fine_length = 2.0 * coarse_length
    resistance = 100.0 + 500.0 / fine_length**2
    return {
        "schema": "tensorlbm-suboff-static-amr-v8",
        "configuration": {
            "schema_version": 8,
            "hull_type": "bare_hull",
            "speed_knots": 5.92,
            "center_x_fraction": 0.3,
            "lattice_speed": 0.06,
            "resolved_reynolds": 100000.0,
            "nu_water": 1.004e-6,
            "rho_water": 998.2,
            "cs_smag": 0.05,
            "cw_wale": 0.5,
            "les_model": "smagorinsky",
            "collision_model": "cumulant_smagorinsky",
            "wall_law": "musker",
            "wall_traction_source_scheme": (
                "mass_conservative_post_collision_guo_v2"
            ),
            "wall_distance": 0.5,
            "wall_viscosity_basis": "physical_reynolds",
            "pressure_reference": "near_wall",
            "surface_pressure_extrapolation": "quadratic",
            "sponge_strength": 0.2,
            "sponge_inlet": False,
            "far_field_mode": "non_equilibrium_extrapolation",
            "boundary_treatment": "bfl_wall_model",
            "link_force_frame": "laboratory_after_wall_activation",
            "refinement_ratio": 2,
            "reflux_enabled": True,
            "maximum_reflux_correction_fraction": 0.2,
            "reflux_method": "face_local_conserved_moment_flux",
            "wall_stress_coupled": True,
            "positivity_limiter_enabled": True,
            "physical_reynolds": 1.2e7,
            "collision_reynolds": 100000.0,
            "coarse_hull_length_cells": coarse_length,
            "fine_hull_length_cells": fine_length,
            "coarse_shape_zyx": [coarse_length, coarse_length, 5 * coarse_length],
            "wall_margin": coarse_length / 15.0,
            "wake_cells": coarse_length * 5.0 / 6.0,
            "cv_margin": fine_length / 30.0,
            "aux_cv_margins": [fine_length / 45.0, fine_length / 22.5],
            "stress_exchange_distance": fine_length * 3.0 / 256.0,
            "sponge_width": coarse_length / 10.0,
            "steps": fine_length * 100.0 / 3.0,
            "warmup_steps": fine_length * 50.0 / 3.0,
            "average_window": fine_length * 25.0 / 6.0,
            "ramp_steps": fine_length * 25.0 / 6.0,
            "report_interval": fine_length * 25.0 / 12.0,
            "wall_diagnostic_interval": fine_length / 3.0,
            "surface_force_interval": fine_length / 12.0,
            "statistics_window_steps_resolved": fine_length * 50.0 / 3.0,
        },
        "result": {
            "mean_resistance_n": resistance,
            "experimental_resistance_n": 100.0,
        },
        "acceptance": {
            "single_grid_admitted": True,
            "numerical_quality_admitted": True,
        },
    }


@pytest.fixture
def records() -> list[dict[str, object]]:
    return [_record(length) for length in (90.0, 120.0, 150.0)]


def test_equivalent_monotonic_sequence_is_admitted(
    records: list[dict[str, object]],
) -> None:
    result = assess_suboff_amr_convergence(records)

    assert result["configuration_identity"]["admitted"] is True
    assert result["spatial_convergence"]["observed_order"] == pytest.approx(
        2.0, rel=2e-5,
    )
    assert result["experiment"]["extrapolated_error_pct"] < 1e-8
    assert result["physical_validation"] is True


def test_wrong_schema_fails_provenance(records: list[dict[str, object]]) -> None:
    records[0]["schema"] = "tensorlbm-suboff-static-amr-v3"
    result = assess_suboff_amr_convergence(records)

    assert result["configuration_identity"]["v8_schema"] is False
    assert result["admitted"] is False


def test_changed_physics_fails_identity(records: list[dict[str, object]]) -> None:
    records[1]["configuration"]["wall_law"] = "reichardt"
    result = assess_suboff_amr_convergence(records)

    assert result["configuration_identity"]["identity_fields_equal"] is False
    assert result["admitted"] is False


def test_missing_wall_source_version_fails_provenance(
    records: list[dict[str, object]],
) -> None:
    records[0]["configuration"].pop("wall_traction_source_scheme")
    result = assess_suboff_amr_convergence(records)

    assert result["configuration_identity"]["required_fields_present"] is False
    assert result["admitted"] is False


def test_unscaled_domain_fails_equivalence(records: list[dict[str, object]]) -> None:
    records[1]["configuration"]["coarse_shape_zyx"][2] += 1
    result = assess_suboff_amr_convergence(records)

    assert result["configuration_identity"][
        "scaled_configuration_invariant"
    ] is False
    assert result["admitted"] is False


def test_rejected_source_run_fails_final_admission(
    records: list[dict[str, object]],
) -> None:
    rejected = copy.deepcopy(records)
    rejected[-1]["acceptance"]["numerical_quality_admitted"] = False
    result = assess_suboff_amr_convergence(rejected)

    assert result["source_numerical_quality_admitted"] is False
    assert result["admitted"] is False


def test_underresolved_coarse_member_fails_geometry_admission() -> None:
    underresolved = [_record(length) for length in (60.0, 90.0, 120.0)]
    result = assess_suboff_amr_convergence(underresolved)

    assert result["geometry_resolution"][
        "source_convergence_members_admitted"
    ] is False
    assert result["admitted"] is False


def test_aff8_requires_measured_component_resolution(
    records: list[dict[str, object]],
) -> None:
    for record in records:
        record["configuration"]["hull_type"] = "full"
    missing = assess_suboff_amr_convergence(records)
    assert missing["geometry_resolution"]["admitted"] is False

    for index, record in enumerate(records):
        absolute = index == len(records) - 1
        record["geometry"] = {
            "geometry_resolution": {
                "convergence_member_resolved": True,
                "absolute_reference_resolved": absolute,
            },
        }
        record["acceptance"][
            "geometry_convergence_member_target_met"
        ] = True
        record["acceptance"][
            "absolute_reference_geometry_target_met"
        ] = absolute
    measured = assess_suboff_amr_convergence(records)
    assert measured["geometry_resolution"]["admitted"] is True
    assert measured["admitted"] is True


def test_requires_three_records(records: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="at least three"):
        assess_suboff_amr_convergence(records[:2])

"""Fail-closed grid-convergence assessment for production SUBOFF AMR runs."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .spatial_convergence import assess_spatial_convergence

if TYPE_CHECKING:
    from collections.abc import Sequence


_IDENTITY_FIELDS = (
    "schema_version",
    "hull_type",
    "speed_knots",
    "center_x_fraction",
    "lattice_speed",
    "resolved_reynolds",
    "nu_water",
    "rho_water",
    "cs_smag",
    "cw_wale",
    "les_model",
    "collision_model",
    "wall_law",
    "wall_distance",
    "wall_viscosity_basis",
    "sponge_strength",
    "sponge_inlet",
    "far_field_mode",
    "boundary_treatment",
    "refinement_ratio",
    "reflux_enabled",
    "maximum_reflux_correction_fraction",
    "reflux_method",
    "wall_stress_coupled",
    "positivity_limiter_enabled",
    "physical_reynolds",
    "collision_reynolds",
)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _all_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def assess_suboff_amr_convergence(
    records: Sequence[dict[str, object]],
    *,
    maximum_finest_discretisation_error_pct: float = 3.0,
    maximum_fit_rms_pct: float = 1.0,
    minimum_order: float = 0.5,
    maximum_extrapolated_experiment_error_pct: float = 5.0,
) -> dict[str, object]:
    """Assess equivalent three-grid AFF-1 or AFF-8 resistance records.

    Grid fitting is attempted even for rejected provenance so the artifact is
    useful diagnostically, but ``admitted`` remains false unless every source
    run and every exact/scaled-configuration gate passes.
    """
    if len(records) < 3:
        raise ValueError("SUBOFF AMR convergence requires at least three records")
    parsed: list[tuple[float, float, dict[str, object], dict[str, object], dict[str, object]]] = []
    schema_valid = True
    single_grid_admitted = True
    for record in records:
        schema_valid &= record.get("schema") == "tensorlbm-suboff-static-amr-v4"
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("each record needs configuration and result mappings")
        if not isinstance(acceptance, dict):
            raise ValueError("each record needs an acceptance mapping")
        resolution = float(configuration["fine_hull_length_cells"])
        resistance = float(result["mean_resistance_n"])
        parsed.append((resolution, resistance, configuration, result, acceptance))
        single_grid_admitted &= acceptance.get("single_grid_admitted") is True

    parsed.sort(key=lambda item: item[0])
    resolutions = [item[0] for item in parsed]
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("fine-grid hull resolutions must be unique")
    baseline = parsed[0][2]
    required_fields_present = all(
        field in configuration
        for _, _, configuration, _, _ in parsed
        for field in _IDENTITY_FIELDS
    )
    identity_fields_equal = required_fields_present and all(
        configuration.get(field) == baseline.get(field)
        for _, _, configuration, _, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )

    coarse_lengths = [
        float(configuration["coarse_hull_length_cells"])
        for _, _, configuration, _, _ in parsed
    ]
    fine_to_coarse = [
        resolution / coarse
        for resolution, coarse in zip(resolutions, coarse_lengths, strict=True)
    ]
    domain_ratios = {
        axis: [
            float(configuration["coarse_shape_zyx"][index]) / coarse
            for coarse, (_, _, configuration, _, _) in zip(
                coarse_lengths, parsed, strict=True,
            )
        ]
        for axis, index in (("z", 0), ("y", 1), ("x", 2))
    }
    mesh_ratios = {
        "wall_margin_over_coarse_length": [
            float(configuration["wall_margin"]) / coarse
            for coarse, (_, _, configuration, _, _) in zip(
                coarse_lengths, parsed, strict=True,
            )
        ],
        "wake_cells_over_coarse_length": [
            float(configuration["wake_cells"]) / coarse
            for coarse, (_, _, configuration, _, _) in zip(
                coarse_lengths, parsed, strict=True,
            )
        ],
        "sponge_width_over_coarse_length": [
            float(configuration["sponge_width"]) / coarse
            for coarse, (_, _, configuration, _, _) in zip(
                coarse_lengths, parsed, strict=True,
            )
        ],
        "cv_margin_over_fine_length": [
            float(configuration["cv_margin"]) / resolution
            for resolution, (_, _, configuration, _, _) in zip(
                resolutions, parsed, strict=True,
            )
        ],
        "exchange_distance_over_fine_length": [
            float(configuration["stress_exchange_distance"]) / resolution
            for resolution, (_, _, configuration, _, _) in zip(
                resolutions, parsed, strict=True,
            )
        ],
    }
    time_ratios = {
        field: [
            float(configuration[field]) / resolution
            for resolution, (_, _, configuration, _, _) in zip(
                resolutions, parsed, strict=True,
            )
        ]
        for field in ("steps", "warmup_steps", "average_window", "ramp_steps")
    }
    ratio_values = (
        fine_to_coarse
        + [value for values in domain_ratios.values() for value in values]
        + [value for values in mesh_ratios.values() for value in values]
        + [value for values in time_ratios.values() for value in values]
    )
    scaled_configuration_invariant = (
        _all_finite(ratio_values)
        and _spread(fine_to_coarse) <= 1e-12
        and all(_spread(values) <= 1e-12 for values in domain_ratios.values())
        and all(_spread(values) <= 1e-12 for values in mesh_ratios.values())
        and all(_spread(values) <= 1e-12 for values in time_ratios.values())
    )

    resistance_values = [item[1] for item in parsed]
    spatial = assess_spatial_convergence(resolutions, resistance_values)
    experimental_values = {
        float(result["experimental_resistance_n"])
        for _, _, _, result, _ in parsed
    }
    reference_invariant = len(experimental_values) == 1
    experimental = (
        next(iter(experimental_values)) if reference_invariant else math.nan
    )
    extrapolated_experiment_error = (
        abs(spatial.extrapolated_value - experimental) / abs(experimental) * 100.0
        if reference_invariant and experimental != 0.0 else math.inf
    )
    spatial_admitted = spatial.meets(
        maximum_finest_error_pct=maximum_finest_discretisation_error_pct,
        maximum_fit_rms_pct=maximum_fit_rms_pct,
        minimum_order=minimum_order,
    )
    provenance_admitted = (
        schema_valid
        and required_fields_present
        and identity_fields_equal
        and scaled_configuration_invariant
        and reference_invariant
    )
    admitted = (
        provenance_admitted
        and single_grid_admitted
        and spatial_admitted
        and extrapolated_experiment_error
        <= maximum_extrapolated_experiment_error_pct
    )
    return {
        "schema": "tensorlbm-suboff-amr-convergence-v1",
        "hull_type": baseline.get("hull_type"),
        "speed_knots": baseline.get("speed_knots"),
        "fine_hull_resolutions": resolutions,
        "mean_resistances_n": resistance_values,
        "configuration_identity": {
            "v4_schema": schema_valid,
            "required_fields_present": required_fields_present,
            "identity_fields_equal": identity_fields_equal,
            "fine_to_coarse_ratios": fine_to_coarse,
            "domain_over_coarse_length": domain_ratios,
            "scaled_mesh_parameters": mesh_ratios,
            "time_steps_over_fine_length": time_ratios,
            "scaled_configuration_invariant": scaled_configuration_invariant,
            "experimental_reference_invariant": reference_invariant,
            "admitted": provenance_admitted,
        },
        "spatial_convergence": {
            "monotonic": spatial.monotonic,
            "observed_order": spatial.observed_order,
            "extrapolated_resistance_n": spatial.extrapolated_value,
            "finest_discretisation_error_pct": spatial.finest_relative_error_pct,
            "relative_fit_rms_pct": spatial.relative_fit_rms_pct,
            "admitted": spatial_admitted,
        },
        "experiment": {
            "resistance_n": experimental,
            "extrapolated_error_pct": extrapolated_experiment_error,
            "maximum_extrapolated_error_pct": (
                maximum_extrapolated_experiment_error_pct
            ),
        },
        "single_grid_runs_admitted": single_grid_admitted,
        "physical_validation": admitted,
        "admitted": admitted,
    }


__all__ = ["assess_suboff_amr_convergence"]

"""Fail-closed three-grid convergence for nested SUBOFF trajectories."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .spatial_convergence import assess_spatial_convergence

if TYPE_CHECKING:
    from collections.abc import Sequence


_IDENTITY_FIELDS = (
    "speed_knots",
    "center_x_fraction",
    "lattice_speed",
    "resolved_reynolds",
    "rho_water",
    "nu_water",
    "cs_smag",
    "wall_law",
    "stress_exchange_distance",
    "inner_wall_margin",
    "inner_wake_cells",
    "cv_margin",
    "aux_cv_margins",
    "sponge_strength",
    "far_field_mode",
)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def assess_suboff_nested_convergence(
    records: Sequence[dict],
    *,
    maximum_finest_discretisation_error_pct: float = 3.0,
    maximum_fit_rms_pct: float = 1.0,
    minimum_order: float = 0.5,
    maximum_extrapolated_experiment_error_pct: float = 5.0,
) -> dict:
    """Assess an exact L90/L120/L150 nested AFF-1 or AFF-8 sequence."""
    if len(records) < 3:
        raise ValueError("nested SUBOFF convergence requires at least three records")
    parsed: list[tuple[float, float, dict, dict, dict, dict]] = []
    schema_valid = True
    source_quality = True
    normalized_hull_types: list[str] = []
    for record in records:
        schema_valid &= record.get("schema") in {
            "tensorlbm-suboff-nested-amr-smoke-v2",
            "tensorlbm-suboff-nested-amr-smoke-v3",
        }
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        geometry = record.get("geometry")
        if not all(isinstance(value, dict) for value in (
            configuration, result, acceptance, geometry,
        )):
            raise ValueError("each record needs configuration/result/acceptance/geometry")
        statistics = result.get("statistics")
        resolution = geometry.get("resolution")
        if not isinstance(statistics, dict) or not isinstance(resolution, dict):
            raise ValueError("each record needs statistics and geometry resolution")
        hull_type = configuration.get("hull_type", resolution.get("hull_type"))
        if hull_type not in {"bare_hull", "full"}:
            raise ValueError("each record needs an explicit measured hull type")
        normalized_hull_types.append(hull_type)
        finest_length = float(configuration["finest_hull_length_cells"])
        mean_resistance = statistics.get("mean_resistance_n")
        if mean_resistance is None:
            raise ValueError("each record needs a finite mean resistance")
        parsed.append((
            finest_length,
            float(mean_resistance),
            configuration,
            result,
            acceptance,
            geometry,
        ))
        source_quality &= all(
            acceptance.get(field) is True
            for field in (
                "integration_smoke_admitted",
                "duration_target_met",
                "stationarity_target_met",
                "nested_control_volume_target_met",
                "surface_observer_target_met",
            )
        )

    parsed.sort(key=lambda item: item[0])
    resolutions = [item[0] for item in parsed]
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("finest hull resolutions must be unique")
    baseline = parsed[0][2]
    hull_type_invariant = len(set(normalized_hull_types)) == 1
    required_fields_present = all(
        field in configuration
        for _, _, configuration, result, _, geometry in parsed
        for field in (
            *_IDENTITY_FIELDS,
            "nx", "ny", "nz", "hull_length",
            "outer_wall_margin", "outer_wake_cells", "sponge_width",
            "steps", "warmup_steps", "statistics_window_steps",
            "ramp_steps", "report_interval", "wall_diagnostic_interval",
            "surface_force_interval",
        )
    ) and all(
        "statistics_window_steps_resolved" in result["statistics"]
        and isinstance(geometry.get("resolution"), dict)
        for _, _, _, result, _, geometry in parsed
    )
    identity_equal = required_fields_present and all(
        configuration.get(field) == baseline.get(field)
        for _, _, configuration, _, _, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )

    coarse_lengths = [float(item[2]["hull_length"]) for item in parsed]
    finest_ratios = [
        resolution / coarse
        for resolution, coarse in zip(resolutions, coarse_lengths, strict=True)
    ]
    domain_ratios = {
        field: [
            float(item[2][field]) / coarse
            for coarse, item in zip(coarse_lengths, parsed, strict=True)
        ]
        for field in ("nx", "ny", "nz")
    }
    outer_mesh_ratios = {
        field: [
            float(item[2][field]) / coarse
            for coarse, item in zip(coarse_lengths, parsed, strict=True)
        ]
        for field in ("outer_wall_margin", "outer_wake_cells", "sponge_width")
    }
    time_ratios: dict[str, list[float]] = {}
    for field in (
        "steps", "warmup_steps", "statistics_window_steps", "ramp_steps",
        "report_interval", "wall_diagnostic_interval", "surface_force_interval",
    ):
        time_ratios[field] = [
            float(item[2][field]) / coarse
            for coarse, item in zip(coarse_lengths, parsed, strict=True)
        ]
    time_ratios["statistics_window_steps_resolved"] = [
        float(item[3]["statistics"]["statistics_window_steps_resolved"]) / coarse
        for coarse, item in zip(coarse_lengths, parsed, strict=True)
    ]
    ratio_groups = [
        finest_ratios,
        *domain_ratios.values(),
        *outer_mesh_ratios.values(),
        *time_ratios.values(),
    ]
    scaled_configuration_invariant = (
        all(all(math.isfinite(value) for value in values) for values in ratio_groups)
        and all(_spread(values) <= 1.0e-12 for values in ratio_groups)
    )

    geometry_convergence_members = all(
        item[5]["resolution"].get("convergence_member_resolved") is True
        for item in parsed
    )
    finest_absolute_geometry = (
        parsed[-1][5]["resolution"].get("absolute_reference_resolved") is True
    )
    geometry_admitted = geometry_convergence_members and finest_absolute_geometry

    resistances = [item[1] for item in parsed]
    spatial = assess_spatial_convergence(resolutions, resistances)
    spatial_admitted = spatial.meets(
        maximum_finest_error_pct=maximum_finest_discretisation_error_pct,
        maximum_fit_rms_pct=maximum_fit_rms_pct,
        minimum_order=minimum_order,
    )
    experimental_values = {
        float(item[3]["statistics"]["experimental_resistance_n"])
        for item in parsed
    }
    reference_invariant = len(experimental_values) == 1
    experimental = next(iter(experimental_values)) if reference_invariant else math.nan
    extrapolated_error = (
        abs(spatial.extrapolated_value - experimental) / abs(experimental) * 100.0
        if reference_invariant and experimental != 0.0 else math.inf
    )
    provenance_admitted = (
        schema_valid
        and required_fields_present
        and identity_equal
        and scaled_configuration_invariant
        and hull_type_invariant
        and reference_invariant
    )
    admitted = (
        provenance_admitted
        and source_quality
        and geometry_admitted
        and spatial_admitted
        and extrapolated_error <= maximum_extrapolated_experiment_error_pct
    )
    return {
        "schema": "tensorlbm-suboff-nested-convergence-v1",
        "hull_type": (
            normalized_hull_types[0] if hull_type_invariant else None
        ),
        "fine_hull_resolutions": resolutions,
        "mean_resistances_n": resistances,
        "configuration_identity": {
            "source_schema_valid": schema_valid,
            "required_fields_present": required_fields_present,
            "identity_fields_equal": identity_equal,
            "finest_to_coarse_ratios": finest_ratios,
            "domain_over_coarse_length": domain_ratios,
            "outer_mesh_over_coarse_length": outer_mesh_ratios,
            "time_steps_over_coarse_length": time_ratios,
            "scaled_configuration_invariant": scaled_configuration_invariant,
            "experimental_reference_invariant": reference_invariant,
            "measured_hull_type_invariant": hull_type_invariant,
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
            "extrapolated_error_pct": extrapolated_error,
            "maximum_extrapolated_error_pct": (
                maximum_extrapolated_experiment_error_pct
            ),
        },
        "source_numerical_quality_admitted": source_quality,
        "geometry_resolution": {
            "source_convergence_members_admitted": geometry_convergence_members,
            "finest_absolute_reference_admitted": finest_absolute_geometry,
            "admitted": geometry_admitted,
        },
        "physical_validation": admitted,
        "admitted": admitted,
    }


__all__ = ["assess_suboff_nested_convergence"]

"""Fail-closed spatial-convergence assessment for canonical sphere drag."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .spatial_convergence import assess_spatial_convergence

if TYPE_CHECKING:
    from collections.abc import Sequence


_IDENTITY_FIELDS = (
    "schema_version",
    "center_x_fraction",
    "reynolds",
    "lattice_speed",
    "collision_model",
    "collision_chunk_cells",
    "compile_natural_kbc",
    "sponge_strength",
    "sponge_inlet",
    "far_field_mode",
    "minimum_statistics_convective_times",
)

_SCALED_TIME_FIELDS = (
    "steps",
    "warmup_steps",
    "ramp_steps",
    "statistics_window_steps",
    "report_interval",
)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def assess_sphere_grid_convergence(
    records: Sequence[dict[str, object]],
    *,
    maximum_finest_discretisation_error_pct: float = 3.0,
    maximum_fit_rms_pct: float = 1.0,
    minimum_order: float = 0.5,
    maximum_extrapolated_reference_error_pct: float = 5.0,
) -> dict[str, object]:
    """Assess three or more equivalent sphere grid records."""
    if len(records) < 3:
        raise ValueError("sphere grid convergence requires at least three records")
    parsed = []
    schema_valid = True
    source_quality = True
    for record in records:
        schema_valid &= record.get("schema") == "tensorlbm-sphere-bfl-control-volume-v3"
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("each record needs configuration and result mappings")
        if not isinstance(acceptance, dict):
            raise ValueError("each record needs an acceptance mapping")
        radius = float(configuration["radius"])
        parsed.append((radius, float(result["cd_control_volume"]), configuration, result))
        source_quality &= acceptance.get("numerical_quality_admitted") is True
    parsed.sort(key=lambda item: item[0])
    radii = [item[0] for item in parsed]
    if len(set(radii)) != len(radii) or any(radius <= 0.0 for radius in radii):
        raise ValueError("sphere radii must be unique and positive")

    baseline = parsed[0][2]
    required_fields = (
        *_IDENTITY_FIELDS, *_SCALED_TIME_FIELDS,
        "shape_zyx", "cv_margin", "sponge_width",
    )
    required_present = all(
        field in configuration
        for _, _, configuration, _ in parsed
        for field in required_fields
    )
    identity_equal = required_present and all(
        configuration.get(field) == baseline.get(field)
        for _, _, configuration, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )
    domain_ratios = {
        axis: [
            float(configuration["shape_zyx"][index]) / radius
            for radius, (_, _, configuration, _) in zip(radii, parsed, strict=True)
        ]
        for axis, index in (("z", 0), ("y", 1), ("x", 2))
    }
    spatial_ratios = {
        "cv_margin_over_radius": [
            float(configuration["cv_margin"]) / radius
            for radius, (_, _, configuration, _) in zip(radii, parsed, strict=True)
        ],
        "sponge_width_over_radius": [
            float(configuration["sponge_width"]) / radius
            for radius, (_, _, configuration, _) in zip(radii, parsed, strict=True)
        ],
    }
    time_ratios = {
        field: [
            float(configuration[field]) / radius
            for radius, (_, _, configuration, _) in zip(radii, parsed, strict=True)
        ]
        for field in _SCALED_TIME_FIELDS
    }
    ratio_groups = (*domain_ratios.values(), *spatial_ratios.values(), *time_ratios.values())
    scaled_invariant = (
        required_present
        and all(all(math.isfinite(value) for value in group) for group in ratio_groups)
        and all(_spread(group) <= 1e-12 for group in ratio_groups)
    )

    coefficients = [item[1] for item in parsed]
    spatial = assess_spatial_convergence([2.0 * radius for radius in radii], coefficients)
    references = {
        float(result["cd_reference_schiller_naumann"])
        for _, _, _, result in parsed
    }
    reference_invariant = len(references) == 1
    reference = next(iter(references)) if reference_invariant else math.nan
    reference_error = (
        abs(spatial.extrapolated_value - reference) / abs(reference) * 100.0
        if reference_invariant and reference != 0.0 else math.inf
    )
    spatial_admitted = spatial.meets(
        maximum_finest_error_pct=maximum_finest_discretisation_error_pct,
        maximum_fit_rms_pct=maximum_fit_rms_pct,
        minimum_order=minimum_order,
    )
    provenance_admitted = (
        schema_valid and required_present and identity_equal
        and scaled_invariant and reference_invariant
    )
    admitted = (
        provenance_admitted and source_quality and spatial_admitted
        and reference_error <= maximum_extrapolated_reference_error_pct
    )
    return {
        "schema": "tensorlbm-sphere-grid-convergence-v1",
        "radii_cells": radii,
        "diameters_cells": [2.0 * radius for radius in radii],
        "cd_control_volume": coefficients,
        "configuration_identity": {
            "v3_schema": schema_valid,
            "required_fields_present": required_present,
            "identity_fields_equal": identity_equal,
            "domain_over_radius": domain_ratios,
            "scaled_spatial_parameters": spatial_ratios,
            "time_steps_over_radius": time_ratios,
            "scaled_configuration_invariant": scaled_invariant,
            "reference_invariant": reference_invariant,
            "admitted": provenance_admitted,
        },
        "spatial_convergence": {
            "monotonic": spatial.monotonic,
            "observed_order": spatial.observed_order,
            "extrapolated_cd": spatial.extrapolated_value,
            "finest_discretisation_error_pct": spatial.finest_relative_error_pct,
            "relative_fit_rms_pct": spatial.relative_fit_rms_pct,
            "admitted": spatial_admitted,
        },
        "reference": {
            "schiller_naumann_cd": reference,
            "extrapolated_error_pct": reference_error,
            "maximum_error_pct": maximum_extrapolated_reference_error_pct,
        },
        "source_numerical_quality_admitted": source_quality,
        "physical_validation": admitted,
        "admitted": admitted,
    }


__all__ = ["assess_sphere_grid_convergence"]

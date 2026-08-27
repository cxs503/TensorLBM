"""Fail-closed grid convergence for Re=100 cylinder drag and shedding."""

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
    "periodic_axes",
    "minimum_shedding_cycles",
    "link_force_frame",
)
_TIME_FIELDS = (
    "steps",
    "warmup_steps",
    "ramp_steps",
    "statistics_window_steps_resolved",
    "report_interval",
)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def assess_cylinder_grid_convergence(
    records: Sequence[dict[str, object]],
    *,
    maximum_finest_discretisation_error_pct: float = 3.0,
    maximum_fit_rms_pct: float = 1.0,
    minimum_order: float = 0.5,
    maximum_reference_error_pct: float = 5.0,
) -> dict[str, object]:
    if len(records) < 3:
        raise ValueError("cylinder grid convergence requires at least three records")
    parsed = []
    schema_valid = True
    source_quality = True
    legacy_execution_defaults_normalized = 0
    for record in records:
        schema_valid &= record.get("schema") == "tensorlbm-cylinder-bfl-control-volume-v4"
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("each record needs configuration and result mappings")
        if not isinstance(acceptance, dict):
            raise ValueError("each record needs an acceptance mapping")
        configuration = dict(configuration)
        if (
            "collision_chunk_cells" not in configuration
            or "compile_natural_kbc" not in configuration
        ):
            legacy_execution_defaults_normalized += 1
            configuration.setdefault("collision_chunk_cells", 0)
            configuration.setdefault("compile_natural_kbc", False)
        radius = float(configuration["radius"])
        parsed.append(
            (
                radius,
                float(result["cd_control_volume"]),
                float(result["strouhal"]),
                configuration,
                result,
            )
        )
        source_quality &= acceptance.get("numerical_quality_admitted") is True
    parsed.sort(key=lambda item: item[0])
    radii = [item[0] for item in parsed]
    if len(set(radii)) != len(radii) or any(radius <= 0.0 for radius in radii):
        raise ValueError("cylinder radii must be unique and positive")

    baseline = parsed[0][3]
    required_fields = (
        *_IDENTITY_FIELDS,
        *_TIME_FIELDS,
        "shape_zyx",
        "cv_margin",
        "sponge_width",
    )
    required_present = all(
        field in configuration for *_, configuration, _ in parsed for field in required_fields
    )
    identity_equal = required_present and all(
        configuration.get(field) == baseline.get(field)
        for *_, configuration, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )
    spanwise_invariant = required_present and all(
        configuration["shape_zyx"][0] == baseline["shape_zyx"][0]
        for *_, configuration, _ in parsed[1:]
    )
    domain_ratios = {
        axis: [
            float(configuration["shape_zyx"][index]) / radius
            for radius, (*_, configuration, _) in zip(radii, parsed, strict=True)
        ]
        for axis, index in (("y", 1), ("x", 2))
    }
    spatial_ratios = {
        "cv_margin_over_radius": [
            float(configuration["cv_margin"]) / radius
            for radius, (*_, configuration, _) in zip(radii, parsed, strict=True)
        ],
        "sponge_width_over_radius": [
            float(configuration["sponge_width"]) / radius
            for radius, (*_, configuration, _) in zip(radii, parsed, strict=True)
        ],
    }
    time_ratios = {
        field: [
            float(configuration[field]) / radius
            for radius, (*_, configuration, _) in zip(radii, parsed, strict=True)
        ]
        for field in _TIME_FIELDS
    }
    ratio_groups = (*domain_ratios.values(), *spatial_ratios.values(), *time_ratios.values())
    scaled_invariant = (
        required_present
        and spanwise_invariant
        and all(all(math.isfinite(value) for value in group) for group in ratio_groups)
        and all(_spread(group) <= 1e-12 for group in ratio_groups)
    )

    diameters = [2.0 * radius for radius in radii]
    cd_values = [item[1] for item in parsed]
    st_values = [item[2] for item in parsed]
    cd_spatial = assess_spatial_convergence(diameters, cd_values)
    st_spatial = assess_spatial_convergence(diameters, st_values)
    cd_references = {float(item[4]["cd_reference"]) for item in parsed}
    st_references = {float(item[4]["strouhal_reference"]) for item in parsed}
    references_invariant = len(cd_references) == 1 and len(st_references) == 1
    cd_reference = next(iter(cd_references)) if len(cd_references) == 1 else math.nan
    st_reference = next(iter(st_references)) if len(st_references) == 1 else math.nan

    def reference_error(value: float, reference: float) -> float:
        return abs(value - reference) / abs(reference) * 100.0 if reference else math.inf

    cd_reference_error = reference_error(cd_spatial.extrapolated_value, cd_reference)
    st_reference_error = reference_error(st_spatial.extrapolated_value, st_reference)
    cd_spatial_admitted = cd_spatial.meets(
        maximum_finest_error_pct=maximum_finest_discretisation_error_pct,
        maximum_fit_rms_pct=maximum_fit_rms_pct,
        minimum_order=minimum_order,
    )
    st_spatial_admitted = st_spatial.meets(
        maximum_finest_error_pct=maximum_finest_discretisation_error_pct,
        maximum_fit_rms_pct=maximum_fit_rms_pct,
        minimum_order=minimum_order,
    )
    provenance_admitted = (
        schema_valid
        and required_present
        and identity_equal
        and scaled_invariant
        and references_invariant
    )
    admitted = (
        provenance_admitted
        and source_quality
        and cd_spatial_admitted
        and st_spatial_admitted
        and cd_reference_error <= maximum_reference_error_pct
        and st_reference_error <= maximum_reference_error_pct
    )

    def spatial_dict(report: object, accepted: bool) -> dict[str, object]:
        return {
            "monotonic": report.monotonic,
            "observed_order": report.observed_order,
            "extrapolated_value": report.extrapolated_value,
            "finest_discretisation_error_pct": report.finest_relative_error_pct,
            "relative_fit_rms_pct": report.relative_fit_rms_pct,
            "admitted": accepted,
        }

    return {
        "schema": "tensorlbm-cylinder-grid-convergence-v1",
        "radii_cells": radii,
        "diameters_cells": diameters,
        "cd_control_volume": cd_values,
        "strouhal": st_values,
        "configuration_identity": {
            "v4_schema": schema_valid,
            "required_fields_present": required_present,
            "legacy_execution_defaults_normalized": (legacy_execution_defaults_normalized),
            "identity_fields_equal": identity_equal,
            "spanwise_cells_invariant": spanwise_invariant,
            "domain_over_radius": domain_ratios,
            "scaled_spatial_parameters": spatial_ratios,
            "time_steps_over_radius": time_ratios,
            "scaled_configuration_invariant": scaled_invariant,
            "references_invariant": references_invariant,
            "admitted": provenance_admitted,
        },
        "drag_spatial_convergence": spatial_dict(cd_spatial, cd_spatial_admitted),
        "strouhal_spatial_convergence": spatial_dict(st_spatial, st_spatial_admitted),
        "reference": {
            "cd": cd_reference,
            "strouhal": st_reference,
            "cd_extrapolated_error_pct": cd_reference_error,
            "strouhal_extrapolated_error_pct": st_reference_error,
            "maximum_error_pct": maximum_reference_error_pct,
        },
        "source_numerical_quality_admitted": source_quality,
        "physical_validation": admitted,
        "admitted": admitted,
    }


__all__ = ["assess_cylinder_grid_convergence"]

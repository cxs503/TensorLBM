"""Two-by-two transverse-domain and inlet-sponge sphere audit."""
from __future__ import annotations

import math

_IDENTITY_FIELDS = (
    "schema_version",
    "radius",
    "center_x_fraction",
    "reynolds",
    "lattice_speed",
    "collision_model",
    "collision_chunk_cells",
    "compile_natural_kbc",
    "steps",
    "warmup_steps",
    "ramp_steps",
    "statistics_window_steps",
    "minimum_statistics_convective_times",
    "report_interval",
    "sponge_width",
    "sponge_strength",
    "cv_margin",
    "far_field_mode",
)


def assess_sphere_domain_inlet_factorial(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Audit all four cells of width×inlet-sponge without fitting drag."""
    if len(records) != 4:
        raise ValueError("sphere boundary factorial requires exactly four records")
    parsed = []
    schema_valid = True
    source_quality = True
    for record in records:
        schema_valid &= record.get("schema") == "tensorlbm-sphere-bfl-control-volume-v3"
        configuration = record.get("configuration")
        result = record.get("result")
        acceptance = record.get("acceptance")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("each sphere record needs configuration and result mappings")
        if not isinstance(acceptance, dict):
            raise ValueError("each sphere record needs an acceptance mapping")
        shape = configuration.get("shape_zyx")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError("each sphere record needs shape_zyx")
        if not isinstance(configuration.get("sponge_inlet"), bool):
            raise ValueError("each sphere record must declare sponge_inlet")
        radius = float(configuration["radius"])
        width = min(float(shape[0]), float(shape[1])) / (2.0 * radius)
        parsed.append((width, configuration["sponge_inlet"], configuration, result))
        source_quality &= acceptance.get("numerical_quality_admitted") is True

    widths = sorted({item[0] for item in parsed})
    if len(widths) != 2 or not widths[1] > widths[0]:
        raise ValueError("factorial requires exactly two ordered transverse widths")
    cells = {(item[0], item[1]): item for item in parsed}
    expected = {(width, enabled) for width in widths for enabled in (False, True)}
    if set(cells) != expected:
        raise ValueError("factorial requires both sponge states at both widths")
    baseline = parsed[0][2]
    required_present = all(
        field in configuration
        for _, _, configuration, _ in parsed
        for field in (*_IDENTITY_FIELDS, "shape_zyx", "sponge_inlet")
    )
    identity_equal = required_present and all(
        configuration.get(field) == baseline.get(field)
        for _, _, configuration, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )
    streamwise_fixed = required_present and all(
        configuration["shape_zyx"][2] == baseline["shape_zyx"][2]
        for _, _, configuration, _ in parsed
    )
    transverse_isotropic = required_present and all(
        configuration["shape_zyx"][0] == configuration["shape_zyx"][1]
        for _, _, configuration, _ in parsed
    )
    references = {
        float(result["cd_reference_schiller_naumann"])
        for _, _, _, result in parsed
    }
    reference_invariant = len(references) == 1
    reference = next(iter(references)) if reference_invariant else math.nan
    provenance_admitted = (
        schema_valid and required_present and identity_equal
        and streamwise_fixed and transverse_isotropic and reference_invariant
    )

    cd = {
        (width, enabled): float(cells[(width, enabled)][3]["cd_control_volume"])
        for width, enabled in expected
    }
    narrow, wide = widths
    grand_mean = sum(cd.values()) / 4.0
    inlet_effect_narrow = cd[(narrow, True)] - cd[(narrow, False)]
    inlet_effect_wide = cd[(wide, True)] - cd[(wide, False)]
    domain_effect_without = cd[(wide, False)] - cd[(narrow, False)]
    domain_effect_with = cd[(wide, True)] - cd[(narrow, True)]
    interaction = inlet_effect_wide - inlet_effect_narrow

    def relative(value: float) -> float:
        return value / max(abs(grand_mean), 1.0e-30) * 100.0

    ordered_cells = [
        {
            "lateral_width_diameters": width,
            "sponge_inlet": enabled,
            "cd_control_volume": cd[(width, enabled)],
            "reference_error_pct": (
                abs(cd[(width, enabled)] - reference)
                / max(abs(reference), 1.0e-30)
                * 100.0
            ) if reference_invariant else math.inf,
        }
        for width in widths for enabled in (False, True)
    ]
    return {
        "schema": "tensorlbm-sphere-domain-inlet-factorial-v1",
        "physical_validation": False,
        "factorial_only_not_boundary_validation": True,
        "configuration_identity": {
            "source_schema_valid": schema_valid,
            "required_fields_present": required_present,
            "identity_fields_equal": identity_equal,
            "streamwise_cells_fixed": streamwise_fixed,
            "transverse_axes_isotropic": transverse_isotropic,
            "reference_invariant": reference_invariant,
            "admitted": provenance_admitted,
        },
        "cells": ordered_cells,
        "effects": {
            "grand_mean_cd": grand_mean,
            "inlet_sponge_effect_at_narrow_width_cd": inlet_effect_narrow,
            "inlet_sponge_effect_at_narrow_width_pct": relative(inlet_effect_narrow),
            "inlet_sponge_effect_at_wide_width_cd": inlet_effect_wide,
            "inlet_sponge_effect_at_wide_width_pct": relative(inlet_effect_wide),
            "domain_effect_without_inlet_sponge_cd": domain_effect_without,
            "domain_effect_without_inlet_sponge_pct": relative(domain_effect_without),
            "domain_effect_with_inlet_sponge_cd": domain_effect_with,
            "domain_effect_with_inlet_sponge_pct": relative(domain_effect_with),
            "width_by_inlet_interaction_cd": interaction,
            "width_by_inlet_interaction_pct": relative(interaction),
        },
        "source_numerical_quality_admitted": source_quality,
        "admitted_as_factorial": provenance_admitted and source_quality,
        "claim_boundary": (
            "Effects are direct signed contrasts; no cell is selected or "
            "corrected by proximity to the reference."
        ),
    }


__all__ = ["assess_sphere_domain_inlet_factorial"]

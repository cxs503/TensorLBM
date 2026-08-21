"""Fail-closed two-domain sensitivity audit for canonical sphere drag."""

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
    "sponge_inlet",
    "cv_margin",
    "far_field_mode",
)


def assess_sphere_domain_sensitivity_pair(
    records: list[dict[str, object]],
    *,
    maximum_drag_change_pct: float = 1.0,
) -> dict[str, object]:
    """Compare fixed-resolution sphere runs in two transverse domains.

    A pair can bound sensitivity but cannot establish domain convergence, so
    ``physical_validation`` is always false.  No blockage correction is
    applied to either direct CFD force.
    """
    if len(records) != 2:
        raise ValueError("sphere domain sensitivity requires exactly two records")
    if not math.isfinite(maximum_drag_change_pct) or maximum_drag_change_pct <= 0.0:
        raise ValueError("maximum drag change must be finite and positive")

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
            raise ValueError("each sphere record needs a three-dimensional shape_zyx")
        radius = float(configuration["radius"])
        if radius <= 0.0:
            raise ValueError("sphere radius must be positive")
        width = min(float(shape[0]), float(shape[1])) / (2.0 * radius)
        parsed.append((width, configuration, result, acceptance))
        source_quality &= acceptance.get("numerical_quality_admitted") is True
    parsed.sort(key=lambda item: item[0])
    if not parsed[1][0] > parsed[0][0]:
        raise ValueError("expanded transverse domain must be wider")

    baseline_configuration = parsed[0][1]
    required_present = all(
        field in configuration
        for _, configuration, _, _ in parsed
        for field in (*_IDENTITY_FIELDS, "shape_zyx")
    )
    identity_equal = required_present and all(
        parsed[1][1].get(field) == baseline_configuration.get(field) for field in _IDENTITY_FIELDS
    )
    baseline_shape = baseline_configuration["shape_zyx"]
    expanded_shape = parsed[1][1]["shape_zyx"]
    streamwise_fixed = expanded_shape[2] == baseline_shape[2]
    transverse_isotropic = (
        baseline_shape[0] == baseline_shape[1] and expanded_shape[0] == expanded_shape[1]
    )
    expanded_only_transverse = (
        streamwise_fixed and transverse_isotropic and expanded_shape[0] > baseline_shape[0]
    )

    cd_values = [float(item[2]["cd_control_volume"]) for item in parsed]
    drag_change_pct = abs(cd_values[1] - cd_values[0]) / max(abs(cd_values[1]), 1.0e-30) * 100.0
    references = {float(item[2]["cd_reference_schiller_naumann"]) for item in parsed}
    reference_invariant = len(references) == 1
    reference = next(iter(references)) if reference_invariant else math.nan
    reference_errors = (
        [abs(value - reference) / max(abs(reference), 1.0e-30) * 100.0 for value in cd_values]
        if reference_invariant
        else [math.inf, math.inf]
    )
    provenance_admitted = (
        schema_valid
        and required_present
        and identity_equal
        and expanded_only_transverse
        and reference_invariant
    )
    sensitivity_within_tolerance = (
        provenance_admitted and source_quality and drag_change_pct <= maximum_drag_change_pct
    )
    return {
        "schema": "tensorlbm-sphere-domain-sensitivity-pair-v1",
        "physical_validation": False,
        "pair_only_not_domain_convergence": True,
        "lateral_width_diameters": [item[0] for item in parsed],
        "cd_control_volume": cd_values,
        "configuration_identity": {
            "source_schema_valid": schema_valid,
            "required_fields_present": required_present,
            "identity_fields_equal": identity_equal,
            "streamwise_cells_fixed": streamwise_fixed,
            "transverse_axes_isotropic": transverse_isotropic,
            "expanded_only_transverse": expanded_only_transverse,
            "admitted": provenance_admitted,
        },
        "domain_sensitivity": {
            "drag_change_pct": drag_change_pct,
            "maximum_drag_change_pct": maximum_drag_change_pct,
            "within_tolerance": sensitivity_within_tolerance,
        },
        "reference": {
            "schiller_naumann_cd": reference,
            "reference_error_pct_by_width": reference_errors,
            "invariant": reference_invariant,
        },
        "source_numerical_quality_admitted": source_quality,
        "admitted_as_pair_sensitivity": sensitivity_within_tolerance,
        "next_required_evidence": ("Add a third, wider domain before claiming domain convergence."),
    }


def assess_sphere_domain_convergence(
    records: list[dict[str, object]],
    *,
    maximum_finest_drag_change_pct: float = 1.0,
    maximum_reference_error_pct: float = 5.0,
) -> dict[str, object]:
    """Assess three or more direct transverse-domain sphere trajectories."""
    if len(records) < 3:
        raise ValueError("sphere domain convergence requires at least three records")
    if min(maximum_finest_drag_change_pct, maximum_reference_error_pct) <= 0.0:
        raise ValueError("domain and reference tolerances must be positive")

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
            raise ValueError("each sphere record needs a three-dimensional shape_zyx")
        radius = float(configuration["radius"])
        width = min(float(shape[0]), float(shape[1])) / (2.0 * radius)
        parsed.append((width, configuration, result))
        source_quality &= acceptance.get("numerical_quality_admitted") is True
    parsed.sort(key=lambda item: item[0])
    widths = [item[0] for item in parsed]
    if len(set(widths)) != len(widths) or any(width <= 0.0 for width in widths):
        raise ValueError("lateral domain widths must be unique and positive")

    baseline = parsed[0][1]
    required_present = all(
        field in configuration
        for _, configuration, _ in parsed
        for field in (*_IDENTITY_FIELDS, "shape_zyx")
    )
    identity_equal = required_present and all(
        configuration.get(field) == baseline.get(field)
        for _, configuration, _ in parsed[1:]
        for field in _IDENTITY_FIELDS
    )
    streamwise_fixed = required_present and all(
        configuration["shape_zyx"][2] == baseline["shape_zyx"][2]
        for _, configuration, _ in parsed[1:]
    )
    transverse_isotropic = required_present and all(
        configuration["shape_zyx"][0] == configuration["shape_zyx"][1]
        for _, configuration, _ in parsed
    )
    provenance_admitted = (
        schema_valid
        and required_present
        and identity_equal
        and streamwise_fixed
        and transverse_isotropic
    )

    cd_values = [float(item[2]["cd_control_volume"]) for item in parsed]
    differences = [right - left for left, right in zip(cd_values, cd_values[1:], strict=False)]
    monotonic = all(value >= 0.0 for value in differences) or all(
        value <= 0.0 for value in differences
    )
    finest_change_pct = (
        abs(cd_values[-1] - cd_values[-2]) / max(abs(cd_values[-1]), 1.0e-30) * 100.0
    )
    domain_convergence_admitted = (
        provenance_admitted
        and source_quality
        and monotonic
        and finest_change_pct <= maximum_finest_drag_change_pct
    )
    references = {float(item[2]["cd_reference_schiller_naumann"]) for item in parsed}
    reference_invariant = len(references) == 1
    reference = next(iter(references)) if reference_invariant else math.nan
    finest_reference_error_pct = (
        abs(cd_values[-1] - reference) / max(abs(reference), 1.0e-30) * 100.0
        if reference_invariant
        else math.inf
    )
    physical_validation = (
        domain_convergence_admitted
        and reference_invariant
        and finest_reference_error_pct <= maximum_reference_error_pct
    )
    return {
        "schema": "tensorlbm-sphere-domain-convergence-v1",
        "physical_validation": physical_validation,
        "lateral_width_diameters": widths,
        "cd_control_volume": cd_values,
        "configuration_identity": {
            "source_schema_valid": schema_valid,
            "required_fields_present": required_present,
            "identity_fields_equal": identity_equal,
            "streamwise_cells_fixed": streamwise_fixed,
            "transverse_axes_isotropic": transverse_isotropic,
            "admitted": provenance_admitted,
        },
        "domain_convergence": {
            "drag_monotonic": monotonic,
            "finest_drag_change_pct": finest_change_pct,
            "maximum_finest_drag_change_pct": maximum_finest_drag_change_pct,
            "admitted": domain_convergence_admitted,
        },
        "reference": {
            "schiller_naumann_cd": reference,
            "finest_reference_error_pct": finest_reference_error_pct,
            "maximum_reference_error_pct": maximum_reference_error_pct,
            "invariant": reference_invariant,
            "admitted": (
                reference_invariant and finest_reference_error_pct <= maximum_reference_error_pct
            ),
        },
        "source_numerical_quality_admitted": source_quality,
        "admitted": physical_validation,
        "claim_boundary": (
            "No empirical blockage correction or assumed asymptotic order "
            "is applied to the direct CFD drag sequence."
        ),
    }


__all__ = [
    "assess_sphere_domain_convergence",
    "assess_sphere_domain_sensitivity_pair",
]

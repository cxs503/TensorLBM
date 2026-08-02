"""Matched inlet-sponge sensitivity audit for canonical sphere drag."""
from __future__ import annotations

import math

_IDENTITY_FIELDS = (
    "schema_version",
    "shape_zyx",
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


def assess_sphere_inlet_sponge_pair(
    records: list[dict[str, object]],
    *,
    maximum_drag_change_pct: float = 1.0,
) -> dict[str, object]:
    """Audit two direct CFD runs differing only by upstream sponge damping."""
    if len(records) != 2:
        raise ValueError("sphere inlet-sponge sensitivity requires exactly two records")
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
        if not isinstance(configuration.get("sponge_inlet"), bool):
            raise ValueError("each sphere configuration must declare sponge_inlet")
        parsed.append((configuration["sponge_inlet"], configuration, result))
        source_quality &= acceptance.get("numerical_quality_admitted") is True
    parsed.sort(key=lambda item: item[0])
    if [item[0] for item in parsed] != [False, True]:
        raise ValueError("pair must contain one disabled and one enabled inlet sponge")

    baseline = parsed[0][1]
    required_present = all(
        field in configuration
        for _, configuration, _ in parsed
        for field in _IDENTITY_FIELDS
    )
    identity_equal = required_present and all(
        parsed[1][1].get(field) == baseline.get(field)
        for field in _IDENTITY_FIELDS
    )
    cd_values = [float(item[2]["cd_control_volume"]) for item in parsed]
    drag_change_pct = (
        abs(cd_values[1] - cd_values[0])
        / max(abs(cd_values[1]), 1.0e-30)
        * 100.0
    )
    references = {
        float(item[2]["cd_reference_schiller_naumann"]) for item in parsed
    }
    reference_invariant = len(references) == 1
    reference = next(iter(references)) if reference_invariant else math.nan
    reference_errors = [
        abs(value - reference) / max(abs(reference), 1.0e-30) * 100.0
        for value in cd_values
    ] if reference_invariant else [math.inf, math.inf]
    provenance_admitted = (
        schema_valid and required_present and identity_equal and reference_invariant
    )
    sensitivity_within_tolerance = (
        provenance_admitted and source_quality
        and drag_change_pct <= maximum_drag_change_pct
    )
    return {
        "schema": "tensorlbm-sphere-inlet-sponge-sensitivity-v1",
        "physical_validation": False,
        "ab_only_not_boundary_validation": True,
        "sponge_inlet": [False, True],
        "cd_control_volume": cd_values,
        "configuration_identity": {
            "source_schema_valid": schema_valid,
            "required_fields_present": required_present,
            "identity_fields_equal": identity_equal,
            "reference_invariant": reference_invariant,
            "admitted": provenance_admitted,
        },
        "boundary_sensitivity": {
            "drag_change_pct": drag_change_pct,
            "maximum_drag_change_pct": maximum_drag_change_pct,
            "within_tolerance": sensitivity_within_tolerance,
        },
        "reference": {
            "schiller_naumann_cd": reference,
            "reference_error_pct_without_and_with_inlet_sponge": reference_errors,
        },
        "source_numerical_quality_admitted": source_quality,
        "admitted_as_boundary_sensitivity": sensitivity_within_tolerance,
        "claim_boundary": (
            "This A/B measures sensitivity; it does not select the result "
            "closest to reference or validate either open boundary."
        ),
    }


__all__ = ["assess_sphere_inlet_sponge_pair"]

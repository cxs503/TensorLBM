from __future__ import annotations

import copy

import pytest

from tensorlbm.sphere_boundary_factorial import (
    assess_sphere_domain_inlet_factorial,
)


def _record(width_diameters: float, sponge_inlet: bool, cd: float) -> dict:
    radius = 9.0
    cross = int(width_diameters * 2.0 * radius)
    return {
        "schema": "tensorlbm-sphere-bfl-control-volume-v3",
        "configuration": {
            "schema_version": 3,
            "shape_zyx": [cross, cross, 216],
            "radius": radius,
            "center_x_fraction": 0.3,
            "reynolds": 100.0,
            "lattice_speed": 0.06,
            "collision_model": "natural_kbc_d3q19",
            "collision_chunk_cells": 262144,
            "compile_natural_kbc": True,
            "steps": 7200,
            "warmup_steps": 4800,
            "ramp_steps": 720,
            "statistics_window_steps": 2400,
            "minimum_statistics_convective_times": 5.0,
            "report_interval": 360,
            "sponge_width": 18,
            "sponge_strength": 0.2,
            "sponge_inlet": sponge_inlet,
            "cv_margin": 6,
            "far_field_mode": "non_equilibrium_extrapolation",
        },
        "result": {
            "cd_control_volume": cd,
            "cd_reference_schiller_naumann": 1.0917310910948732,
        },
        "acceptance": {"numerical_quality_admitted": True},
    }


def _factorial() -> list[dict]:
    return [
        _record(8.0, False, 1.16),
        _record(8.0, True, 1.17),
        _record(12.0, False, 1.15),
        _record(12.0, True, 1.16),
    ]


def test_sphere_factorial_reports_zero_additive_interaction() -> None:
    result = assess_sphere_domain_inlet_factorial(_factorial())

    assert result["configuration_identity"]["admitted"] is True
    assert result["effects"]["width_by_inlet_interaction_cd"] == pytest.approx(0.0)
    assert result["effects"]["domain_effect_with_inlet_sponge_cd"] == pytest.approx(-0.01)
    assert result["effects"]["inlet_sponge_effect_at_wide_width_cd"] == pytest.approx(0.01)
    assert result["admitted_as_factorial"] is True
    assert result["physical_validation"] is False


def test_sphere_factorial_rejects_changed_numerics() -> None:
    records = _factorial()
    records[-1] = copy.deepcopy(records[-1])
    records[-1]["configuration"]["compile_natural_kbc"] = False
    result = assess_sphere_domain_inlet_factorial(records)

    assert result["configuration_identity"]["identity_fields_equal"] is False
    assert result["admitted_as_factorial"] is False


def test_sphere_factorial_requires_all_four_cells() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        assess_sphere_domain_inlet_factorial(_factorial()[:3])

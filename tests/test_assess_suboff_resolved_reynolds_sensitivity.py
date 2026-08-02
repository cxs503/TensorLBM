from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "assess_suboff_resolved_reynolds_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("resolved_re_assessor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _result(path: Path, resolved_reynolds: float, resistance: float) -> Path:
    configuration = dict.fromkeys(MODULE.IDENTITY_FIELDS, 1)
    configuration.update({
        "hull_type": "bare_hull",
        "aux_cv_margins": [4, 12],
        "collision_model": "natural_kbc",
        "compile_natural_kbc": True,
        "natural_kbc_compute_dtype": "storage",
        "population_storage_dtype": "float32",
        "tau_by_level": [
            0.5 + 16.2 * 2**level / resolved_reynolds
            for level in range(4)
        ],
        "wall_law": "musker",
        "sponge_inlet": True,
        "far_field_mode": "non_equilibrium_extrapolation",
        "ghost_interpolation": "trilinear",
        "reflux_correction_stencil": "exterior_cells",
        "resolved_reynolds": resolved_reynolds,
    })
    path.write_text(json.dumps({
        "schema": "tensorlbm-suboff-nested-amr-smoke-v3",
        "configuration": configuration,
        "result": {"statistics": {
            "mean_resistance_n": resistance,
            "mean_bfl_plus_wall_stress_n": resistance + 0.1,
            "mean_bfl_pressure_n": resistance - 80.0,
            "mean_wall_shear_n": 80.1,
            "sampling_convective_times": 1.0,
        }},
        "acceptance": {
            "integration_smoke_admitted": True,
            "population_health_target_met": True,
            "nested_control_volume_target_met": True,
            "stationarity_target_met": False,
        },
    }), encoding="utf-8")
    return path


def _schedule(path: Path, result_path: Path, *, admitted: bool = True) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    configuration = result["configuration"]
    path.write_text(json.dumps({
        "schema": "tensorlbm-collision-viscosity-schedule-v1",
        "status": "admitted" if admitted else "rejected",
        "dtype": configuration["population_storage_dtype"],
        "natural_kbc_compute_dtype": configuration[
            "natural_kbc_compute_dtype"
        ],
        "taus": configuration["tau_by_level"],
        "acceptance": {
            "all_levels_recover_configured_viscosity": admitted,
        },
    }), encoding="utf-8")
    return path


def test_assessor_accepts_matched_causal_sequence_without_physical_claim(tmp_path) -> None:
    paths = [
        _result(tmp_path / "r200.json", 200_000.0, 116.0),
        _result(tmp_path / "r500.json", 500_000.0, 112.0),
        _result(tmp_path / "r1000.json", 1_000_000.0, 109.0),
    ]
    schedules = [
        _schedule(tmp_path / f"schedule-{index}.json", path)
        for index, path in enumerate(paths)
    ]
    result = MODULE.assess(paths, viscosity_schedule_paths=schedules)
    assert result["status"] == "causal_sequence_admitted"
    assert result["physical_validation"] is False
    assert result["component_trends"]["mean_resistance_n"]["trend"] == (
        "strictly_decreasing"
    )
    assert result["acceptance"]["continuum_reynolds_extrapolation_admitted"] is False


def test_assessor_rejects_uncertified_configured_reynolds(tmp_path) -> None:
    paths = [
        _result(tmp_path / "r200.json", 200_000.0, 116.0),
        _result(tmp_path / "r500.json", 500_000.0, 112.0),
        _result(tmp_path / "r1000.json", 1_000_000.0, 109.0),
    ]
    result = MODULE.assess(paths)
    assert result["status"] == "rejected_uncertified_collision_viscosity"
    assert result["acceptance"][
        "all_collision_viscosity_schedules_admitted"
    ] is False
    assert result["acceptance"]["causal_sensitivity_admitted"] is False


def test_assessor_rejects_identity_mismatch(tmp_path) -> None:
    paths = [
        _result(tmp_path / "a.json", 200_000.0, 116.0),
        _result(tmp_path / "b.json", 500_000.0, 112.0),
        _result(tmp_path / "c.json", 1_000_000.0, 109.0),
    ]
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["configuration"]["nx"] = 2
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        MODULE.assess(paths)


def test_assessor_requires_three_cases(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        MODULE.assess([
            _result(tmp_path / "a.json", 200_000.0, 116.0),
            _result(tmp_path / "b.json", 500_000.0, 112.0),
        ])

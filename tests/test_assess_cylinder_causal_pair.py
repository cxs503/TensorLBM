from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / ("assess_cylinder_causal_pair.py")
SPEC = importlib.util.spec_from_file_location("cylinder_pair_assessor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _result(
    path: Path,
    *,
    speed: float = 0.06,
    collision: str = "natural_kbc_d3q19",
    time_scale: int = 1,
) -> Path:
    configuration = {
        "shape_zyx": [3, 360, 360],
        "radius": 9.0,
        "center_x_fraction": 0.3,
        "reynolds": 100.0,
        "lattice_speed": speed,
        "collision_model": collision,
        "warmup_steps": 31_500 * time_scale,
        "ramp_steps": 450 * time_scale,
        "sponge_width": 18,
        "sponge_strength": 0.2,
        "sponge_inlet": False,
        "cv_margin": 6,
        "far_field_mode": "non_equilibrium_extrapolation",
        "periodic_axes": ["z"],
        "link_force_frame": "laboratory_after_wall_activation",
        "tau": 0.5 + 3.0 * speed * 18.0 / 100.0,
        "steps": 54_000 * time_scale,
        "statistics_window_steps_resolved": 22_500 * time_scale,
        "minimum_shedding_cycles": 8.0,
        "domain_clearance_diameters": {
            "upstream_center_distance": 6.0,
            "downstream_center_distance": 14.0,
            "lateral_center_distance": 10.0,
        },
    }
    path.write_text(
        json.dumps(
            {
                "schema": "tensorlbm-cylinder-bfl-control-volume-v4",
                "configuration": configuration,
                "result": {
                    "cd_control_volume": 1.44,
                    "cd_bfl_link": 1.44001,
                    "strouhal": 0.173,
                    "observer_difference_pct": 0.001,
                    "shedding_cycles_observed": 13.0,
                    "drag_stationarity": {"relative_range_pct": 0.7},
                },
                "acceptance": {
                    "numerical_quality_admitted": True,
                    "stationarity_target_met": True,
                    "force_observer_target_met": True,
                    "cycle_target_met": True,
                    "domain_reference_target_met": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_admits_only_collision_change(tmp_path: Path) -> None:
    baseline = _result(tmp_path / "a.json", collision="natural_kbc_d3q19")
    candidate = _result(tmp_path / "b.json", collision="planar_cumulant_d2q9")
    result = MODULE.assess(baseline, candidate, "collision_model")
    assert result["status"] == "causal_pair_admitted"
    assert result["physical_validation"] is False


def test_admits_lower_mach_at_equal_convective_time(tmp_path: Path) -> None:
    baseline = _result(tmp_path / "a.json", speed=0.06, time_scale=1)
    candidate = _result(tmp_path / "b.json", speed=0.03, time_scale=2)
    result = MODULE.assess(baseline, candidate, "lattice_mach")
    assert result["status"] == "causal_pair_admitted"
    assert result["candidate"]["lattice_mach"] < result["baseline"]["lattice_mach"]


def test_rejects_hidden_geometry_change(tmp_path: Path) -> None:
    baseline = _result(tmp_path / "a.json")
    candidate = _result(tmp_path / "b.json", collision="planar_cumulant_d2q9")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["configuration"]["radius"] = 10.0
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        MODULE.assess(baseline, candidate, "collision_model")


def test_rejects_unmatched_low_mach_time_window(tmp_path: Path) -> None:
    baseline = _result(tmp_path / "a.json", speed=0.06)
    candidate = _result(tmp_path / "b.json", speed=0.03)
    with pytest.raises(ValueError, match="convective-time equivalence"):
        MODULE.assess(baseline, candidate, "lattice_mach")

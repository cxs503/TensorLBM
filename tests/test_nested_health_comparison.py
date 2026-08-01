from __future__ import annotations

import pytest

from tensorlbm.nested_health_comparison import (
    compare_nested_health,
    read_nested_health_log,
)


def _record(step: int, speed: float, population: float) -> dict:
    return {
        "step": step,
        "target_reynolds_reached": step >= 2,
        "maximum_collision_limited_fraction": 0.0,
        "levels": [{
            "finite": True,
            "maximum_speed": speed,
            "minimum_population": population,
        }],
        "interfaces": [{"maximum_reflux_residual": 1.0e-10 * step}],
    }


def test_health_log_reader_ignores_reports_and_partial_json(tmp_path) -> None:
    path = tmp_path / "health.log"
    path.write_text(
        "startup\n"
        'nested health {"step":1,"levels":[],"interfaces":[]}\n'
        "nested smoke step=1/2\n"
        'nested health {"step":2\n',
        encoding="utf-8",
    )

    assert read_nested_health_log(path) == [{
        "step": 1,
        "levels": [],
        "interfaces": [],
    }]


def test_comparison_aligns_steps_and_reports_worst_and_latest_ratios() -> None:
    baseline = [_record(1, 0.1, 0.02), _record(2, 0.2, 0.01)]
    candidate = [_record(2, 0.12, 0.018), _record(3, 0.11, 0.019)]

    comparison = compare_nested_health(baseline, candidate)

    assert comparison["common_step_count"] == 1
    assert comparison["latest_common_step"] == 2
    assert comparison["baseline_maximum_speed"] == pytest.approx(0.2)
    assert comparison["candidate_maximum_speed"] == pytest.approx(0.12)
    assert comparison["latest_candidate_to_baseline_speed_ratio"] == pytest.approx(0.6)
    assert comparison["candidate_minimum_population"] == pytest.approx(0.018)


def test_comparison_requires_a_common_step() -> None:
    with pytest.raises(ValueError, match="no common steps"):
        compare_nested_health([_record(1, 0.1, 0.02)], [_record(2, 0.1, 0.02)])

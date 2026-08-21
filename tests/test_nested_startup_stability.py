from __future__ import annotations

from copy import deepcopy

import pytest

from tensorlbm.nested_startup_stability import assess_nested_startup


def _result() -> dict:
    return {
        "configuration": {"resolved_reynolds": 100000.0},
        "result": {
            "maximum_positivity_limited_fraction": 0.0,
            "steps": [
                {"step": 1, "collision_resolved_reynolds": 5000.0},
                {"step": 2, "collision_resolved_reynolds": 100000.0},
            ],
            "population_health": [
                {
                    "step": 2,
                    "target_reynolds_reached": True,
                    "levels": [
                        {
                            "finite": True,
                            "minimum_population": 0.01,
                            "minimum_density": 0.9,
                            "maximum_density": 1.1,
                            "maximum_speed": 0.12,
                        }
                    ],
                    "interfaces": [
                        {
                            "maximum_reflux_residual": 1.0e-10,
                            "restriction_limited_fraction": 0.0,
                        }
                    ],
                }
            ],
        },
    }


def test_clean_target_reynolds_startup_passes_without_force_claim() -> None:
    assessment = assess_nested_startup(_result())

    assert assessment.status == "startup_stability_pass"
    assert assessment.startup_stability_pass is True
    assert assessment.target_reynolds_steps == 1
    assert assessment.maximum_speed == pytest.approx(0.12)
    assert assessment.failure_reasons == ()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("speed", "weakly_compressible_speed_gate_failed"),
        ("population", "nonpositive_or_missing_population"),
        ("transfer", "transfer_limiter_gate_failed"),
        ("collision", "collision_limiter_gate_failed"),
    ),
)
def test_health_and_limiter_failures_are_fail_closed(
    mutation: str,
    reason: str,
) -> None:
    result = deepcopy(_result())
    if mutation == "speed":
        result["result"]["population_health"][0]["levels"][0]["maximum_speed"] = 0.4
    elif mutation == "population":
        result["result"]["population_health"][0]["levels"][0]["minimum_population"] = 0.0
    elif mutation == "transfer":
        interface = result["result"]["population_health"][0]["interfaces"][0]
        interface["restriction_limited_fraction"] = 0.002
    else:
        result["result"]["maximum_positivity_limited_fraction"] = 0.002

    assessment = assess_nested_startup(result)

    assert assessment.startup_stability_pass is False
    assert reason in assessment.failure_reasons


def test_target_reynolds_is_required_unless_explicitly_waived() -> None:
    result = _result()
    result["result"]["steps"] = result["result"]["steps"][:1]
    result["result"]["population_health"][0]["target_reynolds_reached"] = False

    strict = assess_nested_startup(result)
    diagnostic = assess_nested_startup(result, require_target_reynolds=False)

    assert "target_reynolds_not_reached" in strict.failure_reasons
    assert diagnostic.startup_stability_pass is True


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="maximum speed"):
        assess_nested_startup(_result(), maximum_speed=1.0)

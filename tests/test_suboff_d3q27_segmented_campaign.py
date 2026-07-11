"""TDD coverage for D3Q27 SUBOFF segmented campaign planning only."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tensorlbm.suboff_segmented_campaign import build_suboff_segmented_campaign_manifest


def _static_config() -> dict[str, object]:
    return {
        "checkpoint_metadata": {
            "format": "suboff-d3q27-cumulant-xslab-v1",
            "nx": 416, "ny": 208, "nz": 208, "hull_length": 206.0,
            "re": 2_000_000.0, "u_in": 0.06, "y_val": 0.5,
            "world_size": 4, "rank": 0, "nx_local": 104, "q": 27,
        },
        "far_field": {
            "required_transient_steps": 1_200,
            "transient_steps_satisfy_outlet_convection": True,
        },
        "transient_steps": 1_200,
        "warmup_steps": 1_500,
        "dynamic_pressure_wetted_area": 3.25,
        "checkpoint_path_template": "checkpoints/suboff/step-{end_step}",
    }


def test_planner_generates_exact_gap_free_segments_and_post_gate_blocks():
    manifest = build_suboff_segmented_campaign_manifest(
        _static_config(), segment_steps=1_000, segment_count=4
    )

    assert manifest["schema"] == "suboff-d3q27-segmented-run-v1"
    assert manifest["transient_steps"] == 1_200
    assert manifest["warmup_steps"] == 1_500
    assert manifest["expected_ranks"] == [0, 1, 2, 3]
    assert manifest["segments"] == [
        {"start_step": 0, "end_step": 1_000, "checkpoint": "checkpoints/suboff/step-1000"},
        {"start_step": 1_000, "end_step": 2_000, "checkpoint": "checkpoints/suboff/step-2000"},
        {"start_step": 2_000, "end_step": 3_000, "checkpoint": "checkpoints/suboff/step-3000"},
        {"start_step": 3_000, "end_step": 4_000, "checkpoint": "checkpoints/suboff/step-4000"},
    ]
    # The first Ct block is clipped only by an explicit safe requested start;
    # it is never allowed to include the transient/warmup gate.
    assert manifest["blocks"] == [
        {"first_sample_step": 1_501, "last_sample_step": 2_000,
         "friction_sum": None, "pressure_sum": None, "drag_samples": 500},
        {"first_sample_step": 2_001, "last_sample_step": 3_000,
         "friction_sum": None, "pressure_sum": None, "drag_samples": 1_000},
        {"first_sample_step": 3_001, "last_sample_step": 4_000,
         "friction_sum": None, "pressure_sum": None, "drag_samples": 1_000},
    ]
    assert manifest["planned_completed_step"] == 4_000
    assert manifest["planning_only"] is True


def test_planner_can_start_ct_later_but_rejects_ct_before_outlet_convection():
    config = _static_config()
    config["ct_start_step"] = 2_001
    manifest = build_suboff_segmented_campaign_manifest(config, segment_steps=1_000, segment_count=3)
    assert manifest["blocks"][0]["first_sample_step"] == 2_001

    config = _static_config()
    config["ct_start_step"] = 1_200
    with pytest.raises(ValueError, match="outlet convection"):
        build_suboff_segmented_campaign_manifest(config, segment_steps=1_000, segment_count=3)


def test_planner_rejects_incomplete_outlet_transient_and_insufficient_campaign():
    config = _static_config()
    config["transient_steps"] = 1_199
    with pytest.raises(ValueError, match="outlet convection"):
        build_suboff_segmented_campaign_manifest(config, segment_steps=1_000, segment_count=3)

    with pytest.raises(ValueError, match="no wholly post-gate Ct samples"):
        build_suboff_segmented_campaign_manifest(_static_config(), segment_steps=500, segment_count=3)


@pytest.mark.parametrize("field, value", [
    ("checkpoint_path_template", "/absolute/step-{end_step}"),
    ("checkpoint_path_template", "checkpoints/../step-{end_step}"),
    ("checkpoint_path_template", "checkpoints/suboff/fixed"),
])
def test_planner_fails_closed_for_unsafe_or_ambiguous_checkpoint_paths(field, value):
    config = _static_config()
    config[field] = value
    with pytest.raises(ValueError, match="checkpoint_path_template"):
        build_suboff_segmented_campaign_manifest(config, segment_steps=1_000, segment_count=3)


def test_planner_marks_measurement_sums_unavailable_instead_of_claiming_ct():
    manifest = build_suboff_segmented_campaign_manifest(
        _static_config(), segment_steps=1_000, segment_count=3
    )
    blocks = manifest["blocks"]
    assert isinstance(blocks, list)
    assert all(block["friction_sum"] is None for block in blocks)
    assert all(block["pressure_sum"] is None for block in blocks)


def test_planner_does_not_mutate_validated_static_config():
    config = _static_config()
    before = deepcopy(config)
    build_suboff_segmented_campaign_manifest(config, segment_steps=1_000, segment_count=3)
    assert config == before

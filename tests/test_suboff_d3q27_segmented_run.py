"""TDD coverage for fail-closed D3Q27 SUBOFF checkpoint campaigns."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorlbm.suboff_segmented_run import evaluate_suboff_segmented_run


def _metadata(rank: int = 0) -> dict[str, object]:
    return {
        "format": "suboff-d3q27-cumulant-xslab-v1",
        "nx": 96, "ny": 48, "nz": 48, "hull_length": 206.0,
        "re": 2_000_000.0, "u_in": 0.06, "y_val": 0.5,
        "world_size": 1, "rank": rank, "nx_local": 96, "q": 27,
    }


def _checkpoint(directory: Path, step: int, metadata: dict[str, object] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": _metadata() if metadata is None else metadata,
        "step": step,
        "owned_populations": torch.ones((27, 48, 48, 96)),
        "target_mass": 1.0, "mass_cadence": 100,
        "friction_sum": 0.0, "pressure_sum": 0.0, "drag_samples": 0,
    }, directory / "rank0000.pt")


def _manifest(tmp_path: Path) -> dict[str, object]:
    _checkpoint(tmp_path / "step-100", 100)
    _checkpoint(tmp_path / "step-200", 200)
    return {
        "schema": "suboff-d3q27-segmented-run-v1",
        "checkpoint_metadata": _metadata(),
        "far_field": {
            "required_transient_steps": 60,
            "transient_steps_satisfy_outlet_convection": True,
        },
        "transient_steps": 60,
        "warmup_steps": 80,
        "dynamic_pressure_wetted_area": 2.0,
        "segments": [
            {"start_step": 0, "end_step": 100, "checkpoint": "step-100"},
            {"start_step": 100, "end_step": 200, "checkpoint": "step-200"},
        ],
        "blocks": [
            {"first_sample_step": 81, "last_sample_step": 100,
             "friction_sum": 20.0, "pressure_sum": 10.0, "drag_samples": 20},
            {"first_sample_step": 101, "last_sample_step": 200,
             "friction_sum": 100.0, "pressure_sum": 50.0, "drag_samples": 100},
        ],
    }


def test_evaluator_validates_checkpoints_and_combines_only_post_warmup_blocks(tmp_path):
    result = evaluate_suboff_segmented_run(_manifest(tmp_path), root=tmp_path)

    assert result.accepted
    assert result.completed_step == 200
    assert result.included_blocks == 2
    assert result.drag_samples == 120
    assert result.ct_friction == pytest.approx(0.5)
    assert result.ct_pressure == pytest.approx(0.25)
    assert result.ct_total == pytest.approx(0.75)


@pytest.mark.parametrize("mutation, reason", [
    (lambda m: m["segments"].__setitem__(1, {"start_step": 101, "end_step": 200, "checkpoint": "step-200"}),
     "continuous"),
    (lambda m: m["segments"].__setitem__(1, {"start_step": 99, "end_step": 200, "checkpoint": "step-200"}),
     "continuous"),
    (lambda m: m["segments"].__setitem__(1, {"start_step": 100, "end_step": 200, "checkpoint": "missing"}),
     "missing checkpoint"),
    (lambda m: m["far_field"].update({"transient_steps_satisfy_outlet_convection": False}),
     "outlet convection"),
])
def test_evaluator_fails_closed_for_broken_campaign_chain(tmp_path, mutation, reason):
    manifest = _manifest(tmp_path)
    mutation(manifest)
    with pytest.raises(ValueError, match=reason):
        evaluate_suboff_segmented_run(manifest, root=tmp_path)


def test_evaluator_rejects_checkpoint_step_and_metadata_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    _checkpoint(tmp_path / "step-200", 199)
    with pytest.raises(ValueError, match="step"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path)

    _checkpoint(tmp_path / "step-200", 200, {**_metadata(), "nx": 97})
    with pytest.raises(ValueError, match="metadata"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path)


def test_evaluator_refuses_ct_until_transient_and_outlet_convection_are_complete(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["transient_steps"] = 59
    with pytest.raises(ValueError, match="required transient"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path)

    manifest = _manifest(tmp_path / "second")
    manifest["blocks"][0]["first_sample_step"] = 60
    with pytest.raises(ValueError, match="before warmup"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path / "second")


def test_evaluator_rejects_partial_warmup_block_instead_of_silently_combining_it(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["blocks"][0]["first_sample_step"] = 80
    with pytest.raises(ValueError, match="post-warmup"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path)


def test_evaluator_rejects_overlapping_blocks_and_rank_population_shape_mismatch(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["blocks"].append({
        "first_sample_step": 100, "last_sample_step": 120,
        "friction_sum": 21.0, "pressure_sum": 10.5, "drag_samples": 21,
    })
    with pytest.raises(ValueError, match="overlap"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path)

    manifest = _manifest(tmp_path / "shape")
    checkpoint_file = tmp_path / "shape" / "step-100" / "rank0000.pt"
    checkpoint = torch.load(checkpoint_file, weights_only=True)
    checkpoint["owned_populations"] = torch.ones((26, 48, 48, 96))
    torch.save(checkpoint, checkpoint_file)
    with pytest.raises(ValueError, match="population shape"):
        evaluate_suboff_segmented_run(manifest, root=tmp_path / "shape")

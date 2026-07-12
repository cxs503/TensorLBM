"""TDD for fail-closed SUBOFF segmented-campaign audit artifacts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tensorlbm.suboff_campaign_artifact import (
    build_suboff_campaign_audit_artifact,
    validate_suboff_campaign_audit_artifact,
)


def _metadata(rank: int) -> dict[str, object]:
    return {
        "format": "suboff-d3q27-cumulant-xslab-v1", "nx": 4, "ny": 2,
        "nz": 1, "hull_length": 2.0, "re": 1000.0, "u_in": 0.1,
        "y_val": 0.5, "world_size": 2, "rank": rank, "nx_local": 2, "q": 27,
    }


def _save_checkpoint(directory: Path, step: int, rank: int, *, friction: float,
                     pressure: float, samples: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": _metadata(rank), "step": step,
        "owned_populations": torch.full((27, 1, 2, 2), float(rank + step)),
        "target_mass": 4.0, "mass_cadence": 1,
        "friction_sum": friction, "pressure_sum": pressure, "drag_samples": samples,
    }, directory / f"rank{rank:04d}.pt")


def _short_completed_campaign(root: Path) -> dict[str, object]:
    for rank in range(2):
        # These are all_reduce results: every rank stores the same global
        # accumulator, rather than one rank-local contribution each.
        _save_checkpoint(root / "step-2", 2, rank, friction=10.0,
                         pressure=20.0, samples=2)
        _save_checkpoint(root / "step-4", 4, rank, friction=16.0,
                         pressure=28.0, samples=4)
    return {
        "schema": "suboff-d3q27-segmented-run-v1",
        "checkpoint_metadata": _metadata(0), "expected_ranks": [0, 1],
        "dynamic_pressure_wetted_area": 2.0,
        "segments": [
            {"start_step": 0, "end_step": 2, "checkpoint": "step-2"},
            {"start_step": 2, "end_step": 4, "checkpoint": "step-4"},
        ],
        "blocks": [{"first_sample_step": 3, "last_sample_step": 4,
                    "friction_sum": 6.0, "pressure_sum": 8.0,
                    "drag_samples": 2}],
    }


def test_withheld_short_run_builds_and_revalidates_a_hash_bound_artifact(tmp_path):
    manifest = _short_completed_campaign(tmp_path)

    artifact = build_suboff_campaign_audit_artifact(manifest, root=tmp_path)

    assert artifact["schema"] == "suboff-d3q27-campaign-audit-v1"
    assert artifact["manifest_schema"] == manifest["schema"]
    assert len(artifact["checkpoints"]) == 4
    assert {record["sha256"] for record in artifact["checkpoints"]}
    assert all(record["population_shape"] == [27, 1, 2, 2]
               for record in artifact["checkpoints"])
    assert artifact["blocks"] == [{
        "first_sample_step": 3, "last_sample_step": 4,
        "friction_sum": 6.0, "pressure_sum": 8.0, "drag_samples": 2,
    }]
    assert artifact["ct"]["status"] == "computed"
    assert artifact["ct"]["physical_validation"] == "not_verified"
    assert validate_suboff_campaign_audit_artifact(artifact, manifest, root=tmp_path) is True


def test_delta_mismatch_fails_closed_even_when_manifest_claims_a_valid_block(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    manifest["blocks"][0]["friction_sum"] = 12.5

    with pytest.raises(ValueError, match="friction_sum delta"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


def test_validator_rejects_checkpoint_tampering_after_artifact_is_built(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    artifact = build_suboff_campaign_audit_artifact(manifest, root=tmp_path)
    payload = torch.load(tmp_path / "step-4" / "rank0000.pt", weights_only=True)
    payload["pressure_sum"] = 999.0
    torch.save(payload, tmp_path / "step-4" / "rank0000.pt")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_suboff_campaign_audit_artifact(artifact, manifest, root=tmp_path)


def test_block_must_be_exactly_between_adjacent_checkpoint_boundaries(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    invalid = deepcopy(manifest)
    invalid["blocks"][0]["first_sample_step"] = 4
    invalid["blocks"][0]["drag_samples"] = 1

    with pytest.raises(ValueError, match="adjacent checkpoint"):
        build_suboff_campaign_audit_artifact(invalid, root=tmp_path)


def test_step_200_short_run_without_post_gate_blocks_is_explicitly_withheld(tmp_path):
    for rank in range(2):
        _save_checkpoint(tmp_path / "step-200", 200, rank, friction=123.0,
                         pressure=456.0, samples=200)
    manifest = {
        "schema": "suboff-d3q27-segmented-run-v1",
        "checkpoint_metadata": _metadata(0), "expected_ranks": [0, 1],
        "segments": [{"start_step": 0, "end_step": 200,
                      "checkpoint": "step-200"}],
        "blocks": [],
    }

    artifact = build_suboff_campaign_audit_artifact(manifest, root=tmp_path)

    assert artifact["checkpoints"]
    assert artifact["blocks"] == []
    assert artifact["ct"] == {
        "status": "withheld", "reason": "no_post_gate_blocks",
    }
    assert validate_suboff_campaign_audit_artifact(artifact, manifest, root=tmp_path) is True


def test_rank_accumulator_mismatch_fails_closed_before_canonical_delta_is_used(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    payload = torch.load(tmp_path / "step-4" / "rank0001.pt", weights_only=True)
    payload["friction_sum"] = 17.0
    torch.save(payload, tmp_path / "step-4" / "rank0001.pt")

    with pytest.raises(ValueError, match="rank friction_sum accumulator mismatch"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


def test_rank_continuation_state_mismatch_fails_closed(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    payload = torch.load(tmp_path / "step-4" / "rank0001.pt", weights_only=True)
    payload["target_mass"] = 5.0
    torch.save(payload, tmp_path / "step-4" / "rank0001.pt")

    with pytest.raises(ValueError, match="rank target_mass continuation mismatch"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


def test_post_gate_blocks_require_finite_positive_dynamic_pressure_wetted_area_and_emit_ct(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    manifest["dynamic_pressure_wetted_area"] = 2.0

    artifact = build_suboff_campaign_audit_artifact(manifest, root=tmp_path)

    assert artifact["ct"] == {
        "status": "computed",
        "physical_validation": "not_verified",
        "dynamic_pressure_wetted_area": 2.0,
        "ct_friction": 1.5,
        "ct_pressure": 2.0,
        "ct_total": 3.5,
    }


@pytest.mark.parametrize("area", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_post_gate_blocks_fail_closed_without_finite_positive_ct_denominator(tmp_path, area):
    manifest = _short_completed_campaign(tmp_path)
    if area is None:
        manifest.pop("dynamic_pressure_wetted_area")
    else:
        manifest["dynamic_pressure_wetted_area"] = area

    with pytest.raises(ValueError, match="dynamic_pressure_wetted_area"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


@pytest.mark.parametrize("second", [
    {"first_sample_step": 3, "last_sample_step": 4, "friction_sum": 6.0, "pressure_sum": 8.0, "drag_samples": 2},
    {"first_sample_step": 4, "last_sample_step": 4, "friction_sum": 3.0, "pressure_sum": 4.0, "drag_samples": 1},
])
def test_blocks_must_be_strictly_ordered_nonoverlapping_and_use_each_boundary_once(tmp_path, second):
    manifest = _short_completed_campaign(tmp_path)
    manifest["blocks"].append(second)

    with pytest.raises(ValueError, match="blocks must be strictly ordered and non-overlapping"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


@pytest.mark.parametrize("field, value", [("target_mass", 5.0), ("mass_cadence", 2)])
def test_continuation_controls_must_match_across_checkpoint_boundaries(tmp_path, field, value):
    manifest = _short_completed_campaign(tmp_path)
    for rank in range(2):
        path = tmp_path / "step-4" / f"rank{rank:04d}.pt"
        payload = torch.load(path, weights_only=True)
        payload[field] = value
        torch.save(payload, path)

    with pytest.raises(ValueError, match=f"{field} checkpoint-boundary continuation mismatch"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


@pytest.mark.parametrize("population", [torch.ones((27, 1, 2, 2), dtype=torch.int64),
                                          torch.full((27, 1, 2, 2), float("nan"))])
def test_owned_populations_must_be_floating_and_finite(tmp_path, population):
    manifest = _short_completed_campaign(tmp_path)
    path = tmp_path / "step-4" / "rank0000.pt"
    payload = torch.load(path, weights_only=True)
    payload["owned_populations"] = population
    torch.save(payload, path)

    with pytest.raises(ValueError, match="populations must be floating and finite"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


def test_ct_overflow_fails_closed(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    for rank in range(2):
        path = tmp_path / "step-4" / f"rank{rank:04d}.pt"
        payload = torch.load(path, weights_only=True)
        payload["friction_sum"] = 1.0e308
        payload["pressure_sum"] = 1.0e308
        torch.save(payload, path)
    manifest["dynamic_pressure_wetted_area"] = 1.0e-308
    manifest["blocks"][0]["friction_sum"] = 1.0e308 - 10.0
    manifest["blocks"][0]["pressure_sum"] = 1.0e308 - 20.0

    with pytest.raises(ValueError, match="Ct must be finite"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)


def test_roundoff_scale_rank_and_manifest_deltas_are_accepted_with_policy(tmp_path):
    manifest = _short_completed_campaign(tmp_path)
    for rank in range(2):
        path = tmp_path / "step-4" / f"rank{rank:04d}.pt"
        payload = torch.load(path, weights_only=True)
        payload["friction_sum"] = 16.0 + (5.0e-13 if rank else 0.0)
        payload["pressure_sum"] = 28.0 + (5.0e-13 if rank else 0.0)
        torch.save(payload, path)
    manifest["blocks"][0]["friction_sum"] = 6.0 + 5.0e-13
    manifest["blocks"][0]["pressure_sum"] = 8.0 + 5.0e-13

    artifact = build_suboff_campaign_audit_artifact(manifest, root=tmp_path)

    assert artifact["ct"]["status"] == "computed"


@pytest.mark.parametrize("field, rank0, rank1", [
    ("mass_cadence", 1_000_000_000_000_000, 1_000_000_000_000_001),
    ("drag_samples", 1_000_000_000_000_000, 1_000_000_000_000_001),
])
def test_rank_integer_global_state_must_match_exactly(tmp_path, field, rank0, rank1):
    manifest = _short_completed_campaign(tmp_path)
    for rank, value in enumerate((rank0, rank1)):
        for checkpoint in ("step-2", "step-4"):
            path = tmp_path / checkpoint / f"rank{rank:04d}.pt"
            payload = torch.load(path, weights_only=True)
            payload[field] = value
            torch.save(payload, path)

    with pytest.raises(ValueError, match=f"rank {field} .*mismatch"):
        build_suboff_campaign_audit_artifact(manifest, root=tmp_path)

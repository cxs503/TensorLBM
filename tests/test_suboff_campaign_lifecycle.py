"""TDD for durable, non-launching SUBOFF campaign lifecycle artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tensorlbm.suboff_campaign_lifecycle import materialize_suboff_campaign_lifecycle
from test_suboff_campaign_artifact import _short_completed_campaign


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_completed_checkpoint_set_writes_restartable_status_progress_manifest_and_block_telemetry(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    manifest = _short_completed_campaign(checkpoint_root)
    output = tmp_path / "campaign-artifacts"

    result = materialize_suboff_campaign_lifecycle(
        manifest, checkpoint_root=checkpoint_root, artifact_root=output,
    )

    assert result["status"] == "completed"
    status = _load_json(output / "run_status.json")
    assert status["status"] == "completed"
    assert status["checkpoint_set_complete"] is True
    assert status["restart_checkpoint"] == "step-4"
    completed = _load_json(output / "completed_manifest.json")
    assert completed["audit_artifact"]["ct"]["status"] == "computed"
    assert completed["audit_artifact"]["ct"]["physical_validation"] == "not_verified"
    telemetry = _load_json(output / "block_telemetry.json")
    assert telemetry["ct"]["status"] == "computed"
    with (output / "progress.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["end_step"], row["checkpoint"], row["checkpoint_set_complete"])
            for row in rows] == [("2", "step-2", "true"), ("4", "step-4", "true")]


def test_invalid_checkpoint_set_writes_blocked_status_without_completion_claim(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    manifest = _short_completed_campaign(checkpoint_root)
    (checkpoint_root / "step-4" / "rank0001.pt").unlink()
    output = tmp_path / "campaign-artifacts"

    with pytest.raises(ValueError, match="missing checkpoint rank file"):
        materialize_suboff_campaign_lifecycle(
            manifest, checkpoint_root=checkpoint_root, artifact_root=output,
        )

    status = _load_json(output / "run_status.json")
    assert status["status"] == "blocked"
    assert status["checkpoint_set_complete"] is False
    assert "missing checkpoint rank file" in status["reason"]
    assert not (output / "completed_manifest.json").exists()


def test_restartable_lifecycle_is_idempotent_and_rewrites_same_evidence(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    manifest = _short_completed_campaign(checkpoint_root)
    output = tmp_path / "campaign-artifacts"

    first = materialize_suboff_campaign_lifecycle(
        manifest, checkpoint_root=checkpoint_root, artifact_root=output,
    )
    first_progress = (output / "progress.csv").read_text(encoding="utf-8")
    second = materialize_suboff_campaign_lifecycle(
        manifest, checkpoint_root=checkpoint_root, artifact_root=output,
    )

    assert first == second
    assert (output / "progress.csv").read_text(encoding="utf-8") == first_progress

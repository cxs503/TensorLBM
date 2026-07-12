"""TDD for durable, non-launching SUBOFF campaign lifecycle artifacts."""
from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import tensorlbm.suboff_campaign_lifecycle as lifecycle

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


def test_two_callers_never_share_a_temporary_artifact_name(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    manifest = _short_completed_campaign(checkpoint_root)
    output = tmp_path / "campaign-artifacts"
    original_replace = lifecycle.os.replace
    progress_temporaries: list[Path] = []
    barrier = threading.Barrier(2)

    def synchronized_replace(source, destination):
        source_path = Path(source)
        if Path(destination).name == "progress.csv":
            progress_temporaries.append(source_path)
            barrier.wait(timeout=5)
        return original_replace(source, destination)

    monkeypatch.setattr(lifecycle.os, "replace", synchronized_replace)
    errors: list[Exception] = []

    def materialize() -> None:
        try:
            materialize_suboff_campaign_lifecycle(
                manifest, checkpoint_root=checkpoint_root, artifact_root=output,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    callers = [threading.Thread(target=materialize) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=10)

    assert not any(caller.is_alive() for caller in callers)
    assert not errors
    assert len(progress_temporaries) == 2
    assert progress_temporaries[0] != progress_temporaries[1]
    assert _load_json(output / "run_status.json")["status"] == "completed"
    assert not list(output.glob(".*.tmp*"))

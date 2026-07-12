"""Durable, non-launching lifecycle records for segmented SUBOFF checkpoints.

This module never invokes a solver.  It converts an already-produced checkpoint
set into restart/status/progress/completion artifacts, and fails closed when the
set cannot be independently audited.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .suboff_campaign_artifact import build_suboff_campaign_audit_artifact


_STATUS_SCHEMA = "suboff-d3q27-campaign-lifecycle-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    return Path(temporary)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _segments(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("manifest requires at least one segment")
    normalized: list[dict[str, object]] = []
    for raw in segments:
        if not isinstance(raw, Mapping):
            raise ValueError("manifest segment must be a mapping")
        checkpoint = raw.get("checkpoint")
        end = raw.get("end_step")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("manifest segment checkpoint must be a non-empty string")
        if isinstance(end, bool) or not isinstance(end, int) or end < 1:
            raise ValueError("manifest segment end_step must be a positive integer")
        normalized.append({"end_step": end, "checkpoint": checkpoint})
    return normalized


def _write_progress(path: Path, segments: list[dict[str, object]]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("end_step", "checkpoint", "checkpoint_set_complete"))
            writer.writeheader()
            for segment in segments:
                writer.writerow({**segment, "checkpoint_set_complete": "true"})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_suboff_campaign_lifecycle(
    manifest: Mapping[str, object], *, checkpoint_root: str | Path, artifact_root: str | Path,
) -> dict[str, object]:
    """Record an auditable checkpoint lifecycle without launching any compute.

    On an audit failure a durable ``blocked`` run_status is written before the
    original error is re-raised; no completed manifest is emitted.  Successful
    calls rewrite deterministic progress/completion evidence, which makes an
    interrupted reporting step restartable without duplicating telemetry.
    """
    root = Path(checkpoint_root)
    destination = Path(artifact_root)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        segments = _segments(manifest)
        audit = build_suboff_campaign_audit_artifact(manifest, root=root)
    except Exception as exc:
        _atomic_json(destination / "run_status.json", {
            "schema": _STATUS_SCHEMA, "status": "blocked", "checkpoint_set_complete": False,
            "reason": str(exc), "updated_at": _utc_now(),
        })
        raise

    restart = segments[-1]["checkpoint"]
    telemetry = {"schema": _STATUS_SCHEMA, "status": "completed", "ct": audit["ct"],
                 "blocks": audit["blocks"], "updated_at": _utc_now()}
    completed = {"schema": _STATUS_SCHEMA, "status": "completed", "checkpoint_set_complete": True,
                 "restart_checkpoint": restart, "audit_artifact": audit, "updated_at": _utc_now()}
    status = {"schema": _STATUS_SCHEMA, "status": "completed", "checkpoint_set_complete": True,
              "restart_checkpoint": restart, "completed_manifest": "completed_manifest.json",
              "progress": "progress.csv", "block_telemetry": "block_telemetry.json", "updated_at": _utc_now()}
    _write_progress(destination / "progress.csv", segments)
    _atomic_json(destination / "block_telemetry.json", telemetry)
    _atomic_json(destination / "completed_manifest.json", completed)
    _atomic_json(destination / "run_status.json", status)
    return {"status": "completed", "restart_checkpoint": restart, "ct": audit["ct"]}

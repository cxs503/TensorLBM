"""Run-completion evidence contracts with cryptographically bound artifacts.

This module is deliberately a standard-library-only cold-path boundary.  It
records evidence after a run and never participates in solver timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256 as _sha256
import json
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

_CODE_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _require_text(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze(value: Any) -> Any:
    """Return a recursively immutable snapshot made only from safe value types."""
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({_require_text(key, "mapping key"): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    raise TypeError(f"unsupported mutable or non-serializable evidence value: {type(value).__name__}")


def _immutable_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _json_pointer_value(payload: bytes, pointer: str) -> float:
    """Resolve a JSON Pointer to a finite number exactly representable as float."""
    try:
        value: Any = json.loads(payload.decode("utf-8"), parse_int=Decimal, parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation) as error:
        raise ValueError("metric artifact must contain UTF-8 JSON") from error
    if pointer == "":
        target = value
    else:
        if not pointer.startswith("/"):
            raise ValueError("evidence_pointer must be a JSON Pointer")
        target = value
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(target, Mapping):
                if token not in target:
                    raise ValueError("evidence_pointer does not exist in artifact payload")
                target = target[token]
            elif isinstance(target, list):
                if not token.isdigit() or int(token) >= len(target):
                    raise ValueError("evidence_pointer does not exist in artifact payload")
                target = target[int(token)]
            else:
                raise ValueError("evidence_pointer does not exist in artifact payload")
    if isinstance(target, bool) or not isinstance(target, Decimal) or not target.is_finite():
        raise ValueError("evidence_pointer must resolve to a finite numeric JSON value")
    candidate = float(target)
    if not isfinite(candidate) or Decimal.from_float(candidate) != target:
        raise ValueError("evidence_pointer is not exactly representable as MetricEvidence float")
    return candidate


class ValidationStatus(str, Enum):
    """Explicit validation disposition; no status is inferred from metadata."""

    PASS = "PASS"
    FAIL = "FAIL"
    WITHHELD = "WITHHELD"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True, init=False)
class ArtifactManifest:
    """In-memory evidence payload whose identity and digest are internally derived."""

    artifact_id: str
    media_type: str
    payload: bytes
    sha256: str
    size_bytes: int
    metadata: Mapping[str, Any]

    @classmethod
    def from_bytes(
        cls,
        artifact_id: str,
        media_type: str,
        payload: bytes,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactManifest:
        artifact_id = _require_text(artifact_id, "artifact_id")
        media_type = _require_text(media_type, "media_type")
        if isinstance(payload, bool) or not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes")
        frozen_metadata = _immutable_mapping(metadata or {}, "metadata")
        return cls._create(artifact_id, media_type, payload, frozen_metadata)

    @classmethod
    def _create(
        cls,
        artifact_id: str,
        media_type: str,
        payload: bytes,
        metadata: Mapping[str, Any],
    ) -> ArtifactManifest:
        instance = object.__new__(cls)
        object.__setattr__(instance, "artifact_id", artifact_id)
        object.__setattr__(instance, "media_type", media_type)
        object.__setattr__(instance, "payload", payload)
        object.__setattr__(instance, "sha256", _sha256(payload).hexdigest())
        object.__setattr__(instance, "size_bytes", len(payload))
        object.__setattr__(instance, "metadata", metadata)
        return instance

    def verify_integrity(self) -> bool:
        """Recompute both payload-derived fields; return false rather than trusting state."""
        return (
            isinstance(self.payload, bytes)
            and bool(self.payload)
            and isinstance(self.size_bytes, int)
            and not isinstance(self.size_bytes, bool)
            and self.size_bytes == len(self.payload)
            and isinstance(self.sha256, str)
            and self.sha256 == _sha256(self.payload).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    metric_id: str
    value: float
    unit: str
    artifact_id: str
    evidence_pointer: str

    def __post_init__(self) -> None:
        _require_text(self.metric_id, "metric_id")
        _require_text(self.unit, "unit")
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.evidence_pointer, "evidence_pointer")
        if isinstance(self.value, bool) or not isinstance(self.value, float):
            raise TypeError("value must be a non-boolean float")
        if not isfinite(self.value):
            raise ValueError("value must be finite")


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    model_identity: Mapping[str, Any]
    config: Mapping[str, Any]
    code_sha: str
    environment: Mapping[str, Any]
    artifacts: tuple[ArtifactManifest, ...]
    metrics: tuple[MetricEvidence, ...]
    validation_status: ValidationStatus
    validation_reason: str

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        if not isinstance(self.code_sha, str) or not _CODE_SHA_PATTERN.fullmatch(self.code_sha):
            raise ValueError("code_sha must be exactly 40 lowercase hexadecimal characters")
        if not isinstance(self.validation_status, ValidationStatus):
            raise TypeError("validation_status must be a ValidationStatus")
        if not isinstance(self.validation_reason, str):
            raise TypeError("validation_reason must be a string")
        object.__setattr__(self, "model_identity", _immutable_mapping(self.model_identity, "model_identity"))
        object.__setattr__(self, "config", _immutable_mapping(self.config, "config"))
        object.__setattr__(self, "environment", _immutable_mapping(self.environment, "environment"))
        artifacts = tuple(self.artifacts)
        metrics = tuple(self.metrics)
        if not all(isinstance(artifact, ArtifactManifest) for artifact in artifacts):
            raise TypeError("artifacts must contain ArtifactManifest values")
        if not all(isinstance(metric, MetricEvidence) for metric in metrics):
            raise TypeError("metrics must contain MetricEvidence values")
        artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("artifact_id values must be unique")
        known_artifacts = set(artifact_ids)
        if any(metric.artifact_id not in known_artifacts for metric in metrics):
            raise ValueError("every metric must bind an existing artifact")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metrics", metrics)

    def verify_integrity(self) -> bool:
        return all(artifact.verify_integrity() for artifact in self.artifacts)

    def verify_structure(self) -> bool:
        if not all(isinstance(artifact, ArtifactManifest) for artifact in self.artifacts):
            return False
        if not all(isinstance(metric, MetricEvidence) for metric in self.metrics):
            return False
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if any(not isinstance(artifact_id, str) or not artifact_id.strip() for artifact_id in artifact_ids):
            return False
        if len(set(artifact_ids)) != len(artifact_ids):
            return False
        known_artifacts = set(artifact_ids)
        return all(metric.artifact_id in known_artifacts for metric in self.metrics)

    def verify_metric_evidence(self) -> bool:
        artifacts = {artifact.artifact_id: artifact for artifact in self.artifacts}
        try:
            for metric in self.metrics:
                artifact = artifacts[metric.artifact_id]
                if artifact.media_type != "application/json":
                    return False
                if _json_pointer_value(artifact.payload, metric.evidence_pointer) != metric.value:
                    return False
        except (KeyError, ValueError, TypeError):
            return False
        return True


def validate_run_manifest(manifest: RunManifest) -> RunManifest:
    """Fail closed; validate declared status without inferring physical validity."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    if not manifest.verify_structure():
        raise ValueError("manifest has invalid artifact or metric binding structure")
    if manifest.validation_status is ValidationStatus.PASS:
        if not manifest.validation_reason.strip():
            raise ValueError("PASS requires a validation reason")
        if not manifest.artifacts or not manifest.metrics:
            raise ValueError("PASS requires at least one artifact and metric")
        if not manifest.verify_integrity():
            raise ValueError("PASS requires verified artifact integrity")
        if not manifest.verify_metric_evidence():
            raise ValueError("PASS requires metrics bound to artifact payload evidence")
    elif manifest.validation_status in {
        ValidationStatus.FAIL,
        ValidationStatus.WITHHELD,
        ValidationStatus.NOT_APPLICABLE,
    }:
        if not manifest.validation_reason.strip():
            raise ValueError(f"{manifest.validation_status.value} requires a validation reason")
    else:  # Defensive for object.__setattr__ tampering.
        raise ValueError("unknown validation status")
    return manifest


def build_run_manifest_from_artifacts(
    *,
    run_id: str,
    model_identity: Mapping[str, Any],
    config: Mapping[str, Any],
    code_sha: str,
    environment: Mapping[str, Any],
    artifacts: Sequence[ArtifactManifest],
    metrics: Sequence[MetricEvidence],
    validation_status: ValidationStatus,
    validation_reason: str,
) -> RunManifest:
    """Cold-path constructor; callers supply completed run evidence only."""
    return RunManifest(
        run_id=run_id,
        model_identity=model_identity,
        config=config,
        code_sha=code_sha,
        environment=environment,
        artifacts=tuple(artifacts),
        metrics=tuple(metrics),
        validation_status=validation_status,
        validation_reason=validation_reason,
    )


__all__ = [
    "ArtifactManifest",
    "MetricEvidence",
    "RunManifest",
    "ValidationStatus",
    "build_run_manifest_from_artifacts",
    "validate_run_manifest",
]

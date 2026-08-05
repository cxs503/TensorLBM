"""Runtime Evidence Contract R2: fail-closed cold-path provenance tests."""

from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tensorlbm.runtime.evidence import (
    ArtifactManifest,
    MetricEvidence,
    RunManifest,
    ValidationStatus,
    build_run_manifest_from_artifacts,
    validate_run_manifest,
)

CODE_SHA = "a" * 40


def _artifact() -> ArtifactManifest:
    payload = json.dumps(
        {"accounting": {"mass_before": 1.0, "mass_after": 1.0}, "run": "demo"},
        sort_keys=True,
    ).encode("utf-8")
    return ArtifactManifest.from_bytes(
        artifact_id="mass-accounting.json",
        media_type="application/json",
        payload=payload,
        metadata={"source": {"stage": "post-run"}},
    )


def _metric() -> MetricEvidence:
    return MetricEvidence(
        metric_id="relative_mass_error",
        value=1.0,
        unit="1",
        artifact_id="mass-accounting.json",
        evidence_pointer="/accounting/mass_after",
    )


def _manifest(**overrides: object) -> RunManifest:
    fields: dict[str, object] = {
        "run_id": "run-001",
        "model_identity": {"name": "D3Q19", "collision": "MRT"},
        "config": {"grid": {"nx": 32}},
        "code_sha": CODE_SHA,
        "environment": {"python": {"implementation": "CPython"}},
        "artifacts": (_artifact(),),
        "metrics": (_metric(),),
        "validation_status": ValidationStatus.PASS,
        "validation_reason": "post-run accounting evidence verified",
    }
    fields.update(overrides)
    return RunManifest(**fields)  # type: ignore[arg-type]


def test_real_bytes_evidence_binds_metric_and_validates_pass() -> None:
    manifest = build_run_manifest_from_artifacts(
        run_id="run-001",
        model_identity={"name": "D3Q19", "collision": "MRT"},
        config={"grid": {"nx": 32}},
        code_sha=CODE_SHA,
        environment={"python": {"implementation": "CPython"}},
        artifacts=(_artifact(),),
        metrics=(_metric(),),
        validation_status=ValidationStatus.PASS,
        validation_reason="post-run accounting evidence verified",
    )

    assert validate_run_manifest(manifest) is manifest
    assert manifest.verify_integrity() is True
    assert manifest.artifacts[0].size_bytes == len(manifest.artifacts[0].payload)
    assert len(manifest.artifacts[0].sha256) == 64


def test_artifact_hash_is_internal_and_tampering_fails_closed() -> None:
    artifact = _artifact()
    with pytest.raises(TypeError):
        ArtifactManifest(  # type: ignore[call-arg]
            "other", "application/json", b"{}", "0" * 64, 2, {}
        )

    object.__setattr__(artifact, "payload", b"tampered")
    assert artifact.verify_integrity() is False
    with pytest.raises(ValueError, match="integrity"):
        validate_run_manifest(_manifest(artifacts=(artifact,)))

    artifact = _artifact()
    object.__setattr__(artifact, "sha256", "0" * 64)
    assert artifact.verify_integrity() is False
    with pytest.raises(ValueError, match="integrity"):
        validate_run_manifest(_manifest(artifacts=(artifact,)))


@pytest.mark.parametrize("artifact_id,payload", [("", b"x"), ("id", b""), (True, b"x")])
def test_artifact_rejects_empty_or_boolean_identity_and_payload(
    artifact_id: object, payload: bytes
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ArtifactManifest.from_bytes(  # type: ignore[arg-type]
            artifact_id=artifact_id,
            media_type="application/octet-stream",
            payload=payload,
        )


def test_nested_metadata_and_manifest_mappings_are_immutable_snapshots() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    config = {"nested": {"values": [3, 4]}}
    artifact = ArtifactManifest.from_bytes(
        "mass-accounting.json", "application/json", b"{}", metadata
    )
    manifest = _manifest(artifacts=(artifact,), config=config)
    metadata["nested"]["values"].append(9)
    config["nested"]["values"].append(9)

    assert artifact.metadata["nested"]["values"] == (1, 2)
    assert manifest.config["nested"]["values"] == (3, 4)
    with pytest.raises(TypeError):
        artifact.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        manifest.run_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        ArtifactManifest.from_bytes("mutable-leaf", "application/json", b"{}", {"leaf": bytearray(b"x")})


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf")])
def test_metric_rejects_boolean_and_nonfinite_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MetricEvidence("metric", value, "1", "artifact", "/value")  # type: ignore[arg-type]


@pytest.mark.parametrize("code_sha", ["A" * 40, "a" * 39, "g" * 40, ""])
def test_code_sha_requires_exact_lowercase_git_hex(code_sha: str) -> None:
    with pytest.raises(ValueError, match="code_sha"):
        _manifest(code_sha=code_sha)


def test_metric_pointer_and_value_must_bind_to_json_artifact_payload() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="bound"):
        validate_run_manifest(_manifest(metrics=(MetricEvidence("forged", 999.0, "1", artifact.artifact_id, "/missing"),)))
    with pytest.raises(ValueError, match="bound"):
        validate_run_manifest(_manifest(metrics=(MetricEvidence("forged", 999.0, "1", artifact.artifact_id, "/accounting/mass_after"),)))


def test_large_json_integer_evidence_is_rejected_before_float_rounding_can_bind_it() -> None:
    artifact = ArtifactManifest.from_bytes("large", "application/json", b'{"v":9007199254740993}')
    metric = MetricEvidence("forged", float(9007199254740992), "1", "large", "/v")
    with pytest.raises(ValueError, match="bound"):
        validate_run_manifest(_manifest(artifacts=(artifact,), metrics=(metric,)))


@pytest.mark.parametrize("payload", [b'{"v":9007199254740993.0}', b'{"v":9.007199254740993e15}'])
def test_large_json_decimal_or_exponent_evidence_is_rejected_before_float_rounding_can_bind_it(payload: bytes) -> None:
    artifact = ArtifactManifest.from_bytes("large-decimal", "application/json", payload)
    metric = MetricEvidence("forged", float(9007199254740992), "1", "large-decimal", "/v")
    with pytest.raises(ValueError, match="bound"):
        validate_run_manifest(_manifest(artifacts=(artifact,), metrics=(metric,)))


def test_validator_rechecks_binding_after_in_process_tampering() -> None:
    manifest = _manifest()
    object.__setattr__(manifest.artifacts[0], "artifact_id", "tampered")
    with pytest.raises(ValueError, match="binding"):
        validate_run_manifest(manifest)


def test_duplicate_artifact_and_missing_metric_binding_are_rejected() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="unique"):
        _manifest(artifacts=(artifact, artifact))
    with pytest.raises(ValueError, match="existing artifact"):
        _manifest(metrics=(MetricEvidence("m", 1.0, "1", "missing", "/v"),))


@pytest.mark.parametrize(
    ("status", "reason", "artifacts", "metrics", "matches"),
    [
        (ValidationStatus.PASS, "", (_artifact(),), (_metric(),), "reason"),
        (ValidationStatus.PASS, "reason", (), (), "artifact"),
        (ValidationStatus.FAIL, "", (), (), "reason"),
        (ValidationStatus.WITHHELD, "", (), (), "reason"),
        (ValidationStatus.NOT_APPLICABLE, "", (), (), "reason"),
        (ValidationStatus.FAIL, "solver diverged", (), (), None),
        (ValidationStatus.WITHHELD, "evidence unavailable", (), (), None),
        (ValidationStatus.NOT_APPLICABLE, "no validation requested", (), (), None),
    ],
)
def test_status_requirements_fail_closed(
    status: ValidationStatus,
    reason: str,
    artifacts: tuple[ArtifactManifest, ...],
    metrics: tuple[MetricEvidence, ...],
    matches: str | None,
) -> None:
    manifest = _manifest(
        validation_status=status,
        validation_reason=reason,
        artifacts=artifacts,
        metrics=metrics,
    )
    if matches is None:
        assert validate_run_manifest(manifest) is manifest
    else:
        with pytest.raises(ValueError, match=matches):
            validate_run_manifest(manifest)


def test_runtime_core_is_stdlib_only_and_has_no_solver_step_loop() -> None:
    source_path = Path(__file__).parents[2] / "src/tensorlbm/runtime/evidence.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "torch" not in imported_roots
    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor))
        and isinstance(node.target, ast.Name)
        and node.target.id in {"step", "timestep"}
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module and "solver" in node.module
        for node in ast.walk(tree)
    )

"""Tests for cold-path, evidence-gated ML data product contracts."""

import ast
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tensorlbm.runtime import (
    ArtifactManifest,
    MetricEvidence,
    RunManifest,
    ValidationStatus,
)

from tensorlbm.data.contracts import DatasetManifest, DatasetSampleRef, FieldProduct


_CODE_SHA = "a" * 40


def _manifest(status=ValidationStatus.PASS):
    payload = json.dumps({"drag": 1.25}).encode("utf-8")
    artifact = ArtifactManifest.from_bytes("metrics-json", "application/json", payload)
    metrics = (
        MetricEvidence(
            metric_id="drag-coefficient",
            value=1.25,
            unit="1",
            artifact_id=artifact.artifact_id,
            evidence_pointer="/drag",
        ),
    )
    return RunManifest(
        run_id="run-001",
        model_identity={"case": "reference"},
        config={"resolution": 64},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=metrics,
        validation_status=status,
        validation_reason="runtime evidence reviewed",
    )


def _product(manifest=None, *, product_id="velocity-field", quality_status=ValidationStatus.PASS):
    return FieldProduct(
        product_id=product_id,
        run_manifest=manifest or _manifest(),
        artifact_id="metrics-json",
        field_name="velocity",
        shape=(8, 4, 2),
        dtype="float32",
        units="m/s",
        quality_status=quality_status,
        lineage={"source": {"campaign": "baseline"}},
    )


def _dataset(*products, splits=None):
    products = products or (_product(),)
    return DatasetManifest(
        dataset_id="training-set",
        version="r1",
        products=tuple(products),
        task_name="field-reconstruction",
        group_splits=splits or {"train": (products[0].product_id,), "val": (), "test": ()},
        lineage={"curation": {"owner": "data-governance"}},
    )


def test_pass_runtime_evidence_can_create_traceable_training_ready_dataset():
    product = _product()
    dataset = _dataset(product)

    assert product.is_training_eligible is True
    dataset.require_training_ready() is None
    assert DatasetSampleRef.from_product(product) == DatasetSampleRef(
        product_id="velocity-field",
        run_id="run-001",
        artifact_id="metrics-json",
        field_name="velocity",
        shape=(8, 4, 2),
        dtype="float32",
        units="m/s",
    )


@pytest.mark.parametrize("status", [ValidationStatus.WITHHELD, ValidationStatus.FAIL, ValidationStatus.NOT_APPLICABLE])
def test_non_pass_runtime_statuses_cannot_be_training_ready(status):
    product = _product(_manifest(status), quality_status=status)
    dataset = _dataset(product)

    assert product.is_training_eligible is False
    with pytest.raises(ValueError, match="velocity-field"):
        dataset.require_training_ready()


def test_field_product_revalidates_runtime_evidence_and_rejects_promotion_or_missing_artifact():
    manifest = _manifest()
    object.__setattr__(manifest.artifacts[0], "payload", b'{"drag": 99.0}')
    with pytest.raises(ValueError, match="run manifest"):
        _product(manifest)

    withheld = _manifest(ValidationStatus.WITHHELD)
    with pytest.raises(ValueError, match="quality_status"):
        _product(withheld, quality_status=ValidationStatus.PASS)
    with pytest.raises(ValueError, match="artifact_id"):
        FieldProduct(
            product_id="bad-artifact",
            run_manifest=_manifest(),
            artifact_id="missing",
            field_name="velocity",
            shape=(1,),
            dtype="float32",
            units="m/s",
            quality_status=ValidationStatus.PASS,
            lineage={},
        )


def test_dataset_rejects_duplicate_products_unknown_or_overlapping_splits():
    product = _product()
    with pytest.raises(ValueError, match="unique"):
        _dataset(product, product)
    with pytest.raises(ValueError, match="unknown"):
        _dataset(product, splits={"train": ("unknown",), "val": (), "test": ()})
    with pytest.raises(ValueError, match="overlap"):
        _dataset(product, splits={"train": ("velocity-field",), "val": ("velocity-field",), "test": ()})
    with pytest.raises(ValueError, match="train"):
        _dataset(product, splits={"train": (), "val": ("velocity-field",), "test": ()})
    second = _product(product_id="pressure-field")
    with pytest.raises(ValueError, match="assign every"):
        _dataset(product, second, splits={"train": ("velocity-field",), "val": (), "test": ()})


def test_dataset_training_ready_revalidates_product_status_and_evidence_after_construction() -> None:
    withheld = _product(_manifest(ValidationStatus.WITHHELD), quality_status=ValidationStatus.WITHHELD)
    dataset = _dataset(withheld)
    object.__setattr__(withheld, "quality_status", ValidationStatus.PASS)
    with pytest.raises(ValueError, match="velocity-field"):
        dataset.require_training_ready()

    product = _product()
    dataset = _dataset(product)
    object.__setattr__(product.run_manifest.artifacts[0], "payload", b'{"drag": 9.0}')
    with pytest.raises(ValueError, match="velocity-field"):
        dataset.require_training_ready()


def test_contracts_are_deeply_immutable():
    product = _product()
    dataset = _dataset(product)

    assert isinstance(product.lineage, MappingProxyType)
    assert isinstance(product.lineage["source"], MappingProxyType)
    assert isinstance(dataset.group_splits, MappingProxyType)
    assert isinstance(dataset.lineage["curation"], MappingProxyType)
    with pytest.raises(TypeError):
        product.lineage["source"]["campaign"] = "changed"
    with pytest.raises(TypeError):
        dataset.group_splits["train"] = ()
    with pytest.raises(FrozenInstanceError):
        product.field_name = "pressure"


def test_data_contract_ast_is_stdlib_plus_tensorlbm_runtime_only():
    contracts = Path("src/tensorlbm/data/contracts.py")
    tree = ast.parse(contracts.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= {"__future__", "dataclasses", "types", "typing", "tensorlbm"}
    source = contracts.read_text(encoding="utf-8")
    assert "torch" not in source
    assert "solver" not in source
    assert "timestep" not in source

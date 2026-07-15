"""Tests for cold multi-field product catalogue and leakage-safe splits."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tensorlbm.data.field_dataset_r2 import FieldDatasetR2, FieldSampleRefR2
from tensorlbm.data.field_r2 import (
    ArrayEncoding,
    ArrayManifestR2,
    ArrayRole,
    AxisSemantic,
    ByteOrder,
    AxisSpec,
    BlobRef,
    FieldDataProductR2,
    MemoryOrder,
)
from tensorlbm.runtime import ArtifactManifest, MetricEvidence, RunManifest, ValidationStatus


_CODE_SHA = "a" * 40


def _run(run_id: str, status: ValidationStatus = ValidationStatus.PASS) -> RunManifest:
    evidence = json.dumps({"drag": 1.25, "run": run_id}).encode()
    artifact = ArtifactManifest.from_bytes("metrics", "application/json", evidence)
    return RunManifest(
        run_id=run_id,
        model_identity={"case": run_id},
        config={"grid": 8},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=(MetricEvidence("drag", 1.25, "1", artifact.artifact_id, "/drag"),),
        validation_status=status,
        validation_reason="reviewed completed runtime evidence",
    )


def _product(index: int, status: ValidationStatus = ValidationStatus.PASS) -> FieldDataProductR2:
    payload_hash = hashlib.sha256(f"field-{index}".encode()).hexdigest()
    array = ArrayManifestR2(
        array_id=f"velocity-{index}",
        role=ArrayRole.FEATURE,
        shape=(2, 3),
        axes=(AxisSpec("sample", AxisSemantic.SAMPLE, 2), AxisSpec("component", AxisSemantic.COMPONENT, 3)),
        units="m/s",
        encoding=ArrayEncoding("NPY", "float32", MemoryOrder.C, ByteOrder.LITTLE),
        blob_ref=BlobRef(
            f"velocity-blob-{index}",
            f"file:///absolute/velocity-{index}.npy",
            64,
            payload_hash,
            "application/x-npy",
        ),
        component_labels=("u", "v", "w"),
    )
    return FieldDataProductR2(
        product_id=f"product-{index}",
        run_manifest=_run(f"run-{index}", status),
        source_artifact_id="metrics",
        arrays=(array,),
        lineage={"source": {"fixture": index}},
    )


def _ref(index: int, *, group: str | None = None, case: str | None = None, trajectory: str | None = None) -> FieldSampleRefR2:
    return FieldSampleRefR2(
        sample_id=f"sample-{index}",
        product=_product(index),
        group_id=group or f"group-{index}",
        source_case_id=case or f"case-{index}",
        source_trajectory_id=trajectory or f"trajectory-{index}",
    )


def _dataset(*samples: FieldSampleRefR2, splits=None, lineage=None) -> FieldDatasetR2:
    samples = samples or (_ref(1), _ref(2), _ref(3))
    return FieldDatasetR2(
        dataset_id="field-catalogue",
        version="r2",
        task_name="field-reconstruction",
        samples=tuple(samples),
        splits=splits or {"train": (samples[0].sample_id,), "val": (samples[1].sample_id,), "test": (samples[2].sample_id,)},
        lineage=lineage if lineage is not None else {"curation": {"owner": "data-governance"}},
    )


def test_multi_snapshot_product_refs_create_normal_group_safe_catalogue() -> None:
    dataset = _dataset()

    assert dataset.validate_for_use() is None
    assert dataset.training_input_fingerprint() == dataset.training_input_fingerprint()


def test_splits_reject_duplicate_unknown_overlap_unassigned_and_empty_train() -> None:
    one, two, three = _ref(1), _ref(2), _ref(3)
    with pytest.raises(ValueError, match="unique sample_id"):
        _dataset(one, one, three)
    with pytest.raises(ValueError, match="unknown"):
        _dataset(one, two, three, splits={"train": ("missing",), "val": ("sample-2",), "test": ("sample-3",)})
    with pytest.raises(ValueError, match="overlap"):
        _dataset(one, two, three, splits={"train": ("sample-1",), "val": ("sample-1", "sample-2"), "test": ("sample-3",)})
    with pytest.raises(ValueError, match="assign every"):
        _dataset(one, two, three, splits={"train": ("sample-1",), "val": (), "test": ()})
    with pytest.raises(ValueError, match="train"):
        _dataset(one, two, three, splits={"train": (), "val": ("sample-1", "sample-2"), "test": ("sample-3",)})


@pytest.mark.parametrize("field", ["group_id", "source_case_id", "source_trajectory_id"])
def test_split_rejects_group_case_and_trajectory_leakage(field: str) -> None:
    values = {"group": "group-a", "case": "case-a", "trajectory": "trajectory-a"}
    kwargs = {
        "group": values["group"] if field == "group_id" else None,
        "case": values["case"] if field == "source_case_id" else None,
        "trajectory": values["trajectory"] if field == "source_trajectory_id" else None,
    }
    first = _ref(1, **kwargs)
    second = _ref(2, **kwargs)
    with pytest.raises(ValueError, match=field):
        _dataset(first, second, _ref(3), splits={"train": ("sample-1",), "val": ("sample-2",), "test": ("sample-3",)})


@pytest.mark.parametrize("field", ["group_id", "source_case_id", "source_trajectory_id"])
def test_sample_group_fields_reject_empty_and_bytes(field: str) -> None:
    values = {"group_id": "group", "source_case_id": "case", "source_trajectory_id": "trajectory"}
    values[field] = ""
    with pytest.raises(ValueError, match=field):
        FieldSampleRefR2("sample", _product(1), **values)
    values[field] = b"payload"
    with pytest.raises(ValueError, match=field):
        FieldSampleRefR2("sample", _product(1), **values)


def test_product_gate_and_dataset_fields_are_revalidated_at_use_time() -> None:
    with pytest.raises(ValueError, match="PASS"):
        _product(1, ValidationStatus.WITHHELD)

    dataset = _dataset()
    object.__setattr__(dataset.samples[0].product.run_manifest, "validation_status", ValidationStatus.WITHHELD)
    with pytest.raises(ValueError, match="runtime evidence"):
        dataset.validate_for_use()

    dataset = _dataset()
    object.__setattr__(dataset.samples[0], "group_id", "")
    with pytest.raises(ValueError, match="group_id"):
        dataset.validate_for_use()

    dataset = _dataset()
    object.__setattr__(dataset, "splits", {"train": ("sample-1",), "val": ("sample-1", "sample-2"), "test": ("sample-3",)})
    with pytest.raises(ValueError, match="overlap"):
        dataset.validate_for_use()


def test_use_time_rejects_axis_encoding_and_blob_mutation() -> None:
    for target_name, attribute, value in (
        ("axis", "name", ""),
        ("encoding", "dtype", "not-a-dtype"),
        ("blob", "sha256", "Z" * 64),
    ):
        dataset = _dataset()
        array = dataset.samples[0].product.arrays[0]
        target = {"axis": array.axes[0], "encoding": array.encoding, "blob": array.blob_ref}[target_name]
        object.__setattr__(target, attribute, value)
        with pytest.raises((TypeError, ValueError)):
            dataset.validate_for_use()


def test_fingerprint_binds_complete_run_and_array_encoding_closure() -> None:
    baseline = _dataset()
    baseline_fingerprint = baseline.training_input_fingerprint()

    changed_run = _dataset()
    object.__setattr__(changed_run.samples[0].product.run_manifest, "config", {"grid": 9})
    assert changed_run.training_input_fingerprint() != baseline_fingerprint

    changed_encoding = _dataset()
    object.__setattr__(
        changed_encoding.samples[0].product.arrays[0].encoding,
        "byte_order",
        ByteOrder.BIG,
    )
    assert changed_encoding.training_input_fingerprint() != baseline_fingerprint



def test_fingerprint_canonicalizes_legal_product_and_runtime_immutable_values() -> None:
    product_lineage = _product(1)
    object.__setattr__(product_lineage, "lineage", {"labels": frozenset({"one", "two"})})
    ref = FieldSampleRefR2("sample-1", product_lineage, "group-1", "case-1", "trajectory-1")
    dataset = _dataset(ref, _ref(2), _ref(3))
    assert dataset.training_input_fingerprint() == dataset.training_input_fingerprint()

    runtime_bytes = _dataset()
    object.__setattr__(runtime_bytes.samples[0].product.run_manifest, "config", {"token": b"secret"})
    assert runtime_bytes.training_input_fingerprint() == runtime_bytes.training_input_fingerprint()

    with pytest.raises(TypeError, match="unsupported"):
        _dataset(lineage={"labels": {"one", "two"}})

    dataset = _dataset()
    with pytest.raises(TypeError):
        dataset.lineage["curation"]["owner"] = "changed"
    with pytest.raises(TypeError, match="bytes"):
        _dataset(lineage={"payload": b"forbidden"})
    object.__setattr__(dataset, "lineage", {"payload": b"forbidden"})
    with pytest.raises(TypeError, match="bytes"):
        dataset.validate_for_use()


def test_fingerprint_is_canonical_and_changes_for_bound_reference_or_split() -> None:
    baseline = _dataset()
    same = _dataset()
    assert baseline.training_input_fingerprint() == same.training_input_fingerprint()

    changed_ref = _dataset(_ref(1, group="changed-group"), _ref(2), _ref(3))
    assert baseline.training_input_fingerprint() != changed_ref.training_input_fingerprint()

    changed_split = _dataset(splits={"train": ("sample-1", "sample-2"), "val": (), "test": ("sample-3",)})
    assert baseline.training_input_fingerprint() != changed_split.training_input_fingerprint()


def test_production_boundary_is_stdlib_and_runtime_data_only() -> None:
    source = Path("src/tensorlbm/data/field_dataset_r2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "dataclasses", "hashlib", "json", "types", "typing", "tensorlbm"}
    lowered = source.lower()
    for forbidden in ("torch", "numpy", "solver", "timestep"):
        assert forbidden not in lowered

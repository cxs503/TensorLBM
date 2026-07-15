"""Tests for R2 field-byte bindings; this is not a training-ready data claim."""

import ast
import hashlib
import json
import struct
from pathlib import Path

import pytest

from tensorlbm.runtime import ArtifactManifest, MetricEvidence, RunManifest, ValidationStatus

from tensorlbm.data.field_r2 import (
    ArrayEncoding,
    ArrayManifestR2,
    ArrayRole,
    AxisSemantic,
    AxisSpec,
    BlobRef,
    ByteOrder,
    FieldDataProductR2,
    MemoryOrder,
)


_CODE_SHA = "a" * 40


def _npy(descr: str, shape: tuple[int, ...], values: bytes, fortran_order: bool = False) -> bytes:
    header = repr({"descr": descr, "fortran_order": fortran_order, "shape": shape}) + "\n"
    padding = (-((10 + len(header)) % 16)) % 16
    encoded = (header[:-1] + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded)) + encoded + values


def _payload(dtype: str = "<f4", *, shape: tuple[int, ...] = (2, 3), order: bool = False) -> bytes:
    item_size = {"<f4": 4, "|f4": 4, "<f8": 8, "|O": 8}[dtype]
    return _npy(dtype, shape, b"\0" * (item_size * 6), order)


def _run(status: ValidationStatus = ValidationStatus.PASS) -> RunManifest:
    evidence = json.dumps({"drag": 1.25}).encode()
    artifact = ArtifactManifest.from_bytes("completed-run-metrics", "application/json", evidence)
    return RunManifest(
        run_id="run-r2", model_identity={"case": "reference"}, config={"grid": 8},
        code_sha=_CODE_SHA, environment={"backend": "recorded"}, artifacts=(artifact,),
        metrics=(MetricEvidence("drag", 1.25, "1", artifact.artifact_id, "/drag"),),
        validation_status=status, validation_reason="reviewed completed runtime evidence",
    )


def _array(payload: bytes | None = None, *, encoding: ArrayEncoding | None = None, axes=None, components=None):
    payload = payload if payload is not None else _payload()
    blob = BlobRef("velocity-npy", "file:///absolute/velocity.npy", len(payload), hashlib.sha256(payload).hexdigest(), "application/x-npy")
    return ArrayManifestR2(
        array_id="velocity", role=ArrayRole.FEATURE, shape=(2, 3),
        axes=axes or (AxisSpec("sample", AxisSemantic.SAMPLE, 2), AxisSpec("component", AxisSemantic.COMPONENT, 3)),
        units="m/s", encoding=encoding or ArrayEncoding.NPY_FLOAT32_C_LITTLE,
        blob_ref=blob, component_labels=components or ("u", "v", "w"),
    )


def _product(array=None, run=None) -> FieldDataProductR2:
    return FieldDataProductR2(
        product_id="r2-velocity", run_manifest=run or _run(), source_artifact_id="completed-run-metrics",
        arrays=(array or _array(),), lineage={"source": {"campaign": "fixture"}},
    )


def test_float32_c_npy_payload_and_expected_dimensions_pass() -> None:
    array = _array()
    payload = _payload()
    assert array.verify_payload(payload) is None
    product = _product(array)
    assert product.validate_for_use({"velocity": payload}) is None


def test_blob_rejects_payload_tamper_and_size_mismatch() -> None:
    payload = _payload()
    blob = _array(payload).blob_ref
    with pytest.raises(ValueError, match="size"):
        blob.validate_blob_bytes(payload[:-1])
    with pytest.raises(ValueError, match="sha256"):
        blob.validate_blob_bytes(payload[:-1] + b"x")


def test_npy_body_must_be_complete_and_have_no_trailing_bytes() -> None:
    payload = _payload()
    for malformed in (payload[:-1], payload + b"attack"):
        with pytest.raises(ValueError, match="truncated or has trailing"):
            _array(malformed).verify_payload(malformed)


def test_bar_byte_order_npy_dtype_is_rejected() -> None:
    payload = _payload("|f4")
    with pytest.raises(ValueError, match="byte-order-not-applicable"):
        _array(payload).verify_payload(payload)


def test_manifest_float32_cannot_accept_actual_float64_payload() -> None:
    with pytest.raises(ValueError, match="dtype"):
        _array(_payload("<f8")).verify_payload(_payload("<f8"))


@pytest.mark.parametrize(
    ("axes", "shape", "components"),
    [
        ((AxisSpec("sample", AxisSemantic.SAMPLE, 2),), (2, 3), None),
        ((AxisSpec("sample", AxisSemantic.SAMPLE, 2), AxisSpec("c", AxisSemantic.COMPONENT, 3)), (2, 3), ("u", "u", "w")),
        ((AxisSpec("sample", AxisSemantic.SAMPLE, 2),), (2,), ("u",)),
    ],
)
def test_axis_rank_and_component_contract_errors(axes, shape, components) -> None:
    payload = _payload(shape=shape)
    blob = BlobRef("blob", "file:///absolute/blob.npy", len(payload), hashlib.sha256(payload).hexdigest(), "application/x-npy")
    with pytest.raises(ValueError):
        ArrayManifestR2("a", ArrayRole.FEATURE, shape, axes, "m/s", ArrayEncoding.NPY_FLOAT32_C_LITTLE, blob, components)


def test_object_dtype_npy_is_rejected_without_loading_or_pickle() -> None:
    payload = _payload("|O")
    blob = BlobRef("object", "file:///absolute/object.npy", len(payload), hashlib.sha256(payload).hexdigest(), "application/x-npy")
    array = ArrayManifestR2(
        "object", ArrayRole.AUXILIARY, (2, 3),
        (AxisSpec("sample", AxisSemantic.SAMPLE, 2), AxisSpec("component", AxisSemantic.COMPONENT, 3)),
        "1", ArrayEncoding.NPY_FLOAT32_C_LITTLE, blob, ("a", "b", "c"),
    )
    with pytest.raises(ValueError, match="object"):
        array.verify_payload(payload)


def test_withheld_run_cannot_create_field_product() -> None:
    with pytest.raises(ValueError, match="PASS"):
        _product(run=_run(ValidationStatus.WITHHELD))


def test_use_time_rechecks_run_status_and_payload_tampering() -> None:
    product = _product()
    payload = _payload()
    object.__setattr__(product.run_manifest, "validation_status", ValidationStatus.WITHHELD)
    with pytest.raises(ValueError, match="runtime evidence"):
        product.validate_for_use({"velocity": payload})

    product = _product()
    with pytest.raises(ValueError, match="sha256"):
        product.validate_for_use({"velocity": payload[:-1] + b"x"})

    product = _product()
    object.__setattr__(product.arrays[0], "component_labels", ("u", "u", "w"))
    with pytest.raises(ValueError, match="component_labels"):
        product.validate_for_use({"velocity": payload})

    product = _product()
    object.__setattr__(product, "lineage", {"payload": b"forbidden"})
    with pytest.raises(TypeError, match="payload bytes"):
        product.validate_for_use({"velocity": payload})

    product = _product()
    blob = product.arrays[0].blob_ref
    object.__setattr__(blob, "uri", "https://attacker.invalid/velocity.npy")
    with pytest.raises(ValueError, match="uri"):
        product.validate_for_use({"velocity": payload})

    product = _product()
    object.__setattr__(product.arrays[0].blob_ref, "media_type", "")
    with pytest.raises(ValueError, match="media_type"):
        product.validate_for_use({"velocity": payload})


def test_field_r2_production_boundary_is_stdlib_only_and_metadata_has_no_payload() -> None:
    source = Path("src/tensorlbm/data/field_r2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "ast", "dataclasses", "enum", "hashlib", "os", "re", "types", "typing", "tensorlbm"}
    lowered = source.lower()
    assert "torch" not in lowered
    assert "numpy" not in lowered
    assert "solver" not in lowered
    assert "timestep" not in lowered
    field = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlobRef")
    assert all(
        not isinstance(item, ast.AnnAssign) or getattr(item.target, "id", None) != "payload"
        for item in field.body
    )

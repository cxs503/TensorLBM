"""Tests for the solver→data export path (HDF5 snapshots → catalog products)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.data.catalog import FieldDataCatalog
from tensorlbm.data.solver_export import (
    EXPORT_SCHEMA,
    load_product,
    load_product_arrays,
    read_snapshot,
    register_product,
    save_fields_hdf5,
    snapshot_group,
)

_CODE_SHA = "b" * 40


def _fields(nz: int = 4, ny: int = 6, nx: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    mask = np.zeros(shape, dtype=bool)
    mask[:, ny // 2, nx // 4 : 3 * nx // 4] = True
    return {
        "rho": (1.0 + 0.001 * rng.standard_normal(shape)).astype(np.float32),
        "ux": (0.05 * rng.standard_normal(shape)).astype(np.float32),
        "uy": (0.01 * rng.standard_normal(shape)).astype(np.float32),
        "uz": (0.01 * rng.standard_normal(shape)).astype(np.float32),
        "solid_mask": mask,
    }


def _metadata(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": "unit-run",
        "case": "unit-case",
        "step": 100,
        "code_sha": _CODE_SHA,
        "collision": "BGK",
        "device": "cpu",
        "re": 100.0,
        "u_in": 0.05,
        "nu": 0.019,
        "tau": 0.557,
        "nx": 8,
        "ny": 6,
        "nz": 4,
        "n_steps": 100,
    }
    base.update(overrides)
    return base


@pytest.fixture
def catalog(tmp_path):
    cat = FieldDataCatalog.open(tmp_path / "catalog.db")
    yield cat
    cat.close()


def _register(tmp_path: Path, catalog: FieldDataCatalog, fields=None, metadata=None) -> str:
    fields = fields if fields is not None else _fields()
    metadata = metadata if metadata is not None else _metadata()
    h5 = save_fields_hdf5(tmp_path / "run.h5", fields, metadata)
    return register_product(catalog, h5, metadata)


# ---------------------------------------------------------------------------
# save_fields_hdf5
# ---------------------------------------------------------------------------


def test_hdf5_round_trip_numeric_equality(tmp_path):
    fields = _fields()
    path = save_fields_hdf5(tmp_path / "run.h5", fields, _metadata())
    import h5py

    assert path.is_file()
    with h5py.File(path, "r") as handle:
        assert handle.attrs["export_schema"] == EXPORT_SCHEMA
        group = handle[snapshot_group(100)]
        assert group.attrs["step"] == 100
        assert group.attrs["case"] == "unit-case"
        for name, arr in fields.items():
            stored = group[name][...]
            if name == "solid_mask":
                assert stored.dtype == np.int8
                assert np.array_equal(stored, arr.astype(np.int8))
            else:
                assert stored.dtype == np.dtype("<f4")
                assert np.array_equal(stored, arr)


def test_hdf5_second_snapshot_adds_group_and_overwrite_replaces(tmp_path):
    fields = _fields()
    path = save_fields_hdf5(tmp_path / "run.h5", fields, _metadata(step=10))
    other = dict(fields)
    other["rho"] = np.full_like(fields["rho"], 1.0)
    save_fields_hdf5(path, other, _metadata(step=20))
    arrays10, _ = read_snapshot(path, 10)
    arrays20, _ = read_snapshot(path, 20)
    assert np.array_equal(arrays10["rho"], fields["rho"])
    assert np.array_equal(arrays20["rho"], other["rho"])
    # Re-saving the same step replaces the group instead of duplicating it.
    save_fields_hdf5(path, fields, _metadata(step=20))
    arrays20_again, _ = read_snapshot(path, 20)
    assert np.array_equal(arrays20_again["rho"], fields["rho"])


def test_save_rejects_mixed_shapes_and_bad_attrs(tmp_path):
    fields = _fields()
    bad = dict(fields)
    bad["rho"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="share one shape"):
        save_fields_hdf5(tmp_path / "x.h5", bad, _metadata())
    with pytest.raises(ValueError, match="step"):
        save_fields_hdf5(tmp_path / "x.h5", fields, {"case": "no-step"})
    with pytest.raises(ValueError, match="scalar"):
        save_fields_hdf5(tmp_path / "x.h5", fields, _metadata(nested={"bad": 1}))


# ---------------------------------------------------------------------------
# register → query → load loop
# ---------------------------------------------------------------------------


def test_register_query_load_closed_loop(tmp_path, catalog):
    fields = _fields()
    product_id = _register(tmp_path, catalog, fields=fields)
    assert product_id == "unit-run:000100"

    asset = catalog.get_asset(product_id)
    assert asset is not None
    assert asset.kind == "field_product"
    assert asset.field_name == "velocity"
    assert json.loads(asset.shape) == [4, 6, 8, 3]
    assert asset.source_run_id == "unit-run"
    assert "case:unit-case" in asset.tags

    found = catalog.find_assets_by_metadata("case", "unit-case", kind="field_product")
    assert [a.asset_id for a in found] == [product_id]
    assert catalog.find_assets_by_metadata("case", "missing") == []

    product = load_product(catalog, product_id)
    assert product.product_id == product_id
    assert product.source_artifact_id == "hdf5-source"
    arrays = load_product_arrays(product)
    assert set(arrays) == {"velocity", "rho", "solid_mask"}
    assert arrays["velocity"].dtype == np.dtype("<f4")
    assert arrays["velocity"].shape == (4, 6, 8, 3)
    expected_velocity = np.stack([fields["ux"], fields["uy"], fields["uz"]], axis=-1)
    assert np.array_equal(arrays["velocity"], expected_velocity)
    assert np.array_equal(arrays["rho"], fields["rho"])
    assert arrays["solid_mask"].dtype == np.dtype("<i4")

    lineage = catalog.get_lineage(product_id)
    assert any(rec.source_id == "run:unit-run" for rec in lineage)


def test_run_manifest_is_pass_gated_with_verifiable_metric_evidence(tmp_path, catalog):
    product_id = _register(tmp_path, catalog)
    product = load_product(catalog, product_id)
    run = product.run_manifest
    assert run.validation_status.value == "PASS"
    assert run.verify_integrity()
    assert run.verify_metric_evidence()
    metric_ids = {metric.metric_id for metric in run.metrics}
    assert {"step", "u_max", "rho_mean", "solid_fraction"} <= metric_ids
    # Every metric is bound by JSON pointer into the evidence artifact.
    pointers = {metric.evidence_pointer for metric in run.metrics}
    assert all(pointer.startswith("/") for pointer in pointers)


def test_quality_checks_recorded_in_catalog(tmp_path, catalog):
    product_id = _register(tmp_path, catalog)
    reports = catalog.get_quality_reports(product_id)
    assert reports
    report = reports[0]
    assert report["status"] == "passed"
    assert report["overall_score"] == 100
    check_names = {check["check_name"] for check in report["checks"]}
    assert "finiteness" in check_names
    assert "mass_conservation" in check_names


# ---------------------------------------------------------------------------
# Export quality gate
# ---------------------------------------------------------------------------


def test_non_finite_snapshot_is_rejected(tmp_path, catalog):
    fields = _fields()
    fields["ux"][0, 0, 0] = np.nan
    h5 = save_fields_hdf5(tmp_path / "bad.h5", fields, _metadata())
    with pytest.raises(ValueError, match="non-finite"):
        register_product(catalog, h5, _metadata())


def test_density_drift_is_rejected(tmp_path, catalog):
    fields = _fields()
    fields["rho"] = np.full_like(fields["rho"], 1.1)
    h5 = save_fields_hdf5(tmp_path / "drift.h5", fields, _metadata())
    with pytest.raises(ValueError, match="drift"):
        register_product(catalog, h5, _metadata())


def test_duplicate_product_id_is_rejected(tmp_path, catalog):
    _register(tmp_path, catalog)
    with pytest.raises(ValueError, match="already registered"):
        _register(tmp_path, catalog)


def test_blob_tampering_is_detected_at_load_time(tmp_path, catalog):
    product_id = _register(tmp_path, catalog)
    product = load_product(catalog, product_id)
    blob_path = Path(product.arrays[0].blob_ref.uri.removeprefix("file://"))
    original = blob_path.read_bytes()
    blob_path.write_bytes(original[:-1] + b"\x00")
    with pytest.raises(ValueError, match="sha256"):
        load_product_arrays(product)


def test_metadata_validation(tmp_path, catalog):
    fields = _fields()
    h5 = save_fields_hdf5(tmp_path / "run.h5", fields, _metadata())
    with pytest.raises(ValueError, match="run_id"):
        register_product(catalog, h5, _metadata(run_id=""))
    with pytest.raises(ValueError, match="code_sha"):
        register_product(catalog, h5, _metadata(code_sha="XYZ"))
    with pytest.raises(ValueError, match="no snapshot group"):
        register_product(catalog, h5, _metadata(step=200))


# ---------------------------------------------------------------------------
# Input flexibility
# ---------------------------------------------------------------------------


def test_torch_tensor_inputs_match_numpy_export(tmp_path):
    import torch

    fields = _fields(seed=3)
    torch_fields = {
        name: torch.from_numpy(arr.copy()) for name, arr in fields.items()
    }
    path_numpy = save_fields_hdf5(tmp_path / "numpy.h5", fields, _metadata())
    path_torch = save_fields_hdf5(tmp_path / "torch.h5", torch_fields, _metadata())
    arrays_numpy, attrs_numpy = read_snapshot(path_numpy, 100)
    arrays_torch, attrs_torch = read_snapshot(path_torch, 100)
    assert attrs_numpy.keys() == attrs_torch.keys()
    for name in arrays_numpy:
        assert np.array_equal(arrays_numpy[name], arrays_torch[name])


def test_exact_decimal_metrics_payload_satisfies_evidence_check():
    from tensorlbm.data.solver_export import _evidence_payload
    from tensorlbm.runtime import ArtifactManifest, MetricEvidence, RunManifest, ValidationStatus

    metrics = {"rho_mean": 1.0003, "step": 400, "u_max": 0.123456789012345}
    payload = _evidence_payload(metrics)
    artifact = ArtifactManifest.from_bytes("m", "application/json", payload)
    # The values that go into MetricEvidence must be recoverable exactly
    # through the JSON-pointer evidence binding (exact decimal expansion).
    evidence = tuple(
        MetricEvidence(key, float(value), "lattice", artifact.artifact_id, f"/{key}")
        for key, value in metrics.items()
    )
    run = RunManifest(
        run_id="evidence-check",
        model_identity={"case": "unit"},
        config={"grid": 8},
        code_sha=_CODE_SHA,
        environment={"backend": "recorded"},
        artifacts=(artifact,),
        metrics=evidence,
        validation_status=ValidationStatus.PASS,
        validation_reason="unit evidence check",
    )
    assert run.verify_metric_evidence()
    with pytest.raises(ValueError, match="finite"):
        _evidence_payload({"bad": float("nan")})


def test_two_dimensional_export_registers_velocity_pair(tmp_path, catalog):
    fields = _fields(nz=1)
    fields_2d = {
        name: arr[0] for name, arr in fields.items() if name != "uz"
    }
    metadata = _metadata(nz=1)
    h5 = save_fields_hdf5(tmp_path / "run2d.h5", fields_2d, metadata)
    product_id = register_product(catalog, h5, metadata)
    product = load_product(catalog, product_id)
    velocity = next(a for a in product.arrays if a.array_id == "velocity")
    assert velocity.shape == (6, 8, 2)
    assert velocity.component_labels == ("u_x", "u_y")
    assert tuple(axis.name for axis in velocity.axes) == ("y", "x", "component")

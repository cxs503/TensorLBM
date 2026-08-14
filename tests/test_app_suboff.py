"""Tests for :class:`tensorlbm.apps.suboff_app.SuboffSurrogateApp`.

The heavy pieces (model build, training loop, inference) are injected as mocks
so nothing touches a real LBM run or a real training loop; the catalog / job /
serving ledgers are exercised for real against a temporary SQLite file.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.apps.base import DataProduct, Prediction, RunReport, TrainingResult
from tensorlbm.apps.suboff_app import SuboffSurrogateApp
from tensorlbm.ai.suboff_train import SuboffTrainConfig


@pytest.fixture
def field_data_dir(tmp_path):
    """Mock SUBOFF LBM output: data_dir/<channel>/{0,1,2}.npy (small arrays)."""
    data_dir = tmp_path / "suboff8"
    for channel in ("p", "ux", "uy", "uz"):
        ch_dir = data_dir / channel
        ch_dir.mkdir(parents=True)
        for i in range(3):
            arr = np.random.rand(4, 4, 4).astype(np.float32)
            np.save(ch_dir / f"{i}.npy", arr)
    return data_dir


# ---------------------------------------------------------------------------
# Injected mocks
# ---------------------------------------------------------------------------

def _fake_train(cfg):
    assert isinstance(cfg, SuboffTrainConfig)
    return {"best_loss_1e4": 3.21, "final_iter": 42, "checkpoint_dir": "/tmp/ckpt"}


def _fake_predict(cfg):
    return {"pred": np.zeros((3, 5)), "mape": 1.5, "error": np.ones((3, 5))}


def _fake_build(arch):
    return object()  # stand-in for (encoder, decoder)


@pytest.fixture
def app():
    return SuboffSurrogateApp(
        train_fn=_fake_train,
        predict_fn=_fake_predict,
        build_fn=_fake_build,
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_identity():
    app = SuboffSurrogateApp()
    assert app.name == "suboff_surrogate"
    assert app.family == "suboff_surrogate"
    assert app.version == "1.0"


def test_is_aifour_s_application():
    from tensorlbm.apps.base import AI4SApplication

    assert issubclass(SuboffSurrogateApp, AI4SApplication)


# ---------------------------------------------------------------------------
# produce_data / make_dataset
# ---------------------------------------------------------------------------

def test_produce_data(app, field_data_dir):
    product = app.produce_data({"data_dir": str(field_data_dir)})
    assert isinstance(product, DataProduct)
    assert product.field_name == "flow_field"
    assert product.shape == (4, 4, 4)
    assert product.dtype == "float32"
    assert product.path == str(field_data_dir)
    assert set(product.metadata["channels"]) == {"p", "ux", "uy", "uz"}
    assert product.metadata["n_snapshots"] == 3


def test_produce_data_missing_dir_raises(app, tmp_path):
    with pytest.raises(ValueError):
        app.produce_data({"data_dir": str(tmp_path / "nope")})


def test_make_dataset(app, field_data_dir):
    product = app.produce_data({"data_dir": str(field_data_dir)})
    dataset = app.make_dataset(product)
    assert dataset["channels"] == ["p", "ux", "uy", "uz"]
    assert dataset["n_snapshots"] == 3
    # [n_snap, n_points, 4] with n_points = 4*4*4 = 64
    assert dataset["data"].shape == (3, 64, 4)


# ---------------------------------------------------------------------------
# train / infer wrapping
# ---------------------------------------------------------------------------

def test_train_converts_cfg_and_returns_result(app, field_data_dir):
    product = app.produce_data({"data_dir": str(field_data_dir)})
    dataset = app.make_dataset(product)
    result = app.train(
        dataset,
        object(),
        {"data_dir": str(field_data_dir), "n_train": 10, "n_test": 2, "iters": 100},
    )
    assert isinstance(result, TrainingResult)
    assert result.model_path == "/tmp/ckpt"
    assert result.metrics["best_loss_1e4"] == 3.21
    assert result.metrics["final_iter"] == 42.0
    assert result.arch["family"] == "suboff_surrogate"


def test_train_accepts_existing_dataclass(app):
    cfg = SuboffTrainConfig(data_dir="/x", iters=5)
    result = app.train({}, object(), cfg)
    assert result.metrics["best_loss_1e4"] == 3.21


def test_infer_wraps_predict(app):
    prediction = app.infer(object(), {"snap_idx": 0, "data_dir": "/x"})
    assert isinstance(prediction, Prediction)
    assert prediction.output.shape == (3, 5)
    assert prediction.metadata["mape"] == 1.5


# ---------------------------------------------------------------------------
# run() full-stack closed loop + lineage
# ---------------------------------------------------------------------------

def test_run_closed_loop(app, field_data_dir, tmp_path):
    db_path = tmp_path / "platform.db"
    report = app.run(
        db_path,
        {"data_dir": str(field_data_dir)},
        {"arch": {}, "data_dir": str(field_data_dir), "iters": 100},
    )

    assert isinstance(report, RunReport)
    assert report.name == "suboff_surrogate"
    assert report.family == "suboff_surrogate"
    assert report.data_asset_id == "suboff_surrogate:flow_field"
    assert report.dataset_asset_id == "suboff_surrogate:dataset"
    assert report.job_id.startswith("job_")
    assert report.model_id > 0
    assert report.metrics["best_loss_1e4"] == 3.21

    # lineage: job -> dataset -> field product (transitive)
    assert set(report.lineage_upstream) == {
        "suboff_surrogate:dataset",
        "suboff_surrogate:flow_field",
    }


def test_run_registers_catalog_assets_and_job(app, field_data_dir, tmp_path):
    db_path = tmp_path / "platform.db"
    report = app.run(
        db_path,
        {"data_dir": str(field_data_dir)},
        {"arch": {}, "data_dir": str(field_data_dir), "iters": 100},
        name_prefix="suboff",
    )

    from tensorlbm.data.catalog import FieldDataCatalog
    from tensorlbm.ml.training_job import TrainingJobRegistry

    catalog = FieldDataCatalog.open(db_path)
    try:
        # field product asset registered with its shape
        asset = catalog.get_asset("suboff:flow_field")
        assert asset is not None and asset.kind == "field_product"
        assert asset.shape == "(4, 4, 4)"
        # dataset asset registered
        assert catalog.get_asset("suboff:dataset") is not None
        # quality report recorded for the field product
        reports = catalog.get_quality_reports("suboff:flow_field")
        assert reports and reports[0]["status"] == "passed"
    finally:
        catalog.close()

    training = TrainingJobRegistry.open(db_path)
    try:
        job = training.get_job(report.job_id)
        assert job is not None and job.status == "completed"
        assert (job.metrics or {}).get("best_loss_1e4") == 3.21
    finally:
        training.close()


def test_run_registers_model_with_family(app, field_data_dir, tmp_path):
    db_path = tmp_path / "platform.db"
    report = app.run(
        db_path,
        {"data_dir": str(field_data_dir)},
        {"arch": {}, "data_dir": str(field_data_dir), "iters": 100},
    )

    from tensorlbm.ml.serving import ModelRegistry

    serving = ModelRegistry.open(db_path)
    try:
        record = serving.get_model(report.model_id)
        assert record is not None
        # family is embedded in the serving descriptor
        arch = record.get("arch") or {}
        assert arch.get("family") == "suboff_surrogate"
    finally:
        serving.close()


def test_run_passes_arch_to_build_fn(field_data_dir, tmp_path):
    seen = {}

    def _build(arch):
        seen["arch"] = arch
        return object()

    app = SuboffSurrogateApp(
        train_fn=_fake_train, predict_fn=_fake_predict, build_fn=_build
    )
    app.run(
        tmp_path / "platform.db",
        {"data_dir": str(field_data_dir)},
        {"arch": {"device": "cpu"}, "data_dir": str(field_data_dir), "iters": 100},
    )
    assert seen["arch"] == {"device": "cpu"}


def test_default_train_fn_is_real_suboff_train():
    """The un-injected app resolves to the real train/predict implementations."""
    import tensorlbm.ai.suboff_train as st
    import tensorlbm.ai.suboff_inference as si

    app = SuboffSurrogateApp()
    assert app._train_fn is None  # resolves lazily inside train()
    # resolution target functions are importable
    assert callable(st.train_suboff)
    assert callable(si.predict_suboff)

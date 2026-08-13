"""Tests for training-job management (clean-room impl)."""

import pytest

from tensorlbm.ai.database import list_models
from tensorlbm.data.catalog import AssetRecord, FieldDataCatalog
from tensorlbm.ml.training_job import TrainingJob, TrainingJobRegistry


@pytest.fixture
def registry(tmp_path):
    reg = TrainingJobRegistry.open(tmp_path / "training.db")
    yield reg
    reg.close()


@pytest.fixture
def catalog(tmp_path):
    cat = FieldDataCatalog.open(tmp_path / "training.db")
    yield cat
    cat.close()


def _config(**kw):
    base = dict(
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        task="FIELD_RECONSTRUCTION",
    )
    base.update(kw)
    return base


def test_create_and_get(registry):
    job = registry.create_job(_config(), job_id="job_a", dataset_id=7)
    assert isinstance(job, TrainingJob)
    assert job.job_id == "job_a"
    assert job.status == "created"
    assert job.dataset_id == 7
    assert job.config["epochs"] == 10

    fetched = registry.get_job("job_a")
    assert fetched == job


def test_create_auto_job_id(registry):
    job = registry.create_job(_config())
    assert job.job_id.startswith("job_")
    assert len(job.job_id) == len("job_") + 12


def test_create_duplicate_rejected(registry):
    registry.create_job(_config(), job_id="dup")
    with pytest.raises(ValueError):
        registry.create_job(_config(), job_id="dup")


def test_get_missing_returns_none(registry):
    assert registry.get_job("nope") is None


def test_list_filter_by_status(registry):
    registry.create_job(_config(), job_id="j1")
    registry.create_job(_config(), job_id="j2")
    registry.update_status("j1", "running")
    running = registry.list_jobs(status="running")
    assert [j.job_id for j in running] == ["j1"]
    created = registry.list_jobs(status="created")
    assert [j.job_id for j in created] == ["j2"]
    assert len(registry.list_jobs()) == 2


def test_status_transition_path(registry):
    registry.create_job(_config(), job_id="job")
    job = registry.update_status("job", "queued")
    assert job.status == "queued"
    job = registry.update_status("job", "running")
    assert job.status == "running"
    job = registry.update_status("job", "completed")
    assert job.status == "completed"


def test_invalid_transition_rejected(registry):
    registry.create_job(_config(), job_id="job")
    with pytest.raises(ValueError):
        registry.update_status("job", "completed")  # created -> completed illegal


def test_terminal_state_rejects_further_transitions(registry):
    registry.create_job(_config(), job_id="job")
    registry.update_status("job", "failed", error="boom")
    with pytest.raises(ValueError):
        registry.update_status("job", "running")


def test_failed_records_error(registry):
    registry.create_job(_config(), job_id="job")
    registry.update_status("job", "running")
    job = registry.update_status("job", "failed", error="out of memory")
    assert job.error == "out of memory"


def test_update_missing_job_raises(registry):
    with pytest.raises(KeyError):
        registry.update_status("ghost", "running")


def test_record_metrics_merges(registry):
    registry.create_job(_config(), job_id="job")
    registry.record_metrics("job", {"loss": 0.5, "val_loss": 0.6})
    job = registry.record_metrics("job", {"accuracy": 0.9})
    assert job.metrics == {"loss": 0.5, "val_loss": 0.6, "accuracy": 0.9}
    # persisted metrics survive a fresh fetch
    assert registry.get_job("job").metrics == job.metrics


def test_register_model_backfills_model_id(registry):
    registry.create_job(_config(), job_id="job", dataset_id=3)
    model_id = registry.register_model(
        "job",
        name="flow-transformer",
        path="/tmp/model.pt",
        arch={"family": "flow_transformer_ssl", "backend": "torch"},
        metrics={"val_loss": 0.1},
    )
    assert isinstance(model_id, int) and model_id > 0
    job = registry.get_job("job")
    assert job.model_id == model_id

    models = list_models(registry._conn)
    assert len(models) == 1
    assert models[0]["id"] == model_id
    assert models[0]["dataset_id"] == 3  # inherited from the job
    assert models[0]["metrics"]["val_loss"] == 0.1


def test_register_model_missing_job_raises(registry):
    with pytest.raises(KeyError):
        registry.register_model("ghost", name="m", path="/p", arch={})


def test_lineage_job_dataset_product(registry, catalog):
    # Register assets in the catalog (dataset <- product), then record the
    # training-job <- dataset <- product lineage through the registry.
    catalog.register_asset(AssetRecord("prod1", "u-velocity", kind="field_product"))
    catalog.register_asset(AssetRecord("ds1", "train-set", kind="dataset"))
    catalog.register_asset(AssetRecord("job_a", "flow-transformer-run", kind="model"))

    registry.record_lineage(
        catalog,
        job_asset_id="job_a",
        dataset_asset_id="ds1",
        product_asset_id="prod1",
    )

    lineage = catalog.get_lineage("job_a")
    assert len(lineage) == 1
    assert lineage[0].source_id == "ds1"
    assert lineage[0].relation_type == "trained_on"

    upstream = catalog.upstream("job_a")
    assert upstream == ["ds1", "prod1"]


def test_lineage_dataset_only(registry, catalog):
    catalog.register_asset(AssetRecord("ds1", "train-set", kind="dataset"))
    catalog.register_asset(AssetRecord("job_a", "run", kind="model"))
    registry.record_lineage(catalog, job_asset_id="job_a", dataset_asset_id="ds1")
    assert catalog.upstream("job_a") == ["ds1"]

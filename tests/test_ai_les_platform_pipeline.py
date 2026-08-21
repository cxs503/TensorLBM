"""Tests for AI-LES full-pipeline integration (run->dataset->job->model)."""

from types import SimpleNamespace

import pytest

from tensorlbm.ai.ai_les_platform_pipeline import AILesPlatformPipeline


@pytest.fixture
def pipeline(tmp_path):
    p = AILesPlatformPipeline(tmp_path / "ailes.db")
    yield p
    p.close()


def _fake_pipeline(work_dir, **_kw):
    return SimpleNamespace(
        run_id=7,
        dataset_id=9,
        model_id=11,
        n_samples=120,
        data_source="les",
        n_snapshots=12,
        training_time_s=3.1,
        training={"final_train_mse": 0.0023, "epochs": 30},
        validation={"rel_error": 0.04},
        db_path=f"{work_dir}/ai_pipeline.db",
        dataset_path=f"{work_dir}/dataset.pt",
        model_path=f"{work_dir}/model.pt",
    )


def test_full_integration(pipeline, tmp_path):
    out = pipeline.run(
        tmp_path,
        name_prefix="ai_les",
        pipeline_fn=_fake_pipeline,
        pipeline_kwargs={"nx": 16, "ny": 16},
    )
    assert out["run_asset_id"] == "ai_les:run:7"
    assert out["dataset_asset_id"] == "ai_les:dataset:9"
    assert out["job_id"].startswith("job_")
    assert out["model_id"] > 0

    # training job recorded with completed status + metrics
    job = pipeline.training.get_job(out["job_id"])
    assert job.status == "completed"
    assert job.metrics.get("final_train_mse") == 0.0023

    # serving model registered with the eddy-viscosity MLP family
    model = pipeline.serving.get_model(out["model_id"])
    assert model is not None
    arch = model.get("arch") or {}
    assert arch.get("family") == "eddy_viscosity_mlp"

    # lineage: run -> dataset -> job
    upstream = pipeline.upstream_assets(f"ai_les:job:{out['job_id']}")
    assert "ai_les:dataset:9" in upstream
    assert "ai_les:run:7" in upstream


def test_dataset_upstream(pipeline, tmp_path):
    pipeline.run(tmp_path, name_prefix="ai_les", pipeline_fn=_fake_pipeline)
    upstream = pipeline.upstream_assets("ai_les:dataset:9")
    assert upstream == ["ai_les:run:7"]


def test_default_pipeline_fn_importable(pipeline):
    """run_ai_les_pipeline is importable as the default fn (not executed)."""
    from tensorlbm.ai.pipeline import run_ai_les_pipeline

    assert callable(run_ai_les_pipeline)

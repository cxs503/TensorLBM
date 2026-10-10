"""Tests for the SUBOFF full-pipeline integration (data->train->serve)."""

from pathlib import Path

import numpy as np
import pytest

from tensorlbm.ai.suboff_platform_pipeline import SuboffPlatformPipeline


@pytest.fixture
def field_data_dir(tmp_path):
    """Mock SUBOFF LBM output: data_dir/<channel>/*.npy."""
    data_dir = tmp_path / "suboff8"
    for channel in ("p", "ux", "uy", "uz"):
        ch_dir = data_dir / channel
        ch_dir.mkdir(parents=True)
        for i in range(3):
            arr = np.random.rand(8, 8, 8).astype(np.float32)
            np.save(ch_dir / f"snap_{i:05d}.npy", arr)
    return data_dir


@pytest.fixture
def pipeline(tmp_path):
    p = SuboffPlatformPipeline(tmp_path / "platform.db")
    yield p
    p.close()


def _fake_train(cfg):
    return {"best_loss_1e4": 3.21, "final_iter": 42, "checkpoint_dir": "/tmp/ckpt"}


def _fake_predict(cfg):
    return {"mape": 1.5, "rel_l2_avg": 2.3, "pred": np.zeros((3, 5))}


def _fake_predict_demo(cfg):
    n = 100 * 50 * 100
    real = np.zeros((n, 5), dtype=np.float32)
    pred = np.zeros((n, 5), dtype=np.float32)
    pred[:, 4] = 0.1
    error = pred - real
    return {
        "coords": np.zeros((n, 3), dtype=np.float32),
        "real": real,
        "pred": pred,
        "error": error,
        "input": np.zeros((20_000, 8), dtype=np.float32),
        "mape": 1.0,
        "rel_l2_avg": 2.0,
        "mse_avg": 3.0,
        "checkpoint": cfg.checkpoint_path,
        "snap_idx": cfg.snap_idx,
    }


def test_suboff_v03_checkpoint_contract_loadable():
    torch = pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_path = repo_root / "models" / "suboff_v0.3.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert {"encoder", "decoder", "n_iter", "enc_optim", "enc_sched"} <= set(ckpt)


def test_register_field_data(pipeline, field_data_dir):
    reg = pipeline.register_field_data(field_data_dir, name_prefix="suboff")
    assert reg["asset_ids"] == ["suboff:p", "suboff:ux", "suboff:uy", "suboff:uz"]
    assert reg["dataset_asset_id"] == "suboff:dataset"
    assert reg["dataset_id"] > 0
    # each channel asset has a quality report
    rec = pipeline.catalog.get_asset("suboff:ux")
    assert rec is not None and rec.kind == "field_product"
    reports = pipeline.catalog.get_quality_reports("suboff:ux")
    assert reports and reports[0]["status"] == "passed"
    # lineage: every channel -> dataset
    upstream = pipeline.upstream_assets("suboff:dataset")
    assert set(upstream) == {"suboff:p", "suboff:ux", "suboff:uy", "suboff:uz"}


def test_run_training_lifecycle(pipeline, field_data_dir):
    reg = pipeline.register_field_data(field_data_dir)
    job, model_id = pipeline.run_training(
        {"lr": 6e-4, "iters": 10, "data_dir": str(field_data_dir)},
        dataset_id=reg["dataset_id"],
        dataset_asset_id=reg["dataset_asset_id"],
        product_asset_ids=reg["asset_ids"],
        train_fn=_fake_train,
    )
    assert job.job_id.startswith("job_")
    assert model_id > 0
    stored = pipeline.training.get_job(job.job_id)
    assert stored.status == "completed"
    assert stored.metrics.get("best_loss_1e4") == 3.21
    # lineage: dataset -> job
    upstream = pipeline.upstream_assets(f"job:{job.job_id}")
    assert reg["dataset_asset_id"] in upstream
    assert "suboff:p" in upstream  # transitive through dataset


def test_run_training_failure_marks_failed(pipeline, field_data_dir):
    reg = pipeline.register_field_data(field_data_dir)

    def _boom(cfg):
        raise RuntimeError("training exploded")

    with pytest.raises(RuntimeError):
        pipeline.run_training(
            {"iters": 10},
            dataset_id=reg["dataset_id"],
            train_fn=_boom,
        )
    jobs = pipeline.training.list_jobs()
    assert jobs and jobs[0].status == "failed"
    assert "exploded" in (jobs[0].error or "")


def test_full_pipeline(pipeline, field_data_dir):
    out = pipeline.run_full_pipeline(
        field_data_dir,
        {"lr": 6e-4, "iters": 10, "data_dir": str(field_data_dir)},
        {"snap_idx": 0},
        train_fn=_fake_train,
        predict_fn=_fake_predict,
    )
    assert out["job_id"].startswith("job_")
    assert out["model_id"] > 0
    assert out["inference"]["mape"] == 1.5
    # full lineage chain: channel -> dataset -> job
    upstream = pipeline.upstream_assets(f"job:{out['job_id']}")
    assert "suboff:dataset" in upstream
    assert "suboff:p" in upstream


def test_default_inference_fn(pipeline, field_data_dir):
    """Default predict path is importable (real suboff_inference.predict_suboff)."""
    out = pipeline.run_full_pipeline(
        field_data_dir,
        {"iters": 1},
        None,
        train_fn=_fake_train,
    )
    assert out["model_id"] > 0
    assert "inference" not in out  # predict_cfg was None


def test_checkpoint_inference_demo_artifacts(pipeline, field_data_dir, tmp_path):
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parent.parent
    out = pipeline.run_checkpoint_inference_demo(
        data_dir=field_data_dir,
        output_dir=tmp_path / "demo_out",
        checkpoint_path=repo_root / "models" / "suboff_v0.3.pt",
        snap_idx=0,
        test_set_offset=0,
        predict_fn=_fake_predict_demo,
    )

    meta_path = tmp_path / "demo_out" / "run_metadata.json"
    assert meta_path.is_file()
    assert out["contract"]["expected_layout"] == "data_dir/{p,ux,uy,uz}/*.npy"
    assert out["inference"]["metrics"]["mape"] == 1.0
    figure_files = out["artifacts"]["flowfield_figures"]
    assert len(figure_files) == 6
    assert all(Path(p).is_file() for p in figure_files)

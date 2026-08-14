"""Tests for Flow-Transformer full-pipeline integration."""

import torch
import pytest

from tensorlbm.ai.flow_transformer_platform_pipeline import (
    FlowTransformerPlatformPipeline,
)


@pytest.fixture
def pipeline(tmp_path):
    p = FlowTransformerPlatformPipeline(tmp_path / "flowtf.db")
    yield p
    p.close()


def _snapshots(n=3, shape=(8, 8)):
    return [
        (torch.rand(shape), torch.rand(shape))
        for _ in range(n)
    ]


def _fake_train(snapshots, out_path, arch=None, config=None):
    return {
        "path": str(out_path),
        "family": "flow_transformer_ssl",
        "arch": {"d_model": 32, "n_layers": 2},
        "config": {"epochs": 5, "batch_size": 4},
        "backend": "torch",
        "n_snapshots": len(snapshots),
        "n_tokens": 64,
        "grid": [8, 8],
        "final_train_loss": 0.41,
        "final_val_loss": 0.45,
    }


def test_full_integration(pipeline, tmp_path):
    out = pipeline.run(
        tmp_path, _snapshots(), name_prefix="flow_tf", train_fn=_fake_train,
    )
    assert out["data_asset_id"] == "flow_tf:flow-field"
    assert out["dataset_asset_id"] == "flow_tf:dataset"
    assert out["job_id"].startswith("job_")
    assert out["model_id"] > 0

    job = pipeline.training.get_job(out["job_id"])
    assert job.status == "completed"
    assert job.metrics.get("final_train_loss") == 0.41

    model = pipeline.serving.get_model(out["model_id"])
    arch = (model or {}).get("arch") or {}
    assert arch.get("family") == "flow_transformer_ssl"

    # lineage: flow-field -> dataset -> job
    upstream = pipeline.upstream_assets(f"flow_tf:job:{out['job_id']}")
    assert "flow_tf:dataset" in upstream
    assert "flow_tf:flow-field" in upstream


def test_dataset_upstream(pipeline, tmp_path):
    pipeline.run(tmp_path, _snapshots(), train_fn=_fake_train)
    assert pipeline.upstream_assets("flow_tf:dataset") == ["flow_tf:flow-field"]


def test_empty_snapshots_rejected(pipeline, tmp_path):
    with pytest.raises(ValueError):
        pipeline.run(tmp_path, [], train_fn=_fake_train)


def test_default_train_fn_importable(pipeline):
    from tensorlbm.ai.transformer import train_flow_transformer_self_supervised
    assert callable(train_flow_transformer_self_supervised)

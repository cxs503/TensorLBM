"""Tests for platform transformer self-supervised AI endpoints."""
from __future__ import annotations

from pathlib import Path


def test_transformer_train_list_infer(client):
    train_req = {
        "nx": 16,
        "ny": 16,
        "data_steps": 4,
        "sample_every": 2,
        "epochs": 2,
        "batch_size": 2,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "ffn_dim": 32,
    }
    r = client.post("/api/ai/transformer/train", json=train_req)
    assert r.status_code == 200, r.text
    train = r.json()
    assert train["ok"] is True
    assert Path(train["model_path"]).exists()
    assert train["n_snapshots"] >= 1

    r = client.get("/api/ai/transformer/models")
    assert r.status_code == 200, r.text
    models = r.json()
    assert models["count"] >= 1

    ir = client.post(
        "/api/ai/transformer/infer",
        json={"model_path": train["model_path"], "nx": 16, "ny": 16, "seed": 1},
    )
    assert ir.status_code == 200, ir.text
    infer = ir.json()
    assert infer["ok"] is True
    assert infer["mse"] >= 0.0
    assert infer["max_abs_error"] >= 0.0

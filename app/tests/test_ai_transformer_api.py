"""Tests for platform transformer self-supervised AI endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient


def test_transformer_train_list_infer(
    client: TestClient,
    waiter: Callable[[str, float], dict],
) -> None:
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
    queued = r.json()
    assert queued["ok"] is True
    assert queued["status"] == "queued"

    job_id = queued["job_id"]
    status_r = client.get(f"/api/ai/transformer/train/{job_id}")
    assert status_r.status_code == 200, status_r.text
    assert status_r.json()["job_id"] == job_id

    job = waiter(job_id, timeout=60)
    assert job["status"] == "completed"
    assert job["result"]["ok"] is True
    assert Path(job["result"]["model_path"]).exists()
    assert job["result"]["n_snapshots"] >= 2
    assert job["diagnostics"]

    r = client.get("/api/ai/transformer/models")
    assert r.status_code == 200, r.text
    models = r.json()
    assert models["count"] >= 1

    ir = client.post(
        "/api/ai/transformer/infer",
        json={"model_id": job["result"]["model_id"], "nx": 16, "ny": 16, "seed": 1},
    )
    assert ir.status_code == 200, ir.text
    infer = ir.json()
    assert infer["ok"] is True
    assert infer["mse"] >= 0.0
    assert infer["max_abs_error"] >= 0.0


def test_transformer_train_requires_two_snapshots(client: TestClient) -> None:
    r = client.post(
        "/api/ai/transformer/train",
        json={"nx": 16, "ny": 16, "data_steps": 3, "sample_every": 2, "epochs": 1},
    )
    assert r.status_code == 422
    assert "two sampled snapshots" in r.text

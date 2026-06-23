"""Tests for multi-backend AI transformer API.

Verifies that the ``backend`` field in POST /api/ai/transformer/train:
  1. Is accepted without error (schema).
  2. Defaults to "torch" when omitted.
  3. Is reflected in the training result and API response.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from fastapi.testclient import TestClient


def _train_and_wait(client, waiter, extra: dict | None = None, *, timeout: float = 60):
    req = {
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
    if extra:
        req.update(extra)
    r = client.post("/api/ai/transformer/train", json=req)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    return waiter(job_id, timeout=timeout)


def test_default_backend_is_torch(
    client: TestClient,
    waiter: Callable[[str, float], dict],
) -> None:
    """Omitting ``backend`` should default to 'torch' in the result."""
    job = _train_and_wait(client, waiter)
    assert job["status"] == "completed"
    result = job["result"]
    assert result.get("backend", "torch") == "torch"


def test_explicit_torch_backend(
    client: TestClient,
    waiter: Callable[[str, float], dict],
) -> None:
    """Explicitly passing backend='torch' should succeed and echo the value."""
    job = _train_and_wait(client, waiter, extra={"backend": "torch"})
    assert job["status"] == "completed"
    result = job["result"]
    assert result["backend"] == "torch"


def test_invalid_backend_returns_error(client: TestClient) -> None:
    """An unknown backend should be rejected by request validation."""
    req = {
        "nx": 16,
        "ny": 16,
        "data_steps": 4,
        "sample_every": 2,
        "epochs": 1,
        "batch_size": 2,
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "ffn_dim": 32,
        "backend": "tensorflow",
    }
    r = client.post("/api/ai/transformer/train", json=req)
    assert r.status_code == 422, r.text


def test_backend_field_appears_in_queued_response(client: TestClient) -> None:
    """The initial queued response should not error and job_id is present."""
    req = {
        "nx": 16,
        "ny": 16,
        "data_steps": 2,
        "sample_every": 1,
        "epochs": 1,
        "batch_size": 2,
        "d_model": 8,
        "n_heads": 1,
        "n_layers": 1,
        "ffn_dim": 16,
        "backend": "torch",
    }
    r = client.post("/api/ai/transformer/train", json=req)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert "job_id" in data


def test_infer_response_reports_backend(
    client: TestClient,
    waiter: Callable[[str, float], dict],
) -> None:
    job = _train_and_wait(client, waiter, extra={"backend": "torch"})
    model_id = job["result"]["model_id"]
    r = client.post("/api/ai/transformer/infer", json={"model_id": model_id, "nx": 16, "ny": 16})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["backend"] == "torch"

"""Regression checks for important OpenAPI schema contracts."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_openapi_exposes_new_jobs_control_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/jobs/{job_id}/cancel" in paths
    assert "/api/jobs/cleanup" in paths
    assert "/api/ai/transformer/train/{job_id}" in paths


def test_openapi_preprocess_random_porosity_uses_sigma(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    req_ref = schema["paths"]["/api/preprocess/random-porosity-2d"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]["$ref"]
    req_name = req_ref.rsplit("/", maxsplit=1)[-1]
    props = schema["components"]["schemas"][req_name]["properties"]
    assert "sigma" in props
    assert "corr_length" not in props


def test_openapi_solver_cylinder_has_physics_field(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    req_ref = schema["paths"]["/api/solve/cylinder-flow"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    req_name = req_ref.rsplit("/", maxsplit=1)[-1]
    props = schema["components"]["schemas"][req_name]["properties"]
    assert "physics" in props

"""Tests for the AI4S application management API (``/api/apps``).

Verifies ``GET /api/apps`` (application discovery via the framework registry),
``POST /api/apps/{name}/run`` (full-stack closed-loop run returning a
``RunReport``), and ``GET /api/apps/{name}/run/{job_id}`` (run-status query).
A lightweight demo application with a mocked ``run`` is registered so the
tests exercise the real router wiring without a real solver / training loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.main import app as fastapi_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tensorlbm.apps.base import (  # noqa: E402
    AI4SApplication,
    DataProduct,
    Prediction,
    RunReport,
    TrainingResult,
    registry,
)


class DemoApp(AI4SApplication):
    """Minimal demo app whose ``run`` returns a fixed report (no compute)."""

    name = "demo_app"
    family = "demo"
    version = "1.0"

    def produce_data(self, cfg):
        return DataProduct(name="demo", field_name="u", shape=(1,), dtype="float32")

    def build_model(self, arch):
        return torch.nn.Linear(1, 1)

    def make_dataset(self, product):
        return {"x": [1.0]}

    def train(self, dataset, model, cfg):
        return TrainingResult(model_path="/tmp/demo.pt", metrics={"loss": 0.0}, arch={})

    def infer(self, model, sample):
        return Prediction(output=0.0)

    def run(self, db_path, produce_cfg, train_cfg, **kwargs):
        return RunReport(
            name=self.name,
            family=self.family,
            data_asset_id="demo_app:u",
            dataset_asset_id="demo_app:dataset",
            job_id="job_demo_1",
            model_id=1,
            metrics={"loss": 0.0},
            lineage_upstream=("demo_app:dataset", "demo_app:u"),
        )


@pytest.fixture(scope="module")
def client():
    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture()
def demo_app():
    registry.register(DemoApp)
    yield DemoApp
    registry._apps.pop("demo_app", None)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_list_apps_includes_registered_app(client, demo_app):
    resp = client.get("/api/apps")
    assert resp.status_code == 200
    body = resp.json()
    names = {a["name"] for a in body["apps"]}
    assert "demo_app" in names
    demo = next(a for a in body["apps"] if a["name"] == "demo_app")
    assert demo["family"] == "demo"
    assert demo["version"] == "1.0"
    assert body["total"] == len(body["apps"]) >= 1


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def test_run_app_returns_run_report(client, demo_app):
    resp = client.post(
        "/api/apps/demo_app/run",
        json={"produce_cfg": {"nx": 4}, "train_cfg": {"epochs": 1}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "demo_app"
    assert body["family"] == "demo"
    assert body["job_id"] == "job_demo_1"
    assert body["model_id"] == 1
    assert body["data_asset_id"] == "demo_app:u"
    assert body["dataset_asset_id"] == "demo_app:dataset"
    assert body["metrics"] == {"loss": 0.0}
    assert set(body["lineage_upstream"]) == {"demo_app:dataset", "demo_app:u"}


def test_run_unknown_app_returns_404(client, demo_app):
    resp = client.post("/api/apps/does_not_exist/run", json={})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Status query
# ---------------------------------------------------------------------------

def test_get_run_status_after_run(client, demo_app):
    client.post("/api/apps/demo_app/run", json={})
    resp = client.get("/api/apps/demo_app/run/job_demo_1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job_demo_1"
    assert body["app_name"] == "demo_app"
    assert body["status"] == "completed"


def test_get_unknown_run_status_returns_404(client, demo_app):
    resp = client.get("/api/apps/demo_app/run/nope")
    assert resp.status_code == 404

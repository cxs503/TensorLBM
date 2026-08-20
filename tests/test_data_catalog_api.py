"""API tests for the field-data catalog router (``/api/data``).

These exercise the FastAPI layer built over ``tensorlbm.data.catalog`` using
a temporary SQLite database via dependency override, plus a registration
check against the real platform application in ``backend.main``.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Make the ``backend`` package (under ``app/``) importable.  ``src/`` is
# already on ``sys.path`` via ``PYTHONPATH=src``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from backend.routers.data_catalog import get_catalog, router  # noqa: E402

from tensorlbm.data.catalog import FieldDataCatalog  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient bound to the catalog router with a temp SQLite database."""
    db_path = tmp_path / "data_catalog.db"

    def override_catalog() -> Iterator[FieldDataCatalog]:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        catalog = FieldDataCatalog(conn)
        try:
            yield catalog
        finally:
            catalog.close()

    app = FastAPI()
    app.dependency_overrides[get_catalog] = override_catalog
    app.include_router(router, prefix="/api/data")

    with TestClient(app) as c:
        yield c


def _asset_payload(asset_id: str = "p1", **overrides) -> dict:
    payload = {
        "asset_id": asset_id,
        "name": "u-velocity",
        "kind": "field_product",
        "description": "x velocity magnitude",
        "field_name": "ux",
        "units": "lu",
        "shape": "[4, 4, 4]",
        "dtype": "float32",
        "tags": ["velocity", "lbm"],
    }
    payload.update(overrides)
    return payload


def _cube(value: float = 0.0, shape: tuple[int, ...] = (4, 4, 4)) -> list:
    """Build a nested list of the given shape filled with ``value``."""

    def build(dims: tuple[int, ...]) -> list:
        if len(dims) == 1:
            return [value for _ in range(dims[0])]
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(shape)


# ---------------------------------------------------------------------------
# Asset CRUD
# ---------------------------------------------------------------------------


def test_register_and_get_asset(client):
    r = client.post("/api/data/assets", json=_asset_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["asset_id"] == "p1"
    assert body["field_name"] == "ux"
    assert body["tags"] == ["velocity", "lbm"]

    r2 = client.get("/api/data/assets/p1")
    assert r2.status_code == 200
    assert r2.json()["name"] == "u-velocity"


def test_get_missing_asset_returns_404(client):
    r = client.get("/api/data/assets/nope")
    assert r.status_code == 404


def test_register_invalid_asset_returns_422(client):
    r = client.post("/api/data/assets", json=_asset_payload(kind="bogus"))
    assert r.status_code == 422


def test_list_assets_filters_and_pagination(client):
    client.post("/api/data/assets", json=_asset_payload("p1"))
    client.post("/api/data/assets", json=_asset_payload("p2", name="pressure", field_name="p"))
    client.post("/api/data/assets", json=_asset_payload("p3", kind="dataset", field_name=None))

    # filter by kind
    r = client.get("/api/data/assets", params={"kind": "field_product"})
    assert r.status_code == 200
    assert r.json()["total"] == 2

    # filter by field_name
    r = client.get("/api/data/assets", params={"field_name": "p"})
    assert r.json()["total"] == 1
    assert r.json()["assets"][0]["asset_id"] == "p2"

    # pagination
    r = client.get("/api/data/assets", params={"limit": 1, "offset": 1})
    assert r.json()["total"] == 3
    assert len(r.json()["assets"]) == 1

    # invalid status filter
    r = client.get("/api/data/assets", params={"status": "bogus"})
    assert r.status_code == 422


def test_update_asset(client):
    client.post("/api/data/assets", json=_asset_payload())
    r = client.put(
        "/api/data/assets/p1",
        json={"name": "u-renamed", "tags": ["new"], "quality_score": 42},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "u-renamed"
    assert body["tags"] == ["new"]
    assert body["quality_score"] == 42
    # untouched fields preserved
    assert body["field_name"] == "ux"


def test_update_asset_invalid_status_returns_422(client):
    client.post("/api/data/assets", json=_asset_payload())
    r = client.put("/api/data/assets/p1", json={"status": "bogus"})
    assert r.status_code == 422


def test_archive_asset_soft_delete(client):
    client.post("/api/data/assets", json=_asset_payload())
    r = client.delete("/api/data/assets/p1")
    assert r.status_code == 200
    assert r.json()["status"] == "archived"

    # default listing hides archived assets
    assert client.get("/api/data/assets").json()["total"] == 0
    # but they remain queryable
    r = client.get("/api/data/assets", params={"status": "archived"})
    assert r.json()["total"] == 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_roundtrip(client):
    client.post("/api/data/assets", json=_asset_payload())
    r = client.post(
        "/api/data/assets/p1/metadata",
        json={"key": "solver", "value": "octree-shell", "source": "auto"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["key"] == "solver"

    r = client.get("/api/data/assets/p1/metadata")
    assert r.status_code == 200
    assert len(r.json()) == 1 and r.json()[0]["value"] == "octree-shell"

    r = client.delete("/api/data/assets/p1/metadata", params={"key": "solver"})
    assert r.status_code == 200
    assert client.get("/api/data/assets/p1/metadata").json() == []


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


def test_lineage_and_upstream(client):
    client.post("/api/data/assets", json=_asset_payload("run1", kind="run"))
    client.post("/api/data/assets", json=_asset_payload("prod1", source_run_id="run1"))
    client.post("/api/data/assets", json=_asset_payload("ds1", kind="dataset"))

    r = client.post(
        "/api/data/assets/run1/lineage",
        json={"target_id": "prod1", "relation_type": "derived_from"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["source_id"] == "run1"

    client.post(
        "/api/data/assets/prod1/lineage",
        json={"target_id": "ds1", "relation_type": "split_of"},
    )

    r = client.get("/api/data/assets/ds1/lineage")
    assert r.status_code == 200
    body = r.json()
    assert len(body["lineage"]) == 1
    assert body["upstream"] == ["prod1", "run1"]


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def test_quality_check_passes_on_clean_field(client):
    client.post("/api/data/assets", json=_asset_payload(shape="[4, 4, 4]"))
    r = client.post(
        "/api/data/quality/check",
        json={"asset_id": "p1", "data": _cube(0.0)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall_score"] == 100
    assert body["status"] == "passed"
    assert {c["check_name"] for c in body["checks"]} == {"finiteness", "shape_conformance"}

    # the asset's quality_score is updated
    assert client.get("/api/data/assets/p1").json()["quality_score"] == 100


def test_quality_check_flags_mass_drift(client):
    client.post("/api/data/assets", json=_asset_payload(shape="[4, 4, 4]"))
    r = client.post(
        "/api/data/quality/check",
        json={"asset_id": "p1", "data": _cube(1.05), "mass_field": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "warning"
    mass = [c for c in body["checks"] if c["check_name"] == "mass_conservation"][0]
    assert mass["passed"] is False


def test_quality_check_shape_mismatch(client):
    client.post("/api/data/assets", json=_asset_payload(shape="[4, 4, 4]"))
    r = client.post(
        "/api/data/quality/check",
        json={"asset_id": "p1", "data": _cube(0.0, shape=(3, 4, 4))},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    shape_check = [c for c in body["checks"] if c["check_name"] == "shape_conformance"][0]
    assert shape_check["passed"] is False


def test_quality_reports_endpoint(client):
    client.post("/api/data/assets", json=_asset_payload())
    client.post("/api/data/quality/check", json={"asset_id": "p1", "data": _cube(0.0)})
    client.post(
        "/api/data/quality/check", json={"asset_id": "p1", "data": _cube(1.05), "mass_field": True}
    )

    r = client.get("/api/data/quality/p1/reports")
    assert r.status_code == 200
    reports = r.json()
    assert len(reports) == 2
    # most recent first
    assert reports[0]["status"] == "warning"
    assert reports[1]["status"] == "passed"


def test_quality_check_missing_asset_returns_404(client):
    r = client.post("/api/data/quality/check", json={"asset_id": "nope", "data": _cube(0.0)})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Registration in the real platform app
# ---------------------------------------------------------------------------


def test_router_registered_in_main_app():
    from backend.main import app as platform_app

    paths = {route.path for route in platform_app.routes}
    assert "/api/data/assets" in paths
    assert "/api/data/assets/{asset_id}" in paths
    assert "/api/data/assets/{asset_id}/metadata" in paths
    assert "/api/data/assets/{asset_id}/lineage" in paths
    assert "/api/data/quality/check" in paths
    assert "/api/data/quality/{asset_id}/reports" in paths

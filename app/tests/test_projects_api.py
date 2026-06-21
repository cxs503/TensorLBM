"""Tests for the TensorLBM Project/Case management API."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_db(tmp_path, monkeypatch):
    """Redirect the SQLite DB to a temporary directory per test."""
    monkeypatch.setenv("TENSORLBM_OUTPUT_ROOT", str(tmp_path))
    # Force the module to re-compute _DB_PATH using the new env var
    from backend.routers import projects as proj_mod
    proj_mod._OUTPUT_ROOT = tmp_path
    proj_mod._DB_PATH = tmp_path / "projects.db"
    yield
    # Cleanup handled by tmp_path fixture


# ---------------------------------------------------------------------------
# Projects CRUD
# ---------------------------------------------------------------------------

class TestProjectsCRUD:
    def test_list_empty(self, client):
        r = client.get("/api/projects/")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_and_list(self, client):
        payload = {"name": "Test Project", "description": "desc", "owner": "alice", "tags": ["lbm", "marine"]}
        r = client.post("/api/projects/", json=payload)
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "Test Project"
        assert data["owner"] == "alice"
        assert "lbm" in data["tags"]
        assert "id" in data

        # List should now contain one project
        lst = client.get("/api/projects/").json()
        assert len(lst) == 1
        assert lst[0]["name"] == "Test Project"

    def test_get_project(self, client):
        pid = client.post("/api/projects/", json={"name": "Proj A"}).json()["id"]
        r = client.get(f"/api/projects/{pid}")
        assert r.status_code == 200
        assert r.json()["name"] == "Proj A"

    def test_get_project_not_found(self, client):
        r = client.get("/api/projects/nonexistent")
        assert r.status_code == 404

    def test_update_project(self, client):
        pid = client.post("/api/projects/", json={"name": "Old Name"}).json()["id"]
        r = client.put(f"/api/projects/{pid}", json={"name": "New Name", "tags": ["updated"]})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "New Name"
        assert "updated" in d["tags"]

    def test_delete_project(self, client):
        pid = client.post("/api/projects/", json={"name": "To Delete"}).json()["id"]
        r = client.delete(f"/api/projects/{pid}")
        assert r.status_code == 204
        assert client.get(f"/api/projects/{pid}").status_code == 404


# ---------------------------------------------------------------------------
# Cases CRUD
# ---------------------------------------------------------------------------

class TestCasesCRUD:
    def _make_project(self, client, name="My Project"):
        return client.post("/api/projects/", json={"name": name}).json()["id"]

    def test_list_cases_empty(self, client):
        pid = self._make_project(client)
        r = client.get(f"/api/projects/{pid}/cases")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_and_list_cases(self, client):
        pid = self._make_project(client)
        payload = {"name": "Re100 Cylinder", "scenario": "cylinder_flow", "description": "baseline"}
        r = client.post(f"/api/projects/{pid}/cases", json=payload)
        assert r.status_code == 201
        d = r.json()
        assert d["name"] == "Re100 Cylinder"
        assert d["scenario"] == "cylinder_flow"
        assert d["status"] == "draft"

        cases = client.get(f"/api/projects/{pid}/cases").json()
        assert len(cases) == 1

    def test_get_case(self, client):
        pid = self._make_project(client)
        cid = client.post(f"/api/projects/{pid}/cases", json={"name": "CaseA"}).json()["id"]
        r = client.get(f"/api/projects/{pid}/cases/{cid}")
        assert r.status_code == 200
        assert r.json()["name"] == "CaseA"

    def test_update_case_status(self, client):
        pid = self._make_project(client)
        cid = client.post(f"/api/projects/{pid}/cases", json={"name": "C"}).json()["id"]
        r = client.put(f"/api/projects/{pid}/cases/{cid}", json={"status": "completed", "job_id": "abc123"})
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "completed"
        assert d["job_id"] == "abc123"

    def test_delete_case(self, client):
        pid = self._make_project(client)
        cid = client.post(f"/api/projects/{pid}/cases", json={"name": "Del"}).json()["id"]
        r = client.delete(f"/api/projects/{pid}/cases/{cid}")
        assert r.status_code == 204
        assert client.get(f"/api/projects/{pid}/cases/{cid}").status_code == 404

    def test_case_not_found_wrong_project(self, client):
        pid = self._make_project(client)
        cid = client.post(f"/api/projects/{pid}/cases", json={"name": "X"}).json()["id"]
        pid2 = self._make_project(client, "Second")
        r = client.get(f"/api/projects/{pid2}/cases/{cid}")
        assert r.status_code == 404

    def test_cases_deleted_with_project(self, client):
        """Cascade: deleting a project removes its cases."""
        pid = self._make_project(client)
        client.post(f"/api/projects/{pid}/cases", json={"name": "X"})
        client.delete(f"/api/projects/{pid}")
        # Project gone – listing cases should 404
        r = client.get(f"/api/projects/{pid}/cases")
        assert r.status_code == 404

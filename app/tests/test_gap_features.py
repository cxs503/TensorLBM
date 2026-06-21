"""Tests for the five PowerFlow/XFlow gap-closure features:

1. Y+ wall-distance calculator  (POST /api/preprocess/yplus)
2. Probe-point time history      (POST /api/postprocess/probe-history)
3. Time-averaged field stats     (GET  /api/postprocess/time-average/{id})
4. Case clone                    (POST /api/projects/{pid}/cases/{cid}/clone)
5. Parametric sensitivity study  (POST /api/solve/parametric-study)
"""
from __future__ import annotations

import json

# ===========================================================================
# 1. Y+ Wall-Distance Calculator
# ===========================================================================

class TestYPlusCalculator:
    def test_flat_plate_basic(self, client):
        body = {
            "re": 1e5,
            "u_ms": 10.0,
            "l_m": 1.0,
            "nu_m2s": 1e-4,
            "target_yplus": 1.0,
            "n_cells": 200,
            "geometry": "flat_plate",
        }
        r = client.post("/api/preprocess/yplus", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "delta_y_m" in data
        assert "delta_y_lbm" in data
        assert "u_tau_ms" in data
        assert "c_f" in data
        assert data["c_f"] > 0
        assert data["u_tau_ms"] > 0
        assert data["delta_y_m"] > 0
        assert data["reynolds_number"] == 1e5

    def test_channel_geometry(self, client):
        body = {
            "re": 5000.0,
            "u_ms": 1.0,
            "l_m": 0.1,
            "nu_m2s": 1e-5,
            "target_yplus": 5.0,
            "n_cells": 100,
            "geometry": "channel",
        }
        r = client.post("/api/preprocess/yplus", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["geometry"] == "channel"
        assert d["delta_y_lbm"] > 0

    def test_invalid_geometry(self, client):
        body = {
            "re": 1000.0,
            "u_ms": 1.0,
            "l_m": 1.0,
            "nu_m2s": 1e-5,
            "target_yplus": 1.0,
            "n_cells": 100,
            "geometry": "unknown_shape",
        }
        r = client.post("/api/preprocess/yplus", json=body)
        assert r.status_code == 422

    def test_cells_inside_bl_positive(self, client):
        body = {
            "re": 1e6,
            "u_ms": 20.0,
            "l_m": 2.0,
            "nu_m2s": 1.5e-5,
            "target_yplus": 1.0,
            "n_cells": 500,
            "geometry": "flat_plate",
        }
        r = client.post("/api/preprocess/yplus", json=body)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["cells_inside_bl"] >= 1
        assert d["bl_thickness_m"] > 0


# ===========================================================================
# 2. Probe-point time history
# ===========================================================================

class TestProbeHistory:
    def test_no_job(self, client):
        body = {
            "job_id": "nonexistent-job-id",
            "probes": [{"x_frac": 0.5, "y_frac": 0.5, "label": "centre"}],
        }
        r = client.post("/api/postprocess/probe-history", json=body)
        assert r.status_code == 404

    def test_schema_validation_empty_probes(self, client):
        # No probes is not invalid by schema (list can be empty), but no job
        body = {"job_id": "nojob", "probes": []}
        r = client.post("/api/postprocess/probe-history", json=body)
        # 404 because job doesn't exist (schema is valid)
        assert r.status_code == 404


# ===========================================================================
# 3. Time-averaged field statistics
# ===========================================================================

class TestTimeAverage:
    def test_no_job(self, client):
        r = client.get("/api/postprocess/time-average/nonexistent")
        assert r.status_code == 404

    def test_unknown_field(self, client):
        # Request against a non-existent job returns 404 before field validation
        r = client.get(
            "/api/postprocess/time-average/nonexistent?field=nonexistent_field"
        )
        assert r.status_code == 404


# ===========================================================================
# 4. Case Clone
# ===========================================================================

class TestCaseClone:
    def _create_project(self, client, name="CloneTestProject"):
        r = client.post("/api/projects/", json={"name": name, "description": ""})
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def _create_case(self, client, project_id, name="BaseCase"):
        body = {
            "name": name,
            "description": "test case",
            "scenario": "cylinder_flow",
            "config": {"nx": 200, "ny": 80, "re": 100.0},
        }
        r = client.post(f"/api/projects/{project_id}/cases", json=body)
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def test_clone_basic(self, client):
        pid = self._create_project(client)
        cid = self._create_case(client, pid)

        r = client.post(f"/api/projects/{pid}/cases/{cid}/clone", json={})
        assert r.status_code == 201, r.text
        data = r.json()
        assert "copy" in data["name"].lower() or data["name"] != "BaseCase"
        assert data["project_id"] == pid
        assert data["id"] != cid
        # Cloned case starts fresh
        assert data["workflow_stage"] == "draft"
        assert data["job_id"] is None

    def test_clone_with_custom_name(self, client):
        pid = self._create_project(client, "CloneTest2")
        cid = self._create_case(client, pid, "OriginalCase")

        body = {"name": "Re200 variant", "config_overrides": {"re": 200.0}}
        r = client.post(f"/api/projects/{pid}/cases/{cid}/clone", json=body)
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["name"] == "Re200 variant"
        # Config override should be applied
        cfg = data["config"] if isinstance(data["config"], dict) else json.loads(data["config"])
        assert cfg.get("re") == 200.0

    def test_clone_nonexistent_case(self, client):
        pid = self._create_project(client, "CloneTest3")
        r = client.post(
            f"/api/projects/{pid}/cases/nonexistent-case-id/clone", json={}
        )
        assert r.status_code == 404

    def test_clone_inherits_scenario(self, client):
        pid = self._create_project(client, "CloneTest4")
        cid = self._create_case(client, pid, "ScenarioCase")
        r = client.post(f"/api/projects/{pid}/cases/{cid}/clone", json={})
        assert r.status_code == 201, r.text
        assert r.json()["scenario"] == "cylinder_flow"


# ===========================================================================
# 5. Parametric Sensitivity Study
# ===========================================================================

class TestParametricStudy:
    _BASE_CONFIG = {
        "nx": 50, "ny": 30, "u_in": 0.08, "re": 100.0,
        "n_steps": 10, "output_interval": 5, "device": "cpu",
    }

    def test_submit_re_sweep(self, client):
        body = {
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "parameter": "re",
            "values": [50.0, 100.0, 200.0],
        }
        r = client.post("/api/solve/parametric-study", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["solver_type"] == "cylinder_flow"
        assert data["parameter"] == "re"
        assert len(data["job_ids"]) == 3
        assert len(data["study_group"]) == 12

    def test_invalid_solver_type(self, client):
        body = {
            "solver_type": "unknown_solver",
            "base_config": self._BASE_CONFIG,
            "parameter": "re",
            "values": [50.0, 100.0],
        }
        r = client.post("/api/solve/parametric-study", json=body)
        assert r.status_code == 422

    def test_disallowed_parameter(self, client):
        body = {
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "parameter": "device",  # not in _ALLOWED_PARAMS
            "values": [1.0, 2.0],
        }
        r = client.post("/api/solve/parametric-study", json=body)
        assert r.status_code == 422

    def test_too_few_values(self, client):
        body = {
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "parameter": "re",
            "values": [100.0],  # only 1, minimum is 2
        }
        r = client.post("/api/solve/parametric-study", json=body)
        assert r.status_code == 422

    def test_study_group_consistent(self, client, job_manager):
        """All jobs in a study must share the same study_group."""
        body = {
            "solver_type": "lid_driven_cavity",
            "base_config": {
                "nx": 30, "ny": 30, "u_lid": 0.1, "re": 100.0,
                "n_steps": 10, "output_interval": 5, "device": "cpu",
            },
            "parameter": "re",
            "values": [100.0, 200.0],
        }
        r = client.post("/api/solve/parametric-study", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        study_group = data["study_group"]
        for jid in data["job_ids"]:
            job = job_manager.get_job(jid)
            assert job is not None
            assert job.config.get("study", {}).get("group") == study_group

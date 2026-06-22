"""Tests for newly platformized solver workflows."""
from __future__ import annotations


class TestRotatingCylinderAPI:
    def test_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/rotating-cylinder",
            json={
                "nx": 20,
                "ny": 10,
                "u_in": 0.05,
                "re": 60.0,
                "radius": 3.0,
                "spin_ratio": 0.8,
                "n_steps": 2,
                "output_interval": 1,
            },
        )
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()


class TestActuatorDiskAPI:
    def test_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/actuator-disk",
            json={
                "diameter": 8.0,
                "hub_diameter_ratio": 0.2,
                "rpm_lu": 0.01,
                "inflow_velocities": [0.02, 0.03],
                "nx": 40,
                "ny": 20,
                "nz": 20,
                "tau": 0.58,
                "n_steps": 2,
                "warmup_steps": 0,
            },
        )
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()


class TestPropellerWorkflowAPIs:
    def test_propeller_open_water_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/propeller-open-water",
            json={
                "inflow_velocities": [0.005],
                "rpm": 0.001,
                "nx": 40,
                "ny": 20,
                "nz": 20,
                "tau": 0.8,
                "n_revolutions": 1,
                "warmup_steps": 0,
            },
        )
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_ibm_propeller_submit_returns_job_id(self, client):
        r = client.post(
            "/api/solve/ibm-propeller",
            json={
                "inflow_velocities": [0.005],
                "rpm": 0.001,
                "nx": 40,
                "ny": 20,
                "nz": 20,
                "tau": 0.58,
                "n_revolutions": 1,
                "warmup_steps": 0,
                "marker_spacing": 2.0,
                "ibm_dt_substeps": 1,
            },
        )
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()


class TestExpandedParametricStudy:
    def test_rotating_cylinder_study(self, client):
        r = client.post(
            "/api/solve/parametric-study",
            json={
                "solver_type": "rotating_cylinder",
                "base_config": {
                    "nx": 20,
                    "ny": 10,
                    "u_in": 0.05,
                    "re": 60.0,
                    "radius": 3.0,
                    "spin_ratio": 0.5,
                    "n_steps": 2,
                    "output_interval": 1,
                    "device": "cpu",
                },
                "parameter": "spin_ratio",
                "values": [0.5, 1.0],
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["solver_type"] == "rotating_cylinder"
        assert data["parameter"] == "spin_ratio"
        assert len(data["job_ids"]) == 2

    def test_actuator_disk_study(self, client):
        r = client.post(
            "/api/solve/parametric-study",
            json={
                "solver_type": "actuator_disk",
                "base_config": {
                    "diameter": 8.0,
                    "hub_diameter_ratio": 0.2,
                    "rpm_lu": 0.01,
                    "inflow_velocities": [0.02, 0.03],
                    "nx": 40,
                    "ny": 20,
                    "nz": 20,
                    "tau": 0.58,
                    "smagorinsky_cs": 0.1,
                    "n_steps": 2,
                    "warmup_steps": 0,
                    "device": "cpu",
                },
                "parameter": "diameter",
                "values": [8.0, 10.0],
            },
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["job_ids"]) == 2

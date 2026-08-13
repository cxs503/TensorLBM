"""Contract tests for the additive marine preflight API.

These tests deliberately exercise only the backend contract: no solver run,
geometry upload, or frontend dependency is required.
"""
from __future__ import annotations


def _valid_ship_case() -> dict:
    return {
        "case_name": "dtmb-5415-design-speed",
        "vessel_type": "surface_ship",
        "length_between_perpendiculars_m": 142.0,
        "beam_m": 19.0,
        "draft_m": 6.2,
        "design_speed_ms": 12.0,
        "water": {"density_kg_m3": 1025.0, "kinematic_viscosity_m2_s": 1.05e-6},
        "mesh": {"cells_per_length": 160, "domain_length_factor": 3.0},
    }


class TestMarinePreflightContracts:
    def test_valid_ship_case_returns_versioned_decision_contract(self, client):
        response = client.post("/api/marine/preflight", json=_valid_ship_case())

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["contract_version"] == "marine-preflight/v1"
        assert body["case_name"] == "dtmb-5415-design-speed"
        assert body["decision"] == "ready"
        assert body["blocking_issues"] == []
        assert body["derived"]["reynolds_number"] > 0
        assert body["derived"]["froude_number"] > 0
        assert body["resource_estimate"]["total_cells"] > 0
        assert {check["code"] for check in body["checks"]} >= {
            "geometry_aspect_ratio", "lattice_mach", "mesh_resolution", "domain_extent",
        }

    def test_unsafe_lattice_speed_is_blocking_and_never_claims_ready(self, client):
        case = _valid_ship_case()
        case["numerics"] = {"lattice_inlet_velocity": 0.25}

        response = client.post("/api/marine/preflight", json=case)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["decision"] == "blocked"
        assert any(issue["code"] == "lattice_mach" for issue in body["blocking_issues"])

    def test_invalid_dimensions_are_rejected_by_ship_case_schema(self, client):
        case = _valid_ship_case()
        case["beam_m"] = -1.0

        response = client.post("/api/marine/preflight", json=case)

        assert response.status_code == 422

    def test_submarine_case_accepts_depth_and_reports_warning_for_missing_clearance(self, client):
        case = _valid_ship_case()
        case["vessel_type"] = "submarine"
        case["operating_depth_m"] = 30.0
        case["mesh"] = {"cells_per_length": 160, "domain_length_factor": 3.0}

        response = client.post("/api/marine/preflight", json=case)

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["decision"] in {"ready", "review"}
        assert any(check["code"] == "submergence_clearance" for check in body["checks"])


def test_marine_result_contract_requires_traceability_fields():
    from backend.schemas.marine import MarinePreflightResult

    result = MarinePreflightResult.model_validate({
        "contract_version": "marine-preflight/v1",
        "case_name": "case",
        "decision": "ready",
        "checks": [],
        "blocking_issues": [],
        "warnings": [],
        "derived": {"reynolds_number": 1.0, "froude_number": 0.1},
        "resource_estimate": {"total_cells": 1, "distribution_count": 19, "memory_mb": 0.1},
    })
    assert result.contract_version == "marine-preflight/v1"

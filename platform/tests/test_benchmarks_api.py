"""Tests for benchmark endpoints (smoke-runs using fast=True / tiny grids).

Marked as ``slow`` and opt-in via ``PLATFORM_SLOW_TESTS=1`` because each
benchmark spins up several LBM simulations.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PLATFORM_SLOW_TESTS") != "1",
    reason="Benchmark tests opt-in via PLATFORM_SLOW_TESTS=1",
)


def test_marine_benchmark_single_case(client, waiter):
    """Run the cheapest marine case (cylinder) only, in fast mode."""
    r = client.post(
        "/api/benchmarks/marine",
        json={"cases": ["cylinder"], "fast": True, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=600.0)
    assert final["status"] == "completed", final.get("error")
    assert final["result"].get("cylinder") == "ok"


def test_marine_benchmark_suboff_case(client, waiter):
    """Run SUBOFF resistance case and verify the <3% target is met."""
    r = client.post(
        "/api/benchmarks/marine",
        json={"cases": ["suboff"], "fast": True, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=180.0)
    assert final["status"] == "completed", final.get("error")
    suboff = final["result"].get("suboff")
    assert isinstance(suboff, dict)
    assert suboff["name"] == "suboff_resistance"
    assert suboff["target_met"] is True
    assert float(suboff["final_error_pct"]) <= 3.0


def test_marine_benchmark_geometry_library_case(client, waiter):
    """Run CAD-library consistency case and verify all checks pass."""
    r = client.post(
        "/api/benchmarks/marine",
        json={"cases": ["geometry_library"], "fast": True, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=180.0)
    assert final["status"] == "completed", final.get("error")
    geometry = final["result"].get("geometry_library")
    assert isinstance(geometry, dict)
    assert geometry["name"] == "marine_geometry_library"
    assert geometry["ship_ok"] is True
    assert geometry["cb_order_ok"] is True
    assert geometry["suboff_ok"] is True
    assert geometry["all_ok"] is True


def test_multiphase_benchmark(client, waiter):
    r = client.post(
        "/api/benchmarks/multiphase",
        json={"fast": True, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=600.0)
    assert final["status"] == "completed", final.get("error")


def test_ghia_benchmark(client, waiter):
    r = client.post(
        "/api/benchmarks/ghia",
        json={"nx": 32, "re": 100, "n_steps": 200, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=300.0)
    assert final["status"] == "completed", final.get("error")
    assert final["result"]["re"] == 100


def test_mlups_benchmark(client, waiter):
    r = client.post(
        "/api/benchmarks/mlups",
        json={"sizes": [32, 48], "steps": 20, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=300.0)
    assert final["status"] == "completed", final.get("error")
    results = final["result"]["results"]
    assert len(results) == 2
    assert {r["size"] for r in results} == {32, 48}
    for r in results:
        assert r["mlups"] > 0


def test_porous_benchmark(client, waiter):
    r = client.post(
        "/api/benchmarks/porous",
        json={"fast": True, "device": "cpu"},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    final = waiter(job_id, timeout=600.0)
    assert final["status"] == "completed", final.get("error")

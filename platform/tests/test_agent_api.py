"""Tests for the conversational LLM-agent endpoints (`/api/agent/*`).

These tests cover the offline (rule-based) path: no ``TENSORLBM_LLM_API_KEY``
is set in the environment, so the agent uses its deterministic intent
parser and built-in summary text.  This lets the tests run reliably on
any CI runner without network access.

Default tests do **not** actually launch LBM simulations – instead they
monkey-patch the agent's ``submit_*`` tools to return synthetic job
descriptors.  Set ``PLATFORM_SLOW_TESTS=1`` to also exercise the full
chat → submit → solver path end-to-end.
"""
from __future__ import annotations

import os
import time

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """Force the rule-based offline path for every test in this file."""
    monkeypatch.delenv("TENSORLBM_LLM_API_KEY", raising=False)


@pytest.fixture()
def stub_submitters(monkeypatch):
    """Replace every ``submit_*`` tool with a stub that doesn't run a solver.

    This keeps the fast test-path purely a pre-processing / chat exercise:
    no torch tensors are allocated and no background threads are started.
    The stub still returns the same shape of response (``job_id``,
    ``name``, ``job_type``, ``config``) so the agent's summary code paths
    run unchanged.
    """
    from backend import agent_core  # type: ignore[import-not-found]

    counter = {"n": 0}

    for tool_name, tool_obj in list(agent_core._TOOLS.items()):
        if not tool_name.startswith("submit_"):
            continue

        def _stub(_name=tool_name, **kwargs):
            counter["n"] += 1
            job_type = _name.removeprefix("submit_")
            return {
                "job_id": f"stub-{counter['n']:04d}",
                "name": f"stub {job_type}",
                "job_type": job_type,
                "config": kwargs,
            }

        monkeypatch.setattr(tool_obj, "handler", _stub)
    return counter


def _chat(client, message: str, history=None) -> dict:
    r = client.post("/api/agent/chat", json={
        "message": message,
        "history": history or [],
    })
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Info / capabilities
# ---------------------------------------------------------------------------

def test_info(client):
    r = client.get("/api/agent/info")
    assert r.status_code == 200
    d = r.json()
    assert d["llm_enabled"] is False
    assert d["tools_count"] > 0
    assert "fallback" in d


def test_capabilities(client):
    r = client.get("/api/agent/capabilities")
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d["tools"], list)
    tool_names = [t["name"] for t in d["tools"]]
    for expected in (
        "submit_cylinder_flow",
        "submit_lid_driven_cavity",
        "submit_dam_break",
        "submit_sloshing_tank",
        "submit_ship_hull",
        "submit_pipeline_flow",
        "submit_turbulent_channel",
        "get_job_status",
        "list_recent_jobs",
        "analyze_job",
        "velocity_profile",
    ):
        assert expected in tool_names, expected
    assert "cylinder_flow" in d["scenarios"]


def test_openapi_includes_agent_routes(client):
    r = client.get("/openapi.json")
    paths = r.json()["paths"]
    for p in ("/api/agent/chat", "/api/agent/capabilities", "/api/agent/info"):
        assert p in paths, p


# ---------------------------------------------------------------------------
# Help / no-match behaviour
# ---------------------------------------------------------------------------

def test_help_intent(client):
    d = _chat(client, "What can you do?")
    assert "TensorLBM" in d["reply"] or "modelling" in d["reply"].lower() \
        or "modeling" in d["reply"].lower()
    assert d["actions"] == []
    assert d["suggestions"]
    assert d["used_llm"] is False


def test_chinese_help_intent(client):
    d = _chat(client, "你能帮我做什么？帮助")
    assert "TensorLBM" in d["reply"]
    assert d["intent"]["tool"] == "_help"


def test_no_match(client):
    d = _chat(client, "tell me a joke")
    assert d["actions"] == []
    assert d["intent"]["tool"] is None
    # Should still give a helpful nudge
    assert "scenario" in d["reply"].lower() or "help" in d["reply"].lower()


# ---------------------------------------------------------------------------
# List jobs
# ---------------------------------------------------------------------------

def test_list_jobs_intent(client):
    d = _chat(client, "show me all jobs")
    assert len(d["actions"]) == 1
    a = d["actions"][0]
    assert a["tool"] == "list_recent_jobs"
    assert "jobs" in a["result"]


# ---------------------------------------------------------------------------
# Scenario submission (using stub submitters)
# ---------------------------------------------------------------------------

def test_cylinder_intent(client, stub_submitters):
    """Intent parser routes to the cylinder flow tool with extracted Re."""
    d = _chat(client, "Run a cylinder flow Re=75 nx=30 n_steps=5")
    assert d["actions"], d
    a = d["actions"][0]
    assert a["tool"] == "submit_cylinder_flow"
    assert a["args"]["re"] == 75
    assert a["args"]["nx"] == 30
    assert a["args"]["n_steps"] == 5
    assert "job_id" in a["result"]
    assert d["suggestions"]


def test_chinese_scenario(client, stub_submitters):
    """Chinese keyword '圆柱绕流' maps to cylinder flow."""
    d = _chat(client, "用圆柱绕流做一个 Re=120 的算例 步数=5")
    assert d["actions"][0]["tool"] == "submit_cylinder_flow"
    assert d["actions"][0]["args"]["re"] == 120
    assert d["actions"][0]["args"]["n_steps"] == 5


def test_chinese_cavity(client, stub_submitters):
    """Chinese keyword '方腔' maps to lid-driven cavity."""
    d = _chat(client, "做一个方腔算例 Re=400")
    assert d["actions"][0]["tool"] == "submit_lid_driven_cavity"
    assert d["actions"][0]["args"]["re"] == 400


def test_dam_break_keyword(client, stub_submitters):
    d = _chat(client, "Run a dam-break n_steps=2")
    assert d["actions"][0]["tool"] == "submit_dam_break"


def test_lid_driven_cavity_keyword(client, stub_submitters):
    d = _chat(client, "Lid-driven cavity Re=100 n_steps=2")
    assert d["actions"][0]["tool"] == "submit_lid_driven_cavity"


def test_sloshing_keyword(client, stub_submitters):
    d = _chat(client, "Run a sloshing tank simulation")
    assert d["actions"][0]["tool"] == "submit_sloshing_tank"


def test_ship_hull_keyword(client, stub_submitters):
    d = _chat(client, "Simulate a Wigley ship hull at Re=300")
    assert d["actions"][0]["tool"] == "submit_ship_hull"
    assert d["actions"][0]["args"]["re"] == 300


def test_pipeline_keyword(client, stub_submitters):
    d = _chat(client, "Run a near-bed pipeline flow at Re=200")
    assert d["actions"][0]["tool"] == "submit_pipeline_flow"


def test_turbulent_channel_keyword(client, stub_submitters):
    d = _chat(client, "Turbulent channel at re_tau=180 n_steps=10")
    assert d["actions"][0]["tool"] == "submit_turbulent_channel"
    assert d["actions"][0]["args"]["re_tau"] == 180


# ---------------------------------------------------------------------------
# History references the most recent job
# ---------------------------------------------------------------------------

def test_status_references_previous_job(client, stub_submitters):
    """After a submission, 'status' resolves to that job_id via history."""
    first = _chat(client, "Run a cylinder flow nx=30 n_steps=2")
    jid = first["actions"][0]["result"]["job_id"]
    second = _chat(
        client,
        "what is the status?",
        history=[
            {"role": "user", "content": "Run a cylinder flow nx=30 n_steps=2"},
            {"role": "assistant",
             "content": first["reply"],
             "actions": first["actions"]},
        ],
    )
    # The status tool dispatches against the real job_manager; the stub
    # job_id won't be found there but the agent should still resolve
    # the reference and call the tool with the right argument.
    assert second["actions"], second
    a = second["actions"][0]
    assert a["tool"] == "get_job_status"
    assert a["args"]["job_id"] == jid


def test_analyze_references_previous_job(client, stub_submitters):
    first = _chat(client, "Run a dam-break n_steps=2")
    jid = first["actions"][0]["result"]["job_id"]
    second = _chat(
        client,
        "please summarize",
        history=[
            {"role": "user", "content": "Run a dam-break n_steps=2"},
            {"role": "assistant",
             "content": first["reply"],
             "actions": first["actions"]},
        ],
    )
    assert second["actions"][0]["tool"] == "analyze_job"
    assert second["actions"][0]["args"]["job_id"] == jid


# ---------------------------------------------------------------------------
# Safety caps (no submission needed)
# ---------------------------------------------------------------------------

def test_safety_caps_applied():
    """Hard caps in the agent's tool layer clip oversized inputs."""
    from backend import agent_core  # type: ignore[import-not-found]

    assert agent_core._clip(99999, 40, agent_core.MAX_GRID_2D) == agent_core.MAX_GRID_2D
    assert agent_core._clip(-5, 40, agent_core.MAX_GRID_2D) == 40
    assert agent_core._clip(123, 40, agent_core.MAX_GRID_2D) == 123
    assert agent_core.MAX_GRID_2D <= 1024
    assert agent_core.MAX_STEPS <= 200_000


# ---------------------------------------------------------------------------
# Slow end-to-end (only with PLATFORM_SLOW_TESTS=1)
# ---------------------------------------------------------------------------

def _wait_terminal(job_manager_mod, job_id, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = job_manager_mod.get_job(job_id)
        if job is not None:
            status = job.status.value
            if status in ("completed", "failed", "cancelled"):
                return job
        time.sleep(0.1)
    return None


@pytest.mark.skipif(
    os.environ.get("PLATFORM_SLOW_TESTS") != "1",
    reason="Solver execution opt-in via PLATFORM_SLOW_TESTS=1",
)
def test_cylinder_submission_runs(client, job_manager):
    """End-to-end: chat → real submit → solver completes."""
    d = _chat(
        client,
        "Run a cylinder flow at Re=50 with nx=40 ny=20 n_steps=5",
    )
    assert d["actions"][0]["tool"] == "submit_cylinder_flow"
    jid = d["actions"][0]["result"]["job_id"]
    job = _wait_terminal(job_manager, jid, timeout=120.0)
    assert job is not None
    assert job.status.value == "completed"


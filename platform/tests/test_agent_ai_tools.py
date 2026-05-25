"""Platform-level tests for the AI turbulence agent tools."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    monkeypatch.delenv("TENSORLBM_LLM_API_KEY", raising=False)


def _chat(client, message: str) -> dict:
    r = client.post("/api/agent/chat", json={"message": message, "history": []})
    assert r.status_code == 200, r.text
    return r.json()


def test_capabilities_include_ai_tools(client):
    r = client.get("/api/agent/capabilities")
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["tools"]]
    for expected in (
        "ai_generate_dataset",
        "ai_train_turbulence_model",
        "ai_list_models",
        "ai_run_pipeline",
    ):
        assert expected in names, expected


def test_intent_list_ai_models(client):
    d = _chat(client, "list ai models")
    actions = d.get("actions", [])
    assert actions, "expected at least one action"
    assert actions[0]["tool"] == "ai_list_models"


def test_intent_generate_dataset_routes(client):
    # Use the agent's parser directly to avoid kicking off an actual run.
    from backend import agent_core as ac  # type: ignore[import-not-found]
    intent = ac._parse_intent("generate dataset nx=24 ny=24", [])
    assert intent["tool"] == "ai_generate_dataset"
    assert intent["args"].get("nx") == 24
    assert intent["args"].get("ny") == 24


def test_intent_train_model_routes_chinese():
    from backend import agent_core as ac  # type: ignore[import-not-found]
    intent = ac._parse_intent("训练模型 epochs=15", [])
    assert intent["tool"] == "ai_train_turbulence_model"
    assert intent["args"].get("epochs") == 15


def test_intent_pipeline_routes_chinese():
    from backend import agent_core as ac  # type: ignore[import-not-found]
    intent = ac._parse_intent("跑一遍 AI 湍流闭环", [])
    assert intent["tool"] == "ai_run_pipeline"


def test_ai_list_models_empty_db(client, tmp_path, monkeypatch):
    """When no AI runs exist yet, ai_list_models replies gracefully."""
    from backend import agent_core as ac  # type: ignore[import-not-found]
    monkeypatch.setattr(ac, "_AI_WORKROOT", str(tmp_path / "ai"))
    d = _chat(client, "list ai models")
    assert d["actions"][0]["result"]["count"] == 0
    assert "No AI turbulence models" in d["reply"] or "no" in d["reply"].lower()

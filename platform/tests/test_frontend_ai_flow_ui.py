"""Frontend smoke checks for AI Flow transformer UI wiring."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_index_contains_ai_flow_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "panel-ai-flow" in html
    assert "aiFlowTrain" in html
    assert "aiFlowInfer" in html
    assert "aiFlowListModels" in html
    assert "aiFlowPollJob" in html
    assert "aiflow-history-chart" in html

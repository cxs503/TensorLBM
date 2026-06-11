"""Frontend smoke checks for orchestration/governance UI wiring."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_index_contains_orchestration_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "panel-orchestration" in html
    assert "orchLoadTemplates" in html
    assert "orchSubmitExperiment" in html
    assert "orchLoadKpis" in html
    assert "orchRunConfidenceGate" in html
    assert "orchRunActiveLearning" in html

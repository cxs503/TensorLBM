"""Frontend smoke checks for workflow operations and notification wiring."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_index_contains_workflow_ops_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "dashboard-live-metrics-result" in html
    assert "dashboard-auto-stop-result" in html
    assert "dashboard-hpc-result" in html
    assert "dashboard-timeline-table" in html
    assert "dashboard-notify-result" in html
    assert "dashboardLoadLiveMetrics" in html
    assert "dashboardApplyAutoStop" in html
    assert "dashboardSubmitHpc" in html
    assert "dashboardLoadTimeline" in html
    assert "dashboardSaveNotifications" in html

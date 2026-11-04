"""Frontend smoke checks for workflow operations and notification wiring."""
from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import pytest

_LEGACY_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

pytestmark = pytest.mark.skipif(
    not _LEGACY_FRONTEND.is_dir(),
    reason=(
        "legacy native-JS frontend (app/frontend) was deleted in 66efae5e "
        "'Remove old native-JS frontend, serve Vue3 dist from backend'; the Vue 3 "
        "app in frontend-vue/ has no equivalent assertions yet, so these checks "
        "target code that no longer exists"
    ),
)

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
    assert "cumulant_cylinder_flow" in html
    assert "sliding_mesh_rotor" in html
    assert "passive_scalar_transport" in html
    assert "cavitation_flow" in html
    assert "oscillating_airfoil" in html
    assert "conjugate_ht" in html
    assert "pp-advanced" in html
    assert "loadUnifiedMonitor" in html
    assert "runAdvancedEndpoint" in html

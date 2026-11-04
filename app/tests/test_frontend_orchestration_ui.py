"""Frontend smoke checks for orchestration/governance UI wiring."""
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
    assert "orchLoadStudySummary" in html
    assert "orchLoadSobol" in html
    assert "orchRunBayesianOpt" in html

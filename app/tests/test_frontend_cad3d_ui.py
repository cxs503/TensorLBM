"""Frontend smoke checks for CAD3D UI wiring."""
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


def test_index_contains_cad3d_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "id=\"cad3d-canvas\"" in html
    assert "id=\"cad3d-source-type\"" in html
    assert "id=\"cad3d-file-input\"" in html
    assert "cad3dCreateOrUpdate" in html
    assert "cad3dOnSourceTypeChange" in html
    assert "cad3dToggleWireframe" in html
    assert "cad3dToggleClip" in html

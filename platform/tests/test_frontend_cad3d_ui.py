"""Frontend smoke checks for CAD3D UI wiring."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_index_contains_cad3d_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "id=\"cad3d-canvas\"" in html
    assert "cad3dCreateOrUpdate" in html
    assert "cad3dToggleWireframe" in html
    assert "cad3dToggleClip" in html

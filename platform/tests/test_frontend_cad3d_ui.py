"""Frontend smoke checks for CAD3D UI wiring."""
from __future__ import annotations


def test_index_contains_cad3d_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    assert "id=\"cad3d-canvas\"" in html
    assert "cad3dCreateOrUpdate" in html
    assert "cad3dToggleWireframe" in html
    assert "cad3dToggleClip" in html

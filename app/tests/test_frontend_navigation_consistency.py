"""Frontend regression checks for navigation/state consistency refactor."""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_top_nav_uses_data_tab_binding(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    html = r.text
    nav_match = re.search(r'<div class="top-navbar">.*?</nav>', html, flags=re.S)
    assert nav_match is not None
    nav_html = nav_match.group(0)
    assert 'data-tab="geo3d"' in nav_html
    assert "onclick=" not in nav_html
    assert 'class="lang-btn" data-lang="en"' in html
    assert 'class="lang-btn" data-lang="zh"' in html


def test_tab_sequence_includes_geo3d_and_event_binding() -> None:
    core_js = (
        Path(__file__).resolve().parent.parent
        / "frontend"
        / "static"
        / "js"
        / "app_core.js"
    ).read_text(encoding="utf-8")
    assert "'geo3d'" in core_js
    assert "function bindTopNavEvents()" in core_js
    assert "TAB_ENTER_HANDLERS" in core_js


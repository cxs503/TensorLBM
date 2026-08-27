"""Tests for the ``GET /demo`` mount of ``demos/echo_slider.html``.

PR #245 follow-up: the single-file drag-echo demo page is served by the
platform FastAPI app itself, same-origin, so its relative ``/api/drag/echo/*``
calls work without the standalone launcher or a ``?api=`` override.  Checks
200 + ``text/html`` + key page markers, byte-identity with the file on disk,
a clear 404 (not a bare 500) when ``demos/`` is missing, and that ``/demo``
is not swallowed by the SPA catch-all.  Mirrors the app import conventions
of ``tests/test_drag_echo_api.py`` (skipped where fastapi or the platform
deps are not installed, e.g. in CI which installs only ``.[dev]``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

# Make the ``backend`` package (under ``app/``) importable — mirrors
# tests/test_drag_echo_api.py.  ``src/`` is on sys.path via pythonpath=src.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

main_mod = pytest.importorskip("backend.main", reason="platform app deps unavailable")

_PAGE = _REPO_ROOT / "demos" / "echo_slider.html"
# What makes the page work when served from the app itself: the inline
# script, the apiBase field (empty default = same-origin relative calls) and
# the verdict banner carrying the honesty contract.
_PAGE_MARKERS = ("<script>", 'id="apiBase"', 'id="verdict"', "/api/drag/echo/health")


class TestDemoMount:
    def test_page_served_200_html(self) -> None:
        r = TestClient(main_mod.app).get("/demo")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        for marker in _PAGE_MARKERS:
            assert marker in r.text, f"missing marker {marker!r}"

    def test_served_body_is_the_demos_file(self) -> None:
        r = TestClient(main_mod.app).get("/demo")
        assert r.content == _PAGE.read_bytes()

    def test_missing_page_is_clear_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main_mod, "_DEMO_PAGE", _REPO_ROOT / "demos" / "nope.html")
        r = TestClient(main_mod.app).get("/demo")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]
        assert "demos" in r.json()["detail"]

    def test_registered_before_spa_catch_all(self) -> None:
        """/demo must be the only exact match and precede the catch-all route."""
        paths = [getattr(route, "path", "") for route in main_mod.app.routes]
        assert paths.count("/demo") == 1
        assert paths.index("/demo") < paths.index("/{full_path:path}")

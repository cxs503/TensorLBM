"""Tests for the drag-echo demo deliverables under ``demos/``.

- ``echo_slider.html``: parses with the stdlib parser, targets the PR #241
  endpoints, contains no external URLs outside comments (offline lab), and
  stays under 100 KB.
- ``serve_demo.py``: imports standalone (no package context) and serves the
  page over plain HTTP with permissive CORS headers.

Everything is stdlib-only and touches no GPU, no /nfs artifacts and no
network beyond a loopback ephemeral port.
"""

from __future__ import annotations

import html.parser
import importlib.util
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEMOS = REPO / "demos"
PAGE = DEMOS / "echo_slider.html"
LAUNCHER = DEMOS / "serve_demo.py"

EXPECTED_ENDPOINTS = (
    "/api/drag/echo/params",
    "/api/drag/echo/sweep",
    "/api/drag/echo/health",
)


class _TagCollector(html.parser.HTMLParser):
    """Structural check: records tags so the test can assert on shape."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)


def _load_launcher() -> object:
    spec = importlib.util.spec_from_file_location("serve_demo_under_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDemoPage:
    def test_html_parses_and_has_expected_structure(self) -> None:
        parser = _TagCollector()
        parser.feed(PAGE.read_text(encoding="utf-8"))
        parser.close()  # raises on grossly malformed input
        for tag, minimum in (
            ("html", 1),
            ("head", 1),
            ("body", 1),
            ("script", 1),
            ("svg", 2),
            ("input", 6),
            ("select", 2),
        ):
            # slider inputs are built by JS; 6 static ones remain (Re range,
            # point count, api base)
            assert parser.tags.count(tag) >= minimum, f"expected >= {minimum} <{tag}>"

    def test_targets_pr241_endpoints(self) -> None:
        text = PAGE.read_text(encoding="utf-8")
        for endpoint in EXPECTED_ENDPOINTS:
            assert endpoint in text, f"missing endpoint {endpoint}"

    def test_no_external_urls_outside_comments(self) -> None:
        text = PAGE.read_text(encoding="utf-8")
        stripped = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        hits = re.findall(r"https?://[^\s\"'<>]+", stripped)
        assert hits == [], f"external URL references found: {hits}"

    def test_page_size_under_100kb(self) -> None:
        assert PAGE.stat().st_size < 100 * 1024


class TestDemoLauncher:
    def test_importable_standalone(self) -> None:
        """``python -c`` import check — the launcher must not need a package."""
        code = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('serve_demo', sys.argv[1]); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "assert callable(m.serve) and callable(m.main)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code, str(LAUNCHER)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr

    def test_serves_page_with_cors_on_ephemeral_port(self) -> None:
        module = _load_launcher()
        httpd = module.serve(port=0)
        assert isinstance(httpd.server_address, tuple)
        port = int(httpd.server_address[1])
        assert port > 0
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(base + "/echo_slider.html", timeout=5) as resp:
                body = resp.read()
                assert resp.status == 200
                assert resp.headers["Access-Control-Allow-Origin"] == "*"
                assert b"drag echo" in body.lower()
            req = urllib.request.Request(base + "/echo_slider.html", method="OPTIONS")
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 204  # CORS preflight
            try:
                urllib.request.urlopen(base + "/nope.html", timeout=5)
                raise AssertionError("missing file should 404")
            except urllib.error.HTTPError as err:
                assert err.code == 404
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_lan_ip_is_ipv4_shape(self) -> None:
        module = _load_launcher()
        ip = module.lan_ip()
        assert re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ip), ip

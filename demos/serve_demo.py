#!/usr/bin/env python3
"""Static launcher for the TensorLBM drag-echo demo page (stdlib only).

Two run modes, detailed in demos/README.md: (a) DEV — start the PR #241
backend (branch exp/b4-echo) separately, then `python demos/serve_demo.py
8765` and open the printed URL, which carries ?api= pointing at that
backend; (b) POST-MERGE — serve echo_slider.html from the SAME origin as
the FastAPI app (documented in the README, not implemented here; no app/
changes in this branch)."""

from __future__ import annotations

import argparse
import socket
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent


class DemoHandler(SimpleHTTPRequestHandler):
    """Static files from demos/ with permissive CORS (dev-mode cross-origin)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(DEMO_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self.end_headers()


def lan_ip() -> str:
    """Best-effort LAN address for the printed ?api= hint (offline-safe)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no traffic leaves the socket
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"


def serve(port: int = 8765) -> ThreadingHTTPServer:
    """Create (but do not start) the demo server — also used by the tests."""
    return ThreadingHTTPServer(("0.0.0.0", port), DemoHandler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("port", nargs="?", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    args = ap.parse_args(argv)
    httpd = serve(args.port)
    page = "http://localhost:%d/echo_slider.html" % args.port
    hint = page + "?api=http://%s:8000" % lan_ip()
    print("serving demos/ at %s (Ctrl-C to stop)" % page)
    print("open with a backend on this machine:  %s" % hint)
    if args.open:
        webbrowser.open(hint)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

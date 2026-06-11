"""Tests for the /ws WebSocket endpoint.

These tests are skipped by default because Starlette's ``TestClient``
WebSocket support can hang on some CI configurations.  Enable them with
``PLATFORM_WS_TESTS=1``.
"""
from __future__ import annotations

import json
import os

import pytest

_RUN_WS = os.environ.get("PLATFORM_WS_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_WS,
    reason="WebSocket tests opt-in via PLATFORM_WS_TESTS=1",
)


def test_websocket_init_message(client):
    """On connect the server pushes an ``init`` message with the current job list."""
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["type"] == "init"
        assert isinstance(data["jobs"], list)

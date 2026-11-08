#!/usr/bin/env python3
"""Publish the mission report to the TensorPlatform reports API.

Credentials and endpoint come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``); nothing is hard-coded.
Payload is ``/tmp/report.json`` built by
``scripts/make_platform_report_d3q27.py``.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    tokens = _post(f"{BASE}/auth/login", {"username": USER, "password": PASS})
    token = (
        tokens.get("access_token")
        or tokens.get("token")
        or tokens.get("data", {}).get("access_token")
        or tokens.get("data", {}).get("token")
    )
    if not token:
        print("login response without token:", list(tokens)[:6], file=sys.stderr)
        return 2

    with open("/tmp/report.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    res = _post(f"{BASE}/tensor/reports", payload, token=token)
    ident = res.get("id") or res.get("report_id") or res.get("data", {}).get("id")
    print("report published, id =", ident)
    print(json.dumps({k: v for k, v in res.items() if k != "markdown"}, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:  # pragma: no cover
        print("HTTP", exc.code, exc.read().decode("utf-8", "replace")[:800], file=sys.stderr)
        raise SystemExit(1) from exc
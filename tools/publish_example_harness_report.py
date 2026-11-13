#!/usr/bin/env python3
"""Publish the mission report for XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN.

Reads the JSON payload from /tmp/report.json (built by the caller from the
run artifacts and figures), logs in with the agent credentials from the
environment and POSTs it to the platform reports API.
"""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        raise SystemExit(f"login failed: {tok}")
    with open("/tmp/report.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    out = post("/tensor/reports", payload, token)
    print(json.dumps({k: v for k, v in out.items() if k in ("id", "status", "mission_id", "title")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
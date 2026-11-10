#!/usr/bin/env python
"""Publish the P154W3-N11 report to the platform API.

Credentials come from the environment (``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS``, endpoint ``TENSOR_API_BASE``); nothing is hard-coded and
no secret is written to disk.  The payload is built by
``scripts/p154w3_n11_publish.py``.
"""

from __future__ import annotations

import json
import os
import urllib.request


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def main() -> None:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    tok = _post(f"{base}/auth/login", {"username": user, "password": password})
    token = tok.get("access_token") or tok.get("token")
    if not token:
        raise SystemExit(f"login did not return a token: {list(tok)[:6]}")
    with open("/tmp/report_p154w3_n11.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    res = _post(f"{base}/tensor/reports", payload, token)
    print(json.dumps(res)[:600])


if __name__ == "__main__":
    main()
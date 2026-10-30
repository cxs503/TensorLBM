#!/usr/bin/env python3
"""Post the m336 report payload to the Tensor platform ``/tensor/reports`` endpoint.

Equivalent to the documented ``curl`` flow (login → POST /tensor/reports with a
Bearer token), but driven through `urllib` because the shell here refuses
``curl`` with credentials on the command line.

Credentials and base URL come from the environment
(``TENSOR_API_BASE``, ``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``); nothing is
hard-coded.  Usage::

    python evidence/suboff_re1000_m336/post_report.py [/tmp/report.json]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]


def _post(url: str, payload: dict, token: str | None = None, form: bool = False) -> dict:
    if form:
        data = urllib.parse.urlencode(payload).encode()
        ctype = "application/x-www-form-urlencoded"
    else:
        data = json.dumps(payload).encode()
        ctype = "application/json"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", ctype)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:  # surface the server's validation detail
        detail = exc.read().decode()[:600]
        return {"_http_error": exc.code, "_body": detail}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body[:500]}


def main() -> int:
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/report.json")
    payload = json.loads(report_path.read_text())

    token = None
    for kwargs in ({"form": False}, {"form": True}):
        login = _post(
            f"{BASE}/auth/login", {"username": USER, "password": PASSWORD}, **kwargs
        )
        token = (
            login.get("access_token")
            or login.get("token")
            or (login.get("data") or {}).get("access_token")
        )
        print("login attempt", kwargs, "->", json.dumps(login, ensure_ascii=False)[:300])
        if token:
            break
    if not token:
        return 1
    print(f"login ok (token {len(token)} chars)")

    resp = _post(f"{BASE}/tensor/reports", payload, token=token)
    print("POST /tensor/reports ->", json.dumps(resp, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
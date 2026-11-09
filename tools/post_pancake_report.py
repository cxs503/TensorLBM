"""Post the IN69-N07 report payload to the platform report API.

Credentials come from the environment (``TENSOR_API_BASE``,
``TENSOR_AGENT_USER``, ``TENSOR_AGENT_PASS``) so nothing secret is ever written
to the repository or to a command line.

Usage::

    python3 tools/post_pancake_report.py /tmp/report.json
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def post(url: str, payload: dict[str, object], token: str | None = None) -> dict[str, object]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "/tmp/report.json"
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    payload = json.loads(open(path, encoding="utf-8").read())

    login_paths = ("/auth/login", "/api/v1/auth/login", "/tensor/auth/login")
    token = None
    for suffix in login_paths:
        try:
            out = post(base + suffix, {"username": user, "password": password})
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        token = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
        if token:
            print(f"login via {suffix}")
            break
    if not token:
        print("login failed on all known endpoints", file=sys.stderr)
        return 1

    out = post(base + "/tensor/reports", payload, token)
    print(json.dumps(out)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
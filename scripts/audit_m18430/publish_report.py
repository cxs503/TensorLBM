"""Publish the m18430 audit report to the platform report ledger (POST /tensor/reports)."""
from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def login() -> str:
    body = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d.get("access_token") or d.get("token")


def post(path: str, payload: dict, token: str) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> int:
    payload = json.loads(open("/tmp/report.json").read())
    token = login()
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
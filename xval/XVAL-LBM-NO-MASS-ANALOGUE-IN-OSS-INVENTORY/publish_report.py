"""Publish /tmp/report.json to the platform report API (credentials from env)."""
from __future__ import annotations

import json
import os
import urllib.request


def post(path: str, body: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://localhost:8000/api/v1").rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    pwd = os.environ["TENSOR_AGENT_PASS"]
    tok = post(f"{base}/auth/login", {"username": user, "password": pwd})
    token = tok.get("access_token") or tok.get("token") or ""
    payload = json.loads(open("/tmp/report.json", encoding="utf-8").read())
    res = post(f"{base}/tensor/reports", payload, token)
    print(json.dumps({k: v for k, v in res.items() if k in
                      ("id", "report_id", "status", "title", "mission_id")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())